#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License
"""Generate linker .def files for the HRR Windows proxy DLL.

Reads the export table from the real amdhip64_7.dll and produces TWO files:

  amdhip64_7.def       — proxy DLL exports:
                           intercepted functions listed bare (implemented in proxy)
                           all other functions forwarded to amdhip64_7_orig.dll

  amdhip64_7_orig.def  — companion def for generating a stub import library.
                         MSVC requires amdhip64_7_orig.lib at link time to emit
                         PE forwarder entries; lib.exe creates it from this def
                         with no actual DLL needed on the build machine.

Data exports (e.g. AMD_CPU_AFFINITY, HIP_VISIBLE_DEVICES — exported char[]
strings) are silently omitted from both files because MSVC does not support the
  name = other_dll.name
forwarder syntax for data symbols; using it triggers LNK2001.  Omitting them
is safe: applications read these values via getenv(), not GetProcAddress().

Usage:
    python gen_proxy_exports.py <path-to-real-amdhip64_7.dll> [proxy.def]

    The companion amdhip64_7_orig.def is written to the same directory as
    proxy.def (default: current directory).
"""

import struct
import sys
from typing import NamedTuple

# Functions implemented in hrr_proxy_win.c — these are NOT forwarded.
INTERCEPTED = {
    # Core runtime
    "hipInit",
    "hipMalloc",
    "hipFree",
    "hipMemcpy",
    "hipMemset",
    # Module / code-object path
    "hipModuleLoad",        # file-path loader used by hipBLASLt at runtime
    "hipModuleLoadData",    # in-memory loader
    "hipModuleUnload",
    "hipModuleGetFunction",
    "hipModuleLaunchKernel",
    "hipExtModuleLaunchKernel",
    # Fat-binary path (hipblaslt / composable_kernel embedded code objects)
    "__hipRegisterFatBinary",
    "__hipRegisterFunction",
    "hipLaunchKernel",
    # Synchronisation
    "hipDeviceSynchronize",
    "hipStreamSynchronize",
}

FORWARD_TARGET = "amdhip64_7_orig"

# PE section flags
_SCN_CNT_CODE = 0x00000020
_SCN_MEM_EXECUTE = 0x20000000


class _Section(NamedTuple):
    vaddr: int
    vsize: int
    raw_off: int
    raw_size: int
    characteristics: int


def _rva_to_offset(rva: int, sections: list) -> int:
    for s in sections:
        if s.vaddr <= rva < s.vaddr + max(s.vsize, s.raw_size):
            return rva - s.vaddr + s.raw_off
    raise ValueError(f"Cannot resolve RVA {hex(rva)}")


def _section_for_rva(rva: int, sections: list):
    for s in sections:
        if s.vaddr <= rva < s.vaddr + max(s.vsize, s.raw_size):
            return s
    return None


def read_pe_exports(dll_path: str):
    """Parse the PE export table.

    Returns:
        (func_names, data_names) — both sorted lists of strings.
        func_names: exports whose RVA falls in an executable section, or that
                    are already forwarders in the original DLL.
        data_names: exports whose RVA falls in a non-executable section.
    """
    with open(dll_path, "rb") as f:
        data = f.read()

    if data[:2] != b"MZ":
        raise ValueError("Not a valid PE file (missing MZ header)")

    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    if data[e_lfanew : e_lfanew + 4] != b"PE\0\0":
        raise ValueError("Invalid PE signature")

    coff_off = e_lfanew + 4
    num_sections = struct.unpack_from("<H", data, coff_off + 2)[0]
    opt_hdr_size = struct.unpack_from("<H", data, coff_off + 16)[0]
    opt_off = coff_off + 20

    magic = struct.unpack_from("<H", data, opt_off)[0]
    if magic == 0x20B:      # PE32+ (64-bit)
        data_dir_off = opt_off + 112
    elif magic == 0x10B:    # PE32 (32-bit)
        data_dir_off = opt_off + 96
    else:
        raise ValueError(f"Unknown optional header magic: {hex(magic)}")

    export_rva  = struct.unpack_from("<I", data, data_dir_off)[0]
    export_size = struct.unpack_from("<I", data, data_dir_off + 4)[0]
    if export_rva == 0:
        return [], []

    sec_off = opt_off + opt_hdr_size
    sections: list[_Section] = []
    for i in range(num_sections):
        s = sec_off + i * 40
        sections.append(_Section(
            vaddr           = struct.unpack_from("<I", data, s + 12)[0],
            vsize           = struct.unpack_from("<I", data, s +  8)[0],
            raw_off         = struct.unpack_from("<I", data, s + 20)[0],
            raw_size        = struct.unpack_from("<I", data, s + 16)[0],
            characteristics = struct.unpack_from("<I", data, s + 36)[0],
        ))

    exp_off      = _rva_to_offset(export_rva, sections)
    num_names    = struct.unpack_from("<I", data, exp_off + 24)[0]
    funcs_rva    = struct.unpack_from("<I", data, exp_off + 28)[0]
    names_rva    = struct.unpack_from("<I", data, exp_off + 32)[0]
    ordinals_rva = struct.unpack_from("<I", data, exp_off + 36)[0]

    funcs_off    = _rva_to_offset(funcs_rva,    sections)
    names_off    = _rva_to_offset(names_rva,    sections)
    ordinals_off = _rva_to_offset(ordinals_rva, sections)

    func_names: list[str] = []
    data_names: list[str] = []

    for i in range(num_names):
        name_rva = struct.unpack_from("<I", data, names_off + i * 4)[0]
        name_off = _rva_to_offset(name_rva, sections)
        end      = data.index(b"\0", name_off)
        name     = data[name_off:end].decode("ascii")

        # ordinals_rva table holds 0-based indices into the EAT (funcs_rva)
        ordinal  = struct.unpack_from("<H", data, ordinals_off + i * 2)[0]
        func_rva = struct.unpack_from("<I", data, funcs_off + ordinal * 4)[0]

        # An RVA inside the export directory means it's already a forwarder
        # string ("other.dll.symbol") in the original DLL — treat as function.
        if export_rva <= func_rva < export_rva + export_size:
            func_names.append(name)
            continue

        sec = _section_for_rva(func_rva, sections)
        if sec is not None and (sec.characteristics & (_SCN_CNT_CODE | _SCN_MEM_EXECUTE)):
            func_names.append(name)
        else:
            data_names.append(name)

    return sorted(func_names), sorted(data_names)


def generate_def(dll_path: str, proxy_def_path: str) -> None:
    """Write proxy.def and the companion amdhip64_7_orig.def."""
    import os

    func_names, data_names = read_pe_exports(dll_path)
    if not func_names and not data_names:
        print(f"ERROR: No exports found in {dll_path}", file=sys.stderr)
        sys.exit(1)

    orig_def_path = os.path.join(
        os.path.dirname(os.path.abspath(proxy_def_path)),
        f"{FORWARD_TARGET}.def",
    )

    intercepted = 0
    forwarded   = 0

    # --- proxy def -----------------------------------------------------------
    with open(proxy_def_path, "w") as f:
        f.write("LIBRARY amdhip64_7\n")
        f.write("EXPORTS\n")
        for name in func_names:
            if name in INTERCEPTED:
                f.write(f"    {name}\n")
                intercepted += 1
            else:
                # PE forwarder: Windows loader resolves this at load time.
                # MSVC emits the forwarder entry only when amdhip64_7_orig.lib
                # is present at link time (see companion def below).
                f.write(f"    {name} = {FORWARD_TARGET}.{name}\n")
                forwarded += 1

    # --- companion def for stub import library --------------------------------
    # lib.exe uses this to create amdhip64_7_orig.lib with no actual DLL.
    with open(orig_def_path, "w") as f:
        f.write(f"LIBRARY {FORWARD_TARGET}\n")
        f.write("EXPORTS\n")
        for name in func_names:
            f.write(f"    {name}\n")

    print(f"Generated {proxy_def_path}:")
    print(f"  {intercepted} intercepted  (implemented in proxy)")
    print(f"  {forwarded} forwarded     -> {FORWARD_TARGET}.dll")
    print(f"  {len(data_names)} data exports  (skipped — MSVC forwarder syntax unsupported for data)")
    print(f"  {intercepted + forwarded} total function exports")
    print(f"Generated {orig_def_path}  (use with lib.exe to create stub import library)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path-to-real-amdhip64_7.dll> [proxy.def]")
        sys.exit(1)

    dll = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "amdhip64_7.def"
    generate_def(dll, out)
