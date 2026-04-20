#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License
"""Generate hrr_symver.h and hrr_record.ver from a real libamdhip64.so.

ELF symbol versioning works like glibc: the version tag on a symbol is
the version in which the symbol was *introduced*, frozen forever for
ABI stability.  hipMalloc was added in HIP 4.2 and is still tagged
hip_4.2 in ROCm 7.x; later versions only get new tags for newly added
APIs (hipMallocAsync@hip_5.1, hipModuleLaunchCooperativeKernel@hip_5.5,
...).

Hardcoding hip_4.2 in the source therefore works *today* but would
break the moment AMD republished an existing API at a new version
(analogous to glibc's memcpy@GLIBC_2.14 ABI break).  This script
removes that fragility by reading the actual versions from the
installed libamdhip64.so at CMake configure time.

Inputs:
  libamdhip64.so.7    -- real HIP runtime to read versions from
  out_header          -- destination path for hrr_symver.h
  out_verscript       -- destination path for hrr_record.ver

The set of symbols to version-alias is hardcoded below; it must stay
in sync with the interceptions implemented in hrr_interposer_linux.c
and hrr_interposer_cxx.cpp.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# C-linkage HIP symbols intercepted by hrr_interposer_linux.c and
# hrr_interposer_cxx.cpp.  Order is purely cosmetic (groups in the
# generated header).  Every name MUST exist in libamdhip64; missing
# symbols cause a hard build failure so we don't silently leak
# unversioned interceptions.
INTERCEPTED_SYMBOLS = [
    # ("group label", [symbols])
    ("Memory allocation", [
        "hipMalloc",
        "hipExtMallocWithFlags",
        "hipMallocManaged",
        "hipMallocAsync",
        "hipMallocFromPoolAsync",
        "hipFreeAsync",
        "hipFree",
    ]),
    ("Memory copy", [
        "hipMemcpy",
        "hipMemcpyAsync",
        "hipMemcpyWithStream",
        "hipMemcpyHtoD",
        "hipMemcpyHtoDAsync",
        "hipMemcpyDtoH",
        "hipMemcpyDtoD",
        "hipMemcpyDtoDAsync",
        "hipMemcpy2D",
        "hipMemcpy2DAsync",
    ]),
    ("Memory set", [
        "hipMemset",
        "hipMemsetAsync",
        "hipMemsetD8",
        "hipMemsetD8Async",
        "hipMemsetD16",
        "hipMemsetD16Async",
        "hipMemsetD32",
        "hipMemsetD32Async",
    ]),
    ("Host memory", [
        "hipHostMalloc",
        "hipHostFree",
        "hipHostRegister",
        "hipHostUnregister",
        "hipHostGetDevicePointer",
    ]),
    ("Module load", [
        "hipModuleLoad",
        "hipModuleLoadData",
        "hipModuleUnload",
        "hipModuleGetFunction",
    ]),
    ("Kernel launch", [
        "hipModuleLaunchKernel",
        "hipModuleLaunchCooperativeKernel",
        "hipLaunchKernel",
        "hipExtModuleLaunchKernel",
    ]),
    ("Synchronization", [
        "hipDeviceSynchronize",
        "hipStreamSynchronize",
        "hipInit",
    ]),
]

# Symbols defined in hrr_interposer_cxx.cpp instead of the C TU.
# .symver directives emit an assembler alias of an existing symbol, so
# they MUST live in the same translation unit as the symbol's
# definition — otherwise the linker reports "undefined reference".
# The header only emits HRR_SYMVER() lines for C-defined symbols; the
# cxx file emits its own using the HRR_VER_<name> macros below.
CXX_DEFINED_SYMBOLS = {
    "hipLaunchKernel",
    "hipExtModuleLaunchKernel",
}

# Extra macro emitted for the Itanium C++-mangled alias of
# hipExtModuleLaunchKernel.  The mangled name "_Z24hipExtModule..."
# can't be a macro identifier, so the cxx file spells it out verbatim
# and just needs the version to attach.
CXX_MANGLED_VERSION_MACROS = {
    "HRR_VER_hipExtModuleLaunchKernel_mangled": "hipExtModuleLaunchKernel",
}


def read_symbol_versions(libpath: Path) -> dict[str, str]:
    """Parse `objdump -T libpath` and return {symbol_name: version}.

    objdump -T output columns:
        addr flags section size version symbol_name
    Version "Base" or "*UND*" means unversioned/undefined and is
    skipped.  We keep only versions matching ^hip_ (the HIP ABI)."""
    try:
        out = subprocess.run(
            ["objdump", "-T", str(libpath)],
            check=True, capture_output=True, text=True,
        ).stdout
    except FileNotFoundError:
        sys.exit("[gen_symver] error: objdump not found in PATH")
    except subprocess.CalledProcessError as e:
        sys.exit(f"[gen_symver] objdump failed on {libpath}: {e.stderr}")

    versions: dict[str, str] = {}
    # objdump -T row example:
    #   00000000003d82b0 g    DF .text  000000000000002a  hip_4.2  hipMalloc
    # The flags column ("g    DF") has internal whitespace, so column
    # counts vary; split() collapses runs of whitespace and the symbol
    # name + version are always the last two tokens.
    for line in out.splitlines():
        toks = line.split()
        if len(toks) < 2:
            continue
        ver, name = toks[-2], toks[-1]
        if not ver.startswith("hip_"):
            continue
        # First definition wins (binutils lists default version first).
        versions.setdefault(name, ver)
    return versions


def emit_header(out_path: Path, versions: dict[str, str], libpath: Path) -> None:
    """Write hrr_symver.h with HRR_VER_<name> macros and HRR_SYMVER list."""
    lines: list[str] = []
    lines.append("/* AUTO-GENERATED by gen_symver.py — DO NOT EDIT.")
    lines.append(f" * Source: {libpath}")
    lines.append(" *")
    lines.append(" * Versioned symbol aliases for the HRR LD_PRELOAD interposer.")
    lines.append(" * libamdhip64 exports its API as VERSIONED symbols")
    lines.append(" * (hipMalloc@hip_4.2, hipModuleLaunchKernel@hip_4.2, ...).")
    lines.append(" * Callers built against hip_runtime_api.h link against the")
    lines.append(" * versioned reference, so an LD_PRELOAD library that exports")
    lines.append(" * only the unversioned name is silently bypassed by the")
    lines.append(" * dynamic linker — calls fall through to libamdhip64 and")
    lines.append(" * never reach our recorder.  This file plugs that hole by")
    lines.append(" * emitting a .symver alias for every intercepted API at the")
    lines.append(" * exact version libamdhip64 ships on this build host.")
    lines.append(" *")
    lines.append(" * Versions are extracted from the actual libamdhip64.so at")
    lines.append(" * configure time so a future ROCm release that rebumps an")
    lines.append(" * existing symbol (e.g. hipMalloc@hip_8.0) will be picked up")
    lines.append(" * automatically — no source edits required. */")
    lines.append("")
    lines.append("#ifndef HRR_SYMVER_H")
    lines.append("#define HRR_SYMVER_H")
    lines.append("")
    lines.append("/* Per-symbol version macros (string literals so .symver")
    lines.append(" * directives can concatenate them).  Exposed for use by")
    lines.append(" * hrr_interposer_cxx.cpp which versions C++-mangled aliases. */")

    missing: list[str] = []
    for group, names in INTERCEPTED_SYMBOLS:
        lines.append(f"/* --- {group} --- */")
        for name in names:
            ver = versions.get(name)
            if ver is None:
                missing.append(name)
                continue
            lines.append(f'#define HRR_VER_{name} "{ver}"')
        lines.append("")

    # Extra macro for the C++-mangled alias of hipExtModuleLaunchKernel
    # (libmigraphx_gpu links against the mangled form; libamdhip64
    # exports it at the same version as the C symbol).
    lines.append("/* C++-mangled alias version (libamdhip64 ships both the")
    lines.append(" * C and Itanium-mangled forms at the same node). */")
    for cxx_macro, c_name in CXX_MANGLED_VERSION_MACROS.items():
        ver = versions.get(c_name)
        if ver is None:
            continue
        lines.append(f'#define {cxx_macro} "{ver}"')
    lines.append("")

    if missing:
        sys.exit(
            "[gen_symver] error: the following intercepted symbols were "
            f"NOT found in {libpath}:\n  " + "\n  ".join(missing) +
            "\nUpdate INTERCEPTED_SYMBOLS in gen_symver.py if the API "
            "was renamed/removed, or check that the right libamdhip64 "
            "is being inspected."
        )

    lines.append("/* Single '@' adds a versioned ALIAS to an existing")
    lines.append(" * unversioned definition.  '@@' would replace the default")
    lines.append(" * symbol entirely and conflict with the unversioned export"
                 " our")
    lines.append(" * own initialization code uses for dlsym lookups. */")
    lines.append("#define HRR_SYMVER_ALIAS(name, ver) \\")
    lines.append("    __asm__(\".symver \" #name \",\" #name \"@\" ver)")
    lines.append("")
    lines.append("/* HRR_SYMVER(name) — picks up the version from the")
    lines.append(" * HRR_VER_<name> macro emitted above. */")
    lines.append("#define HRR_SYMVER(name) HRR_SYMVER_ALIAS(name, HRR_VER_##name)")
    lines.append("")

    for group, names in INTERCEPTED_SYMBOLS:
        # The C++-only entry points (hipLaunchKernel,
        # hipExtModuleLaunchKernel) are aliased from inside
        # hrr_interposer_cxx.cpp where the symbol definition lives —
        # skip them here so we don't reference a symbol the C
        # translation unit hasn't defined.
        emit_names = [n for n in names if n not in CXX_DEFINED_SYMBOLS]
        if not emit_names:
            continue
        lines.append(f"/* {group} */")
        for name in emit_names:
            lines.append(f"HRR_SYMVER({name});")
        lines.append("")

    lines.append("#endif  /* HRR_SYMVER_H */")
    out_path.write_text("\n".join(lines) + "\n")


def emit_version_script(out_path: Path, versions: dict[str, str],
                         libpath: Path) -> None:
    """Write hrr_record.ver declaring every version node referenced.

    The linker requires each version tag mentioned in a .symver
    directive to be declared in the version script.  We emit them in
    ascending numeric order so each node depends on the previous one
    (matches how glibc / libamdhip64 chain their version trees)."""
    used: set[str] = set()
    for _, names in INTERCEPTED_SYMBOLS:
        for name in names:
            v = versions.get(name)
            if v is not None:
                used.add(v)

    def parse_ver(v: str) -> tuple[int, ...]:
        # "hip_5.10" -> (5, 10)
        return tuple(int(x) for x in v[len("hip_"):].split("."))

    sorted_vers = sorted(used, key=parse_ver)

    lines: list[str] = []
    lines.append("/* AUTO-GENERATED by gen_symver.py — DO NOT EDIT.")
    lines.append(f" * Source: {libpath}")
    lines.append(" *")
    lines.append(" * Declares the version nodes referenced by .symver")
    lines.append(" * directives in hrr_symver.h.  The linker rejects any")
    lines.append(" * .symver tag not declared here, so the set MUST stay in")
    lines.append(" * sync with the symbol → version mapping above. */")
    lines.append("")
    prev: str | None = None
    for v in sorted_vers:
        if prev is None:
            lines.append(f"{v} {{ }};")
        else:
            lines.append(f"{v} {{ }} {prev};")
        prev = v
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("libamdhip64",
                     help="Path to libamdhip64.so (real HIP runtime)")
    ap.add_argument("--out-header", required=True, type=Path,
                     help="Destination path for hrr_symver.h")
    ap.add_argument("--out-verscript", required=True, type=Path,
                     help="Destination path for hrr_record.ver")
    args = ap.parse_args()

    libpath = Path(args.libamdhip64).resolve()
    if not libpath.exists():
        sys.exit(f"[gen_symver] error: {libpath} does not exist")

    versions = read_symbol_versions(libpath)
    if not versions:
        sys.exit(f"[gen_symver] error: no hip_* versioned symbols found "
                  f"in {libpath} — wrong file?")

    args.out_header.parent.mkdir(parents=True, exist_ok=True)
    args.out_verscript.parent.mkdir(parents=True, exist_ok=True)
    emit_header(args.out_header, versions, libpath)
    emit_version_script(args.out_verscript, versions, libpath)
    print(f"[gen_symver] wrote {args.out_header} and "
          f"{args.out_verscript} from {libpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
