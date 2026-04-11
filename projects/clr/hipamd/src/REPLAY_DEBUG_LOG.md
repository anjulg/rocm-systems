# HRR Replay Debug Session Log

## Problem

`hrr_replay.exe` fails during playback with `hipErrorInvalidValue` at
`hipModuleLaunchKernel`:

```
[HRR] Loaded archive: 23 events, 1 kernels, 3 blobs, 5 code objects
[HRR] Replaying on: AMD Radeon(TM) 890M Graphics (gfx1150)
[HRR] Pre-loaded 5 / 5 code objects
[HRR] HIP error 1 (invalid argument) at hrr_replay.cpp:268
[HRR] Replay failed at event 14 (KERNEL_LAUNCH)
```

The failing kernel is a Tensile GEMM kernel with a ~550-char name:
`Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT64x16x32_...WG64_2_1`

## Root Cause Chain (three bugs found iteratively)

### Bug 1: Kernel name buffer too small (hrr_code_object.h)

`hrr_kernel_meta_t.name` was `char[256]` but Tensile kernel names are ~550
chars. The name was truncated during code object metadata parsing, so
`hrr_find_kernel()` (which uses `strcmp`) never matched, returning
`meta == NULL`. With no metadata, 0 args were recorded.

**Fix:** Increased to `char[1024]`.

**Result:** Still failed — metadata lookup returned NULL for a different
reason.

### Bug 2: Incomplete msgpack parser (hrr_code_object.c)

The `mp_skip()` function was missing handlers for several msgpack types:
`float32` (0xca), `float64` (0xcb), `bin8/16/32` (0xc4-c6),
`ext8/16/32` (0xc7-c9), `fixext1-16` (0xd4-d8). When any of these
appeared in AMDGPU metadata, `mp_skip()` consumed only the 1-byte tag,
leaving the payload unread and desynchronizing the parser.

Additional fixes in the same file:
- `mp_read_str()` now accepts `bin` types (some encoders use bin for strings)
- Error recovery in map iteration fixed: `mp_read_str` returning -1 already
  consumed the tag byte, so the caller should `mp_skip` once (the value), not
  twice
- Added `.symbol` fallback: if `.name` is empty, derive from `.symbol`
  (stripping `.kd` suffix)

**Result:** Still failed — 0 kernels parsed from the specific code object.

### Bug 3: CCOB (Compressed Clang Offload Bundle) not handled (hrr_proxy_win.c)

Verbose log (`HRR_VERBOSE=1`) revealed:

```
[HRR] hipModuleLoad(.../TensileLibrary_..._gfx1150.co) -> module=000001FA63688130
[HRR]   format=unknown (magic=43434f42), capturing raw 176274 bytes
[HRR] module_load: mod=000001FA63688130 handle=5 image_size=176274 parsed 0 kernels
```

Magic `43434f42` = ASCII "CCOB" = **Compressed Clang Offload Bundle**.
Tensile `.co` files on recent ROCm are zstd-compressed. The HIP runtime
decompresses internally during `hipModuleLoad`, but the proxy captured the
raw compressed bytes. `hrr_parse_code_object` saw "CCOB" instead of ELF
magic and returned 0.

**Fix:** Added CCOB decompression to `hrr_proxy_win.c`:
- `hrr_decompress_ccob()` — parses CCOB header, dynamically loads
  `zstd.dll`, decompresses
- `hrr_process_cob()` — extracts GPU ELFs from decompressed COB
- Both `hipModuleLoad` and `hipModuleLoadData` handlers updated
- Requires `zstd.dll` at runtime (ships with ROCm)

### Bug 3b: Wrong CCOB header format (hrr_proxy_win.c)

After deploying the initial CCOB fix, verbose logs showed:

```
[HRR]   format=CCOB (compressed offload bundle)
[HRR] module_load: mod=... handle=5 image_size=176274 parsed 0 kernels
```

CCOB was detected but decompression failed silently — `image_size=176274`
is the raw compressed size, confirming the fallback path was taken. The
`[HRR] CCOB: version=... method=...` diagnostic line was never printed,
meaning `hrr_decompress_ccob()` returned NULL before reaching it.

**Root cause:** The parser assumed a wrong header layout for V2+. It treated
the `TotalFileSize(4)` + `UncompressedSize(4)` fields as a single
`uint64_t file_id_size` (variable-length file identifier), causing `p` to
jump astronomically past `end` and trigger an early NULL return.

Actual LLVM CCOB header layouts (from `OffloadBundle.cpp`):

| Version | Layout | Total |
|---------|--------|-------|
| V1 | Common(8) + UncompressedSize(4) + Hash(8) | 20 bytes |
| V2 | Common(8) + TotalFileSize(4) + UncompressedSize(4) + Hash(8) | 24 bytes |
| V3 | Common(8) + TotalFileSize(8) + UncompressedSize(8) + Hash(8) | 32 bytes |

**Fix:** Rewrote `hrr_decompress_ccob()` with a version switch that matches
the packed struct layouts from LLVM (`RawCompressedBundleHeader::V1Header`,
`V2Header`, `V3Header`).

## All Files Modified

### `out_of_tree/hrr_code_object.h`
- `hrr_kernel_meta_t.name`: `char[256]` → `char[1024]`

### `out_of_tree/hrr_code_object.c`
- `mp_skip()`: Added handlers for float32, float64, bin8/16/32, ext8/16/32,
  fixext1/2/4/8/16
- `mp_read_str()`: Accept `bin` types (0xc4-c6) as strings
- `parse_kernel_args()`: Fixed error recovery (`mp_skip(r)` not
  `mp_skip(r); mp_skip(r)`)
- `parse_kernels_array()`: Same error recovery fix; added `.symbol` fallback
  with `.kd` suffix stripping
- `parse_metadata()`: Same error recovery fix

### `out_of_tree/hrr_trace_writer.c`
- `hrr_record_kernel_launch()`: Added WARNING with module/kernel inventory
  when metadata lookup fails
- `hrr_record_kernel_launch_packed()`: Same WARNING
- `hrr_record_module_load()`: Added verbose logging of parse results

### `out_of_tree/hrr_proxy_win.c`
- Added CCOB constants, zstd dynamic loading (`ensure_zstd()`)
- Added `hrr_decompress_ccob()` — CCOB header parsing + zstd decompression
- Added `hrr_process_cob()` — COB parsing extracted into reusable helper
- `hipModuleLoad` handler: CCOB detection between COB and unknown-format
  branches
- `hipModuleLoadData` handler: CCOB fallback when `elf64_true_size` returns 0

### `hrr_replay.cpp`
- Added WARNING when kernel launch event has 0 recorded args

### Bug 4: Only first NT_AMDGPU_METADATA note parsed (hrr_code_object.c)

Diagnostic output with `HRR_VERBOSE=1` revealed:

```
[HRR]   metadata: amdhsa.kernels array has 1 entries (max_kernels=128)
[HRR] module_load: ... handle=5 image_size=2248256 parsed 1 kernels
```

The 2.2 MB Tensile ELF genuinely has only **1 kernel per metadata note**.
Tensile compiles each kernel variant into a separate object file, each with
its own `NT_AMDGPU_METADATA` note. When linked together, the `.note` section
contains **N separate metadata notes** (one per variant). The parser had
`return parse_metadata(...)` which returned after the first match, missing
all subsequent notes.

Also revealed: the `.hsaco` module has **428 kernels** in a single metadata
note, exceeding the `MAX_CO_KERNELS=128` storage limit.

**Fix:**

1. `hrr_parse_code_object()` now iterates ALL `NT_AMDGPU_METADATA` notes
   and accumulates kernels across them (instead of returning after the first)
2. Module kernel storage switched from fixed `kernels[128]` to dynamically
   allocated (`calloc(1024, ...)` per module) to handle 428+ kernels without
   static memory bloat
3. `hrr_record_module_unload()` now frees the allocation

### Also fixed in this round

- `mp_read_str()`: on type mismatch, puts tag byte back (`pos--`) so callers
  can `mp_skip()` the entire non-string value correctly
- `parse_kernel_args()`: when `nargs > 64`, remaining args are now consumed
  from the stream to keep it in sync (previously left unread, corrupting
  subsequent kernel parsing)
- All `mp_read_str` callers updated: failed key reads do `mp_skip; mp_skip`
  (skip both key and value); failed value reads do `mp_skip` on the value

## Current Status

All five bugs (1-4) are fixed. Next steps:
1. Rebuild proxy DLL
2. Ensure `zstd.dll` is accessible (copy next to proxy or set `ROCM_PATH`)
3. Re-capture with `HRR_RECORD=1 HRR_VERBOSE=1`
4. Verify Tensile module now shows `parsed N kernels` with N >> 1
5. Re-replay with `hrr-replay.exe capture.hrr`

## Diagnostic Tips

- `HRR_VERBOSE=1` — prints per-module kernel parse counts, format detection,
  and `amdhsa.kernels array has N entries` for each metadata note
- The WARNING message shows `N modules registered, M kernels parsed total`
- If metadata still missing, compare the exact kernel name in the WARNING
  against the names printed in per-module kernel inventory
