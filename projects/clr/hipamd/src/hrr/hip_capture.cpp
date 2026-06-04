/* Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License */

/*
 * hip_capture.cpp — Hand-written capture shims for complex HIP APIs.
 *
 * Covers the MANUAL_CAPTURE_APIS set defined in gen_hrr_api_args.py:
 *   - Memcpy H2D variants with blob snapshotting (hipMemcpy, hipMemcpyAsync,
 *     hipMemcpyHtoD, hipMemcpyHtoDAsync, hipMemcpyWithStream)
 *   - Module load with code object snapshotting (hipModuleLoad*)
 *   - Kernel launch with arg introspection via kernel->signature()
 *   - Fat binary registration (__hipRegisterFatBinary)
 *   - Host memory registration (hipHostRegister / hipHostUnregister)
 *
 * Everything else (malloc/free, stream/event, memset, device sync, etc.)
 * is auto-generated in hip_capture_generated.cpp.
 *
 * All shims write hrr_args_* structs as event payloads (uniform format).
 * Kernel launch is the exception — it uses a variable-length binary payload
 * (the format defined in hrr_reader.h parse_kernel_launch).
 *
 * g_real_table, g_cap_table, g_compiler_installed are defined here (non-static)
 * so hip_capture_generated.cpp can extern them.
 *
 * Independence rule: zero dependency on hipamd/src/profiler/.
 */

#include "hip_capture.h"
#include "hip_capture_writer.h"

// hrr_api_args.h — for hrr_args_* struct types and hrr_api_id_t enum
#include "hrr_api_args.h"

// HIP runtime internals
#include "../hip_global.hpp"       // hip::asKernel()
#include "../hip_internal.hpp"
#include "hip/amd_detail/hip_api_trace.hpp"
#include "utils/flags.hpp"         // HIP_HRR_CAPTURE_OUTPUT flag

// ROCclr kernel introspection
#include "device/devkernel.hpp"    // amd::Kernel, KernelParameterDescriptor
#include "platform/kernel.hpp"     // amd::KernelSignature
#include "platform/memory.hpp"     // amd::Memory::getSize()
#include "device/device.hpp"       // amd::MemObjMap::FindMemObj()
#include "opencl/amdocl/cl_kernel.h"  // T_POINTER enum

// Fat binary format structs (ClangOffloadBundleUncompressedHeader, etc.)
#include "../hip_code_object.hpp"
#include "../hip_platform.hpp"   // PlatformState::Instance()

#include <atomic>
#include <climits>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>


// GetHipDispatchTable / GetHipCompilerDispatchTable
namespace hip {
const HipDispatchTable*         GetHipDispatchTable();
const HipCompilerDispatchTable* GetHipCompilerDispatchTable();
}

// ---------------------------------------------------------------------------
// Global tables — non-static: extern'd in hip_capture_generated.cpp
// ---------------------------------------------------------------------------

HipDispatchTable         g_real_table{};
HipDispatchTable         g_cap_table{};
std::atomic<bool>        g_installed{false};
std::atomic<bool>        g_table_built{false};  // guard for hip_capture_build_table()

HipCompilerDispatchTable g_real_compiler_table{};
std::atomic<bool>        g_compiler_installed{false};  // guard for hip_capture_build_compiler_table()

// TLS dims saved by __hipPushCallConfiguration / hipConfigureCall for use by
// hipLaunchByPtr and the <<< >>> (hipLaunchKernel) path.
static thread_local dim3        g_pushed_grid{};
static thread_local dim3        g_pushed_block{};
static thread_local size_t      g_pushed_shared{};
static thread_local hipStream_t g_pushed_stream{};

// TLS kernarg buffer accumulated by hipSetupArgument calls; consumed and
// cleared by capture_hipLaunchByPtr.
static thread_local std::vector<uint8_t> g_setuparg_buf{};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

bool hip_capture_enabled() {
  return !flagIsDefault(HIP_HRR_CAPTURE_OUTPUT) &&
         HIP_HRR_CAPTURE_OUTPUT[0] != '\0';
}

const char* hip_capture_output_dir() {
  return HIP_HRR_CAPTURE_OUTPUT;
}

// Parsed once at first call; the ROCclr flag is a c-string so we map to int.
// "timeline"=0, "inputs"=1, "full"=2.  Unknown values warn and fall back to
// timeline so a typo never silently changes archive contents.
int hip_capture_record_mode() {
  static const int cached = []() -> int {
    const char* s = HIP_HRR_RECORD_MODE;
    if (!s || !*s) return HRR_RECORD_TIMELINE;
    if (std::strcmp(s, "timeline") == 0) return HRR_RECORD_TIMELINE;
    if (std::strcmp(s, "inputs")   == 0) return HRR_RECORD_INPUTS;
    if (std::strcmp(s, "full")     == 0) return HRR_RECORD_FULL;
    LogPrintfWarning("[HRR capture] unknown HIP_HRR_RECORD_MODE='%s' "
                     "(expected: timeline|inputs|full) — using timeline", s);
    return HRR_RECORD_TIMELINE;
  }();
  return cached;
}

size_t hip_capture_max_snap_bytes() {
  // 0 = unlimited.  Default 16 MiB matches hip_replay's HRR_MAX_SNAP_MB.
  const uint32_t mb = HIP_HRR_MAX_SNAP_MB;
  if (mb == 0) return 0;
  return static_cast<size_t>(mb) * 1024u * 1024u;
}

// Parse the extra[] sentinel format for packed kernarg buffers.
static bool parse_kernel_extra(void** extra, const void*& out_buf, size_t& out_size) {
  if (!extra) return false;
  if (extra[0] != HIP_LAUNCH_PARAM_BUFFER_POINTER) return false;
  if (extra[2] != HIP_LAUNCH_PARAM_BUFFER_SIZE)    return false;
  if (extra[4] != HIP_LAUNCH_PARAM_END)             return false;
  out_buf  = extra[1];
  out_size = *reinterpret_cast<const size_t*>(extra[3]);
  return out_buf != nullptr && out_size > 0;
}

// Map from amd::Program* (held by the runtime for a loaded module) to the
// Hash128 we computed at module-load time.  Used by record_launch() to tag
// kernel-launch events with the code-object hash of the function being
// launched, so the playback can disambiguate kernels that share a name
// across multiple modules.  This matters in practice: MIGraphX/rocMLIR
// emits one conv-layer-specific copy of `mlir_convolution_broadcast_add_relu`
// per layer; ResNet-50 produces 30+ same-named kernels with different
// shapes baked into the binary.  Without a per-launch hash, the playback
// would resolve the name to whichever module wins a linear scan and
// silently launch the wrong layer's kernel against the right layer's data.
//
// NOTE: entries are not removed on hipModuleUnload — typical workloads
// load once at init and never unload, so a stale Program* entry would only
// matter if the runtime reused the same heap address for a different
// module before any new launch.  If that becomes a problem in practice,
// add `hipModuleUnload` to MANUAL_CAPTURE_APIS in gen_hrr_api_args.py
// and erase from the map in a hand-written unload shim.
static std::mutex                                        g_prog_hash_mu;
static std::unordered_map<const void*, hrr_cap::Hash128> g_prog_hash_map;

// Resolve the runtime's amd::Program* for a hipModule_t.  We treat it as
// an opaque key; `amd::Program` is only forward-declared by platform/kernel.hpp,
// which is all we need.
static const void* program_from_module(hipModule_t module) {
  if (!module) return nullptr;
  return static_cast<const void*>(as_amd(reinterpret_cast<cl_program>(module)));
}

static void remember_module_hash(hipModule_t module, hrr_cap::Hash128 h) {
  if (!module || (h.lo == 0 && h.hi == 0)) return;
  const void* prog = program_from_module(module);
  if (!prog) return;
  std::lock_guard<std::mutex> lk(g_prog_hash_mu);
  g_prog_hash_map[prog] = h;
}

static hrr_cap::Hash128 hash_for_program(const void* prog) {
  if (!prog) return {0, 0};
  std::lock_guard<std::mutex> lk(g_prog_hash_mu);
  auto it = g_prog_hash_map.find(prog);
  return it != g_prog_hash_map.end() ? it->second : hrr_cap::Hash128{0, 0};
}

// ---------------------------------------------------------------------------
// Kernel launch event serialization
//
// Binary layout of KERNEL_LAUNCH payload (matches hrr_reader.cpp parse_kernel_launch):
//
//   u64  stream_handle (recorded hipStream_t cast to uint64_t)
//   u16  name_len
//   u8[] kernel_name (name_len bytes, no NUL)
//   u64  co_hash_lo   (0 = unknown)
//   u64  co_hash_hi
//   u32[3] grid
//   u32[3] block
//   u32  shared_mem
//   u16  num_args
//   u16  num_snapshots         (HIP_HRR_RECORD_MODE=timeline: 0;
//                               inputs: one direction=0 per pointer arg;
//                               full: also one direction=1 per pointer arg)
//   for each arg:
//     u8   value_kind  (0=scalar, 1=pointer/gpu addr)
//     u16  size
//     u8[] data (size bytes)
//   for each snapshot (num_snapshots entries — between args and full_kbuf;
//                      wire-compatible with hip_replay's snap_record_t):
//     u64  ptr_handle  (recorded device pointer)
//     u64  offset      (always 0 — snapshots start at the alloc base)
//     u64  length      (bytes captured; clamped by HIP_HRR_MAX_SNAP_MB)
//     u64  hash_lo
//     u64  hash_hi
//     u8   direction   (0=input/pre-launch, 1=output/post-launch)
//   --- trailing (v3.1, optional — older archives stop after args+snaps) ---
//   u32  full_kbuf_size (0 = not captured; non-zero when extra[] was used)
//   u8[] full_kbuf      (full_kbuf_size bytes — the entire packed kernarg
//                        buffer, including implicit/hidden args the rocclr
//                        signature does not enumerate; required for correct
//                        replay of MLIR/SP3/etc. kernels launched via extra[])
// ---------------------------------------------------------------------------

// One per-kernel buffer snapshot — built by capture_buffer_snapshot() and
// emitted into the kernel-launch payload. Wire layout matches hip_replay's
// snap_record_t (33 bytes packed, written field-by-field).
struct SnapshotRecord {
  uint64_t ptr_handle;
  uint64_t offset;
  uint64_t length;
  uint64_t hash_lo;
  uint64_t hash_hi;
  uint8_t  direction;
};

static void serialize_kernel_launch(
    const char*                 kernel_name,
    uint64_t                    co_hash_lo,
    uint64_t                    co_hash_hi,
    uint32_t gx, uint32_t gy, uint32_t gz,
    uint32_t bx, uint32_t by, uint32_t bz,
    uint32_t shared_mem,
    hipStream_t stream,
    const amd::KernelSignature& sig,
    void**                      kernel_params,
    const void*                 kbuf,
    size_t                      ksz,
    const std::vector<SnapshotRecord>& snapshots,
    hrr_api_id_t                api_id = HRR_API_HIPMODULELAUNCHKERNEL)
{
  // Reserve space for hrr_event_header at front; payload body follows.
  std::vector<uint8_t> payload(sizeof(hrr_event_header), 0);

  auto push_u8  = [&](uint8_t  v) { payload.push_back(v); };
  auto push_u16 = [&](uint16_t v) {
    payload.push_back(static_cast<uint8_t>(v));
    payload.push_back(static_cast<uint8_t>(v >> 8));
  };
  auto push_u32 = [&](uint32_t v) {
    for (int i = 0; i < 4; i++) payload.push_back(static_cast<uint8_t>(v >> (i*8)));
  };
  auto push_u64 = [&](uint64_t v) {
    for (int i = 0; i < 8; i++) payload.push_back(static_cast<uint8_t>(v >> (i*8)));
  };
  auto push_bytes = [&](const void* data, size_t n) {
    const auto* p = static_cast<const uint8_t*>(data);
    payload.insert(payload.end(), p, p + n);
  };

  // raw stream handle as first payload field after header (for replay stream routing)
  push_u64(reinterpret_cast<uint64_t>(stream));

  uint16_t name_len = static_cast<uint16_t>(std::strlen(kernel_name));
  push_u16(name_len);
  push_bytes(kernel_name, name_len);
  // co_hash identifies the exact code object that owned the function at
  // capture time.  Required for correct replay when multiple modules host
  // kernels of the same name (see Program*->Hash128 map comment above).
  // Both halves remain 0 when the launch came from a path we don't track
  // (e.g. statically registered fat-binary kernels); the playback falls
  // back to a name-only scan in that case.
  push_u64(co_hash_lo); push_u64(co_hash_hi);
  push_u32(gx); push_u32(gy); push_u32(gz);
  push_u32(bx); push_u32(by); push_u32(bz);
  push_u32(shared_mem);

  uint32_t n_all = sig.numParametersAll();
  uint16_t num_args = 0;
  for (uint32_t i = 0; i < n_all; i++)
    if (!sig.at(i).info_.hidden_) num_args++;
  push_u16(num_args);
  // num_snapshots — 0 in timeline mode; one record per pointer arg per
  // direction in inputs/full modes (the actual records follow the args).
  push_u16(static_cast<uint16_t>(snapshots.size()));

  if (kbuf && ksz > 0) {
    const auto* buf_bytes = static_cast<const uint8_t*>(kbuf);
    for (uint32_t i = 0; i < n_all; i++) {
      const auto& desc = sig.at(i);
      if (desc.info_.hidden_) continue;
      uint8_t kind = (desc.type_ == T_POINTER) ? 1 : 0;
      uint16_t sz = static_cast<uint16_t>(desc.size_);
      push_u8(kind); push_u16(sz);
      if (desc.offset_ + sz <= ksz)
        push_bytes(buf_bytes + desc.offset_, sz);
      else
        for (uint16_t j = 0; j < sz; j++) push_u8(0);
    }
  } else if (kernel_params) {
    uint32_t param_idx = 0;
    for (uint32_t i = 0; i < n_all; i++) {
      const auto& desc = sig.at(i);
      if (desc.info_.hidden_) { continue; }
      uint8_t kind = (desc.type_ == T_POINTER) ? 1 : 0;
      uint16_t sz = static_cast<uint16_t>(desc.size_);
      push_u8(kind); push_u16(sz);
      if (kernel_params[param_idx])
        push_bytes(kernel_params[param_idx], sz);
      else
        for (uint16_t j = 0; j < sz; j++) push_u8(0);
      param_idx++;
    }
  }

  // Buffer snapshots — between args and full_kbuf trailer.  Layout matches
  // hip_replay's snap_record_t (33 bytes per record, no padding).  Direction
  // 0 = pre-launch (input restore on replay); 1 = post-launch (--verify).
  for (const auto& s : snapshots) {
    push_u64(s.ptr_handle);
    push_u64(s.offset);
    push_u64(s.length);
    push_u64(s.hash_lo);
    push_u64(s.hash_hi);
    push_u8 (s.direction);
  }

  // Trailing full-kbuf field (v3.1).  Appended even when ksz == 0 so the
  // playback can unambiguously distinguish "extra[] mode with kbuf captured"
  // from "kernelParams[] mode, no kbuf available".  Older readers stop
  // parsing after the per-arg loop and silently ignore the trailer.
  //
  // The full kbuf is required to correctly replay kernels whose actual
  // kernarg layout includes implicit/hidden args the rocclr signature does
  // not enumerate (notably MLIR-compiled kernels from rocMLIR/MIOpen Find 2.0
  // and SP3 hand-assembled MIOpen kernels).  Replay via kernelParams[] would
  // lose those args; replay via extra[] with this exact buffer (after
  // translating embedded GPU pointers) preserves them.
  uint32_t full_kbuf_size = (kbuf && ksz > 0 && ksz <= UINT32_MAX)
                            ? static_cast<uint32_t>(ksz)
                            : 0u;
  push_u32(full_kbuf_size);
  if (full_kbuf_size > 0)
    push_bytes(kbuf, full_kbuf_size);

  if (payload.size() > UINT16_MAX) {
    LogPrintfWarning("[HRR capture] Kernel launch payload too large (%zu bytes) — truncating",
                     payload.size());
    payload.resize(UINT16_MAX);
  }
  hrr_cap::writer::write_event_raw(api_id,
                                   reinterpret_cast<hrr_event_header*>(payload.data()),
                                   static_cast<uint16_t>(payload.size()));
}

// Maximum number of unique pointer-arg handles we'll snapshot per kernel.
// Matches hip_replay's MAX_SNAP_PTRS — kernels with more pointer args than
// this just have their excess args un-snapshotted (the rest still works).
static constexpr size_t kMaxSnapPtrs = 16;

// Walk the kernel signature and harvest unique device-pointer handles from
// either the packed kernarg buffer (extra[] launches) or the kernelParams[]
// array.  Hidden args are skipped.  Used by both pre- and post-launch
// snapshot collection so the two sweeps target the exact same pointer set.
static void collect_pointer_args(
    const amd::KernelSignature& sig,
    void**       kernel_params,
    const void*  kbuf, size_t ksz,
    std::vector<uint64_t>& out)
{
  out.clear();
  const auto push_unique = [&](uint64_t h) {
    if (h == 0 || out.size() >= kMaxSnapPtrs) return;
    for (uint64_t x : out) if (x == h) return;
    out.push_back(h);
  };
  uint32_t n_all = sig.numParametersAll();
  uint32_t param_idx = 0;
  for (uint32_t i = 0; i < n_all; i++) {
    const auto& d = sig.at(i);
    if (d.info_.hidden_) continue;
    if (d.type_ == T_POINTER) {
      uint64_t h = 0;
      if (kbuf && ksz > 0) {
        if (d.offset_ + 8 <= ksz)
          std::memcpy(&h, static_cast<const uint8_t*>(kbuf) + d.offset_, 8);
      } else if (kernel_params && kernel_params[param_idx]) {
        std::memcpy(&h, kernel_params[param_idx], 8);
      }
      push_unique(h);
    }
    param_idx++;  // advances per visible arg (matches serialize_kernel_launch)
  }
}

// D2H-read a single allocation containing `handle`, write it as a content-
// addressed blob, and fill out a SnapshotRecord.  Returns false when the
// pointer doesn't resolve to a runtime-tracked allocation (foreign sysmem,
// graph-managed VAs we don't follow, etc.) — the caller silently drops it.
//
// `should_sync` selects the sync flavour: pre-launch direction=0 syncs the
// submission stream (so the buffer reflects everything submitted before the
// kernel we're about to launch); post-launch direction=1 syncs the same
// stream to flush the kernel's writes.  hipStreamSynchronize on stream==null
// degrades to a device sync, which matches the AMD HIP runtime semantics.
static bool capture_buffer_snapshot(uint64_t handle, uint8_t direction,
                                    hipStream_t stream,
                                    size_t max_snap_bytes,
                                    SnapshotRecord& out)
{
  if (handle == 0) return false;
  void* dev_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(handle));

  size_t offset = 0;
  amd::Memory* mem = amd::MemObjMap::FindMemObj(dev_ptr, &offset, nullptr);
  if (!mem) return false;

  const size_t alloc_sz = mem->getSize();
  if (offset >= alloc_sz) return false;
  size_t snap_sz = alloc_sz - offset;
  if (max_snap_bytes > 0 && snap_sz > max_snap_bytes)
    snap_sz = max_snap_bytes;
  if (snap_sz == 0) return false;

  // Sync first so the captured bytes reflect a stable point in time.
  // Both directions need this: pre-launch wants pending writes drained;
  // post-launch wants the kernel's writes drained.
  hipError_t e = (stream)
      ? g_real_table.hipStreamSynchronize_fn(stream)
      : g_real_table.hipDeviceSynchronize_fn();
  if (e != hipSuccess) {
    LogPrintfWarning("[HRR capture] snapshot dir=%u sync failed (%d) — skip",
                     direction, e);
    return false;
  }

  std::vector<uint8_t> host_buf(snap_sz);
  e = g_real_table.hipMemcpy_fn(host_buf.data(), dev_ptr,
                                snap_sz, hipMemcpyDeviceToHost);
  if (e != hipSuccess) {
    LogPrintfWarning("[HRR capture] snapshot dir=%u D2H failed (%d) — skip",
                     direction, e);
    return false;
  }

  hrr_cap::Hash128 h = hrr_cap::writer::write_blob(host_buf.data(), snap_sz);
  out.ptr_handle = handle;
  out.offset     = 0;
  out.length     = static_cast<uint64_t>(snap_sz);
  out.hash_lo    = h.lo;
  out.hash_hi    = h.hi;
  out.direction  = direction;
  return true;
}

// Pre-launch snapshot sweep.  No-op outside inputs/full mode.  Returns a
// vector that may be empty (timeline mode) or smaller than the pointer-arg
// count (untrackable pointers were dropped).  Used by the launch shims to
// build the snapshot list BEFORE the real call so direction=0 records
// reflect the buffer state at kernel-entry time.
static std::vector<SnapshotRecord> capture_pre_launch_snapshots(
    const amd::KernelSignature& sig,
    void**      kernel_params,
    const void* kbuf, size_t ksz,
    hipStream_t stream)
{
  std::vector<SnapshotRecord> out;
  if (hip_capture_record_mode() == HRR_RECORD_TIMELINE) return out;

  std::vector<uint64_t> handles;
  collect_pointer_args(sig, kernel_params, kbuf, ksz, handles);
  if (handles.empty()) return out;

  const size_t cap = hip_capture_max_snap_bytes();
  out.reserve(handles.size());
  for (uint64_t h : handles) {
    SnapshotRecord r{};
    if (capture_buffer_snapshot(h, /*direction=*/0, stream, cap, r))
      out.push_back(r);
  }
  return out;
}

// Post-launch snapshot sweep.  Only fires in full mode; appends direction=1
// records to `snaps` so the playback's --verify can byte-compare actual vs.
// expected output state.  Same pointer set as pre-launch — that's how
// hip_replay's --verify-outputs-only filtering distinguishes pure outputs
// from buffers whose pre-state already matched (read-only inputs).
static void capture_post_launch_snapshots(
    const amd::KernelSignature& sig,
    void**      kernel_params,
    const void* kbuf, size_t ksz,
    hipStream_t stream,
    std::vector<SnapshotRecord>& snaps)
{
  if (hip_capture_record_mode() != HRR_RECORD_FULL) return;

  std::vector<uint64_t> handles;
  collect_pointer_args(sig, kernel_params, kbuf, ksz, handles);
  if (handles.empty()) return;

  const size_t cap = hip_capture_max_snap_bytes();
  for (uint64_t h : handles) {
    SnapshotRecord r{};
    if (capture_buffer_snapshot(h, /*direction=*/1, stream, cap, r))
      snaps.push_back(r);
  }
}

// record_launch — finalize a kernel-launch event.  Called AFTER the real
// HIP launch returned hipSuccess.  `pre_snaps` is moved in: it was built
// before the real call (so direction=0 reflects the input bytes the kernel
// actually consumed) and is augmented here with direction=1 records when
// HIP_HRR_RECORD_MODE=full.
static void record_launch(
    hipFunction_t f,
    unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz,
    unsigned shared_mem,
    hipStream_t stream,
    void** kernel_params, void** extra,
    std::vector<SnapshotRecord> pre_snaps = {},
    hrr_api_id_t api_id = HRR_API_HIPMODULELAUNCHKERNEL)
{
  if (!hip_capture_enabled()) return;
  amd::Kernel* kernel = hip::asKernel(f);
  if (!kernel) return;

  const amd::KernelSignature& sig = kernel->signature();
  const void* kbuf = nullptr;
  size_t      ksz  = 0;

  if (!kernel_params && extra)
    parse_kernel_extra(extra, kbuf, ksz);

  // Detect launches where the app under-provisioned HIP_LAUNCH_PARAM_BUFFER_SIZE
  // — the captured kbuf is too small to hold the kernel's hidden args (hidden_
  // block_count_*, hidden_group_size_*, hidden_remainder_*, hidden_global_offset_*,
  // etc.).  The GPU instruction stream still issues loads at those high offsets,
  // so it ends up reading bytes off the end of the allocated kbuf — typically
  // whatever the previous launch left in the kernarg ring buffer, or zero.  The
  // launch may run to completion either way, but its output is undefined and
  // non-reproducible across runs.  This is an upstream bug in whatever code is
  // calling hipModuleLaunchKernel (often a JIT launcher template that sized the
  // buffer from the visible arg list only).  HRR records exactly the bytes the
  // app passed and lets replay surface the issue via --verify.  Warned once
  // per kernel name so a busy app doesn't flood the log.
  if (kbuf && ksz > 0) {
    size_t expected = 0;
    uint32_t n_all = sig.numParametersAll();
    for (uint32_t i = 0; i < n_all; i++) {
      const auto& d = sig.at(i);
      size_t end = static_cast<size_t>(d.offset_) + static_cast<size_t>(d.size_);
      if (end > expected) expected = end;
    }
    if (ksz < expected) {
      static std::mutex                       warn_mu;
      static std::unordered_set<std::string>  warned;
      std::lock_guard<std::mutex> lk(warn_mu);
      if (warned.insert(kernel->name()).second) {
        LogPrintfWarning(
            "[HRR capture] '%s': extra[] kernarg buffer is %zu bytes but kernel "
            "descriptor declares kernarg_segment_byte_size=%zu — the upstream "
            "caller under-provisioned HIP_LAUNCH_PARAM_BUFFER_SIZE.  The GPU "
            "reads %zu bytes of hidden args off the end of the buffer "
            "(undefined contents).  HRR captures the bytes as-is; fix the "
            "upstream launcher to either use kernelParams[] or pad the "
            "extra[] buffer to %zu bytes.",
            kernel->name().c_str(),
            static_cast<size_t>(ksz), expected,
            expected - static_cast<size_t>(ksz), expected);
      }
    }
  }

  // Tag the launch with the hash of the code object that owns this function,
  // so the playback can resolve the exact module — kernel-name alone is
  // ambiguous when MLIR/MIGraphX emits multiple shape-specialised copies
  // of the same name across different modules.
  const void* prog = static_cast<const void*>(&kernel->program());
  hrr_cap::Hash128 co_hash = hash_for_program(prog);

  capture_post_launch_snapshots(sig, kernel_params, kbuf, ksz, stream, pre_snaps);

  serialize_kernel_launch(
      kernel->name().c_str(),
      co_hash.lo, co_hash.hi,
      gx, gy, gz, bx, by, bz,
      static_cast<uint32_t>(shared_mem),
      stream, sig, kernel_params, kbuf, ksz,
      pre_snaps, api_id);
}

// Helper: build a pre-launch snapshot list by inspecting f's signature.
// Returns an empty vector in timeline mode without touching the GPU.
static std::vector<SnapshotRecord> pre_launch_snapshots_from_f(
    hipFunction_t f, void** kernel_params, void** extra,
    hipStream_t stream)
{
  if (hip_capture_record_mode() == HRR_RECORD_TIMELINE)
    return {};
  amd::Kernel* kernel = hip::asKernel(f);
  if (!kernel) return {};
  const void* kbuf = nullptr;
  size_t      ksz  = 0;
  if (!kernel_params && extra) parse_kernel_extra(extra, kbuf, ksz);
  // Pre-launch: only collect pointer args from the kbuf (extra[] path).
  // Reading kernel_params[] before HIP validates the launch is unsafe —
  // negative-test callers may pass an undersized array (e.g. {nullptr} for a
  // kernel with multiple args), causing collect_pointer_args to walk off the
  // end of the array and segfault.  Post-launch snapshots (after hipSuccess)
  // ARE safe to use kernel_params because HIP validated the array first.
  return capture_pre_launch_snapshots(kernel->signature(),
                                      nullptr, kbuf, ksz, stream);
}

// ---------------------------------------------------------------------------
// Helper macro: fill common hrr_args_* header fields
// ---------------------------------------------------------------------------

// HRR_FILL_HDR removed — write_event_raw() stamps thread_id/sequence_id into
// the payload's hrr_event_header prefix automatically.

// ---------------------------------------------------------------------------
// Memcpy shims — with H2D blob snapshotting
// ---------------------------------------------------------------------------

hipError_t capture_hipMemcpy(void* dst, const void* src,
                                     size_t sizeBytes, hipMemcpyKind kind) {
  hipError_t r = g_real_table.hipMemcpy_fn(dst, src, sizeBytes, kind);
  if (r == hipSuccess) {
    hrr_cap::Hash128 h{0, 0};
    if (kind == hipMemcpyHostToDevice && src && sizeBytes > 0)
      h = hrr_cap::writer::write_blob(src, sizeBytes);
    else if (kind == hipMemcpyDeviceToHost && dst && sizeBytes > 0)
      h = hrr_cap::writer::write_blob(dst, sizeBytes);  // host dst valid after sync call
    hrr_args_hipMemcpy a{};
    a.ret           = static_cast<int32_t>(r);
    a.dst           = reinterpret_cast<uint64_t>(dst);
    a.src           = reinterpret_cast<uint64_t>(src);
    a.sizeBytes     = static_cast<uint64_t>(sizeBytes);
    a.kind          = static_cast<int32_t>(kind);
    a.blob_hash_lo  = h.lo;
    a.blob_hash_hi  = h.hi;
    hrr_cap::writer::write_event_raw(HRR_API_HIPMEMCPY, &a.hdr, sizeof(a));
  }
  return r;
}

hipError_t capture_hipMemcpyAsync(void* dst, const void* src,
                                          size_t sizeBytes, hipMemcpyKind kind,
                                          hipStream_t stream) {
  hipError_t r = g_real_table.hipMemcpyAsync_fn(dst, src, sizeBytes, kind, stream);
  if (r == hipSuccess) {
    hrr_cap::Hash128 h{0, 0};
    if (kind == hipMemcpyHostToDevice && src && sizeBytes > 0) {
      h = hrr_cap::writer::write_blob(src, sizeBytes);
    } else if (kind == hipMemcpyDeviceToHost && dst && sizeBytes > 0) {
      // Sync the stream so host dst is valid before we snapshot it.
      hipError_t sync_r = g_real_table.hipStreamSynchronize_fn(stream);
      if (sync_r == hipSuccess)
        h = hrr_cap::writer::write_blob(dst, sizeBytes);
      else
        LogPrintfWarning("[HRR capture] hipStreamSynchronize failed (%d) — D2H blob skipped",
                         sync_r);
    }
    hrr_args_hipMemcpyAsync a{};
    a.ret          = static_cast<int32_t>(r);
    a.dst          = reinterpret_cast<uint64_t>(dst);
    a.src          = reinterpret_cast<uint64_t>(src);
    a.sizeBytes    = static_cast<uint64_t>(sizeBytes);
    a.kind         = static_cast<int32_t>(kind);
    a.stream       = reinterpret_cast<uint64_t>(stream);
    a.blob_hash_lo = h.lo;
    a.blob_hash_hi = h.hi;
    hrr_cap::writer::write_event_raw(HRR_API_HIPMEMCPYASYNC, &a.hdr, sizeof(a));
  }
  return r;
}

hipError_t capture_hipMemcpyHtoD(hipDeviceptr_t dst, const void* src, size_t sizeBytes) {
  hipError_t r = g_real_table.hipMemcpyHtoD_fn(dst, src, sizeBytes);
  if (r == hipSuccess) {
    hrr_cap::Hash128 h{0, 0};
    if (src && sizeBytes > 0) h = hrr_cap::writer::write_blob(src, sizeBytes);
    hrr_args_hipMemcpyHtoD a{};
    a.ret          = static_cast<int32_t>(r);
    a.dst          = reinterpret_cast<uint64_t>(dst);
    a.src          = reinterpret_cast<uint64_t>(src);
    a.sizeBytes    = static_cast<uint64_t>(sizeBytes);
    a.blob_hash_lo = h.lo;
    a.blob_hash_hi = h.hi;
    hrr_cap::writer::write_event_raw(HRR_API_HIPMEMCPYHTOD, &a.hdr, sizeof(a));
  }
  return r;
}

hipError_t capture_hipMemcpyHtoDAsync(hipDeviceptr_t dst, const void* src,
                                      size_t sizeBytes, hipStream_t stream) {
  hipError_t r = g_real_table.hipMemcpyHtoDAsync_fn(dst, src, sizeBytes, stream);
  if (r == hipSuccess) {
    hrr_cap::Hash128 h{0, 0};
    if (src && sizeBytes > 0) h = hrr_cap::writer::write_blob(src, sizeBytes);
    hrr_args_hipMemcpyHtoDAsync a{};
    a.ret          = static_cast<int32_t>(r);
    a.dst          = reinterpret_cast<uint64_t>(dst);
    a.src          = reinterpret_cast<uint64_t>(src);
    a.sizeBytes    = static_cast<uint64_t>(sizeBytes);
    a.stream       = reinterpret_cast<uint64_t>(stream);
    a.blob_hash_lo = h.lo;
    a.blob_hash_hi = h.hi;
    hrr_cap::writer::write_event_raw(HRR_API_HIPMEMCPYHTODASYNC, &a.hdr, sizeof(a));
  }
  return r;
}

// hipMemcpyWithStream — synchronous copy with an explicit stream.
// Semantics: blocks until the copy completes (like hipMemcpy but with stream).
// H2D: host src is valid immediately on return — snapshot directly.
// D2H: host dst is valid immediately on return — snapshot directly.
// D2D: no blob data needed (both ends are GPU pointers).
hipError_t capture_hipMemcpyWithStream(void* dst, const void* src,
                                       size_t sizeBytes, hipMemcpyKind kind,
                                       hipStream_t stream) {
  hipError_t r = g_real_table.hipMemcpyWithStream_fn(dst, src, sizeBytes, kind, stream);
  if (r == hipSuccess) {
    hrr_cap::Hash128 h{0, 0};
    if (kind == hipMemcpyHostToDevice && src && sizeBytes > 0) {
      h = hrr_cap::writer::write_blob(src, sizeBytes);
    } else if (kind == hipMemcpyDeviceToHost && dst && sizeBytes > 0) {
      // Call is synchronous — dst is valid immediately after return.
      h = hrr_cap::writer::write_blob(dst, sizeBytes);
    }
    hrr_args_hipMemcpyWithStream a{};
    a.ret          = static_cast<int32_t>(r);
    a.dst          = reinterpret_cast<uint64_t>(dst);
    a.src          = reinterpret_cast<uint64_t>(src);
    a.sizeBytes    = static_cast<uint64_t>(sizeBytes);
    a.kind         = static_cast<int32_t>(kind);
    a.stream       = reinterpret_cast<uint64_t>(stream);
    a.blob_hash_lo = h.lo;
    a.blob_hash_hi = h.hi;
    hrr_cap::writer::write_event_raw(HRR_API_HIPMEMCPYWITHSTREAM, &a.hdr, sizeof(a));
  }
  return r;
}

// ---------------------------------------------------------------------------
// 2D memcpy shims — hipDrvMemcpy2DUnaligned / hipMemcpyParam2D[Async]
// ---------------------------------------------------------------------------
//
// hip_Memcpy2D can carry H2D, D2H, or D2D copies depending on srcMemoryType /
// dstMemoryType.  For H2D copies the source is a pitched host buffer; we
// linearise the copied region (Height rows × WidthInBytes bytes) into a
// contiguous blob so that the playback shim can restore the data without
// needing the original pitch layout.

static hrr_cap::Hash128 snapshot_2d_host_src(const hip_Memcpy2D* p) {
  if (p->srcMemoryType != hipMemoryTypeHost || !p->srcHost ||
      p->Height == 0 || p->WidthInBytes == 0)
    return {0, 0};
  const size_t pitch = p->srcPitch ? p->srcPitch : p->WidthInBytes;
  const size_t total = p->Height * p->WidthInBytes;
  std::vector<uint8_t> linear(total);
  const char* base =
      reinterpret_cast<const char*>(p->srcHost) +
      p->srcY * pitch + p->srcXInBytes;
  for (size_t row = 0; row < p->Height; ++row)
    memcpy(linear.data() + row * p->WidthInBytes, base + row * pitch,
           p->WidthInBytes);
  return hrr_cap::writer::write_blob(linear.data(), total);
}

// Macro to populate the extra hip_Memcpy2D fields that the generator appended to
// each 2D-copy args struct.  Each of the three structs has these fields at a
// different offset (hipMemcpyParam2DAsync has an extra `stream` base field), so
// we use a macro rather than a shared helper to avoid unsafe cross-struct casts.
#define FILL_2D_EXTRA_FIELDS(a, p, h)                                  \
  do {                                                                  \
    (a).src_x_bytes  = static_cast<uint64_t>((p)->srcXInBytes);       \
    (a).src_y        = static_cast<uint64_t>((p)->srcY);               \
    (a).src_mem_type = static_cast<uint32_t>((p)->srcMemoryType);      \
    (a).pad0         = 0;                                               \
    (a).src_host     = reinterpret_cast<uint64_t>((p)->srcHost);       \
    (a).src_device   = reinterpret_cast<uint64_t>((p)->srcDevice);     \
    (a).src_array    = reinterpret_cast<uint64_t>((p)->srcArray);      \
    (a).src_pitch    = static_cast<uint64_t>((p)->srcPitch);           \
    (a).dst_x_bytes  = static_cast<uint64_t>((p)->dstXInBytes);       \
    (a).dst_y        = static_cast<uint64_t>((p)->dstY);               \
    (a).dst_mem_type = static_cast<uint32_t>((p)->dstMemoryType);      \
    (a).pad1         = 0;                                               \
    (a).dst_host     = reinterpret_cast<uint64_t>((p)->dstHost);       \
    (a).dst_device   = reinterpret_cast<uint64_t>((p)->dstDevice);     \
    (a).dst_array    = reinterpret_cast<uint64_t>((p)->dstArray);      \
    (a).dst_pitch    = static_cast<uint64_t>((p)->dstPitch);           \
    (a).width_bytes  = static_cast<uint64_t>((p)->WidthInBytes);       \
    (a).height       = static_cast<uint64_t>((p)->Height);             \
    (a).blob_hash_lo = (h).lo;                                         \
    (a).blob_hash_hi = (h).hi;                                         \
  } while (0)

hipError_t capture_hipDrvMemcpy2DUnaligned(const hip_Memcpy2D* pCopy) {
  hipError_t r = g_real_table.hipDrvMemcpy2DUnaligned_fn(pCopy);
  if (r == hipSuccess && pCopy) {
    hrr_cap::Hash128 h = snapshot_2d_host_src(pCopy);
    hrr_args_hipDrvMemcpy2DUnaligned a{};
    a.ret   = static_cast<int32_t>(r);
    a.pCopy = reinterpret_cast<uint64_t>(pCopy);
    FILL_2D_EXTRA_FIELDS(a, pCopy, h);
    hrr_cap::writer::write_event_raw(HRR_API_HIPDRVMEMCPY2DUNALIGNED, &a.hdr, sizeof(a));
  }
  return r;
}

hipError_t capture_hipMemcpyParam2D(const hip_Memcpy2D* pCopy) {
  hipError_t r = g_real_table.hipMemcpyParam2D_fn(pCopy);
  if (r == hipSuccess && pCopy) {
    hrr_cap::Hash128 h = snapshot_2d_host_src(pCopy);
    hrr_args_hipMemcpyParam2D a{};
    a.ret   = static_cast<int32_t>(r);
    a.pCopy = reinterpret_cast<uint64_t>(pCopy);
    FILL_2D_EXTRA_FIELDS(a, pCopy, h);
    hrr_cap::writer::write_event_raw(HRR_API_HIPMEMCPYPARAM2D, &a.hdr, sizeof(a));
  }
  return r;
}

hipError_t capture_hipMemcpyParam2DAsync(const hip_Memcpy2D* pCopy,
                                         hipStream_t stream) {
  hipError_t r = g_real_table.hipMemcpyParam2DAsync_fn(pCopy, stream);
  if (r == hipSuccess && pCopy) {
    hrr_cap::Hash128 h = snapshot_2d_host_src(pCopy);
    hrr_args_hipMemcpyParam2DAsync a{};
    a.ret    = static_cast<int32_t>(r);
    a.pCopy  = reinterpret_cast<uint64_t>(pCopy);
    a.stream = reinterpret_cast<uint64_t>(stream);
    FILL_2D_EXTRA_FIELDS(a, pCopy, h);
    hrr_cap::writer::write_event_raw(HRR_API_HIPMEMCPYPARAM2DASYNC, &a.hdr, sizeof(a));
  }
  return r;
}

#undef FILL_2D_EXTRA_FIELDS

// ---------------------------------------------------------------------------
// Module shims
// ---------------------------------------------------------------------------

// Helper: get the actual device binary from a successfully loaded hipModule_t.
// The caller passes an in-memory image (which may be a fat binary bundle, not
// a raw ELF).  The runtime unbundles/extracts the device ELF internally and
// stores it in the amd::Program.  We read it back from there so we always
// capture the processed ELF, not the raw (possibly bundled) input image.
static hrr_cap::Hash128 write_module_code_object(hipModule_t module) {
  amd::Program* prog = as_amd(reinterpret_cast<cl_program>(module));
  if (!prog) return {0, 0};
  const int dev = hip::ihipGetDevice();
  const amd::Device* device = hip::g_devices[dev]->devices()[0];
  const auto& bin = prog->binary(*device);
  const uint8_t* data = std::get<0>(bin);
  const size_t   size = std::get<1>(bin).first;
  if (!data || !size) return {0, 0};
  return hrr_cap::writer::write_code_object(data, size);
}

// Infer the byte-size of a raw code-object image by sniffing its magic header.
// Returns 0 when the format is unrecognised.
//
// Three formats accepted (all present in real ROCm workloads):
//   1. Compressed fat bundle  — starts with "CCOB"; totalSize at byte 8
//   2. Uncompressed fat bundle — starts with "__CLANG_OFFLOAD_BUNDLE__";
//      walk all bundle entries, total = max(offset+size) over all entries
//   3. ELF                     — starts with "\x7fELF"; total = e_shoff +
//      e_shnum * e_shentsize (section headers are typically the last bytes)
static size_t infer_image_size(const void* image) {
  if (!image) return 0;
  const auto* b = static_cast<const uint8_t*>(image);

  // Compressed fat binary: "CCOB" magic + u16 version + u16 method + u32 totalSize
  if (b[0]=='C' && b[1]=='C' && b[2]=='O' && b[3]=='B') {
    uint32_t total = 0;
    std::memcpy(&total, b + 8, 4);  // totalSize field
    return static_cast<size_t>(total);
  }

  // Uncompressed fat binary: "__CLANG_OFFLOAD_BUNDLE__" magic
  const char* kMagic = hip::symbols::kOffloadBundleUncompressedMagicStr;
  const size_t kMagicLen = hip::symbols::kOffloadBundleUncompressedMagicStrSize - 1;
  if (std::memcmp(b, kMagic, kMagicLen) == 0) {
    uint64_t num = 0;
    std::memcpy(&num, b + kMagicLen, 8);
    const uint8_t* p = b + kMagicLen + 8;
    size_t end = 0;
    for (uint64_t i = 0; i < num; ++i) {
      uint64_t offset = 0, size = 0, id_size = 0;
      std::memcpy(&offset,  p,     8);
      std::memcpy(&size,    p + 8, 8);
      std::memcpy(&id_size, p + 16, 8);
      size_t entry_end = static_cast<size_t>(offset + size);
      if (entry_end > end) end = entry_end;
      p += 24 + static_cast<size_t>(id_size);
    }
    return end;
  }

  // ELF: "\x7fELF" magic (64-bit LE assumed — all AMD code objects are ELF64 LE)
  if (b[0]==0x7f && b[1]=='E' && b[2]=='L' && b[3]=='F') {
    uint64_t shoff = 0; uint16_t shentsize = 0, shnum = 0;
    std::memcpy(&shoff,    b + 40, 8);  // e_shoff
    std::memcpy(&shentsize, b + 58, 2); // e_shentsize
    std::memcpy(&shnum,    b + 60, 2);  // e_shnum
    if (shoff && shentsize && shnum)
      return static_cast<size_t>(shoff) + shentsize * shnum;
  }

  return 0;  // unrecognised format
}

// When write_module_code_object fails (e.g. lazy-compilation not yet complete),
// fall back to hashing the raw input image.  infer_image_size sniffs the format
// to determine the byte count; if that too fails the hash stays {0,0} and the
// playback shim will skip the module gracefully rather than aborting.
static hrr_cap::Hash128 write_module_or_image(hipModule_t module,
                                               const void* image) {
  hrr_cap::Hash128 h = write_module_code_object(module);
  if (!h.lo && !h.hi && image) {
    size_t sz = infer_image_size(image);
    if (sz > 0)
      h = hrr_cap::writer::write_code_object(image, sz);
    else
      LogPrintfWarning("[HRR capture] hipModuleLoad*: could not infer image size"
                       " — code object not snapshotted");
  }
  return h;
}

hipError_t capture_hipModuleGetFunction(hipFunction_t* function,
                                        hipModule_t module,
                                        const char* kname) {
  hipError_t r = g_real_table.hipModuleGetFunction_fn(function, module, kname);
  if (r == hipSuccess && kname && function) {
    size_t name_len = strlen(kname) + 1;
    hrr_cap::Hash128 h = hrr_cap::writer::write_blob(kname, name_len);
    hrr_args_hipModuleGetFunction a{};
    a.ret          = static_cast<int32_t>(r);
    a.function     = reinterpret_cast<uint64_t>(*function);
    a.module       = reinterpret_cast<uint64_t>(module);
    a.kname        = reinterpret_cast<uint64_t>(kname);
    a.fname_hash_lo = h.lo;
    a.fname_hash_hi = h.hi;
    hrr_cap::writer::write_event_raw(HRR_API_HIPMODULEGETFUNCTION, &a.hdr, sizeof(a));
  }
  return r;
}

hipError_t capture_hipModuleLoadData(hipModule_t* module, const void* image) {
  hipError_t r = g_real_table.hipModuleLoadData_fn(module, image);
  if (r == hipSuccess) {
    hrr_cap::Hash128 h = write_module_or_image(*module, image);
    remember_module_hash(*module, h);
    hrr_args_hipModuleLoadData a{};
    a.ret        = static_cast<int32_t>(r);
    a.module     = reinterpret_cast<uint64_t>(*module);
    a.image      = reinterpret_cast<uint64_t>(image);
    a.co_hash_lo = h.lo;
    a.co_hash_hi = h.hi;
    a.module_id  = 0;
    hrr_cap::writer::write_event_raw(HRR_API_HIPMODULELOADDATA, &a.hdr, sizeof(a));
  }
  return r;
}

hipError_t capture_hipModuleLoadDataEx(hipModule_t* module, const void* image,
                                       unsigned int numOptions,
                                       hipJitOption* options,
                                       void** optionValues) {
  hipError_t r = g_real_table.hipModuleLoadDataEx_fn(
      module, image, numOptions, options, optionValues);
  if (r == hipSuccess) {
    hrr_cap::Hash128 h = write_module_or_image(*module, image);
    remember_module_hash(*module, h);
    hrr_args_hipModuleLoadDataEx a{};
    a.ret          = static_cast<int32_t>(r);
    a.module       = reinterpret_cast<uint64_t>(*module);
    a.image        = reinterpret_cast<uint64_t>(image);
    a.numOptions   = static_cast<uint32_t>(numOptions);
    a.options      = reinterpret_cast<uint64_t>(options);
    a.optionValues = reinterpret_cast<uint64_t>(optionValues);
    a.co_hash_lo   = h.lo;
    a.co_hash_hi   = h.hi;
    a.module_id    = 0;
    hrr_cap::writer::write_event_raw(HRR_API_HIPMODULELOADDATAEX, &a.hdr, sizeof(a));
  }
  return r;
}

hipError_t capture_hipModuleLoadFatBinary(hipModule_t* module, const void* fatbin) {
  hipError_t r = g_real_table.hipModuleLoadFatBinary_fn(module, fatbin);
  if (r == hipSuccess && module) {
    hrr_cap::Hash128 h = write_module_or_image(*module, fatbin);
    remember_module_hash(*module, h);
    hrr_args_hipModuleLoadFatBinary a{};
    a.ret        = static_cast<int32_t>(r);
    a.module     = reinterpret_cast<uint64_t>(*module);
    a.fatbin     = reinterpret_cast<uint64_t>(fatbin);
    a.co_hash_lo = h.lo;
    a.co_hash_hi = h.hi;
    a.module_id  = 0;
    hrr_cap::writer::write_event_raw(HRR_API_HIPMODULELOADFATBINARY, &a.hdr, sizeof(a));
  }
  return r;
}

hipError_t capture_hipModuleLoad(hipModule_t* module, const char* fname) {
  hipError_t r = g_real_table.hipModuleLoad_fn(module, fname);
  if (r == hipSuccess) {
    hrr_cap::Hash128 h{0, 0};
    // Note: h is filled below from the file contents — remember_module_hash
    // is called after that, once we have a real hash to store.
    // Read the file from disk and snapshot it as a code object so the replay
    // can load it by hash. Without this, fname is a capture-time address and
    // is useless at replay time.
    if (fname) {
      if (FILE* fh = fopen(fname, "rb")) {
        fseek(fh, 0, SEEK_END);
        long sz = ftell(fh);
        rewind(fh);
        if (sz > 0) {
          std::vector<uint8_t> buf(static_cast<size_t>(sz));
          if (fread(buf.data(), 1, buf.size(), fh) == buf.size())
            h = hrr_cap::writer::write_code_object(buf.data(), buf.size());
        }
        fclose(fh);
      }
    }
    remember_module_hash(*module, h);
    hrr_args_hipModuleLoad a{};
    a.ret        = static_cast<int32_t>(r);
    a.module     = reinterpret_cast<uint64_t>(*module);
    a.fname      = 0;  // not a valid cross-process address; hash identifies the file
    a.co_hash_lo = h.lo;
    a.co_hash_hi = h.hi;
    a.module_id  = 0;
    hrr_cap::writer::write_event_raw(HRR_API_HIPMODULELOAD, &a.hdr, sizeof(a));
  }
  return r;
}

// ---------------------------------------------------------------------------
// Kernel launch shims — variable-length binary payload (not hrr_args_*)
// ---------------------------------------------------------------------------

hipError_t capture_hipModuleLaunchKernel(
    hipFunction_t f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, hipStream_t stream,
    void** kernelParams, void** extra) {
  // Pre-launch snapshots (no-op outside inputs/full mode).  Must happen
  // BEFORE the real launch so direction=0 captures the bytes the kernel
  // actually consumes, not post-launch outputs.
  auto pre = pre_launch_snapshots_from_f(f, kernelParams, extra, stream);
  hipError_t r = g_real_table.hipModuleLaunchKernel_fn(
      f, gridDimX, gridDimY, gridDimZ,
         blockDimX, blockDimY, blockDimZ,
      sharedMemBytes, stream, kernelParams, extra);
  if (r == hipSuccess) {
    record_launch(f, gridDimX, gridDimY, gridDimZ,
                     blockDimX, blockDimY, blockDimZ,
                  sharedMemBytes, stream, kernelParams, extra,
                  std::move(pre));
  }
  return r;
}

hipError_t capture_hipExtModuleLaunchKernel(
    hipFunction_t f,
    uint32_t globalWorkSizeX, uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
    uint32_t localWorkSizeX,  uint32_t localWorkSizeY,  uint32_t localWorkSizeZ,
    size_t sharedMemBytes, hipStream_t stream,
    void** kernelParams, void** extra,
    hipEvent_t startEvent, hipEvent_t stopEvent, uint32_t flags) {
  auto pre = pre_launch_snapshots_from_f(f, kernelParams, extra, stream);
  hipError_t r = g_real_table.hipExtModuleLaunchKernel_fn(
      f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
         localWorkSizeX,  localWorkSizeY,  localWorkSizeZ,
      sharedMemBytes, stream, kernelParams, extra, startEvent, stopEvent, flags);
  if (r == hipSuccess) {
    // hipExtModuleLaunchKernel takes OpenCL-style global work size (total threads
    // per dim) and local work size (threads per block). The replay layer is
    // uniform across all kernel-launch APIs and re-issues every event via
    // hipModuleLaunchKernel, which takes CUDA-style grid dim (number of blocks).
    // Convert here so the recorded grid[3] is replayable — matches the runtime's
    // own conversion at rocclr/platform/ndrange.hpp:144-146
    // (grid_[i] = global_[i] / local_[i], truncating division).
    auto to_grid = [](uint32_t global, uint32_t local) -> unsigned {
      return local ? global / local : global;
    };
    record_launch(f,
                  to_grid(globalWorkSizeX, localWorkSizeX),
                  to_grid(globalWorkSizeY, localWorkSizeY),
                  to_grid(globalWorkSizeZ, localWorkSizeZ),
                  localWorkSizeX,  localWorkSizeY,  localWorkSizeZ,
                  static_cast<unsigned>(sharedMemBytes), stream, kernelParams, extra,
                  std::move(pre));
  }
  return r;
}

// ---------------------------------------------------------------------------
// Old HIP C launch API: hipConfigureCall → hipSetupArgument×N → hipLaunchByPtr
// ---------------------------------------------------------------------------
// The generated shims for hipConfigureCall and __hipPushCallConfiguration write
// events but do NOT update the TLS dims that capture_hipLaunchByPtr reads.
// The generated shim for hipSetupArgument stores only the stale arg pointer, not
// the bytes it points to — causing a segfault on playback.
//
// Fix:
//   capture_hipConfigureCall / capture___hipPushCallConfiguration:
//     Write the event AND update g_pushed_* so capture_hipLaunchByPtr has the
//     correct grid/block/shared/stream.
//   capture_hipSetupArgument:
//     Copy size bytes from arg into g_setuparg_buf at the given offset so the
//     arg bytes accumulate correctly before hipLaunchByPtr.
//   capture_hipLaunchByPtr (existing, updated below):
//     Passes g_setuparg_buf as kbuf to serialize_kernel_launch so the args end
//     up in the kernel launch event payload.

hipError_t capture_hipConfigureCall(dim3 gridDim, dim3 blockDim,
                                    size_t sharedMem, hipStream_t stream) {
  hipError_t r = g_real_table.hipConfigureCall_fn(gridDim, blockDim, sharedMem, stream);
  if (r == hipSuccess) {
    g_pushed_grid   = gridDim;
    g_pushed_block  = blockDim;
    g_pushed_shared = sharedMem;
    g_pushed_stream = stream;
    g_setuparg_buf.clear();
    hrr_args_hipConfigureCall a{};
    a.ret        = static_cast<int32_t>(r);
    a.gridDim_x  = gridDim.x;  a.gridDim_y  = gridDim.y;  a.gridDim_z  = gridDim.z;
    a.blockDim_x = blockDim.x; a.blockDim_y = blockDim.y; a.blockDim_z = blockDim.z;
    a.sharedMem  = static_cast<decltype(a.sharedMem)>(sharedMem);
    a.stream     = reinterpret_cast<uint64_t>(stream);
    hrr_cap::writer::write_event_raw(HRR_API_HIPCONFIGURECALL, &a.hdr, sizeof(a));
  }
  return r;
}

hipError_t capture___hipPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                              size_t sharedMem, hipStream_t stream) {
  hipError_t r = g_real_compiler_table.__hipPushCallConfiguration_fn(
      gridDim, blockDim, sharedMem, stream);
  if (r == hipSuccess) {
    g_pushed_grid   = gridDim;
    g_pushed_block  = blockDim;
    g_pushed_shared = sharedMem;
    g_pushed_stream = stream;
    hrr_args___hipPushCallConfiguration a{};
    a.ret        = static_cast<int32_t>(r);
    a.gridDim_x  = gridDim.x;  a.gridDim_y  = gridDim.y;  a.gridDim_z  = gridDim.z;
    a.blockDim_x = blockDim.x; a.blockDim_y = blockDim.y; a.blockDim_z = blockDim.z;
    a.sharedMem  = static_cast<decltype(a.sharedMem)>(sharedMem);
    a.stream     = reinterpret_cast<uint64_t>(stream);
    hrr_cap::writer::write_event_raw(HRR_API_HIPPUSHCALLCONFIGURATION, &a.hdr, sizeof(a));
  }
  return r;
}

hipError_t capture_hipSetupArgument(const void* arg, size_t size, size_t offset) {
  hipError_t r = g_real_table.hipSetupArgument_fn(arg, size, offset);
  if (r == hipSuccess && hip_capture_enabled()) {
    // Grow the TLS buffer to cover offset+size and copy the argument bytes in.
    size_t end = offset + size;
    if (end > g_setuparg_buf.size())
      g_setuparg_buf.resize(end, 0);
    if (arg)
      std::memcpy(g_setuparg_buf.data() + offset, arg, size);
    hrr_args_hipSetupArgument a{};
    a.ret    = static_cast<int32_t>(r);
    a.arg    = reinterpret_cast<uint64_t>(arg);
    a.size   = static_cast<decltype(a.size)>(size);
    a.offset = static_cast<decltype(a.offset)>(offset);
    hrr_cap::writer::write_event_raw(HRR_API_HIPSETUPARGUMENT, &a.hdr, sizeof(a));
  }
  return r;
}

hipError_t capture_hipLaunchKernel(const void* function_address,
                                           dim3 numBlocks, dim3 dimBlocks,
                                           void** args, size_t sharedMemBytes,
                                           hipStream_t stream) {
  // Resolve the host-stub address to a real hipFunction_t once up front so
  // we can do the pre-launch snapshot BEFORE the real call.  If resolution
  // fails we still launch normally (matching pre-snapshot behaviour) and
  // skip recording — kernel-name-less launches aren't meaningful for replay.
  hipFunction_t f_pre = nullptr;
  std::vector<SnapshotRecord> pre;
  if (hip_capture_enabled() && g_real_table.hipGetFuncBySymbol_fn &&
      g_real_table.hipGetFuncBySymbol_fn(&f_pre, function_address) == hipSuccess &&
      f_pre) {
    pre = pre_launch_snapshots_from_f(f_pre, args, nullptr, stream);
  }
  hipError_t r = g_real_table.hipLaunchKernel_fn(
      function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream);
  if (r == hipSuccess && f_pre) {
    record_launch(f_pre,
                  numBlocks.x, numBlocks.y, numBlocks.z,
                  dimBlocks.x, dimBlocks.y, dimBlocks.z,
                  static_cast<unsigned>(sharedMemBytes), stream, args, nullptr,
                  std::move(pre));
  }
  return r;
}

hipError_t capture_hipExtLaunchKernel(const void* function_address,
                                      dim3 numBlocks, dim3 dimBlocks,
                                      void** args, size_t sharedMemBytes,
                                      hipStream_t stream,
                                      hipEvent_t startEvent,
                                      hipEvent_t stopEvent,
                                      int flags) {
  // Identical to hipLaunchKernel — resolve the host stub to hipFunction_t,
  // snapshot pointer args, call through, then record as a kernel-launch event.
  // startEvent/stopEvent/flags are timing/hint parameters; they don't affect
  // the kernel itself and are not needed for replay correctness.
  hipFunction_t f_pre = nullptr;
  std::vector<SnapshotRecord> pre;
  if (hip_capture_enabled() && g_real_table.hipGetFuncBySymbol_fn &&
      g_real_table.hipGetFuncBySymbol_fn(&f_pre, function_address) == hipSuccess &&
      f_pre) {
    pre = pre_launch_snapshots_from_f(f_pre, args, nullptr, stream);
  }
  hipError_t r = g_real_table.hipExtLaunchKernel_fn(
      function_address, numBlocks, dimBlocks, args, sharedMemBytes,
      stream, startEvent, stopEvent, flags);
  if (r == hipSuccess && f_pre) {
    record_launch(f_pre,
                  numBlocks.x, numBlocks.y, numBlocks.z,
                  dimBlocks.x, dimBlocks.y, dimBlocks.z,
                  static_cast<unsigned>(sharedMemBytes), stream, args, nullptr,
                  std::move(pre));
  }
  return r;
}

// ---------------------------------------------------------------------------
// Multi-kernel multi-device launches
// ---------------------------------------------------------------------------
// hipLaunchParams[] contains per-device func (host pointer), args (stale
// pointer), and stream (stale handle) — none capturable by the auto-generated
// shim which only stores the pointer to the array, not its contents.
//
// Strategy: call the real function first, then walk the params array and call
// record_launch() for each entry that has a resolvable kernel function.  Each
// kernel is recorded as a standard HRR_API_HIPMODULELAUNCHKERNEL event, so
// playback can replay them independently via replay_kernel_launch().  The
// hipExtLaunchMultiKernelMultiDevice / hipLaunchCooperativeKernelMultiDevice
// events themselves are no-ops at playback time (see NOOP_PLAYBACK_APIS).
//
// Note: pre-launch snapshots are not collected for these APIs because the
// pre-launch state must be sampled before the real call, but we need the real
// call to succeed before we know the kernels are worth recording.  Post-launch
// snapshots are still collected by record_launch() (full-mode only).
static void record_multi_device_launches(hipLaunchParams* params, int n) {
  if (!hip_capture_enabled() || !params || n <= 0) return;
  for (int i = 0; i < n; ++i) {
    const hipLaunchParams& p = params[i];
    if (!p.func) continue;
    hipFunction_t f = nullptr;
    if (!g_real_table.hipGetFuncBySymbol_fn ||
        g_real_table.hipGetFuncBySymbol_fn(&f, p.func) != hipSuccess || !f)
      continue;
    record_launch(f,
                  p.gridDim.x,  p.gridDim.y,  p.gridDim.z,
                  p.blockDim.x, p.blockDim.y, p.blockDim.z,
                  static_cast<unsigned>(p.sharedMem),
                  p.stream, p.args, nullptr);
  }
}

hipError_t capture_hipExtLaunchMultiKernelMultiDevice(
    hipLaunchParams* launchParamsList, int numDevices, unsigned int flags) {
  hipError_t r = g_real_table.hipExtLaunchMultiKernelMultiDevice_fn(
      launchParamsList, numDevices, flags);
  if (r == hipSuccess)
    record_multi_device_launches(launchParamsList, numDevices);
  return r;
}

hipError_t capture_hipLaunchCooperativeKernelMultiDevice(
    hipLaunchParams* launchParamsList, int numDevices, unsigned int flags) {
  hipError_t r = g_real_table.hipLaunchCooperativeKernelMultiDevice_fn(
      launchParamsList, numDevices, flags);
  if (r == hipSuccess)
    record_multi_device_launches(launchParamsList, numDevices);
  return r;
}

hipError_t capture_hipLaunchByPtr(const void* func) {
  // Resolve and pre-snapshot first.  We use the TLS dims pushed by
  // __hipPushCallConfiguration to know the launch grid/block.
  hipFunction_t f_pre = nullptr;
  std::vector<SnapshotRecord> pre;
  if (hip_capture_enabled() && g_real_table.hipGetFuncBySymbol_fn &&
      g_real_table.hipGetFuncBySymbol_fn(&f_pre, func) == hipSuccess && f_pre) {
    // hipLaunchByPtr has no kernel_params/extra — args sit in the runtime's
    // ring-buffer kernarg.  collect_pointer_args sees nothing in that case
    // and the snapshot list comes back empty, which is correct: we have
    // nothing to introspect.  Left here for symmetry with other shims.
    pre = pre_launch_snapshots_from_f(f_pre, nullptr, nullptr, g_pushed_stream);
  }
  // Snapshot the TLS arg buffer NOW (before real call) so pre-launch contents
  // reflect what the kernel actually received.
  const void* kbuf = g_setuparg_buf.empty() ? nullptr : g_setuparg_buf.data();
  size_t      ksz  = g_setuparg_buf.size();

  hipError_t r = g_real_table.hipLaunchByPtr_fn(func);
  if (r == hipSuccess && f_pre) {
    amd::Kernel* kernel = hip::asKernel(f_pre);
    if (kernel) {
      const amd::KernelSignature& sig = kernel->signature();
      const void* prog = static_cast<const void*>(&kernel->program());
      hrr_cap::Hash128 co_hash = hash_for_program(prog);
      // Post-launch sweep (full mode only) — same as record_launch does.
      capture_post_launch_snapshots(sig, nullptr, kbuf, ksz,
                                    g_pushed_stream, pre);
      serialize_kernel_launch(
          kernel->name().c_str(),
          co_hash.lo, co_hash.hi,
          g_pushed_grid.x, g_pushed_grid.y, g_pushed_grid.z,
          g_pushed_block.x, g_pushed_block.y, g_pushed_block.z,
          static_cast<uint32_t>(g_pushed_shared), g_pushed_stream,
          sig, nullptr, kbuf, ksz, pre);
    }
  }
  // Clear the TLS arg buffer so it doesn't leak into the next launch.
  g_setuparg_buf.clear();
  return r;
}

hipError_t capture_hipLaunchCooperativeKernel(const void* f, dim3 gridDim,
                                              dim3 blockDimX, void** kernelParams,
                                              unsigned int sharedMemBytes,
                                              hipStream_t stream) {
  hipFunction_t f_pre = nullptr;
  std::vector<SnapshotRecord> pre;
  if (hip_capture_enabled() && g_real_table.hipGetFuncBySymbol_fn &&
      g_real_table.hipGetFuncBySymbol_fn(&f_pre, f) == hipSuccess && f_pre) {
    pre = pre_launch_snapshots_from_f(f_pre, kernelParams, nullptr, stream);
  }
  hipError_t r = g_real_table.hipLaunchCooperativeKernel_fn(
      f, gridDim, blockDimX, kernelParams, sharedMemBytes, stream);
  if (r == hipSuccess && f_pre) {
    record_launch(f_pre,
                  gridDim.x, gridDim.y, gridDim.z,
                  blockDimX.x, blockDimX.y, blockDimX.z,
                  sharedMemBytes, stream, kernelParams, nullptr,
                  std::move(pre),
                  HRR_API_HIPLAUNCHCOOPERATIVEKERNEL);
  }
  return r;
}

hipError_t capture_hipLaunchCooperativeKernel_spt(const void* f, dim3 gridDim,
                                                  dim3 blockDim, void** kernelParams,
                                                  uint32_t sharedMemBytes,
                                                  hipStream_t hStream) {
  hipFunction_t f_pre = nullptr;
  std::vector<SnapshotRecord> pre;
  if (hip_capture_enabled() && g_real_table.hipGetFuncBySymbol_fn &&
      g_real_table.hipGetFuncBySymbol_fn(&f_pre, f) == hipSuccess && f_pre) {
    pre = pre_launch_snapshots_from_f(f_pre, kernelParams, nullptr, hStream);
  }
  hipError_t r = g_real_table.hipLaunchCooperativeKernel_spt_fn(
      f, gridDim, blockDim, kernelParams, sharedMemBytes, hStream);
  if (r == hipSuccess && f_pre) {
    record_launch(f_pre,
                  gridDim.x, gridDim.y, gridDim.z,
                  blockDim.x, blockDim.y, blockDim.z,
                  sharedMemBytes, hStream, kernelParams, nullptr,
                  std::move(pre),
                  HRR_API_HIPLAUNCHCOOPERATIVEKERNEL_SPT);
  }
  return r;
}

// hipModuleLaunchCooperativeKernel — driver-level cooperative launch.
// Like hipModuleLaunchKernel it takes a live hipFunction_t and kernelParams[].
// Unlike hipModuleLaunchKernel it has no `extra[]` parameter, so we pass
// nullptr for extra in the record_launch call.
hipError_t capture_hipModuleLaunchCooperativeKernel(
    hipFunction_t f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, hipStream_t stream, void** kernelParams) {
  auto pre = pre_launch_snapshots_from_f(f, kernelParams, nullptr, stream);
  hipError_t r = g_real_table.hipModuleLaunchCooperativeKernel_fn(
      f, gridDimX, gridDimY, gridDimZ,
         blockDimX, blockDimY, blockDimZ,
      sharedMemBytes, stream, kernelParams);
  if (r == hipSuccess) {
    record_launch(f, gridDimX, gridDimY, gridDimZ,
                     blockDimX, blockDimY, blockDimZ,
                  sharedMemBytes, stream, kernelParams, /*extra=*/nullptr,
                  std::move(pre),
                  HRR_API_HIPMODULELAUNCHCOOPERATIVEKERNEL);
  }
  return r;
}

// hipDrvLaunchKernelEx — extensible driver launch with HIP_LAUNCH_CONFIG.
// The config struct carries grid/block/shared/stream plus an optional attrs[]
// array that may indicate a cooperative launch.  The generated shim passes the
// stale config pointer straight to hipDrvLaunchKernelEx and also emits
// (void**)&nullptr for params/extra — both cause error 400 at replay.
//
// Fix: extract the launch dimensions from config, detect cooperative mode via
// attrs, and call record_launch() with the right api_id so replay_kernel_launch
// uses the correct launch API.
hipError_t capture_hipDrvLaunchKernelEx(const HIP_LAUNCH_CONFIG* config,
                                        hipFunction_t f,
                                        void** params, void** extra) {
  hipStream_t stream = config ? config->hStream : nullptr;
  auto pre = pre_launch_snapshots_from_f(f, params, extra, stream);
  hipError_t r = g_real_table.hipDrvLaunchKernelEx_fn(config, f, params, extra);
  if (r == hipSuccess && config) {
    // Detect cooperative launch attribute.
    hrr_api_id_t api_id = HRR_API_HIPDRVLAUNCHKERNELEX;
    if (config->attrs && config->numAttrs > 0) {
      for (unsigned i = 0; i < config->numAttrs; ++i) {
        if (config->attrs[i].id == hipLaunchAttributeCooperative &&
            config->attrs[i].val.cooperative) {
          api_id = HRR_API_HIPMODULELAUNCHCOOPERATIVEKERNEL;
          break;
        }
      }
    }
    record_launch(f,
                  config->gridDimX, config->gridDimY, config->gridDimZ,
                  config->blockDimX, config->blockDimY, config->blockDimZ,
                  config->sharedMemBytes, stream,
                  params, extra, std::move(pre), api_id);
  }
  return r;
}

// ---------------------------------------------------------------------------
// Fat binary registration — record the binary blob
//
// We capture the clang offload bundle blob so the replay can load it via
// hipModuleLoadData, making all kernel names resolvable.
// ---------------------------------------------------------------------------

// Compute the total byte size of a clang offload bundle blob.
// Supports both uncompressed ("__CLANG_OFFLOAD_BUNDLE__") and compressed ("CCOB") formats.
// Returns 0 if the format is unrecognised.
static size_t compute_bundle_size(const void* blob) {
  if (!blob) return 0;
  const char* p = static_cast<const char*>(blob);

  // Compressed format: magic "CCOB", header contains totalSize at byte 8.
  if (std::memcmp(p, hip::symbols::kOffloadBundleCompressedMagicStr,
                  hip::symbols::kOffloadBundleCompressedMagicStrSize - 1) == 0) {
    const auto* hdr = static_cast<const hip::symbols::ClangOffloadBundleCompressedHeader*>(blob);
    return static_cast<size_t>(hdr->totalSize);
  }

  // Uncompressed format: magic "__CLANG_OFFLOAD_BUNDLE__" + numOfCodeObjects + entries.
  if (std::memcmp(p, hip::symbols::kOffloadBundleUncompressedMagicStr,
                  hip::symbols::kOffloadBundleUncompressedMagicStrSize - 1) != 0) {
    return 0;  // Unknown format
  }
  const auto* hdr = static_cast<const hip::symbols::ClangOffloadBundleUncompressedHeader*>(blob);
  uint64_t n = hdr->numOfCodeObjects;
  if (n == 0) return 0;

  // Walk entries to find the last offset + size (that is the blob end).
  size_t end = 0;
  const uint8_t* cur = reinterpret_cast<const uint8_t*>(&hdr->desc[0]);
  for (uint64_t i = 0; i < n; i++) {
    const auto* entry = reinterpret_cast<const hip::symbols::ClangOffloadBundleInfo*>(cur);
    size_t entry_end = static_cast<size_t>(entry->offset) + static_cast<size_t>(entry->size);
    if (entry_end > end) end = entry_end;
    // Advance past this entry: three uint64_t fields + bundleEntryIdSize bytes
    cur += 3 * sizeof(uint64_t) + entry->bundleEntryIdSize;
  }
  return end;
}

void** capture___hipRegisterFatBinary(const void* data) {
  void** r = g_real_compiler_table.__hipRegisterFatBinary_fn(data);
  // Shim is only installed when capture is active — no hip_capture_enabled() check needed.

  // data is a __CudaFatBinaryWrapper* { magic, version, binary, dummy }.
  // Capture the fat binary blob (binary field) so replay can load it via hipModuleLoadData.
  struct __HRRFatBinaryWrapper { uint32_t magic; uint32_t version; const void* binary; const void* dummy; };
  const auto* wrapper = static_cast<const __HRRFatBinaryWrapper*>(data);
  const void* blob = (wrapper && (wrapper->magic == 0x48495046u /*HIPF*/ ||
                                   wrapper->magic == 0x4B504948u /*HIPK*/))
                     ? wrapper->binary : nullptr;
  size_t blob_size = blob ? compute_bundle_size(blob) : 0;

  hrr_args___hipRegisterFatBinary a{};
  a.ret      = reinterpret_cast<uint64_t>(r);
  a.blob_size = static_cast<uint64_t>(blob_size);
  if (blob && blob_size > 0) {
    auto h = hrr_cap::writer::write_blob(blob, blob_size);
    a.blob_hash_lo = h.lo;
    a.blob_hash_hi = h.hi;
  }
  hrr_cap::writer::write_event_raw(HRR_API_HIPREGISTERFATBINARY, &a.hdr, sizeof(a));
  return r;
}

// ---------------------------------------------------------------------------
// hipHostRegister / hipHostUnregister — sysmem blob snapshotting
//
// We snapshot the host memory at Register time so the replayer can restore it
// before calling hipHostRegister on a freshly allocated buffer.
// hipHostUnregister doesn't receive a size, so we track it in pinned_reg_map.
// ---------------------------------------------------------------------------

static std::mutex                              g_pinned_reg_mu;
static std::unordered_map<void*, size_t>       g_pinned_reg_map;

hipError_t capture_hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags) {
  hipError_t r = g_real_table.hipHostRegister_fn(hostPtr, sizeBytes, flags);
  if (r == hipSuccess) {
    hrr_cap::Hash128 h{0, 0};
    if (hostPtr && sizeBytes > 0)
      h = hrr_cap::writer::write_blob(hostPtr, sizeBytes);
    hrr_args_hipHostRegister a{};
    a.ret          = static_cast<int32_t>(r);
    a.hostPtr      = reinterpret_cast<uint64_t>(hostPtr);
    a.sizeBytes    = static_cast<uint64_t>(sizeBytes);
    a.flags        = flags;
    a.blob_hash_lo = h.lo;
    a.blob_hash_hi = h.hi;
    hrr_cap::writer::write_event_raw(HRR_API_HIPHOSTREGISTER, &a.hdr, sizeof(a));
    {
      std::lock_guard<std::mutex> lk(g_pinned_reg_mu);
      g_pinned_reg_map[hostPtr] = sizeBytes;
    }
  }
  return r;
}

hipError_t capture_hipHostUnregister(void* hostPtr) {
  hipError_t r = g_real_table.hipHostUnregister_fn(hostPtr);
  if (r == hipSuccess) {
    hrr_args_hipHostUnregister a{};
    a.ret     = static_cast<int32_t>(r);
    a.hostPtr = reinterpret_cast<uint64_t>(hostPtr);
    hrr_cap::writer::write_event_raw(HRR_API_HIPHOSTUNREGISTER, &a.hdr, sizeof(a));
    {
      std::lock_guard<std::mutex> lk(g_pinned_reg_mu);
      g_pinned_reg_map.erase(hostPtr);
    }
  }
  return r;
}

// ---------------------------------------------------------------------------
// Install / uninstall (build_table functions live in hip_capture_generated.cpp)
// ---------------------------------------------------------------------------

void hip_capture_install() {
  if (g_installed.exchange(true)) return;
  std::memcpy(const_cast<HipDispatchTable*>(hip::GetHipDispatchTable()),
              &g_cap_table, sizeof(HipDispatchTable));
}

void hip_capture_uninstall() {
  if (!g_installed.exchange(false)) return;
  std::memcpy(const_cast<HipDispatchTable*>(hip::GetHipDispatchTable()),
              &g_real_table, sizeof(HipDispatchTable));
}

// ---------------------------------------------------------------------------
// Init / Shutdown
// ---------------------------------------------------------------------------

// Record a single fat binary blob as a HRR_API_HIPREGISTERFATBINARY event.
// blob_ptr is the fbwrapper->binary pointer (the actual clang offload bundle).
static void record_fat_binary_blob(const void* blob_ptr) {
  if (!blob_ptr) return;
  size_t blob_size = compute_bundle_size(blob_ptr);
  if (blob_size == 0) return;

  hrr_args___hipRegisterFatBinary a{};
  a.ret      = 0;  // handle not meaningful at init time
  a.blob_size = static_cast<uint64_t>(blob_size);
  auto h = hrr_cap::writer::write_blob(blob_ptr, blob_size);
  a.blob_hash_lo = h.lo;
  a.blob_hash_hi = h.hi;
  hrr_cap::writer::write_event_raw(HRR_API_HIPREGISTERFATBINARY, &a.hdr, sizeof(a));
}

// ---------------------------------------------------------------------------
// hip_capture_init — called from hip::init() (hip_context.cpp), at the END of
// runtime initialization.  By this point amd::Runtime::init() / HSA / device
// enumeration have all completed, the ROCclr flag system is live (so
// HIP_HRR_CAPTURE_OUTPUT can be read via flagIsDefault), PlatformState has
// fat-binary metadata, and the dispatch table is in its final form.
//
// Installing the shims here trades capture of the very first API call (the
// one that triggered hip::init() itself, e.g. hipGetDevice) for robustness:
// snapshotting the dispatch table at static-init time forces ToolsInit() /
// rocprofiler_register_library_api_table() to run before HSA is ready, which
// has been observed to leave amd::Device::getDevices() returning zero devices
// for the rest of the process (manifests in callers as hipErrorNoDevice).
//
// Fat binaries that fired via __hipRegisterFatBinary during app static-init —
// before our compiler-table shim is installed — are recovered retroactively
// via ForEachFatBinaryBlob().
// ---------------------------------------------------------------------------
void hip_capture_init() {
  if (!hip_capture_enabled()) return;
  hip_capture_build_table();
  if (!hrr_cap::writer::open(hip_capture_output_dir())) return;

  // Retroactively record fat binaries that fired before our shims were live.
  // __hipRegisterFatBinary fires at app static-init, before hip_capture_init().
  // Done before install so the recorded events precede any captured launches.
  hip::PlatformState::Instance().StatCO().ForEachFatBinaryBlob(record_fat_binary_blob);

  hip_capture_install();
  hip_capture_build_compiler_table();
  std::atexit(hip_capture_shutdown);
}

void hip_capture_shutdown() {
  hip_capture_uninstall();
  hrr_cap::writer::flush(hip_capture_output_dir());
  hrr_cap::writer::close();

  LogPrintfInfo("[HRR capture] Wrote %llu events, %llu blobs to: %s",
               static_cast<unsigned long long>(hrr_cap::writer::event_count()),
               static_cast<unsigned long long>(hrr_cap::writer::blob_count()),
               hip_capture_output_dir());
}
