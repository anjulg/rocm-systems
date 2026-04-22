/* Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License */

// hrr-replay: Replay a .hrr trace archive on the current GPU.
//
// Usage: hrr-replay <capture.hrr> [--verify] [--timing] [--kernel-filter NAME]

#include "hrr_reader.h"

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <chrono>

enum class PrintDtype { fp32, fp16 };

static float half_to_float(uint16_t h) {
  uint32_t sign = (uint32_t)(h >> 15) << 31;
  uint32_t exp  = (h >> 10) & 0x1f;
  uint32_t mant = h & 0x3ff;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {
      // Denormal: normalize
      exp = 1;
      while (!(mant & 0x400)) { mant <<= 1; exp--; }
      mant &= 0x3ff;
      f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    f = sign | 0x7f800000 | (mant << 13);
  } else {
    f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float result;
  memcpy(&result, &f, sizeof(result));
  return result;
}

// Most-recently-launched kernel info, used to attribute deferred GPU
// errors (e.g. illegal-memory-access page faults that don't surface until
// the next driver call after the faulty launch).
struct LastAsyncOp {
  std::string kind;          // "kernel <name>", "memcpy H2D", ...
  uint64_t event_seq = 0;    // sequence id of the producing event
};
static LastAsyncOp g_last_async;

// Wraps a HIP call and reports both the call's own error and any sticky
// context error that pre-dated it.  Returns hipSuccess only if both are
// clean; returns the first non-success error otherwise.
//
// If the call returns success but a sticky error existed, that means the
// real fault was earlier (an async op).  The return-value error and the
// sticky error may be the same code — we always blame the earlier op
// since it's the actual cause.
#define HIP_CHECK_AT(call, where)                                             \
  do {                                                                        \
    hipError_t _pre  = hipPeekAtLastError();                                  \
    hipError_t _ret  = (call);                                                \
    hipError_t _err  = (_pre != hipSuccess) ? _pre : _ret;                    \
    if (_err != hipSuccess) {                                                 \
      const char* _attr = (_pre != hipSuccess && !g_last_async.kind.empty()) \
          ? g_last_async.kind.c_str() : (where);                              \
      fprintf(stderr,                                                         \
              "[HRR] HIP error %d (%s)\n"                                     \
              "[HRR]   surfaced at: %s (%s:%d)\n"                             \
              "[HRR]   blamed on:   %s%s\n",                                  \
              _err, hipGetErrorString(_err), (where), __FILE__, __LINE__,     \
              _attr,                                                          \
              (_pre != hipSuccess && !g_last_async.kind.empty())              \
                  ? " (deferred from earlier async op)" : "");                \
      (void)hipGetLastError();                                                \
      return _err;                                                            \
    }                                                                         \
  } while (0)

#define HIP_CHECK(call) HIP_CHECK_AT(call, #call)

struct ReplayState {
  // Handle -> live GPU pointer
  std::unordered_map<uint64_t, void*> alloc_map;
  // Handle -> allocation size
  std::unordered_map<uint64_t, size_t> alloc_sizes;
  // Module handle -> hipModule_t
  std::unordered_map<uint64_t, hipModule_t> module_map;
  // Code object hash -> loaded module
  std::unordered_map<std::string, hipModule_t> co_modules;
  // Stream handle -> hipStream_t
  std::unordered_map<uint32_t, hipStream_t> stream_map;

  bool verify = false;
  // FP32 verification tolerances.  GPU work is rarely bit-exact across
  // runs (atomic-op ordering, scheduler differences) so a strict memcmp
  // produces noisy false positives.  Pass if max element-wise diff is
  // within atol + rtol * max(|expected|).
  float verify_atol = 1e-3f;
  float verify_rtol = 1e-3f;
  // Skip snapshots whose pre-launch buffer state already matches the
  // recorded "expected" — those are read-only kernel args (Tensile A/B,
  // bias, weights) that the writer can't tell apart from real outputs at
  // record time.  Reduces verifier noise for kernels with many input
  // pointers (GEMM, conv) without losing any real output check.
  bool verify_outputs_only = false;
  uint64_t verify_inputs_skipped = 0;
  bool fail_fast = false;  // stop at first verify mismatch (--fail-fast)
  // Handles that were ever classified as real outputs (pre-launch state !=
  // expected).  Once a handle is known to be a real output we never re-
  // classify it as an input, even if a later launch finds it already
  // holds the expected value (because the previous launch wrote it).
  std::unordered_set<uint64_t> known_output_handles;
  bool timing = false;
  bool skip_device_sync = false;
  // Sync after every kernel launch by default.  Replay is a correctness
  // tool, not a benchmark — making faults surface at the launch that
  // caused them (instead of at some unrelated later driver call) is
  // worth the per-kernel sync cost.  Disable with --async if needed.
  bool sync_after_launch = true;
  PrintDtype print_dtype = PrintDtype::fp32;
  bool verbose = false;
  std::string kernel_filter;
  std::string params_file;

  // All overrides from run_params.json
  hrr::RunParams run_params;

  size_t kernels_launched = 0;
  size_t verify_pass = 0;
  size_t verify_fail = 0;
  double total_kernel_ms = 0.0;
};

static void* translate_ptr(ReplayState& state, uint64_t handle) {
  auto it = state.alloc_map.find(handle);
  if (it != state.alloc_map.end()) return it->second;
  // Might be a sub-allocation (handle is base + offset)
  // Try range lookup
  for (auto& [h, ptr] : state.alloc_map) {
    auto sz_it = state.alloc_sizes.find(h);
    if (sz_it != state.alloc_sizes.end()) {
      if (handle >= h && handle < h + sz_it->second) {
        return static_cast<char*>(ptr) + (handle - h);
      }
    }
  }
  return nullptr;
}

static hipModule_t get_module_for_co(ReplayState& state,
                                     const hrr::Archive& archive,
                                     uint64_t hash_lo, uint64_t hash_hi) {
  std::string hex = hrr::hash_hex(hash_lo, hash_hi);
  auto it = state.co_modules.find(hex);
  if (it != state.co_modules.end()) return it->second;

  std::vector<uint8_t> co_data;
  if (!hrr::read_code_object(archive, hash_lo, hash_hi, co_data)) {
    fprintf(stderr, "[HRR] Code object %s not found in archive\n", hex.c_str());
    return nullptr;
  }

  fprintf(stderr, "[HRR] Loading code object %s (%zu bytes)\n",
          hex.c_str(), co_data.size());
  hipModule_t mod = nullptr;
  hipError_t err = hipModuleLoadData(&mod, co_data.data());
  if (err != hipSuccess) {
    fprintf(stderr, "[HRR] Failed to load code object %s: %d (%s)\n",
            hex.c_str(), err, hipGetErrorString(err));
    (void)hipGetLastError();
    return nullptr;
  }

  fprintf(stderr, "[HRR] Code object %s loaded OK (mod=%p)\n",
          hex.c_str(), (void*)mod);
  state.co_modules[hex] = mod;
  return mod;
}

// Return codes from replay_event().
// 0              : success, continue replay.
// REPLAY_STOP_OK : --fail-fast triggered; replay should stop cleanly
//                  (verification failed but no hard GPU error).
// Any other value: hard error (hipError_t), replay must abort.
static constexpr int REPLAY_STOP_OK = -1;

static int replay_event(ReplayState& state, const hrr::Archive& archive,
                        const hrr::Event& ev) {
  switch (ev.header.event_type) {
    case hrr::EVENT_MALLOC: {
      uint64_t alloc_size = ev.malloc_ev.size;
      auto ao_it = state.run_params.alloc_overrides.find(ev.malloc_ev.ptr_handle);
      if (ao_it != state.run_params.alloc_overrides.end()) {
        if (state.verbose)
          fprintf(stderr, "[HRR] [params] Alloc override: handle=0x%llx "
                  "size %llu -> %llu\n",
                  (unsigned long long)ev.malloc_ev.ptr_handle,
                  (unsigned long long)ev.malloc_ev.size,
                  (unsigned long long)ao_it->second);
        alloc_size = ao_it->second;
      }
      void* ptr = nullptr;
      HIP_CHECK(hipMalloc(&ptr, alloc_size));
      // Zero-init so that if a downstream H2D copy was missed by the
      // recorder (e.g. a trace from a pre-hipMemcpyAsync interposer),
      // reads return 0 instead of stale GPU memory that may decode as a
      // wild pointer/index and trigger an illegal-memory-access fault.
      (void)hipMemset(ptr, 0, alloc_size);
      state.alloc_map[ev.malloc_ev.ptr_handle] = ptr;
      state.alloc_sizes[ev.malloc_ev.ptr_handle] = alloc_size;
      break;
    }

    case hrr::EVENT_FREE: {
      auto it = state.alloc_map.find(ev.malloc_ev.ptr_handle);
      if (it != state.alloc_map.end()) {
        (void)hipFree(it->second);
        state.alloc_map.erase(it);
        state.alloc_sizes.erase(ev.malloc_ev.ptr_handle);
      }
      break;
    }

    case hrr::EVENT_MEMCPY: {
      const auto& mc = ev.memcpy_ev;
      size_t seq = static_cast<size_t>(ev.header.sequence_id);
      auto do_it = state.run_params.data_overrides.find(seq);

      if (do_it != state.run_params.data_overrides.end()) {
        FILE* df = fopen(do_it->second.c_str(), "rb");
        if (!df) {
          fprintf(stderr, "[HRR] [params] Cannot open data override file: %s\n",
                  do_it->second.c_str());
          break;
        }
        fseek(df, 0, SEEK_END);
        long file_size = ftell(df);
        fseek(df, 0, SEEK_SET);
        std::vector<uint8_t> data(file_size);
        fread(data.data(), 1, file_size, df);
        fclose(df);

        void* dst = translate_ptr(state, mc.dst_addr);
        if (dst) {
          HIP_CHECK(hipMemcpy(dst, data.data(), file_size,
                              hipMemcpyHostToDevice));
          if (state.verbose)
            fprintf(stderr, "[HRR] [params] Data override: event %zu, "
                    "loaded %ld bytes from %s\n",
                    seq, file_size, do_it->second.c_str());
        }
      } else if (mc.kind == 1 && mc.hash_lo != 0) {  // H2D with blob data
        std::vector<uint8_t> blob;
        if (hrr::read_blob(archive, mc.hash_lo, mc.hash_hi, blob)) {
          void* dst = translate_ptr(state, mc.dst_addr);
          if (dst) {
            g_last_async = {"memcpy H2D " + std::to_string(mc.size) + " B",
                            ev.header.sequence_id};
            HIP_CHECK(hipMemcpy(dst, blob.data(), mc.size,
                                hipMemcpyHostToDevice));
          }
        }
      } else if (mc.kind == 3) {  // D2D
        void* dst = translate_ptr(state, mc.dst_addr);
        void* src = translate_ptr(state, mc.src_addr);
        if (dst && src) {
          g_last_async = {"memcpy D2D " + std::to_string(mc.size) + " B",
                          ev.header.sequence_id};
          HIP_CHECK(hipMemcpy(dst, src, mc.size, hipMemcpyDeviceToDevice));
        }
      }
      break;
    }

    case hrr::EVENT_MEMSET: {
      if (ev.raw_payload.size() >= 20) {
        uint64_t dst_addr;
        uint32_t value;
        uint64_t size;
        memcpy(&dst_addr, ev.raw_payload.data(), 8);
        memcpy(&value, ev.raw_payload.data() + 8, 4);
        memcpy(&size, ev.raw_payload.data() + 12, 8);
        void* dst = translate_ptr(state, dst_addr);
        if (dst) {
          g_last_async = {"memset " + std::to_string(size) + " B",
                          ev.header.sequence_id};
          HIP_CHECK(hipMemset(dst, static_cast<int>(value), size));
        }
      }
      break;
    }

    case hrr::EVENT_MODULE_LOAD: {
      hipModule_t mod = get_module_for_co(state, archive,
                                          ev.module_load_ev.hash_lo,
                                          ev.module_load_ev.hash_hi);
      if (mod) {
        state.module_map[ev.module_load_ev.module_handle] = mod;
      }
      break;
    }

    case hrr::EVENT_KERNEL_LAUNCH: {
      if (!ev.kernel_launch) break;
      const auto& kl = *ev.kernel_launch;

      if (!state.kernel_filter.empty()) {
        if (kl.kernel_name.find(state.kernel_filter) == std::string::npos) {
          break;
        }
      }

      // Apply parameter overrides if present
      uint32_t launch_grid[3]  = {kl.grid[0], kl.grid[1], kl.grid[2]};
      uint32_t launch_block[3] = {kl.block[0], kl.block[1], kl.block[2]};
      uint32_t launch_shared   = kl.shared_mem;
      const hrr::KernelOverride* ovr = nullptr;

      auto ovr_it = state.run_params.kernel_overrides.find(
          static_cast<size_t>(ev.header.sequence_id));
      if (ovr_it != state.run_params.kernel_overrides.end()) {
        ovr = &ovr_it->second;
        if (ovr->has_grid) {
          launch_grid[0] = ovr->grid[0];
          launch_grid[1] = ovr->grid[1];
          launch_grid[2] = ovr->grid[2];
        }
        if (ovr->has_block) {
          launch_block[0] = ovr->block[0];
          launch_block[1] = ovr->block[1];
          launch_block[2] = ovr->block[2];
        }
        if (ovr->has_shared)
          launch_shared = ovr->shared_bytes;
        if (state.verbose)
          fprintf(stderr, "[HRR] [params] Override applied for kernel '%s' "
                  "(event %llu)\n", kl.kernel_name.c_str(),
                  (unsigned long long)ev.header.sequence_id);
      }

      // Restore input buffer snapshots
      for (const auto& snap : kl.snapshots) {
        if (snap.direction == 0) {  // input
          void* dst = translate_ptr(state, snap.ptr_handle);
          if (dst) {
            std::vector<uint8_t> blob;
            if (hrr::read_blob(archive, snap.hash_lo, snap.hash_hi, blob)) {
              hipMemcpy(dst, blob.data(), snap.length, hipMemcpyHostToDevice);
            }
          }
        }
      }

      // Find kernel function in loaded modules.
      //
      // Priority 1: the kernel's recorded code-object hash uniquely
      // identifies which code object owned the function at capture time.
      // This is essential when the same kernel name (e.g.
      // "mlir_convolution_broadcast_add_relu") is compiled per-shape into
      // many separate code objects — picking the wrong one returns a
      // valid-looking hipFunction_t but launches a kernel whose memory
      // layout doesn't match the recorded args, causing a GPU page-fault.
      hipFunction_t func = nullptr;
      if (kl.co_hash_lo != 0 || kl.co_hash_hi != 0) {
        std::string co_hex = hrr::hash_hex(kl.co_hash_lo, kl.co_hash_hi);
        auto cit = state.co_modules.find(co_hex);
        if (cit != state.co_modules.end()) {
          hipError_t err = hipModuleGetFunction(&func, cit->second,
                                                kl.kernel_name.c_str());
          if (state.verbose)
            fprintf(stderr, "[HRR]   co_hash[%s] mod=%p -> %s\n",
                    co_hex.c_str(), (void*)cit->second,
                    (err == hipSuccess && func) ? "FOUND" : "miss");
          if (err != hipSuccess) func = nullptr;
        } else if (state.verbose) {
          fprintf(stderr, "[HRR]   co_hash[%s] not pre-loaded, falling back\n",
                  co_hex.c_str());
        }
      }

      // Priority 2 (fallback): modules registered via EVENT_MODULE_LOAD.
      if (!func) {
        for (auto& [handle, mod] : state.module_map) {
          hipError_t err = hipModuleGetFunction(&func, mod,
                                                kl.kernel_name.c_str());
          if (state.verbose)
            fprintf(stderr, "[HRR]   module_map[0x%llx] mod=%p -> %s\n",
                    (unsigned long long)handle, (void*)mod,
                    (err == hipSuccess && func) ? "FOUND" : "miss");
          if (err == hipSuccess && func) break;
          func = nullptr;
        }
      }
      // Priority 3 (last resort): scan every pre-loaded code object.
      // Only safe for traces with no co_hash recorded (legacy capture);
      // for modern traces this would mask the wrong-module bug above.
      if (!func && kl.co_hash_lo == 0 && kl.co_hash_hi == 0) {
        for (auto& [hex, mod] : state.co_modules) {
          hipError_t err = hipModuleGetFunction(&func, mod,
                                                kl.kernel_name.c_str());
          if (state.verbose)
            fprintf(stderr, "[HRR]   co_modules[%s] mod=%p -> %s\n",
                    hex.c_str(), (void*)mod,
                    (err == hipSuccess && func) ? "FOUND" : "miss");
          if (err == hipSuccess && func) break;
          func = nullptr;
        }
      }

      // The search loop above probes every loaded module and an
      // hipErrorNotFound on misses is expected.  Clear the sticky context
      // error so HIP_CHECK on the launch below doesn't misattribute it as
      // a deferred async fault.
      (void)hipGetLastError();

      if (!func) {
        if (kl.co_hash_lo != 0 || kl.co_hash_hi != 0) {
          fprintf(stderr,
                  "[HRR] Kernel '%s' not found via recorded co_hash %s "
                  "(%zu module_map probed)\n",
                  kl.kernel_name.c_str(),
                  hrr::hash_hex(kl.co_hash_lo, kl.co_hash_hi).c_str(),
                  state.module_map.size());
        } else {
          fprintf(stderr,
                  "[HRR] Kernel '%s' not found in any loaded module "
                  "(%zu module_map + %zu co_modules searched)\n",
                  kl.kernel_name.c_str(),
                  state.module_map.size(), state.co_modules.size());
        }
        break;
      }

      if (kl.args.empty()) {
        fprintf(stderr, "[HRR] WARNING: kernel '%s' has 0 recorded args — "
                "trace was likely captured with truncated metadata "
                "(rebuild proxy with larger name buffer and re-capture)\n",
                kl.kernel_name.c_str());
      }

      // Build kernarg buffer from captured args.  Track unresolved pointer
      // args (non-zero handle that translates to NULL) so we can skip the
      // launch instead of having the GPU page-fault on a NULL deref.
      // Also stash per-arg info so we can dump it if the GPU faults.
      struct ArgDump {
        size_t   idx;
        uint8_t  kind;     // 1=ptr, 0=scalar, 2=hidden
        uint64_t handle;   // pointer arg's recorded handle (raw bytes for scalar <=8B)
        void*    live_ptr;
        uint64_t alloc_base;   // 0 if unknown
        size_t   alloc_size;   // 0 if unknown
        size_t   raw_size;
      };
      std::vector<void*> arg_ptrs;
      std::vector<std::vector<uint8_t>> arg_storage;
      std::vector<uint64_t> unresolved_handles;
      std::vector<ArgDump> arg_dumps;

      size_t arg_idx = 0;
      for (const auto& arg : kl.args) {
        if (arg.value_kind == 2) {
          if (state.verbose) {
            fprintf(stderr, "  arg[%zu]: hidden (skipped)\n", arg_idx);
          }
          arg_dumps.push_back({arg_idx, 2, 0, nullptr, 0, 0, arg.data.size()});
          arg_idx++;
          continue;  // skip hidden
        }

        arg_storage.emplace_back();
        auto& storage = arg_storage.back();

        if (arg.value_kind == 1 && arg.data.size() >= 8) {
          // Pointer arg: translate handle to live pointer
          uint64_t handle;
          memcpy(&handle, arg.data.data(), 8);
          void* live_ptr = translate_ptr(state, handle);
          if (!live_ptr && handle != 0) {
            unresolved_handles.push_back(handle);
          }
          // Find which captured allocation this handle came from for diagnostics
          uint64_t alloc_base = 0;
          size_t   alloc_size = 0;
          if (live_ptr) {
            auto exact = state.alloc_map.find(handle);
            if (exact != state.alloc_map.end()) {
              alloc_base = handle;
              auto sz = state.alloc_sizes.find(handle);
              if (sz != state.alloc_sizes.end()) alloc_size = sz->second;
            } else {
              for (const auto& [base, p] : state.alloc_map) {
                auto sz = state.alloc_sizes.find(base);
                if (sz != state.alloc_sizes.end() &&
                    handle >= base && handle < base + sz->second) {
                  alloc_base = base;
                  alloc_size = sz->second;
                  break;
                }
              }
            }
          }
          arg_dumps.push_back({arg_idx, 1, handle, live_ptr,
                               alloc_base, alloc_size, arg.data.size()});
          storage.resize(sizeof(void*));
          memcpy(storage.data(), &live_ptr, sizeof(void*));
          if (state.verbose) {
            fprintf(stderr, "  arg[%zu]: ptr handle=0x%llx -> %p%s\n",
                    arg_idx, (unsigned long long)handle, live_ptr,
                    live_ptr ? "" : (handle == 0 ? " (NULL)" : " (NULL - missing alloc!)"));
          }
        } else {
          // Scalar arg: use raw bytes, apply override if present
          storage = arg.data;
          if (ovr) {
            auto sa_it = ovr->scalar_args.find(static_cast<uint16_t>(arg_idx));
            if (sa_it != ovr->scalar_args.end() &&
                sa_it->second.size() == storage.size()) {
              storage = sa_it->second;
              if (state.verbose)
                fprintf(stderr, "  arg[%zu]: scalar OVERRIDDEN\n", arg_idx);
            }
          }
          uint64_t scalar_val = 0;
          memcpy(&scalar_val, storage.data(),
                 std::min(storage.size(), sizeof(scalar_val)));
          arg_dumps.push_back({arg_idx, 0, scalar_val, nullptr, 0, 0,
                               arg.data.size()});
          if (state.verbose && arg.data.size() <= 8) {
            fprintf(stderr, "  arg[%zu]: scalar size=%u val=0x%llx\n",
                    arg_idx, arg.size, (unsigned long long)scalar_val);
          }
        }
        arg_ptrs.push_back(storage.data());
        arg_idx++;
      }

      // If any required pointer arg couldn't be resolved, skip the launch.
      // Letting the kernel run with NULL for a non-zero handle is a
      // guaranteed GPU page-fault (illegal memory access) that will then
      // poison every subsequent driver call with a sticky error.
      if (!unresolved_handles.empty()) {
        fprintf(stderr,
                "[HRR] SKIP launch '%s' (event %llu): %zu pointer arg(s) "
                "could not be resolved (first: handle=0x%llx).  Re-capture "
                "with the full interposer (in particular hipMemcpyAsync / "
                "hipMallocAsync) to record the missing allocation(s).\n",
                kl.kernel_name.c_str(),
                (unsigned long long)ev.header.sequence_id,
                unresolved_handles.size(),
                (unsigned long long)unresolved_handles.front());
        break;
      }

      // Launch with optional timing
      hipEvent_t start = nullptr, stop = nullptr;
      if (state.timing) {
        hipEventCreate(&start);
        hipEventCreate(&stop);
        hipEventRecord(start);
      }

      // Pre-launch classification of "is this snapshot a read-only input?"
      // The writer snapshots every pointer arg post-launch because it has
      // no metadata to distinguish reads from writes.  Here we read each
      // snapshotted buffer BEFORE the kernel runs and compare to the
      // recorded expected output: if they already match, the kernel
      // can't possibly have written that data, so it's an input arg
      // (e.g. GEMM A/B/bias) and we'll suppress it from the verify
      // report and trace dump.  Active under --verify-outputs-only,
      // independent of --verify, so dump-only sessions can also filter.
      std::vector<bool> snap_is_input(kl.snapshots.size(), false);
      if (state.verify_outputs_only) {
        for (size_t si = 0; si < kl.snapshots.size(); si++) {
          const auto& snap = kl.snapshots[si];
          if (snap.direction != 1) continue;

          // If this handle was already identified as a real output by an
          // earlier launch (e.g. the same kernel run 10 times in a row),
          // never reclassify it as an input.  Without this, from launch 2
          // onward the output buffer already holds the correct result from
          // the previous launch, so pre-launch == expected → falsely
          // suppressed.
          if (state.known_output_handles.count(snap.ptr_handle)) continue;

          void* src = translate_ptr(state, snap.ptr_handle);
          if (!src) continue;
          std::vector<uint8_t> expected;
          if (!hrr::read_blob(archive, snap.hash_lo, snap.hash_hi,
                              expected)) continue;
          size_t cmp_len = std::min((size_t)snap.length, expected.size());
          if (cmp_len == 0) continue;
          std::vector<uint8_t> pre(cmp_len);
          if (hipMemcpy(pre.data(), src, cmp_len,
                        hipMemcpyDeviceToHost) != hipSuccess) {
            (void)hipGetLastError();
            continue;
          }
          if (cmp_len == expected.size() &&
              memcmp(pre.data(), expected.data(), cmp_len) == 0) {
            snap_is_input[si] = true;
            state.verify_inputs_skipped++;
          } else {
            // First time we see this handle as a real output — remember it.
            state.known_output_handles.insert(snap.ptr_handle);
          }
        }
      }

      g_last_async = {std::string("kernel '") + kl.kernel_name + "'",
                      ev.header.sequence_id};
      HIP_CHECK(hipModuleLaunchKernel(
          func,
          launch_grid[0], launch_grid[1], launch_grid[2],
          launch_block[0], launch_block[1], launch_block[2],
          launch_shared, nullptr,
          arg_ptrs.data(), nullptr));

      if (state.timing) {
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        float ms = 0.0f;
        hipEventElapsedTime(&ms, start, stop);
        state.total_kernel_ms += ms;
        hipEventDestroy(start);
        hipEventDestroy(stop);
      }

      state.kernels_launched++;

      // Sync after launch so GPU faults surface at the actual culprit
      // rather than being deferred to the next unrelated driver call.
      if (state.sync_after_launch) {
        hipError_t sync_err = hipDeviceSynchronize();
        if (sync_err != hipSuccess) {
          fprintf(stderr,
                  "[HRR] GPU error after kernel '%s' (event %llu): "
                  "%d (%s)\n"
                  "[HRR]   grid=(%u,%u,%u) block=(%u,%u,%u) shared=%u "
                  "args=%zu\n",
                  kl.kernel_name.c_str(),
                  (unsigned long long)ev.header.sequence_id,
                  sync_err, hipGetErrorString(sync_err),
                  launch_grid[0], launch_grid[1], launch_grid[2],
                  launch_block[0], launch_block[1], launch_block[2],
                  launch_shared, kl.args.size());
          for (const auto& d : arg_dumps) {
            if (d.kind == 1) {  // pointer
              uint64_t off = (d.alloc_base && d.handle >= d.alloc_base)
                  ? d.handle - d.alloc_base : 0;
              fprintf(stderr,
                      "[HRR]   arg[%zu] PTR  handle=0x%llx -> %p  "
                      "alloc_base=0x%llx alloc_size=%zu off=%llu\n",
                      d.idx, (unsigned long long)d.handle, d.live_ptr,
                      (unsigned long long)d.alloc_base, d.alloc_size,
                      (unsigned long long)off);
            } else if (d.kind == 0) {  // scalar
              fprintf(stderr,
                      "[HRR]   arg[%zu] SCAL size=%zu val=0x%llx\n",
                      d.idx, d.raw_size, (unsigned long long)d.handle);
            } else {
              fprintf(stderr, "[HRR]   arg[%zu] HIDDEN size=%zu\n",
                      d.idx, d.raw_size);
            }
          }
          fprintf(stderr,
                  "[HRR]   Hint: most page-faults at this stage are caused "
                  "by stale/uninitialized GPU memory whose H2D upload was "
                  "missed by the recorder.  Re-capture with the updated "
                  "interposer (hipMemcpyAsync interception) and retry.\n");
          (void)hipGetLastError();
          return sync_err;
        } else if (state.verbose) {
          fprintf(stderr, "[HRR] Kernel '%s' OK\n", kl.kernel_name.c_str());
        }
      }

      // HRR_VERIFY_TRACE levels: 0=off, 1=compact 4-element trace,
      // 2=detailed 8-element dump (dtype-aware via --print-dtype).
      const char* vt_env = getenv("HRR_VERIFY_TRACE");
      int trace_level = vt_env ? atoi(vt_env) : 0;

      // Verify output buffers and/or dump trace output
      if (state.verify || trace_level > 0) {
        hipDeviceSynchronize();
        for (size_t si = 0; si < kl.snapshots.size(); si++) {
          const auto& snap = kl.snapshots[si];
          if (snap.direction == 1) {  // output
            // --verify-outputs-only: drop snapshots classified as
            // read-only inputs in the pre-launch sweep above.  Applies
            // to both the verifier and the trace dump so the user
            // doesn't have to pass --verify just to filter the dump.
            if (state.verify_outputs_only &&
                si < snap_is_input.size() && snap_is_input[si]) {
              continue;
            }
            void* src = translate_ptr(state, snap.ptr_handle);
            if (!src) continue;

            std::vector<uint8_t> expected;
            if (!hrr::read_blob(archive, snap.hash_lo, snap.hash_hi,
                                expected)) {
              continue;
            }

            size_t cmp_len = std::min((size_t)snap.length, expected.size());
            std::vector<uint8_t> actual(cmp_len);
            hipMemcpy(actual.data(), src, cmp_len,
                      hipMemcpyDeviceToHost);

            bool exact = (cmp_len == expected.size() &&
                          memcmp(actual.data(), expected.data(), cmp_len) == 0);

            if (trace_level == 1) {
              const float* af = reinterpret_cast<const float*>(actual.data());
              const float* ef = reinterpret_cast<const float*>(expected.data());
              fprintf(stderr,
                      "[VTRACE] kernel '%s' snap handle=0x%llx len=%zu "
                      "exp[0..3]=%.4g,%.4g,%.4g,%.4g got[0..3]=%.4g,%.4g,%.4g,%.4g %s\n",
                      kl.kernel_name.c_str(), (unsigned long long)snap.ptr_handle,
                      cmp_len,
                      cmp_len>=16?ef[0]:0.f, cmp_len>=16?ef[1]:0.f,
                      cmp_len>=16?ef[2]:0.f, cmp_len>=16?ef[3]:0.f,
                      cmp_len>=16?af[0]:0.f, cmp_len>=16?af[1]:0.f,
                      cmp_len>=16?af[2]:0.f, cmp_len>=16?af[3]:0.f,
                      exact ? "EXACT" : "diff");
            } else if (trace_level >= 2) {
              size_t elem_size = (state.print_dtype == PrintDtype::fp16) ? 2 : 4;
              size_t n = std::min<size_t>(8, cmp_len / elem_size);
              fprintf(stderr,
                      "\n[HRR] OUT kernel='%s' ev=%llu handle=0x%llx len=%zu\n"
                      "[HRR]   expected:",
                      kl.kernel_name.c_str(),
                      (unsigned long long)ev.header.sequence_id,
                      (unsigned long long)snap.ptr_handle, cmp_len);
              for (size_t i = 0; i < n; i++) {
                float v;
                if (state.print_dtype == PrintDtype::fp16) {
                  uint16_t h;
                  memcpy(&h, expected.data() + i * 2, 2);
                  v = half_to_float(h);
                } else {
                  memcpy(&v, expected.data() + i * 4, 4);
                }
                fprintf(stderr, " %.6g", v);
              }
              fprintf(stderr, "\n[HRR]   actual:  ");
              for (size_t i = 0; i < n; i++) {
                float v;
                if (state.print_dtype == PrintDtype::fp16) {
                  uint16_t h;
                  memcpy(&h, actual.data() + i * 2, 2);
                  v = half_to_float(h);
                } else {
                  memcpy(&v, actual.data() + i * 4, 4);
                }
                fprintf(stderr, " %.6g", v);
              }
              fprintf(stderr, "\n");
            }

            // Verify pass/fail only when --verify is active
            if (state.verify) {
              if (exact) {
                state.verify_pass++;
              } else {
                size_t num_f32 = cmp_len / 4;
                float max_diff = 0.0f;
                float max_abs_exp = 0.0f;
                const float* a = reinterpret_cast<const float*>(actual.data());
                const float* e = reinterpret_cast<const float*>(expected.data());
                for (size_t i = 0; i < num_f32; i++) {
                  float d = std::fabs(a[i] - e[i]);
                  if (d > max_diff) max_diff = d;
                  float ae = std::fabs(e[i]);
                  if (ae > max_abs_exp) max_abs_exp = ae;
                }
                float threshold = state.verify_atol +
                                  state.verify_rtol * max_abs_exp;
                if (cmp_len == expected.size() &&
                    std::isfinite(max_diff) && max_diff <= threshold) {
                  state.verify_pass++;
                  if (state.verbose && max_diff > 0.0f) {
                    fprintf(stderr,
                            "[HRR] near-match kernel '%s' (handle=0x%llx, "
                            "max_diff=%.6g, threshold=%.6g)\n",
                            kl.kernel_name.c_str(),
                            (unsigned long long)snap.ptr_handle,
                            max_diff, threshold);
                  }
                } else {
                  state.verify_fail++;
                  size_t first_bad = 0;
                  for (size_t i = 0; i < num_f32; i++) {
                    if (std::fabs(a[i] - e[i]) > threshold) { first_bad = i; break; }
                  }
                  // Show 4 elements starting at first_bad so the user sees
                  // the actual diverging values, not always the (often fine)
                  // leading zeros.
                  size_t d0 = first_bad;
                  size_t d1 = (first_bad + 1 < num_f32) ? first_bad + 1 : first_bad;
                  size_t d2 = (first_bad + 2 < num_f32) ? first_bad + 2 : first_bad;
                  size_t d3 = (first_bad + 3 < num_f32) ? first_bad + 3 : first_bad;
                  fprintf(stderr,
                          "[HRR] MISMATCH kernel '%s' output buffer "
                          "(handle=0x%llx, max_diff=%.6g, threshold=%.6g, "
                          "max|expected|=%.6g, len=%zu, first_bad=%zu)\n"
                          "[HRR]   exp[%zu..%zu]=%.4g,%.4g,%.4g,%.4g\n"
                          "[HRR]   got[%zu..%zu]=%.4g,%.4g,%.4g,%.4g\n",
                          kl.kernel_name.c_str(),
                          (unsigned long long)snap.ptr_handle,
                          max_diff, threshold, max_abs_exp,
                          cmp_len, first_bad,
                          d0, d3, e[d0], e[d1], e[d2], e[d3],
                          d0, d3, a[d0], a[d1], a[d2], a[d3]);
                  if (state.fail_fast) return REPLAY_STOP_OK;
                }
              }
            }
          }
        }
      }
      break;
    }

    case hrr::EVENT_DEVICE_SYNC:
      if (!state.skip_device_sync) {
        if (state.verbose) fprintf(stderr, "[HRR] hipDeviceSynchronize()...\n");
        HIP_CHECK(hipDeviceSynchronize());
      }
      break;

    case hrr::EVENT_STREAM_SYNC:
      if (!state.skip_device_sync) {
        if (state.verbose) fprintf(stderr, "[HRR] STREAM_SYNC -> hipDeviceSynchronize()...\n");
        HIP_CHECK(hipDeviceSynchronize());
      }
      break;

    default:
      break;
  }
  return 0;
}

static void print_usage(const char* argv0) {
  fprintf(stderr,
    "Usage: %s <capture.hrr> [options]\n"
    "\n"
    "Options:\n"
    "  --verify            Compare output buffers with recorded snapshots\n"
    "  --fail-fast         Stop replay at the first verification mismatch\n"
    "                      (default: continue and report all mismatches)\n"
    "  --verify-outputs-only\n"
    "                      Suppress snapshots whose pre-launch buffer state\n"
    "                      already matches the recorded blob.  Those are\n"
    "                      read-only kernel inputs (Tensile A/B, bias,\n"
    "                      weights) that the writer can't distinguish from\n"
    "                      real outputs at record time.\n"
    "  --timing            Report per-kernel GPU timing\n"
    "  --kernel-filter STR Only replay kernels containing STR in name\n"
    "  --params FILE       Load run_params.json to override kernel grid/block/\n"
    "                      shared_mem, scalar args, alloc sizes, and memcpy data\n"
    "  --skip-device-sync  Skip all recorded device/stream sync events\n"
    "  --print-dtype TYPE  Element type for HRR_VERIFY_TRACE=2: fp32 (default), fp16\n"
    "  --async             Do NOT sync after every kernel launch (default is\n"
    "                      to sync so faults are attributed to the right kernel)\n"
    "  --sync-after-launch Deprecated; per-kernel sync is now the default\n"
    "  --verbose           Print each event as it is processed\n"
    "  --help              Show this help\n"
    "\n"
    "Environment variables:\n"
    "  HRR_VERIFY_TRACE=1  Compact 4-element FP32 trace per output snapshot\n"
    "  HRR_VERIFY_TRACE=2  Detailed 8-element dump per output snapshot\n"
    "                      (dtype selected by --print-dtype, default fp32)\n",
    argv0);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  std::string archive_path;
  ReplayState state;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--verify") == 0) {
      state.verify = true;
    } else if (strcmp(argv[i], "--fail-fast") == 0) {
      state.fail_fast = true;
    } else if (strcmp(argv[i], "--verify-outputs-only") == 0) {
      state.verify_outputs_only = true;
    } else if (strcmp(argv[i], "--timing") == 0) {
      state.timing = true;
    } else if (strcmp(argv[i], "--skip-device-sync") == 0) {
      state.skip_device_sync = true;
    } else if (strcmp(argv[i], "--sync-after-launch") == 0) {
      state.sync_after_launch = true;  // now the default; kept for compat
    } else if (strcmp(argv[i], "--async") == 0) {
      state.sync_after_launch = false;
    } else if (strcmp(argv[i], "--print-dtype") == 0 && i + 1 < argc) {
      const char* dt = argv[++i];
      if (strcmp(dt, "fp16") == 0) state.print_dtype = PrintDtype::fp16;
      else if (strcmp(dt, "fp32") == 0) state.print_dtype = PrintDtype::fp32;
      else {
        fprintf(stderr, "[HRR] Unknown dtype '%s' (supported: fp16, fp32)\n", dt);
        return 1;
      }
    } else if (strcmp(argv[i], "--verbose") == 0) {
      state.verbose = true;
    } else if (strcmp(argv[i], "--kernel-filter") == 0 && i + 1 < argc) {
      state.kernel_filter = argv[++i];
    } else if (strcmp(argv[i], "--params") == 0 && i + 1 < argc) {
      state.params_file = argv[++i];
    } else if (strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      return 0;
    } else if (argv[i][0] != '-') {
      archive_path = argv[i];
    }
  }

  if (archive_path.empty()) {
    fprintf(stderr, "[HRR] No archive path specified\n");
    return 1;
  }

  // Load archive
  hrr::Archive archive;
  if (!hrr::load_archive(archive_path, archive)) {
    return 1;
  }

  printf("[HRR] Loaded archive: %zu events, %zu kernels, %zu blobs, "
         "%zu code objects\n",
         archive.event_count, archive.kernel_count,
         archive.blob_count, archive.code_object_count);

  // Load parameter overrides
  if (!state.params_file.empty()) {
    if (!hrr::load_run_params(state.params_file, state.run_params)) {
      fprintf(stderr, "[HRR] Failed to load params from %s\n",
              state.params_file.c_str());
      return 1;
    }
  }

  // Init HIP
  HIP_CHECK(hipInit(0));

  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count == 0) {
    fprintf(stderr, "[HRR] No GPU devices found\n");
    return 1;
  }

  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  printf("[HRR] Replaying on: %s (%s)\n", props.name, props.gcnArchName);

  // Pre-load all code objects from the archive directory.
  // This ensures kernels are findable even if the trace lacks MODULE_LOAD
  // events (e.g. fat-binary captures from older proxy versions).
  for (const auto& [hex, co_path] : archive.code_objects) {
    if (state.co_modules.count(hex)) continue;

    FILE* co_f = fopen(co_path.c_str(), "rb");
    if (!co_f) continue;
    fseek(co_f, 0, SEEK_END);
    long co_size = ftell(co_f);
    fseek(co_f, 0, SEEK_SET);
    if (co_size <= 0) { fclose(co_f); continue; }

    std::vector<uint8_t> co_data(co_size);
    size_t co_read = fread(co_data.data(), 1, co_size, co_f);
    fclose(co_f);
    if (co_read != static_cast<size_t>(co_size)) continue;

    hipModule_t mod = nullptr;
    hipError_t err = hipModuleLoadData(&mod, co_data.data());
    if (err == hipSuccess && mod) {
      state.co_modules[hex] = mod;
      if (state.verbose)
        fprintf(stderr, "[HRR] Pre-loaded code object %s (%ld bytes)\n",
                hex.c_str(), co_size);
    } else if (state.verbose) {
      fprintf(stderr, "[HRR] Skipped code object %s (arch mismatch or load error)\n",
              hex.c_str());
    }
  }
  // Pre-load failures (e.g. arch mismatch on a multi-target capture) are
  // expected and shouldn't poison the context for the first replayed event.
  (void)hipGetLastError();

  if (!archive.code_objects.empty())
    printf("[HRR] Pre-loaded %zu / %zu code objects\n",
           state.co_modules.size(), archive.code_objects.size());

  // Replay events
  auto wall_start = std::chrono::high_resolution_clock::now();
  auto wall_end   = wall_start;

  for (size_t i = 0; i < archive.events.size(); i++) {
    if (state.verbose) {
      fprintf(stderr, "[HRR] Event %zu: %s\n",
              i, hrr::event_type_name(archive.events[i].header.event_type));
    }
    int ret = replay_event(state, archive, archive.events[i]);
    if (ret == REPLAY_STOP_OK) {
      // --fail-fast: first verify mismatch hit; stop cleanly.
      break;
    }
    if (ret != 0) {
      fprintf(stderr, "[HRR] Replay failed at event %zu (%s)\n",
              i, hrr::event_type_name(archive.events[i].header.event_type));
      return ret;
    }
  }

  hipDeviceSynchronize();
  wall_end = std::chrono::high_resolution_clock::now();
  double wall_ms = std::chrono::duration<double, std::milli>(
                       wall_end - wall_start).count();

  // Report
  printf("[HRR] Replay complete: %zu kernels launched\n",
         state.kernels_launched);
  printf("[HRR] Wall time: %.1f ms\n", wall_ms);

  if (state.timing) {
    printf("[HRR] Total GPU kernel time: %.1f ms\n", state.total_kernel_ms);
  }

  if (state.verify) {
    if (state.verify_outputs_only && state.verify_inputs_skipped > 0) {
      printf("[HRR] Verification: %zu passed, %zu failed "
             "(%zu read-only input snapshots suppressed)\n",
             state.verify_pass, state.verify_fail,
             (size_t)state.verify_inputs_skipped);
    } else {
      printf("[HRR] Verification: %zu passed, %zu failed\n",
             state.verify_pass, state.verify_fail);
    }
  }

  // Cleanup
  for (auto& [handle, ptr] : state.alloc_map) {
    (void)hipFree(ptr);
  }
  for (auto& [hex, mod] : state.co_modules) {
    (void)hipModuleUnload(mod);
  }

  return state.verify_fail > 0 ? 1 : 0;
}
