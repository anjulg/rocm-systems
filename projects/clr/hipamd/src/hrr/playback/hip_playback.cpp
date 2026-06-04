/* Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License */
//
// hip_playback.cpp — Manual playback implementations for APIs that need
// complex handling: kernel launches, H2D memcpy with blobs, module load
// from code objects, and hipModuleGetFunction name resolution.
//
// Also implements PlaybackContext helpers: load_blob, load_code_object,
// load_module.

#include "hip_playback.h"
#include "hrr_api_args.h"
#include "hrr_reader.h"   // hrr::hash_hex

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

// Thread-local sequence ID — set by dispatch_event before calling any handler.
// Kernel-launch handlers use this to wait for their submission turn and then
// immediately unblock the next thread before doing timing/sync.
thread_local uint64_t hrr_dispatch_seq = 0;

// ---------------------------------------------------------------------------
// HIP error checking — returns the hipError_t so callers can branch on it.
// Usage:  HRR_HIP_CHECK(hipFoo(...));                      // log only
//         if (HRR_HIP_CHECK(hipFoo(...)) != hipSuccess) {} // log + branch
// ---------------------------------------------------------------------------

static inline hipError_t hrr_hip_check(hipError_t e, const char* call,
                                        const char* file, int line) {
    if (e != hipSuccess)
        fprintf(stderr, "[HRR] HIP error %d (%s): %s (%s:%d)\n",
                e, hipGetErrorString(e), call, file, line);
    return e;
}
#define HRR_HIP_CHECK(call) hrr_hip_check((call), #call, __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

// Build path: archive_dir/blobs/<2-char-prefix>/<hex>.blob
static std::string blob_path(const std::string& archive_dir,
                             uint64_t hash_lo, uint64_t hash_hi) {
    std::string hex = hrr::hash_hex(hash_lo, hash_hi);
    return archive_dir + "/blobs/" + hex.substr(0, 2) + "/" + hex + ".blob";
}

// Build path: archive_dir/code_objects/<hex>.hsaco
static std::string co_path(const std::string& archive_dir,
                            uint64_t hash_lo, uint64_t hash_hi) {
    return archive_dir + "/code_objects/" + hrr::hash_hex(hash_lo, hash_hi) + ".hsaco";
}

static std::vector<uint8_t> read_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return {}; }
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    if (fread(buf.data(), 1, buf.size(), f) != buf.size()) { fclose(f); return {}; }
    fclose(f);
    return buf;
}

}  // namespace

// ---------------------------------------------------------------------------
// PlaybackContext: blob / code-object loading
// ---------------------------------------------------------------------------

const void* PlaybackContext::load_blob(uint64_t hash_lo, uint64_t hash_hi,
                                       size_t* sz_out) const {
    if (!hash_lo && !hash_hi) return nullptr;
    std::string key = hrr::hash_hex(hash_lo, hash_hi);
    {
        std::shared_lock lk(map_mutex);
        auto it = blob_cache_.find(key);
        if (it != blob_cache_.end()) {
            if (sz_out) *sz_out = it->second.size();
            return it->second.data();
        }
    }
    // Not cached — read from disk, then insert under exclusive lock
    auto data = read_file(blob_path(archive_dir, hash_lo, hash_hi));
    if (data.empty()) return nullptr;
    std::unique_lock lk(map_mutex);
    auto it = blob_cache_.emplace(key, std::move(data)).first;
    if (sz_out) *sz_out = it->second.size();
    return it->second.data();
}

const void* PlaybackContext::load_code_object(uint64_t hash_lo, uint64_t hash_hi,
                                              size_t* sz_out) const {
    if (!hash_lo && !hash_hi) return nullptr;
    // Prefix with "co:" to avoid colliding with blobs of the same hash in blob_cache_.
    std::string key = "co:" + hrr::hash_hex(hash_lo, hash_hi);
    {
        std::shared_lock lk(map_mutex);
        auto it = blob_cache_.find(key);
        if (it != blob_cache_.end()) {
            if (sz_out) *sz_out = it->second.size();
            return it->second.data();
        }
    }
    auto data = read_file(co_path(archive_dir, hash_lo, hash_hi));
    if (data.empty()) return nullptr;
    std::unique_lock lk(map_mutex);
    auto [it, inserted] = blob_cache_.emplace(key, std::move(data));
    (void)inserted;
    if (sz_out) *sz_out = it->second.size();
    return it->second.data();
}

hipModule_t PlaybackContext::load_module(uint64_t hash_lo, uint64_t hash_hi) {
    std::string hex = hrr::hash_hex(hash_lo, hash_hi);
    {
        std::shared_lock lk(map_mutex);
        auto it = co_modules.find(hex);
        if (it != co_modules.end()) return it->second;
    }

    // Cache miss — load without holding any lock (disk I/O + GPU call)
    // Drain any deferred GPU error before loading: hipModuleLoadData triggers
    // a driver state check that surfaces faults from prior async kernel launches
    // even after hipDeviceSynchronize() has returned success.
    {
        hipError_t pre_sync = hipDeviceSynchronize();
        hipError_t pre_err  = hipGetLastError();
        if (pre_sync != hipSuccess || pre_err != hipSuccess)
            fprintf(stderr, "[HRR] load_module %s: pre-drain sync=%d err=%d\n",
                    hex.c_str(), pre_sync, pre_err);
    }

    size_t sz = 0;
    const void* data = load_code_object(hash_lo, hash_hi, &sz);
    if (!data || sz == 0) {
        fprintf(stderr, "[HRR] Code object %s not found in archive\n", hex.c_str());
        return nullptr;
    }
    hipModule_t mod = nullptr;
    hipError_t err = hipModuleLoadData(&mod, data);
    if (err != hipSuccess) {
        fprintf(stderr, "[HRR] Failed to load code object %s: %d (%s)\n",
                hex.c_str(), err, hipGetErrorString(err));
        return nullptr;
    }

    // Re-acquire with exclusive lock; if a concurrent thread already loaded
    // this module, discard ours to avoid a double-load leak.
    std::unique_lock lk(map_mutex);
    auto [it, inserted] = co_modules.emplace(hex, mod);
    if (!inserted) {
        hipModuleUnload(mod);
        return it->second;
    }
    if (verbose)
        fprintf(stderr, "[HRR] Loaded code object %s (%zu bytes)\n", hex.c_str(), sz);
    return mod;
}

// ---------------------------------------------------------------------------
// Kernel launch — shared implementation used by all four launch APIs
// ---------------------------------------------------------------------------
// Kernel launch payload (raw_payload, bytes after 32-byte EventHeader):
//   [0..7]   stream_handle (uint64_t)
//   [8..9]   name_len (uint16_t)
//   [10..]   kernel_name (name_len bytes, no NUL)
//   [+0..7]  co_hash_lo (uint64_t)
//   [+8..15] co_hash_hi (uint64_t)
//   [+0..11] grid[3]   (uint32_t[3])
//   [+12..23] block[3] (uint32_t[3])
//   [+24..27] shared_mem (uint32_t)
//   [+28..29] num_args (uint16_t)
//   [+30..31] num_snapshots (uint16_t — non-zero only for inputs/full archives)
//   per arg:       u8 value_kind, u16 size, <size> bytes data
//   per snapshot:  u64 ptr_handle, u64 offset, u64 length,
//                  u64 hash_lo, u64 hash_hi, u8 direction
//                  (33 bytes each, wire-compatible with hip_replay snap_record_t)
//   --- trailing (v3.1, optional) ---
//   u32  full_kbuf_size (0 = not captured)
//   u8[] full_kbuf      (full_kbuf_size bytes; the entire packed kernarg
//                        buffer including implicit/hidden args — used for
//                        MLIR/SP3 kernels that need exact extra[] replay)

// Parsed snapshot record — populated from the payload above.  Kept local to
// the replay path so neither the capture-side header nor the public reader
// has to know about it.
struct ReplaySnapshot {
    uint64_t ptr_handle;
    uint64_t offset;
    uint64_t length;
    uint64_t hash_lo;
    uint64_t hash_hi;
    uint8_t  direction;
};

static hipError_t replay_kernel_launch(PlaybackContext& ctx, const uint8_t* pl) {
    // Skip the 32-byte header; kernel launch has a variable-length binary format.
    const auto* hdr = reinterpret_cast<const hrr_event_header*>(pl);
    const uint8_t* p   = pl + sizeof(hrr_event_header);
    const uint8_t* end = pl + hdr->payload_length;

    if (p + 8 > end) return hipErrorInvalidValue;
    uint64_t stream_rec; memcpy(&stream_rec, p, 8); p += 8;

    if (p + 2 > end) return hipErrorInvalidValue;
    uint16_t name_len; memcpy(&name_len, p, 2); p += 2;
    if (p + name_len > end) return hipErrorInvalidValue;
    std::string kernel_name(reinterpret_cast<const char*>(p), name_len);
    p += name_len;

    uint64_t co_hash_lo = 0, co_hash_hi = 0;
    if (p + 16 <= end) {
        memcpy(&co_hash_lo, p, 8); p += 8;
        memcpy(&co_hash_hi, p, 8); p += 8;
    }

    if (p + 32 > end) return hipErrorInvalidValue;
    uint32_t grid[3], block[3], shared_mem;
    memcpy(grid,       p, 12); p += 12;
    memcpy(block,      p, 12); p += 12;
    memcpy(&shared_mem, p, 4); p +=  4;

    uint16_t num_args, num_snapshots;
    memcpy(&num_args,       p, 2); p += 2;
    memcpy(&num_snapshots,  p, 2); p += 2;

    // Apply kernel filter if set
    if (!ctx.kernel_filter.empty() &&
        kernel_name.find(ctx.kernel_filter) == std::string::npos)
        return hipSuccess;

    // Resolve hipFunction_t — cache hit avoids repeated hipModuleGetFunction
    // searches. Locked because multiple threads can now be in kernel launch
    // preparation concurrently (only the HIP call itself is serialized).
    //
    // The cache key includes the captured code-object hash because MLIR /
    // MIGraphX style frameworks emit MANY modules that each define a kernel
    // with the same mangled name (one per shape specialisation); a
    // name-only cache happily returns the first one found and silently
    // launches the wrong layer's kernel against the right layer's data.
    // When the archive doesn't carry a hash (co_hash_lo == co_hash_hi == 0,
    // older archives or untracked load paths), the key falls back to the
    // bare kernel name and the resolution scans modules by name.
    const bool have_hash = (co_hash_lo || co_hash_hi);
    const std::string cache_key = have_hash
        ? hrr::hash_hex(co_hash_lo, co_hash_hi) + ":" + kernel_name
        : kernel_name;

    hipFunction_t func = nullptr;
    {
        std::shared_lock lk(ctx.map_mutex);
        auto it = ctx.func_cache.find(cache_key);
        if (it != ctx.func_cache.end())
            func = it->second;
    }

    if (!func) {
        // Preferred path: the capture recorded the exact code object that
        // owned this function.  Load (or look up) that specific module by
        // hash and resolve the name inside it.  This is the only safe
        // strategy when multiple modules host kernels with the same name.
        if (have_hash) {
            hipModule_t mod = ctx.load_module(co_hash_lo, co_hash_hi);
            if (mod) {
                hipError_t e = hipModuleGetFunction(&func, mod, kernel_name.c_str());
                if (e != hipSuccess) func = nullptr;
            }
        }

        // Fallback: archives produced before the capture-side co_hash fix
        // (or kernels from load paths we don't track on capture) have
        // co_hash == 0,0.  Scan module_map then co_modules by name and
        // take the first match.  Correct only when the name is unique
        // across all loaded modules.
        if (!func) {
            std::shared_lock lk(ctx.map_mutex);
            for (auto& [rec_mod, live_mod] : ctx.module_map) {
                if (hipModuleGetFunction(&func, live_mod, kernel_name.c_str()) == hipSuccess
                    && func) break;
                func = nullptr;
            }
            if (!func) {
                for (auto& [hex, mod] : ctx.co_modules) {
                    if (hipModuleGetFunction(&func, mod, kernel_name.c_str()) == hipSuccess
                        && func) break;
                    func = nullptr;
                }
            }
        }

        if (!func) {
            fprintf(stderr, "[HRR] Kernel '%s' not found in any loaded module%s\n",
                    kernel_name.c_str(),
                    have_hash ? " (hash lookup also failed)" : "");
            return hipErrorNotFound;
        }
        std::unique_lock lk(ctx.map_mutex);
        ctx.func_cache.emplace(cache_key, func);
    }

    // Build kernelParams[] from captured args, translating GPU pointers.
    std::vector<void*>                arg_ptrs;
    std::vector<std::vector<uint8_t>> arg_storage;
    for (uint16_t i = 0; i < num_args; i++) {
        if (p + 3 > end) break;
        uint8_t  value_kind = *p++;
        uint16_t arg_size;
        memcpy(&arg_size, p, 2); p += 2;
        if (p + arg_size > end) break;

        if (value_kind == 2) {  // hidden arg — skip
            p += arg_size;
            continue;
        }
        arg_storage.emplace_back();
        auto& storage = arg_storage.back();
        if (value_kind == 1 && arg_size >= 8) {  // GPU pointer
            uint64_t rec_ptr; memcpy(&rec_ptr, p, 8);
            void* live = ctx.translate_ptr(rec_ptr);
            storage.resize(sizeof(void*));
            memcpy(storage.data(), &live, sizeof(void*));
            if (ctx.verbose)
                fprintf(stderr, "[HRR]   arg[%u]: ptr 0x%llx -> %p%s\n",
                        i, (unsigned long long)rec_ptr, live,
                        live ? "" : " (MISSING!)");
        } else {
            storage.assign(p, p + arg_size);
            if (ctx.verbose) {
                // Print scalar args as hex bytes for debugging
                fprintf(stderr, "[HRR]   arg[%u]: scalar %u bytes = ", i, arg_size);
                for (uint16_t b = 0; b < arg_size && b < 8; b++)
                    fprintf(stderr, "%02x", p[b]);
                if (arg_size > 8) fprintf(stderr, "...");
                // Also print as u32/u64 for convenience
                if (arg_size == 4) { uint32_t v; memcpy(&v, p, 4); fprintf(stderr, " (u32=%u)", v); }
                if (arg_size == 8) { uint64_t v; memcpy(&v, p, 8); fprintf(stderr, " (u64=%llu)", (unsigned long long)v); }
                fprintf(stderr, "\n");
            }

            // Heuristic: scan scalar struct args for embedded GPU pointers and
            // translate them in-place. Kernels that take a struct by value
            // (e.g. MIGraphX tensor_view, Thrust functors) carry GPU pointers
            // inside what the kernel signature reports as T_VALUE.  The capture
            // layer can't see into them, so the raw recorded VA gets copied
            // through and the GPU faults with hipErrorIllegalAddress on first
            // dereference.
            //
            // Scan 8-byte aligned offsets, look up each u64 candidate in
            // alloc_map (exact + range match), and substitute the live address
            // when found.  A minimum-value threshold avoids translating small
            // integers (sizes, counts, strides) that happen to fall within a
            // padded alloc range.
            constexpr uint64_t kMinPtrCandidate = 0x10000ull;  // 64 KiB
            for (uint16_t off = 0; off + 8 <= arg_size; off += 8) {
                uint64_t cand;
                std::memcpy(&cand, storage.data() + off, 8);
                if (cand < kMinPtrCandidate) continue;
                void* live = ctx.translate_ptr(cand);
                if (!live) continue;
                std::memcpy(storage.data() + off, &live, 8);
                if (ctx.verbose)
                    fprintf(stderr, "[HRR]     embedded ptr @off=%u: 0x%llx -> %p\n",
                            off, (unsigned long long)cand, live);
            }
        }
        arg_ptrs.push_back(storage.data());
        p += arg_size;
    }

    // Buffer snapshots — between args and the optional full_kbuf trailer.
    // Layout per record: u64 ptr_handle, u64 offset, u64 length,
    //                    u64 hash_lo, u64 hash_hi, u8 direction
    // direction == 0 -> pre-launch (input restore); 1 -> post-launch (--verify).
    // Older archives wrote num_snapshots = 0 and the loop is a no-op.
    std::vector<ReplaySnapshot> snapshots;
    snapshots.reserve(num_snapshots);
    for (uint16_t i = 0; i < num_snapshots; i++) {
        if (p + 41 > end) {
            fprintf(stderr, "[HRR] Kernel '%s': truncated snapshot record %u\n",
                    kernel_name.c_str(), i);
            return hipErrorInvalidValue;
        }
        ReplaySnapshot s{};
        memcpy(&s.ptr_handle, p, 8); p += 8;
        memcpy(&s.offset,     p, 8); p += 8;
        memcpy(&s.length,     p, 8); p += 8;
        memcpy(&s.hash_lo,    p, 8); p += 8;
        memcpy(&s.hash_hi,    p, 8); p += 8;
        s.direction = *p++;
        snapshots.push_back(s);
    }

    // v3.1 trailing field: the full packed kernarg buffer, when the original
    // launch went through extra[].  Older archives stop here (p == end).
    // We prefer this buffer for replay because it carries every byte the
    // kernel actually saw at capture time — including implicit/hidden args
    // (block_count_*, group_size_*, etc.) that the rocclr KernelSignature
    // does not expose and that MLIR/SP3 kernels rely on for correct execution.
    std::vector<uint8_t> full_kbuf;
    if (p + 4 <= end) {
        uint32_t fk_sz = 0;
        std::memcpy(&fk_sz, p, 4); p += 4;
        if (fk_sz > 0 && p + fk_sz <= end) {
            full_kbuf.assign(p, p + fk_sz);
            p += fk_sz;
            if (ctx.verbose)
                fprintf(stderr, "[HRR]   full_kbuf size=%u bytes\n", fk_sz);
            // Translate any embedded GPU pointers in-place using the same
            // heuristic as the scalar-arg path.  Hidden args (small integers,
            // dim counts, etc.) are below the threshold and left untouched.
            constexpr uint64_t kMinPtrCandidate = 0x10000ull;
            for (size_t off = 0; off + 8 <= full_kbuf.size(); off += 8) {
                uint64_t cand;
                std::memcpy(&cand, full_kbuf.data() + off, 8);
                if (cand < kMinPtrCandidate) continue;
                void* live = ctx.translate_ptr(cand);
                if (!live) continue;
                std::memcpy(full_kbuf.data() + off, &live, 8);
                if (ctx.verbose)
                    fprintf(stderr, "[HRR]   kbuf ptr @off=%zu: 0x%llx -> %p\n",
                            off, (unsigned long long)cand, live);
            }
            // Optional: full hex dump of the post-translation kbuf for a single
            // kernel of interest. Gated on env var HRR_DUMP_KBUF_KERNEL so it
            // only fires for the targeted kernel name (substring match) and
            // stays out of normal verbose output.
            if (const char* tag = std::getenv("HRR_DUMP_KBUF_KERNEL");
                tag && *tag && kernel_name.find(tag) != std::string::npos) {
                fprintf(stderr, "[HRR]   full_kbuf hex dump (%zu bytes) for '%s':\n",
                        full_kbuf.size(), kernel_name.c_str());
                for (size_t off = 0; off < full_kbuf.size(); off += 16) {
                    fprintf(stderr, "[HRR]     %04zx:", off);
                    for (size_t j = 0; j < 16 && off + j < full_kbuf.size(); ++j)
                        fprintf(stderr, " %02x", full_kbuf[off + j]);
                    fprintf(stderr, "\n");
                }
            }
        }
    }

    hipStream_t stream = ctx.translate_stream(stream_rec);

    // Restore direction=0 (pre-launch input) snapshots BEFORE the launch.
    // This is the self-healing core: even if some upstream op replayed
    // non-deterministically and produced different bytes, we overwrite the
    // input buffers with their captured pre-launch contents so this kernel
    // sees exactly what it saw at record time.  H2D copies are issued on the
    // same replay stream so they serialize naturally before the launch.
    if (ctx.restore_inputs && !snapshots.empty()) {
        for (const auto& s : snapshots) {
            if (s.direction != 0) continue;
            void* dst = ctx.translate_ptr(s.ptr_handle);
            if (!dst) {
                if (ctx.verbose)
                    fprintf(stderr,
                            "[HRR]   skip input restore (handle 0x%llx not mapped)\n",
                            (unsigned long long)s.ptr_handle);
                continue;
            }
            size_t blob_sz = 0;
            const void* blob = ctx.load_blob(s.hash_lo, s.hash_hi, &blob_sz);
            if (!blob) continue;
            size_t copy_sz = std::min<size_t>(s.length, blob_sz);
            // Clamp to the live allocation to avoid clobbering adjacent
            // alloc-map entries (padding factor can grow live > recorded).
            if (size_t avail = ctx.alloc_bytes_from(dst); avail && copy_sz > avail)
                copy_sz = avail;
            hipError_t er = stream
                ? hipMemcpyAsync(dst, blob, copy_sz,
                                 hipMemcpyHostToDevice, stream)
                : hipMemcpy     (dst, blob, copy_sz, hipMemcpyHostToDevice);
            if (er != hipSuccess) {
                fprintf(stderr,
                        "[HRR] input restore H2D failed for '%s' handle=0x%llx: "
                        "%d (%s)\n",
                        kernel_name.c_str(),
                        (unsigned long long)s.ptr_handle, er,
                        hipGetErrorString(er));
            } else {
                ctx.verify_input_restored++;
            }
        }
    }


    // Skip HIP event timing during graph capture: recording events on a
    // captured stream inserts them into the graph and invalidates the
    // capture state (error 901 on all subsequent operations).
    const bool do_timing = ctx.timing && !ctx.in_graph_capture;

    // Timing events are created once per replay thread and reused for every
    // kernel launch on that thread — no per-launch create/destroy overhead.
    // thread_local gives each replay thread its own independent pair.
    thread_local hipEvent_t tl_start = nullptr;
    thread_local hipEvent_t tl_stop  = nullptr;

    bool timing_ok = do_timing;
    if (timing_ok && !tl_start) {
        if (HRR_HIP_CHECK(hipEventCreate(&tl_start)) != hipSuccess ||
            HRR_HIP_CHECK(hipEventCreate(&tl_stop))  != hipSuccess) {
            tl_start = tl_stop = nullptr;
            timing_ok = false;
        } else {
            std::unique_lock lk(ctx.map_mutex);
            ctx.owned_timing_events.push_back(tl_start);
            ctx.owned_timing_events.push_back(tl_stop);
        }
    }

    if (timing_ok)
        timing_ok = (HRR_HIP_CHECK(hipEventRecord(tl_start, stream)) == hipSuccess);

    // Launch the kernel.
    //
    // Preferred path (v3.1 archives): if the capture recorded the full packed
    // kernarg buffer (extra[] launches), replay with that exact buffer via
    // extra[].  This preserves every byte the kernel actually saw — including
    // implicit/hidden args (block_count_*, group_size_*, grid_dims, etc.) that
    // the rocclr KernelSignature does not enumerate.  Required for correct
    // replay of MLIR-compiled kernels (mlir_*, MIOpen Find 2.0, MIGraphX) and
    // SP3 hand-assembled MIOpen kernels.
    //
    // Fallback for older archives (no full_kbuf trailer):
    //   SP3 assembly kernels (MIOpen Sp3Asm*) directly index the kernarg buffer
    //   at known offsets and also read hidden args (hidden_global_offset_x/y/z,
    //   etc.) that must be zero. Using kernelParams[] leaves hidden slots
    //   uninitialized in the ring-buffer allocator, which causes GPU faults.
    //   Instead, build a packed kernarg buffer using the AMDGPU ABI layout rule:
    //   each argument is placed at the next offset that is a multiple of its
    //   own size (natural alignment).  This matches exactly what
    //   amd::KernelSignature::at(i).offset_ reports.
    //
    //   HIP C++ kernels (clang-compiled) work correctly with kernelParams[]:
    //   the runtime handles hidden args internally, so no packed buffer is
    //   needed.
    //
    // Truncated-kbuf detection (diagnostic-only):
    //   Some apps (e.g. MIGraphX 2.15.0 MLIR launchers) call hipModuleLaunchKernel
    //   with an extra[] buffer sized only for the visible/explicit args, leaving
    //   no room for the AMDGPU hidden-arg trailer.  The capture-time launch then
    //   had the GPU read those bytes off the kernarg ring buffer — undefined.
    //   We detect this and warn, but we still replay the exact bytes the kernel
    //   saw at capture time.  Doing otherwise (e.g. rerouting through
    //   kernelParams[] so the runtime synthesizes correct hidden args) changes
    //   what the GPU sees vs. capture, which breaks --verify for the majority
    //   of these kernels whose bodies don't actually reference hidden args.
    //   Honest report-on-mismatch is more useful than self-healing here.
    bool capture_was_truncated_kbuf = false;
    if (!full_kbuf.empty()) {
        uint32_t visible_cursor = 0;
        for (const auto& s : arg_storage) {
            uint32_t sz = static_cast<uint32_t>(s.size());
            uint32_t align = (sz >= 8) ? 8u : (sz ? sz : 1u);
            visible_cursor = (visible_cursor + align - 1) & ~(align - 1);
            visible_cursor += sz;
        }
        if (full_kbuf.size() <= visible_cursor) {
            capture_was_truncated_kbuf = true;
            // One-shot warning per kernel: rate-limit so a busy workload
            // doesn't spam.  Verbose mode shows every occurrence.
            static std::mutex warn_mu;
            static std::unordered_set<std::string> warned;
            bool first = false;
            {
                std::lock_guard<std::mutex> lk(warn_mu);
                first = warned.insert(kernel_name).second;
            }
            if (first || ctx.verbose) {
                fprintf(stderr,
                        "[HRR] '%s': captured kbuf (%zu B) <= visible-only (%u B)"
                        " — upstream app under-provisioned HIP_LAUNCH_PARAM_BUFFER_SIZE."
                        "  Replaying the exact captured bytes; any --verify mismatch"
                        " on this kernel reflects capture-time non-determinism, not"
                        " a replay bug.\n",
                        kernel_name.c_str(), full_kbuf.size(), visible_cursor);
            }
        }
    }

    bool is_coop = (hdr->event_type == HRR_API_HIPLAUNCHCOOPERATIVEKERNEL ||
                    hdr->event_type == HRR_API_HIPLAUNCHCOOPERATIVEKERNEL_SPT ||
                    hdr->event_type == HRR_API_HIPMODULELAUNCHCOOPERATIVEKERNEL);

    // Skip launches where any grid dimension is zero or exceeds the hardware limit.
    //
    // Zero: some ROCm versions accept grid=0 silently; others return
    // hipErrorInvalidValue.  Either way no blocks run, so skipping is correct.
    //
    // > INT_MAX for X (e.g. 0x80000000): arises from signed→unsigned int32
    // overflow in the original app.  The capture runtime may have treated it as
    // a no-op; the replay runtime validates the argument and returns error 1.
    //
    // > 65535 for Y or Z: AMD GPU hardware encodes these in 16-bit fields
    // (max 0xFFFF = 65535).  Some capture-time runtime versions accept values
    // slightly above this limit for no-op kernels; the replay runtime does not.
    static constexpr uint32_t kMaxGridDimX  = 0x7FFFFFFFu;  // INT_MAX
    static constexpr uint32_t kMaxGridDimYZ = 65535u;        // 16-bit hardware field
    if (grid[0] == 0 || grid[1] == 0 || grid[2] == 0 ||
        grid[0] > kMaxGridDimX  ||
        grid[1] > kMaxGridDimYZ ||
        grid[2] > kMaxGridDimYZ) {
        if (ctx.verbose)
            fprintf(stderr, "[HRR] Kernel '%s' grid=[%u,%u,%u] out of range"
                    " — skipping (no-op launch)\n",
                    kernel_name.c_str(), grid[0], grid[1], grid[2]);
        return hipSuccess;
    }

    hipError_t r;
    if (is_coop) {
        // Cooperative kernels require hipModuleLaunchCooperativeKernel so the
        // runtime sets up the cooperative-groups sync buffer.  Using the plain
        // hipModuleLaunchKernel path skips that setup and the first
        // grid_group::sync() inside the kernel faults with error 700.
        // hipModuleLaunchCooperativeKernel only accepts kernelParams[], not
        // extra[], so we always use arg_ptrs here (sufficient for all HIP C++
        // cooperative kernels from hip-tests).
        r = hipModuleLaunchCooperativeKernel(
            func,
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
            shared_mem, stream,
            arg_ptrs.empty() ? nullptr : arg_ptrs.data());
    } else if (!full_kbuf.empty()) {
        size_t extra_sz = full_kbuf.size();
        void* extra[5] = {
            HIP_LAUNCH_PARAM_BUFFER_POINTER, full_kbuf.data(),
            HIP_LAUNCH_PARAM_BUFFER_SIZE,    &extra_sz,
            HIP_LAUNCH_PARAM_END
        };
        r = hipModuleLaunchKernel(
            func,
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
            shared_mem, stream,
            nullptr, extra);
    } else {
        bool is_sp3 = (kernel_name.find("Sp3") != std::string::npos ||
                       kernel_name.find("sp3") != std::string::npos);
        if (is_sp3 && !arg_ptrs.empty()) {
            // Compute kernarg layout from captured arg sizes using natural alignment.
            // Each arg aligns to its own size (max 8). Hidden args are zero-padded
            // at the end by over-allocating the buffer.
            uint32_t cursor = 0;
            std::vector<uint32_t> koffsets(arg_storage.size());
            for (size_t i = 0; i < arg_storage.size(); ++i) {
                uint32_t sz = static_cast<uint32_t>(arg_storage[i].size());
                uint32_t align = (sz >= 8) ? 8u : sz ? sz : 1u;
                cursor = (cursor + align - 1) & ~(align - 1);
                koffsets[i] = cursor;
                cursor += sz;
            }
            // Round up to 64-byte alignment; add 256 bytes for hidden arg space.
            uint32_t kbuf_sz = ((cursor + 63) & ~63u) + 256;
            std::vector<uint8_t> kbuf(kbuf_sz, 0);
            for (size_t i = 0; i < arg_storage.size(); ++i) {
                const auto& s = arg_storage[i];
                if (koffsets[i] + s.size() <= kbuf_sz)
                    memcpy(kbuf.data() + koffsets[i], s.data(), s.size());
            }
            size_t extra_sz = kbuf_sz;
            void* extra[5] = {
                HIP_LAUNCH_PARAM_BUFFER_POINTER, kbuf.data(),
                HIP_LAUNCH_PARAM_BUFFER_SIZE,    &extra_sz,
                HIP_LAUNCH_PARAM_END
            };
            r = hipModuleLaunchKernel(
                func,
                grid[0], grid[1], grid[2],
                block[0], block[1], block[2],
                shared_mem, stream,
                nullptr, extra);
        } else {
            // HIP C++ kernels: kernelParams[] path — runtime handles hidden args.
            r = hipModuleLaunchKernel(
                func,
                grid[0], grid[1], grid[2],
                block[0], block[1], block[2],
                shared_mem, stream,
                arg_ptrs.empty() ? nullptr : arg_ptrs.data(),
                nullptr);
        }
    }

    if (timing_ok)
        timing_ok = (HRR_HIP_CHECK(hipEventRecord(tl_stop, stream)) == hipSuccess);

    if (r != hipSuccess) {
        fprintf(stderr, "[HRR] Kernel '%s' launch error: %d (%s) func=%p"
                " grid=[%u,%u,%u] block=[%u,%u,%u]\n",
                kernel_name.c_str(), r, hipGetErrorString(r), (void*)func,
                grid[0], grid[1], grid[2], block[0], block[1], block[2]);
        return r;
    }

    if (timing_ok)
        timing_ok = (HRR_HIP_CHECK(hipEventSynchronize(tl_stop)) == hipSuccess);
    if (timing_ok) {
        float ms = 0.f;
        if (HRR_HIP_CHECK(hipEventElapsedTime(&ms, tl_start, tl_stop)) == hipSuccess) {
            std::unique_lock lk(ctx.map_mutex);
            ctx.total_kernel_ms += ms;
        }
    }

    if (ctx.sync_after_launch && !ctx.in_graph_capture) {
        // Clear any pre-existing error before sync so we get a clean error code.
        // Skipped during stream capture: hipDeviceSynchronize is forbidden while a
        // stream is in capture mode (returns hipErrorStreamCaptureUnsupported 900).
        hipGetLastError();
        r = hipDeviceSynchronize();
        hipError_t last_r = hipGetLastError();
        if (r == hipSuccess && last_r != hipSuccess) r = last_r;
        if (r != hipSuccess)
            fprintf(stderr, "[HRR] GPU error after '%s': %d (%s) last=%d (%s)\n",
                    kernel_name.c_str(), r, hipGetErrorString(r),
                    (int)last_r, hipGetErrorString(last_r));
        else if (ctx.verbose)
            fprintf(stderr, "[HRR] Kernel '%s' OK\n", kernel_name.c_str());
    }

    // Post-launch verify against direction=1 (output) snapshots.  Requires
    // a HIP_HRR_RECORD_MODE=full archive AND ctx.verify true.  The kernel
    // must have completed before D2H read — sync the stream (cheap if the
    // user also set sync_after_launch).  Failures are diagnosed with the
    // first differing byte for compactness; bumping verify_fail ensures the
    // overall exit code reports non-zero so CI can catch regressions.
    if (ctx.verify && !snapshots.empty()) {
        bool sync_ok = true;
        for (const auto& s : snapshots) {
            if (s.direction != 1) continue;
            if (sync_ok) {
                hipError_t se = stream ? hipStreamSynchronize(stream)
                                        : hipDeviceSynchronize();
                if (se != hipSuccess) {
                    fprintf(stderr,
                            "[HRR] verify: sync failed before D2H for '%s': "
                            "%d (%s) — skipping remaining snapshots\n",
                            kernel_name.c_str(), se, hipGetErrorString(se));
                    (void)hipGetLastError();
                    sync_ok = false;
                    ctx.verify_fail++;
                    continue;
                }
                // Sync only once per kernel: amortize cost across all output
                // snapshots for this launch instead of N times.
                sync_ok = true;
            }
            void* src = ctx.translate_ptr(s.ptr_handle);
            if (!src) {
                if (ctx.verbose)
                    fprintf(stderr, "[HRR] verify: handle 0x%llx not mapped — skip\n",
                            (unsigned long long)s.ptr_handle);
                continue;
            }
            size_t blob_sz = 0;
            const void* expected = ctx.load_blob(s.hash_lo, s.hash_hi, &blob_sz);
            if (!expected) {
                fprintf(stderr,
                        "[HRR] verify: blob %016llx%016llx missing for '%s'\n",
                        (unsigned long long)s.hash_hi,
                        (unsigned long long)s.hash_lo,
                        kernel_name.c_str());
                continue;
            }
            size_t cmp_len = std::min<size_t>(s.length, blob_sz);
            if (size_t avail = ctx.alloc_bytes_from(src); avail && cmp_len > avail)
                cmp_len = avail;
            std::vector<uint8_t> actual(cmp_len);
            hipError_t ce = hipMemcpy(actual.data(), src, cmp_len,
                                      hipMemcpyDeviceToHost);
            if (ce != hipSuccess) {
                fprintf(stderr,
                        "[HRR] verify: D2H failed for '%s' handle=0x%llx: %d (%s)\n",
                        kernel_name.c_str(),
                        (unsigned long long)s.ptr_handle, ce,
                        hipGetErrorString(ce));
                ctx.verify_fail++;
                continue;
            }
            if (memcmp(actual.data(), expected, cmp_len) == 0) {
                ctx.verify_pass++;
                if (ctx.verbose)
                    fprintf(stderr,
                            "[HRR] verify OK: '%s' handle=0x%llx %zu bytes\n",
                            kernel_name.c_str(),
                            (unsigned long long)s.ptr_handle, cmp_len);
            } else {
                ctx.verify_fail++;
                size_t first_diff = 0;
                const uint8_t* exp = static_cast<const uint8_t*>(expected);
                while (first_diff < cmp_len &&
                       actual[first_diff] == exp[first_diff])
                    ++first_diff;
                fprintf(stderr,
                        "[HRR] verify FAIL: '%s' handle=0x%llx %zu bytes, "
                        "first diff @byte %zu (got 0x%02x exp 0x%02x)%s\n",
                        kernel_name.c_str(),
                        (unsigned long long)s.ptr_handle, cmp_len, first_diff,
                        actual[first_diff], exp[first_diff],
                        capture_was_truncated_kbuf
                            ? "  [capture had truncated kbuf — likely upstream "
                              "non-determinism, not a replay bug]"
                            : "");
            }
        }
    }

    ctx.kernels_launched++;
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: kernel launches (all four variants share the same payload)
// ---------------------------------------------------------------------------

hipError_t playback_hipModuleLaunchKernel(PlaybackContext& ctx,
                                          const uint8_t* payload) {
    return replay_kernel_launch(ctx, payload);
}

hipError_t playback_hipExtModuleLaunchKernel(PlaybackContext& ctx,
                                             const uint8_t* payload) {
    return replay_kernel_launch(ctx, payload);
}

hipError_t playback_hipLaunchKernel(PlaybackContext& ctx,
                                    const uint8_t* payload) {
    return replay_kernel_launch(ctx, payload);
}

hipError_t playback_hipLaunchByPtr(PlaybackContext& ctx,
                                   const uint8_t* payload) {
    return replay_kernel_launch(ctx, payload);
}

hipError_t playback_hipExtLaunchKernel(PlaybackContext& ctx,
                                       const uint8_t* payload) {
    return replay_kernel_launch(ctx, payload);
}

hipError_t playback_hipLaunchCooperativeKernel(PlaybackContext& ctx,
                                               const uint8_t* payload) {
    return replay_kernel_launch(ctx, payload);
}

hipError_t playback_hipLaunchCooperativeKernel_spt(PlaybackContext& ctx,
                                                   const uint8_t* payload) {
    return replay_kernel_launch(ctx, payload);
}

hipError_t playback_hipModuleLaunchCooperativeKernel(PlaybackContext& ctx,
                                                     const uint8_t* payload) {
    return replay_kernel_launch(ctx, payload);
}

hipError_t playback_hipDrvLaunchKernelEx(PlaybackContext& ctx,
                                          const uint8_t* payload) {
    return replay_kernel_launch(ctx, payload);
}

// ---------------------------------------------------------------------------
// Manual playback: __hipRegisterFatBinary
// ---------------------------------------------------------------------------
// Load the fat binary blob via hipModuleLoadData so all embedded kernel names
// become resolvable at kernel launch replay time.
// Stored in co_modules keyed by the full 32-char hex hash — collision-free
// and consistent with load_module(), so kernel name scans find it automatically.

hipError_t playback___hipRegisterFatBinary(PlaybackContext& ctx,
                                           const uint8_t* payload) {
    const auto* a = reinterpret_cast<const hrr_args___hipRegisterFatBinary*>(payload);
    uint64_t blob_hash_lo = a->blob_hash_lo;
    uint64_t blob_hash_hi = a->blob_hash_hi;

    if (!blob_hash_lo && !blob_hash_hi) return hipSuccess;  // no blob — skip
    if (!a->blob_size) return hipSuccess;

    std::string hex = hrr::hash_hex(blob_hash_lo, blob_hash_hi);

    // Deduplicate: if already loaded (e.g. multiple __hipRegisterFatBinary events
    // for the same binary), skip the load.
    {
        std::shared_lock lk(ctx.map_mutex);
        if (ctx.co_modules.count(hex)) return hipSuccess;
    }

    size_t sz = 0;
    const void* blob = ctx.load_blob(blob_hash_lo, blob_hash_hi, &sz);
    if (!blob || sz == 0) {
        fprintf(stderr, "[HRR] __hipRegisterFatBinary: blob not found in archive\n");
        return hipSuccess;  // non-fatal — kernels will fail at launch but don't abort
    }

    hipModule_t mod = nullptr;
    hipError_t err = hipModuleLoadData(&mod, blob);
    if (err != hipSuccess) {
        fprintf(stderr, "[HRR] __hipRegisterFatBinary: hipModuleLoadData failed: %d (%s)\n",
                err, hipGetErrorString(err));
        return hipSuccess;  // non-fatal
    }

    {
        std::unique_lock lk(ctx.map_mutex);
        auto [it, inserted] = ctx.co_modules.emplace(hex, mod);
        if (!inserted) {
            // Another thread raced us between the shared_lock check and here — discard ours.
            hipModuleUnload(mod);
        }
    }
    if (ctx.verbose)
        fprintf(stderr, "[HRR] Loaded fat binary blob (%zu bytes) -> hipModule_t\n", sz);
    return hipSuccess;
}

// ---------------------------------------------------------------------------
// Manual playback: hipModuleGetFunction
// ---------------------------------------------------------------------------
// Resolves the function handle from the live translated module by name (stored
// as a blob at capture time) and records it in func_map so that subsequent
// APIs like hipFuncGetAttribute can translate the stale recorded handle.

hipError_t playback_hipModuleGetFunction(PlaybackContext& ctx,
                                         const uint8_t* payload) {
    const auto* a = reinterpret_cast<const hrr_args_hipModuleGetFunction*>(payload);

    // Retrieve function name from blob store
    size_t name_sz = 0;
    const char* name = static_cast<const char*>(
        ctx.load_blob(a->fname_hash_lo, a->fname_hash_hi, &name_sz));
    if (!name || name_sz == 0) {
        fprintf(stderr, "[HRR] hipModuleGetFunction: function name blob not found"
                " — function handle will not be translated\n");
        return hipSuccess;
    }

    hipModule_t live_mod = ctx.translate_module(a->module);
    if (!live_mod) {
        fprintf(stderr, "[HRR] hipModuleGetFunction: module handle not found for '%s'"
                " — skipping\n", name);
        return hipSuccess;
    }

    hipFunction_t func = nullptr;
    hipError_t r = hipModuleGetFunction(&func, live_mod, name);
    if (r == hipSuccess && func) {
        ctx.record_func(a->function, func);
    } else {
        fprintf(stderr, "[HRR] hipModuleGetFunction('%s') failed (%d)"
                " — function handle will not be translated\n", name, (int)r);
    }
    return hipSuccess;
}

// ---------------------------------------------------------------------------
// Manual playback: hipModuleLoadData / hipModuleLoadDataEx / hipModuleLoad
// ---------------------------------------------------------------------------
// Payload layout (after 32-byte EventHeader):
//   hipModuleLoadData / hipModuleLoadDataEx:
//     ret(4) module(8) image(8) co_hash_lo(8) co_hash_hi(8) [module_id(4)]
//   hipModuleLoad:
//     ret(4) module(8) fname(8) co_hash_lo(8) co_hash_hi(8) [module_id(4)]
//
// The recorded module handle is at offset +4 (8 bytes).
// co_hash_lo is at offset +20 (8 bytes), co_hash_hi at +28 (8 bytes).

static hipError_t replay_module_load(PlaybackContext& ctx,
                                     const uint8_t* payload) {
    // hipModuleLoad and hipModuleLoadData share the same layout for the fields we need
    const auto* a = reinterpret_cast<const hrr_args_hipModuleLoadData*>(payload);
    uint64_t rec_module = a->module;
    uint64_t co_hash_lo = a->co_hash_lo;
    uint64_t co_hash_hi = a->co_hash_hi;

    if (!co_hash_lo && !co_hash_hi) {
        // The capture shim failed to extract the code object (e.g. lazy
        // compilation not yet done, or unsupported image format).  Rather than
        // aborting, skip the module load gracefully: the module handle won't be
        // in the map, so any kernel launched from it will fail to resolve and
        // will be skipped or reported as not-found — far more informative than
        // an immediate abort here.
        fprintf(stderr, "[HRR] hipModuleLoad/DataEx: no code object hash in payload"
                " — module skipped (kernels from this module will not resolve)\n");
        return hipSuccess;
    }

    hipModule_t mod = ctx.load_module(co_hash_lo, co_hash_hi);
    if (!mod) return hipErrorSharedObjectInitFailed;

    ctx.record_module(rec_module, mod);
    return hipSuccess;
}

hipError_t playback_hipModuleLoadData(PlaybackContext& ctx,
                                      const uint8_t* payload) {
    return replay_module_load(ctx, payload);
}

hipError_t playback_hipModuleLoadDataEx(PlaybackContext& ctx,
                                        const uint8_t* payload) {
    return replay_module_load(ctx, payload);
}

hipError_t playback_hipModuleLoad(PlaybackContext& ctx,
                                  const uint8_t* payload) {
    return replay_module_load(ctx, payload);
}

hipError_t playback_hipModuleLoadFatBinary(PlaybackContext& ctx,
                                            const uint8_t* payload) {
    // hrr_args_hipModuleLoadFatBinary has the same field layout as
    // hrr_args_hipModuleLoadData (module, fatbin/image, co_hash_lo, co_hash_hi,
    // module_id at the same offsets) so the shared helper works directly.
    return replay_module_load(ctx, payload);
}

// ---------------------------------------------------------------------------
// Manual playback: hipMalloc / hipMallocManaged / hipHostMalloc
// ---------------------------------------------------------------------------
// Payload: ret(4) ptr(8) size(8) [additional fields for managed/host variants]
// ptr at +4, size at +12

// GPU allocation padding multiplier for replay.
//
// MIOpen / ROCBlas kernels are often launched with grids larger than the
// batch size (e.g. grid[0] = batch * n_groups with n_groups=96; if batch=1
// but grid is 256×n_groups, the kernel's s88 = blockIdx.x/n_groups sweeps
// 256 "virtual batches" and accesses memory at up to 256× the single-batch
// tensor size).  In the original application, a framework memory pool
// allocates GPU VA contiguously so the adjacent pages are all mapped;
// replay hipMallocs are independent and the adjacent pages are unmapped,
// causing GPU page faults.
//
// Fix: over-allocate all device allocations by HRR_ALLOC_PAD_FACTOR so
// any sub-allocation within the pool block has enough headroom.  The
// extra memory is zero-initialized.
//
// SP3AsmConv stride2 on 64×112×112 input sweeps 256 virtual batches:
//   256 × 64 × 112 × 112 × 4 = ~781 MB from in_ptr.
// A 1 GB cap ensures any sub-allocation has ≥800 MB headroom.
// With 46 GB GPU and ≤30 pool allocations: 30 × 1 GB = 30 GB — within budget.
static constexpr size_t HRR_ALLOC_PAD_FACTOR = 256;
static constexpr size_t HRR_ALLOC_PAD_MAX    = 1ULL * 1024 * 1024 * 1024;  // 1 GB cap

static hipError_t replay_malloc(PlaybackContext& ctx, const uint8_t* pl,
                                bool managed = false) {
    const auto* a = reinterpret_cast<const hrr_args_hipMalloc*>(pl);
    size_t orig_sz = static_cast<size_t>(a->size);
    // Padded size: multiply by factor but cap at 256 MB.
    size_t pad_sz = std::min(orig_sz * HRR_ALLOC_PAD_FACTOR, HRR_ALLOC_PAD_MAX);
    pad_sz = std::max(orig_sz, pad_sz);  // never shrink
    void* live = nullptr;
    hipError_t r;
    if (managed)
        r = hipMallocManaged(&live, pad_sz);
    else
        r = hipMalloc(&live, pad_sz);
    if (r == hipSuccess) {
        // hipMalloc on AMD returns zeroed memory (HIP spec requirement).
        // No explicit hipMemset needed — avoids blocking 256MB zeroing per alloc.
        //
        // Range search in translate_ptr must use the *recorded* (orig_sz) extent.
        // Storing pad_sz as the recorded extent has caused recorded sub-pointers
        // belonging to unrelated allocations to be silently mapped into another
        // allocation's padding region.
        ctx.record_alloc(a->ptr, live, orig_sz, pad_sz);
        if (ctx.verbose && pad_sz > orig_sz)
            fprintf(stderr, "[HRR] hipMalloc 0x%llx: orig=%zu padded=%zu\n",
                    (unsigned long long)a->ptr, orig_sz, pad_sz);
    }
    return r;
}

hipError_t playback_hipMalloc(PlaybackContext& ctx, const uint8_t* pl) {
    return replay_malloc(ctx, pl);
}
hipError_t playback_hipMallocManaged(PlaybackContext& ctx, const uint8_t* pl) {
    return replay_malloc(ctx, pl, /*managed=*/true);
}


// ---------------------------------------------------------------------------
// Manual playback: hipMallocAsync / hipMallocFromPoolAsync
// ---------------------------------------------------------------------------
// hipMallocAsync:  ret(4) dev_ptr(8) size(8) stream(8)
// hipMallocFromPoolAsync: ret(4) dev_ptr(8) size(8) mem_pool(8) stream(8)

hipError_t playback_hipMallocAsync(PlaybackContext& ctx,
                                   const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMallocAsync*>(pl);
    hipStream_t stream = ctx.translate_stream(a->stream);
    void* live = nullptr;
    size_t orig_sz = static_cast<size_t>(a->size);
    size_t pad_sz = std::max(orig_sz, std::min(orig_sz * HRR_ALLOC_PAD_FACTOR, HRR_ALLOC_PAD_MAX));
    hipError_t r = hipMallocAsync(&live, pad_sz, stream);
    if (r == hipSuccess)
        ctx.record_alloc(a->dev_ptr, live, orig_sz, pad_sz);
    return r;
}

hipError_t playback_hipMallocFromPoolAsync(PlaybackContext& ctx,
                                           const uint8_t* pl) {
    const auto* a  = reinterpret_cast<const hrr_args_hipMallocFromPoolAsync*>(pl);
    hipMemPool_t pool   = ctx.translate_mempool(a->mem_pool);
    hipStream_t  stream = ctx.translate_stream(a->stream);
    void* live = nullptr;
    size_t orig_sz = static_cast<size_t>(a->size);
    size_t pad_sz = std::max(orig_sz, std::min(orig_sz * HRR_ALLOC_PAD_FACTOR, HRR_ALLOC_PAD_MAX));
    hipError_t r = hipMallocFromPoolAsync(&live, pad_sz, pool, stream);
    if (r == hipSuccess)
        ctx.record_alloc(a->dev_ptr, live, orig_sz, pad_sz);
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: hipHostMalloc / hipMallocHost
// ---------------------------------------------------------------------------
// hipHostMalloc:  ret(4) ptr(8) size(8) flags(4)
// hipMallocHost:  ret(4) ptr(8) size(8)

hipError_t playback_hipHostMalloc(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipHostMalloc*>(pl);
    void* live = nullptr;
    hipError_t r = hipHostMalloc(&live, static_cast<size_t>(a->size), a->flags);
    if (r == hipSuccess) ctx.record_alloc(a->ptr, live, static_cast<size_t>(a->size));
    return r;
}

hipError_t playback_hipMallocHost(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMallocHost*>(pl);
    void* live = nullptr;
    hipError_t r = hipMallocHost(&live, static_cast<size_t>(a->size));
    if (r == hipSuccess) ctx.record_alloc(a->ptr, live, static_cast<size_t>(a->size));
    return r;
}

hipError_t playback_hipFreeHost(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipFreeHost*>(pl);
    void* live = ctx.translate_ptr(a->ptr);
    if (!live) return hipSuccess;
    hipError_t r = hipFreeHost(live);
    if (r == hipSuccess) ctx.remove_alloc(a->ptr);
    return r;
}

hipError_t playback_hipHostFree(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipHostFree*>(pl);
    void* live = ctx.translate_ptr(a->ptr);
    if (!live) return hipSuccess;
    hipError_t r = hipHostFree(live);
    if (r == hipSuccess) ctx.remove_alloc(a->ptr);
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: hipHostRegister / hipHostUnregister
// ---------------------------------------------------------------------------
// hipHostRegister recorded a snapshot of the host memory as a blob.
// At replay we allocate a fresh host buffer (malloc), restore the blob
// into it, call hipHostRegister on it, and track the (recorded -> live)
// mapping so kernel-arg pointer translations work.
// hipHostUnregister unregisters, frees the backing buffer, and removes the entry.

hipError_t playback_hipHostRegister(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipHostRegister*>(pl);
    size_t sz = static_cast<size_t>(a->sizeBytes);
    if (sz == 0) return hipSuccess;

    // Allocate backing host buffer aligned to 64 bytes (page-register friendly).
    void* buf = nullptr;
#ifdef _WIN32
    buf = _aligned_malloc(sz, 64);
#else
    if (posix_memalign(&buf, 64, sz) != 0) buf = nullptr;
#endif
    if (!buf) return hipErrorMemoryAllocation;

    // Restore snapshot into the buffer.
    if (a->blob_hash_lo || a->blob_hash_hi) {
        size_t blob_sz = 0;
        const void* blob = ctx.load_blob(a->blob_hash_lo, a->blob_hash_hi, &blob_sz);
        if (blob && blob_sz == sz)
            std::memcpy(buf, blob, sz);
        else
            std::memset(buf, 0, sz);
    } else {
        std::memset(buf, 0, sz);
    }

    hipError_t r = hipHostRegister(buf, sz, a->flags);
    if (r == hipSuccess) {
        ctx.record_alloc(a->hostPtr, buf, sz);
        std::unique_lock lk(ctx.map_mutex);
        ctx.host_reg_bufs[a->hostPtr] = buf;
    } else {
#ifdef _WIN32
        _aligned_free(buf);
#else
        free(buf);
#endif
    }
    return r;
}

hipError_t playback_hipHostUnregister(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipHostUnregister*>(pl);

    // Retrieve the backing buffer regardless of whether translate_ptr succeeds —
    // we must free it even if the alloc_map entry was already removed.
    void* buf = nullptr;
    {
        std::unique_lock lk(ctx.map_mutex);
        auto it = ctx.host_reg_bufs.find(a->hostPtr);
        if (it != ctx.host_reg_bufs.end()) {
            buf = it->second;
            ctx.host_reg_bufs.erase(it);
        }
    }

    void* live = buf ? buf : ctx.translate_ptr(a->hostPtr);
    if (!live) return hipSuccess;

    hipError_t r = hipHostUnregister(live);
    if (r == hipSuccess) ctx.remove_alloc(a->hostPtr);

#ifdef _WIN32
    _aligned_free(buf);
#else
    free(buf);
#endif
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: hipHostGetDevicePointer
// ---------------------------------------------------------------------------
// The generated shim would pass the raw recorded host ptr to the real API,
// which fails because it's a stale captured address.  We need to translate
// it through host_reg_bufs first, then record the returned device pointer
// in alloc_map so future translate_ptr calls work.

hipError_t playback_hipHostGetDevicePointer(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipHostGetDevicePointer*>(pl);

    // Check host_reg_bufs first (hipHostRegister path), then alloc_map
    // (hipHostMalloc path — already pinned, no register step needed).
    void* live_host = nullptr;
    {
        std::unique_lock lk(ctx.map_mutex);
        auto it = ctx.host_reg_bufs.find(a->hstPtr);
        if (it != ctx.host_reg_bufs.end())
            live_host = it->second;
    }
    if (!live_host)
        live_host = ctx.translate_ptr(a->hstPtr);

    if (!live_host) {
        fprintf(stderr, "[HRR] hipHostGetDevicePointer: no live buf for recorded hstPtr %llx\n",
                (unsigned long long)a->hstPtr);
        return hipErrorInvalidValue;
    }

    void* dev_ptr = nullptr;
    hipError_t r = hipHostGetDevicePointer(&dev_ptr, live_host, a->flags);
    if (r == hipSuccess) {
        ctx.record_alloc(a->devPtr, dev_ptr, 0);
    }
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: hipFree / hipFreeAsync
// ---------------------------------------------------------------------------
// hipFree:       ret(4) ptr(8)
// hipFreeAsync:  ret(4) dev_ptr(8) stream(8)

hipError_t playback_hipFree(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipFree*>(pl);
    void* live = ctx.translate_ptr(a->ptr);
    if (!live) return hipSuccess;
    hipError_t r = hipFree(live);
    if (r == hipSuccess) ctx.remove_alloc(a->ptr);
    return r;
}

hipError_t playback_hipFreeAsync(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a  = reinterpret_cast<const hrr_args_hipFreeAsync*>(pl);
    void*       live   = ctx.translate_ptr(a->dev_ptr);
    hipStream_t stream = ctx.translate_stream(a->stream);
    if (!live) return hipSuccess;
    hipError_t r = hipFreeAsync(live, stream);
    if (r == hipSuccess) ctx.remove_alloc(a->dev_ptr);
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: hipMemcpy / hipMemcpyAsync / hipMemcpyHtoD / hipMemcpyHtoDAsync
// ---------------------------------------------------------------------------

// is_async: true  -> call hipMemcpyAsync(stream) regardless of whether stream is null
//           false -> call synchronous hipMemcpy (no stream argument)
// This mirrors the captured API exactly — hipMemcpyAsync on the default stream
// (stream_rec==0, translated to nullptr) must still use the async variant.
static hipError_t replay_memcpy_impl(PlaybackContext& ctx,
                                     uint64_t dst_rec, uint64_t src_rec,
                                     uint64_t size, int32_t kind,
                                     bool is_async, hipStream_t stream,
                                     uint64_t hash_lo, uint64_t hash_hi) {
    void*      dst = ctx.translate_ptr(dst_rec);
    hipError_t r   = hipSuccess;


    if (kind == hipMemcpyHostToDevice && (hash_lo || hash_hi)) {
        size_t blob_sz = 0;
        const void* blob = ctx.load_blob(hash_lo, hash_hi, &blob_sz);
        if (!blob) {
            fprintf(stderr, "[HRR] H2D blob %016llx%016llx not found\n",
                    (unsigned long long)hash_lo, (unsigned long long)hash_hi);
            return hipErrorNotFound;
        }
        if (!dst) { fprintf(stderr, "[HRR] H2D dst 0x%llx not mapped (size=%llu blob_sz=%zu)\n",
                            (unsigned long long)dst_rec, (unsigned long long)size, blob_sz);
                    return hipErrorInvalidValue; }
        size_t copy_sz = static_cast<size_t>(size);
        if (copy_sz > blob_sz) copy_sz = blob_sz;
        size_t avail = ctx.alloc_bytes_from(dst);
        if (avail > 0 && copy_sz > avail) {
            fprintf(stderr, "[HRR] H2D dst 0x%llx: copy_sz=%zu > avail=%zu — clamping\n",
                    (unsigned long long)dst_rec, copy_sz, avail);
            copy_sz = avail;
        }
        if (is_async)
            r = hipMemcpyAsync(dst, blob, copy_sz, hipMemcpyHostToDevice, stream);
        else
            r = hipMemcpy(dst, blob, copy_sz, hipMemcpyHostToDevice);
        if (r != hipSuccess)
            fprintf(stderr, "[HRR] H2D memcpy failed: %d (%s) dst=%p copy_sz=%zu blob_sz=%zu avail=%zu\n",
                    r, hipGetErrorString(r), dst, copy_sz, blob_sz, avail);
    } else if (kind == hipMemcpyDeviceToDevice) {
        void* src = ctx.translate_ptr(src_rec);
        if (!dst) fprintf(stderr, "[HRR] D2D dst 0x%llx not mapped\n", (unsigned long long)dst_rec);
        if (!src) fprintf(stderr, "[HRR] D2D src 0x%llx not mapped\n", (unsigned long long)src_rec);
        if (dst && src) {
            size_t copy_sz = static_cast<size_t>(size);
            size_t dst_avail = ctx.alloc_bytes_from(dst);
            size_t src_avail = ctx.alloc_bytes_from(src);
            if (dst_avail > 0 && copy_sz > dst_avail) {
                fprintf(stderr, "[HRR] D2D dst_avail=%zu < copy_sz=%zu — clamping\n", dst_avail, copy_sz);
                copy_sz = dst_avail;
            }
            if (src_avail > 0 && copy_sz > src_avail)
                copy_sz = src_avail;
            if (is_async)
                r = hipMemcpyAsync(dst, src, copy_sz,
                                   hipMemcpyDeviceToDevice, stream);
            else
                r = hipMemcpy(dst, src, copy_sz,
                              hipMemcpyDeviceToDevice);
            if (r != hipSuccess)
                fprintf(stderr, "[HRR] D2D memcpy failed: %d (%s)\n", r, hipGetErrorString(r));
        }
    } else if (kind == hipMemcpyDeviceToHost && ctx.validate_d2h &&
               (hash_lo || hash_hi)) {
        // D2H validation: copy from live device src into a local host buffer,
        // then compare against the expected data blob captured at record time.
        void* src_dev = ctx.translate_ptr(src_rec);
        if (!src_dev) {
            if (ctx.verbose)
                fprintf(stderr, "[HRR] D2H validate: src 0x%llx not mapped — skip\n",
                        (unsigned long long)src_rec);
        } else {
            size_t copy_sz = static_cast<size_t>(size);
            size_t blob_sz = 0;
            const void* expected = ctx.load_blob(hash_lo, hash_hi, &blob_sz);
            if (!expected) {
                fprintf(stderr, "[HRR] D2H validate: expected blob not found — skip\n");
            } else {
                copy_sz = std::min(copy_sz, blob_sz);
                std::vector<uint8_t> actual(copy_sz);
                // For async memcpy the stream may not yet have completed — sync it so
                // all preceding GPU work has finished before reading back.
                // Synchronous hipMemcpy already guarantees completion; no extra sync needed.
                if (is_async) hipStreamSynchronize(stream);
                r = hipMemcpy(actual.data(), src_dev, copy_sz, hipMemcpyDeviceToHost);
                if (r != hipSuccess) {
                    fprintf(stderr, "[HRR] D2H validate: hipMemcpy failed: %d (%s)\n",
                            r, hipGetErrorString(r));
                    ctx.d2h_fail++;
                } else if (memcmp(actual.data(), expected, copy_sz) == 0) {
                    ctx.d2h_pass++;
                    if (ctx.verbose)
                        fprintf(stderr, "[HRR] D2H validate: %zu bytes OK\n", copy_sz);
                } else {
                    ctx.d2h_fail++;
                    // Find first differing byte for diagnostics
                    size_t first_diff = 0;
                    const uint8_t* exp = static_cast<const uint8_t*>(expected);
                    while (first_diff < copy_sz && actual[first_diff] == exp[first_diff])
                        ++first_diff;
                    fprintf(stderr,
                            "[HRR] D2H validate FAIL: %zu bytes, first diff at byte %zu "
                            "(got 0x%02x expected 0x%02x)\n",
                            copy_sz, first_diff,
                            actual[first_diff], exp[first_diff]);
                }
            }
        }
    }
    // H2H / unhandled: no-op
    return r;
}

hipError_t playback_hipMemcpy(PlaybackContext& ctx,
                              const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpy*>(pl);
    return replay_memcpy_impl(ctx, a->dst, a->src, a->sizeBytes, a->kind,
                              /*is_async=*/false, nullptr,
                              a->blob_hash_lo, a->blob_hash_hi);
}

hipError_t playback_hipMemcpyAsync(PlaybackContext& ctx,
                                   const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpyAsync*>(pl);
    return replay_memcpy_impl(ctx, a->dst, a->src, a->sizeBytes, a->kind,
                              /*is_async=*/true, ctx.translate_stream(a->stream),
                              a->blob_hash_lo, a->blob_hash_hi);
}

hipError_t playback_hipMemcpyHtoD(PlaybackContext& ctx,
                                  const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpyHtoD*>(pl);
    return replay_memcpy_impl(ctx, a->dst, a->src, a->sizeBytes,
                              hipMemcpyHostToDevice,
                              /*is_async=*/false, nullptr,
                              a->blob_hash_lo, a->blob_hash_hi);
}

hipError_t playback_hipMemcpyHtoDAsync(PlaybackContext& ctx,
                                       const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpyHtoDAsync*>(pl);
    return replay_memcpy_impl(ctx, a->dst, a->src, a->sizeBytes,
                              hipMemcpyHostToDevice,
                              /*is_async=*/true, ctx.translate_stream(a->stream),
                              a->blob_hash_lo, a->blob_hash_hi);
}

// ---------------------------------------------------------------------------
// Manual playback: hipMemcpyDtoH / hipMemcpyDtoHAsync
// ---------------------------------------------------------------------------
// The generated playback calls hipMemcpyDtoH(ctx.translate_ptr(a->dst), ...).
// For plain malloc/stack/new host buffers, translate_ptr returns nullptr →
// hipErrorInvalidValue.  Route through replay_memcpy_impl instead: for the
// normal (non-validate) case it skips the actual copy (D2H has no blob to
// load and the result isn't used by subsequent HIP calls); for --validate mode
// it allocates its own temp buffer and checks against the captured snapshot.
hipError_t playback_hipMemcpyDtoH(PlaybackContext& ctx, 
                                  const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpyDtoH*>(pl);
    return replay_memcpy_impl(ctx, a->dst, a->src, a->sizeBytes,
                              hipMemcpyDeviceToHost,
                              /*is_async=*/false, nullptr,
                              a->blob_hash_lo, a->blob_hash_hi);
}

hipError_t playback_hipMemcpyDtoHAsync(PlaybackContext& ctx,
                                       const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpyDtoHAsync*>(pl);
    return replay_memcpy_impl(ctx, a->dst, a->src, a->sizeBytes,
                              hipMemcpyDeviceToHost,
                              /*is_async=*/true, ctx.translate_stream(a->stream),
                              a->blob_hash_lo, a->blob_hash_hi);
}

// ---------------------------------------------------------------------------
// Manual playback: hipMemcpyWithStream
// ---------------------------------------------------------------------------
// Synchronous copy with stream. Captured by manual shim (has blob_hash fields).
// Routes through replay_memcpy_impl exactly like hipMemcpy/hipMemcpyAsync.
// is_async=true so the stream is passed through (even if it translates to null).
hipError_t playback_hipMemcpyWithStream(PlaybackContext& ctx,
                                        const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpyWithStream*>(pl);
    return replay_memcpy_impl(ctx, a->dst, a->src, a->sizeBytes, a->kind,
                              /*is_async=*/true, ctx.translate_stream(a->stream),
                              a->blob_hash_lo, a->blob_hash_hi);
}

// ---------------------------------------------------------------------------
// Manual playback: hipDrvMemcpy2DUnaligned / hipMemcpyParam2D[Async]
// ---------------------------------------------------------------------------
// The capture shim serialises the full hip_Memcpy2D struct fields plus a
// linearised H2D source blob (Height × WidthInBytes contiguous bytes).
// During replay we reconstruct the struct, translate device pointers through
// the alloc map, load the H2D blob when present, and execute the copy.
//
// For H2D the blob is linearised (no pitch) so we present it with
// srcXInBytes=0, srcY=0, srcPitch=WidthInBytes, clearing the original offsets.

static hipError_t replay_memcpy_param2d_impl(
        PlaybackContext& ctx,
        const hrr_args_hipDrvMemcpy2DUnaligned* a,
        bool is_async, hipStream_t stream) {

    hip_Memcpy2D desc{};
    desc.srcXInBytes   = static_cast<size_t>(a->src_x_bytes);
    desc.srcY          = static_cast<size_t>(a->src_y);
    desc.srcMemoryType = static_cast<hipMemoryType>(a->src_mem_type);
    desc.srcPitch      = static_cast<size_t>(a->src_pitch);
    desc.dstXInBytes   = static_cast<size_t>(a->dst_x_bytes);
    desc.dstY          = static_cast<size_t>(a->dst_y);
    desc.dstMemoryType = static_cast<hipMemoryType>(a->dst_mem_type);
    desc.dstPitch      = static_cast<size_t>(a->dst_pitch);
    desc.WidthInBytes  = static_cast<size_t>(a->width_bytes);
    desc.Height        = static_cast<size_t>(a->height);

    // Translate device and array pointers through the live alloc map.
    desc.srcDevice = reinterpret_cast<hipDeviceptr_t>(ctx.translate_ptr(a->src_device));
    desc.dstDevice = reinterpret_cast<hipDeviceptr_t>(ctx.translate_ptr(a->dst_device));
    desc.srcArray  = reinterpret_cast<hipArray_t>(ctx.translate_ptr(a->src_array));
    desc.dstArray  = reinterpret_cast<hipArray_t>(ctx.translate_ptr(a->dst_array));

    // H2D: load the linearised source blob and present it as a packed host row.
    std::vector<uint8_t> src_blob;
    if (desc.srcMemoryType == hipMemoryTypeHost &&
        (a->blob_hash_lo || a->blob_hash_hi)) {
        size_t blob_sz = 0;
        const void* blob = ctx.load_blob(a->blob_hash_lo, a->blob_hash_hi, &blob_sz);
        if (!blob) {
            fprintf(stderr, "[HRR] 2D H2D blob %016llx%016llx not found\n",
                    (unsigned long long)a->blob_hash_lo,
                    (unsigned long long)a->blob_hash_hi);
            return hipErrorNotFound;
        }
        src_blob.assign(static_cast<const uint8_t*>(blob),
                        static_cast<const uint8_t*>(blob) + blob_sz);
        // Linearised data: present with zero offsets and packed pitch.
        desc.srcHost     = src_blob.data();
        desc.srcXInBytes = 0;
        desc.srcY        = 0;
        desc.srcPitch    = desc.WidthInBytes;
    }

    // D2H dst: use a temporary buffer (host dst is stale; output not consumed).
    std::vector<uint8_t> dst_buf;
    if (desc.dstMemoryType == hipMemoryTypeHost) {
        const size_t pitch = desc.dstPitch ? desc.dstPitch : desc.WidthInBytes;
        const size_t sz    = (desc.dstY + desc.Height) * pitch + desc.WidthInBytes;
        dst_buf.resize(sz, 0);
        desc.dstHost     = dst_buf.data();
        desc.dstXInBytes = 0;
        desc.dstY        = 0;
        desc.dstPitch    = pitch;
    }

    if (!desc.srcDevice && !desc.srcHost && !desc.srcArray) {
        fprintf(stderr, "[HRR] 2D memcpy: no src mapped (src_mem_type=%u src_dev=0x%llx)\n",
                a->src_mem_type, (unsigned long long)a->src_device);
        return hipSuccess;
    }
    if (!desc.dstDevice && !desc.dstHost && !desc.dstArray) {
        fprintf(stderr, "[HRR] 2D memcpy: no dst mapped (dst_mem_type=%u dst_dev=0x%llx)\n",
                a->dst_mem_type, (unsigned long long)a->dst_device);
        return hipSuccess;
    }

    hipError_t r;
    if (is_async)
        r = hipMemcpyParam2DAsync(&desc, stream);
    else
        r = hipMemcpyParam2D(&desc);

    if (r != hipSuccess)
        fprintf(stderr, "[HRR] 2D memcpy playback failed: %d (%s)\n",
                r, hipGetErrorString(r));
    return r;
}

hipError_t playback_hipDrvMemcpy2DUnaligned(PlaybackContext& ctx,
                                             const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipDrvMemcpy2DUnaligned*>(pl);
    return replay_memcpy_param2d_impl(ctx, a, /*is_async=*/false, nullptr);
}

hipError_t playback_hipMemcpyParam2D(PlaybackContext& ctx,
                                      const uint8_t* pl) {
    // hrr_args_hipMemcpyParam2D has the same base layout as
    // hrr_args_hipDrvMemcpy2DUnaligned (hdr + ret + pCopy + extra fields).
    static_assert(sizeof(hrr_args_hipMemcpyParam2D) ==
                  sizeof(hrr_args_hipDrvMemcpy2DUnaligned),
                  "hrr_args_hipMemcpyParam2D layout mismatch");
    const auto* a = reinterpret_cast<const hrr_args_hipDrvMemcpy2DUnaligned*>(pl);
    return replay_memcpy_param2d_impl(ctx, a, /*is_async=*/false, nullptr);
}

hipError_t playback_hipMemcpyParam2DAsync(PlaybackContext& ctx,
                                           const uint8_t* pl) {
    // hrr_args_hipMemcpyParam2DAsync has an extra `stream` base field before
    // the 2D extra fields, so we must read the extra fields at the right offset.
    const auto* a2 = reinterpret_cast<const hrr_args_hipMemcpyParam2DAsync*>(pl);
    hipStream_t stream = ctx.translate_stream(a2->stream);
    // The extra fields start at a2->src_x_bytes; alias them through a helper
    // pointer that covers only the extra region.
    // Construct a temporary hrr_args_hipDrvMemcpy2DUnaligned on the stack with
    // the extra fields copied in so replay_memcpy_param2d_impl can read them.
    hrr_args_hipDrvMemcpy2DUnaligned tmp{};
    tmp.src_x_bytes  = a2->src_x_bytes;
    tmp.src_y        = a2->src_y;
    tmp.src_mem_type = a2->src_mem_type;
    tmp.pad0         = 0;
    tmp.src_host     = a2->src_host;
    tmp.src_device   = a2->src_device;
    tmp.src_array    = a2->src_array;
    tmp.src_pitch    = a2->src_pitch;
    tmp.dst_x_bytes  = a2->dst_x_bytes;
    tmp.dst_y        = a2->dst_y;
    tmp.dst_mem_type = a2->dst_mem_type;
    tmp.pad1         = 0;
    tmp.dst_host     = a2->dst_host;
    tmp.dst_device   = a2->dst_device;
    tmp.dst_array    = a2->dst_array;
    tmp.dst_pitch    = a2->dst_pitch;
    tmp.width_bytes  = a2->width_bytes;
    tmp.height       = a2->height;
    tmp.blob_hash_lo = a2->blob_hash_lo;
    tmp.blob_hash_hi = a2->blob_hash_hi;
    return replay_memcpy_param2d_impl(ctx, &tmp, /*is_async=*/true, stream);
}

// ---------------------------------------------------------------------------
// Manual playback: stream create/destroy
// ---------------------------------------------------------------------------
// hipStreamCreate:              ret(4) stream(8)
// hipStreamCreateWithFlags:     ret(4) stream(8) flags(4)
// hipStreamCreateWithPriority:  ret(4) stream(8) flags(4) priority(4)
// hipStreamDestroy:             ret(4) stream(8)

hipError_t playback_hipStreamCreate(PlaybackContext& ctx,
                                    const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipStreamCreate*>(pl);
    hipStream_t s = nullptr;
    hipError_t r = hipStreamCreate(&s);
    if (r == hipSuccess) ctx.record_stream(a->stream, s);
    return r;
}

hipError_t playback_hipStreamCreateWithFlags(PlaybackContext& ctx,
                                             const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipStreamCreateWithFlags*>(pl);
    hipStream_t s = nullptr;
    hipError_t r  = hipStreamCreateWithFlags(&s, a->flags);
    if (r == hipSuccess) {
        ctx.record_stream(a->stream, s);
        if (ctx.verbose)
            fprintf(stderr, "[HRR] StreamCreateWithFlags: rec=0x%llx -> live=%p\n",
                    (unsigned long long)a->stream, (void*)s);
    }
    return r;
}

hipError_t playback_hipStreamCreateWithPriority(PlaybackContext& ctx,
                                                const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipStreamCreateWithPriority*>(pl);
    hipStream_t s = nullptr;
    hipError_t  r = hipStreamCreateWithPriority(&s, a->flags, a->priority);
    if (r == hipSuccess) ctx.record_stream(a->stream, s);
    return r;
}

hipError_t playback_hipStreamDestroy(PlaybackContext& ctx,
                                     const uint8_t* pl) {
    const auto* a  = reinterpret_cast<const hrr_args_hipStreamDestroy*>(pl);
    hipStream_t stream = ctx.translate_stream(a->stream);
    hipError_t r = hipSuccess;
    if (stream) r = hipStreamDestroy(stream);
    ctx.remove_stream(a->stream);
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: hipStreamEndCapture / hipGraphInstantiate
// ---------------------------------------------------------------------------
// Stream-capture flow:
//   hipStreamBeginCapture — generated shim calls real API (no handle output)
//   hipStreamEndCapture   — calls real API, records resulting hipGraph_t handle
//   hipGraphInstantiate   — calls real API, records resulting hipGraphExec_t handle
//   hipGraphLaunch        — generated shim translates both handles; works once above succeed
//
// hrr_args_hipStreamEndCapture layout (after 32-byte EventHeader):
//   ret(4) stream(8) pGraph(8)       — pGraph = recorded *pGraph output value
//
// hrr_args_hipGraphInstantiate layout:
//   ret(4) pGraphExec(8) graph(8) pErrorNode(8) pLogBuffer(8) bufferSize(8)

// hrr_args_hipStreamBeginCapture payload (after 32-byte EventHeader):
//   ret(4) stream(8) mode(4)
hipError_t playback_hipStreamBeginCapture(PlaybackContext& ctx,
                                          const uint8_t* payload) {
    const auto* a = reinterpret_cast<const hrr_args_hipStreamBeginCapture*>(payload);
    if (a->ret != hipSuccess) return hipSuccess;  // original failed — skip

    hipStream_t stream = ctx.translate_stream(a->stream);
    if (!stream && a->stream != 0) {
        // Stream handle not in map — create a temporary stream for graph capture
        fprintf(stderr, "[HRR] hipStreamBeginCapture: stream 0x%llx not found, "
                "creating temp stream for graph capture\n",
                (unsigned long long)a->stream);
        hipError_t cr = hipStreamCreate(&stream);
        if (cr != hipSuccess) {
            fprintf(stderr, "[HRR] hipStreamBeginCapture: failed to create temp stream: %d\n", cr);
            return hipSuccess;  // non-fatal
        }
        ctx.record_stream(a->stream, stream);
    }

    hipStreamCaptureMode mode = (hipStreamCaptureMode)a->mode;
    hipError_t r = hipStreamBeginCapture(stream, mode);
    if (r != hipSuccess && mode != hipStreamCaptureModeGlobal) {
        // ThreadLocal may fail in replay context — try Global
        r = hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
        if (r != hipSuccess)
            fprintf(stderr, "[HRR] hipStreamBeginCapture failed (both modes): %d (%s)\n",
                    r, hipGetErrorString(r));
    }
    if (r == hipSuccess)
        ctx.in_graph_capture = true;
    return r;
}

hipError_t playback_hipStreamEndCapture(PlaybackContext& ctx,
                                        const uint8_t* payload) {
    const auto* a = reinterpret_cast<const hrr_args_hipStreamEndCapture*>(payload);
    if (a->ret != hipSuccess) return hipSuccess;  // original call failed — skip

    hipStream_t stream = ctx.translate_stream(a->stream);
    if (!stream) {
        fprintf(stderr, "[HRR] hipStreamEndCapture: stream 0x%llx not found in map\n",
                (unsigned long long)a->stream);
        return hipSuccess;  // non-fatal
    }
    ctx.in_graph_capture = false;
    hipGraph_t live_graph = nullptr;
    hipError_t r = hipStreamEndCapture(stream, &live_graph);
    if (r == hipSuccess && live_graph) {
        ctx.record_graph(a->pGraph, live_graph);
        if (ctx.verbose)
            fprintf(stderr, "[HRR] hipStreamEndCapture: recorded graph 0x%llx\n",
                    (unsigned long long)a->pGraph);
    } else {
        fprintf(stderr, "[HRR] hipStreamEndCapture failed: %d (%s)\n",
                r, hipGetErrorString(r));
    }
    return r;
}

hipError_t playback_hipGraphInstantiate(PlaybackContext& ctx,
                                        const uint8_t* payload) {
    const auto* a = reinterpret_cast<const hrr_args_hipGraphInstantiate*>(payload);
    if (a->ret != hipSuccess) return hipSuccess;  // original call failed — skip

    hipGraph_t graph = ctx.translate_graph(a->graph);
    if (!graph) {
        fprintf(stderr, "[HRR] hipGraphInstantiate: graph 0x%llx not found in map\n",
                (unsigned long long)a->graph);
        return hipSuccess;  // non-fatal — launches will be skipped
    }

    hipGraphExec_t exec = nullptr;
    // Use the simplified WithFlags variant; pErrorNode/pLogBuffer are optional at replay
    hipError_t r = hipGraphInstantiateWithFlags(&exec, graph, 0);
    if (r == hipSuccess && exec) {
        ctx.record_graph_exec(a->pGraphExec, exec);
        if (ctx.verbose)
            fprintf(stderr, "[HRR] hipGraphInstantiate: recorded exec 0x%llx\n",
                    (unsigned long long)a->pGraphExec);
    } else {
        fprintf(stderr, "[HRR] hipGraphInstantiate (via WithFlags) failed: %d (%s)\n",
                r, hipGetErrorString(r));
    }
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: hipGraphLaunch
// ---------------------------------------------------------------------------
// Payload layout (after 32-byte EventHeader):
//   ret(4) graphExec(8) stream(8)
hipError_t playback_hipGraphLaunch(PlaybackContext& ctx,
                                   const uint8_t* payload) {
    const auto* a = reinterpret_cast<const hrr_args_hipGraphLaunch*>(payload);
    if (a->ret != hipSuccess) return hipSuccess;  // original call failed — skip

    hipGraphExec_t exec = ctx.translate_graph_exec(a->graphExec);
    if (!exec) {
        if (ctx.verbose)
            fprintf(stderr, "[HRR] hipGraphLaunch: graphExec 0x%llx not found in map\n",
                    (unsigned long long)a->graphExec);
        return hipSuccess;  // non-fatal — exec not yet created
    }

    hipStream_t stream = ctx.translate_stream(a->stream);

    thread_local hipEvent_t tl_g_start = nullptr;
    thread_local hipEvent_t tl_g_stop  = nullptr;
    bool timing_ok = ctx.timing;
    if (timing_ok && !tl_g_start) {
        if (HRR_HIP_CHECK(hipEventCreate(&tl_g_start)) != hipSuccess ||
            HRR_HIP_CHECK(hipEventCreate(&tl_g_stop))  != hipSuccess) {
            tl_g_start = tl_g_stop = nullptr;
            timing_ok = false;
        } else {
            std::unique_lock lk(ctx.map_mutex);
            ctx.owned_timing_events.push_back(tl_g_start);
            ctx.owned_timing_events.push_back(tl_g_stop);
        }
    }
    if (timing_ok)
        timing_ok = (HRR_HIP_CHECK(hipEventRecord(tl_g_start, stream)) == hipSuccess);

    hipError_t r = hipGraphLaunch(exec, stream);

    if (timing_ok)
        timing_ok = (HRR_HIP_CHECK(hipEventRecord(tl_g_stop, stream)) == hipSuccess);

    if (r != hipSuccess) {
        fprintf(stderr, "[HRR] hipGraphLaunch failed: %d (%s) exec=0x%llx stream=0x%llx\n",
                r, hipGetErrorString(r),
                (unsigned long long)a->graphExec, (unsigned long long)a->stream);
        return r;
    }

    ctx.graphs_launched.fetch_add(1, std::memory_order_relaxed);

    if (timing_ok)
        timing_ok = (HRR_HIP_CHECK(hipEventSynchronize(tl_g_stop)) == hipSuccess);
    if (timing_ok) {
        float ms = 0.f;
        if (HRR_HIP_CHECK(hipEventElapsedTime(&ms, tl_g_start, tl_g_stop)) == hipSuccess) {
            std::unique_lock lk(ctx.map_mutex);
            ctx.total_graph_ms += ms;
        }
    }

    if (ctx.verbose)
        fprintf(stderr, "[HRR] hipGraphLaunch: exec 0x%llx on stream 0x%llx -> OK\n",
                (unsigned long long)a->graphExec, (unsigned long long)a->stream);
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: event create/destroy
// ---------------------------------------------------------------------------
// hipEventCreate:            ret(4) event(8)
// hipEventCreateWithFlags:   ret(4) event(8) flags(4)
// hipEventDestroy:           ret(4) event(8)

hipError_t playback_hipEventCreate(PlaybackContext& ctx,
                                   const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipEventCreate*>(pl);
    hipEvent_t e = nullptr;
    hipError_t r = hipEventCreate(&e);
    if (r == hipSuccess) ctx.record_event(a->event, e);
    return r;
}

hipError_t playback_hipEventCreateWithFlags(PlaybackContext& ctx,
                                            const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipEventCreateWithFlags*>(pl);
    hipEvent_t e  = nullptr;
    hipError_t r  = hipEventCreateWithFlags(&e, a->flags);
    if (r == hipSuccess) ctx.record_event(a->event, e);
    return r;
}

hipError_t playback_hipEventDestroy(PlaybackContext& ctx,
                                    const uint8_t* pl) {
    const auto* a  = reinterpret_cast<const hrr_args_hipEventDestroy*>(pl);
    hipEvent_t event = ctx.translate_event(a->event);
    hipError_t r = hipSuccess;
    if (event) r = hipEventDestroy(event);
    ctx.remove_event(a->event);
    return r;
}
