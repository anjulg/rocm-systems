/* Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License */
#pragma once

/*
 * hip_capture.h — In-tree HIP capture layer public API.
 *
 * Captures HIP API calls to a .hrr archive for later replay.
 * Independent of the profiler layer — no profiler headers included.
 *
 * Binary format is compatible with hrr_reader.h / hrr_replay.cpp.
 * Format constants (HRR_MAGIC, HRR_VERSION, hrr_file_header) are in
 * hrr_api_args.h — included by hip_capture.cpp and hip_capture_writer.cpp.
 */

#include <cstddef>
#include <cstdint>

namespace hrr_cap {

// 128-bit hash (FNV-1a variant) — return type of write_blob / write_code_object
struct Hash128 {
  uint64_t lo;
  uint64_t hi;
};

}  // namespace hrr_cap

// ---------------------------------------------------------------------------
// Public API — called from hip_context.cpp and hip_capture.cpp
// ---------------------------------------------------------------------------

// Check if capture is enabled (HIP_HRR_CAPTURE_OUTPUT env var set and non-empty)
bool hip_capture_enabled();

// Return the output directory from the env var
const char* hip_capture_output_dir();

// HRR snapshot recording mode (parsed from HIP_HRR_RECORD_MODE flag).
// 0 = timeline (no per-kernel snapshots; baseline behaviour)
// 1 = inputs   (pre-launch direction=0 snapshots — restored before each replay)
// 2 = full     (also post-launch direction=1 snapshots — checked by --verify)
enum HrrRecordMode : int {
  HRR_RECORD_TIMELINE = 0,
  HRR_RECORD_INPUTS   = 1,
  HRR_RECORD_FULL     = 2,
};
int   hip_capture_record_mode();

// Per-snapshot byte cap. 0 = no cap (NOT recommended for arena allocators —
// the "snap size = rest-of-alloc" heuristic over-captures aggressively).
size_t hip_capture_max_snap_bytes();

// Snapshot real fn ptrs (must be called while live table holds real ptrs)
void hip_capture_build_table();

// Install capture shims into live dispatch tables
void hip_capture_install();

// Restore real dispatch tables
void hip_capture_uninstall();

// Hook compiler dispatch table for <<<>>> launch path
void hip_capture_build_compiler_table();

// Called from hip_context.cpp init() — performs build + conditional install
void hip_capture_init();

// Called at atexit — uninstalls shims and flushes the archive to disk
void hip_capture_shutdown();
