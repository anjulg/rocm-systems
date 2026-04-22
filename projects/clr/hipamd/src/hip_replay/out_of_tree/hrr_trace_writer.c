/* Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License */

/* Out-of-tree trace writer. Same .hrr format as in-tree hip_hrr.cpp.
 * Pure C for maximum portability (no C++ ABI issues in LD_PRELOAD). */

#include "hrr_trace_writer.h"
#include "hrr_code_object.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#define HRR_MKDIR(p) _mkdir(p)
#define HRR_SEP "\\"
#define HRR_MUTEX CRITICAL_SECTION
#define HRR_MUTEX_INIT(m) InitializeCriticalSection(m)
#define HRR_MUTEX_LOCK(m) EnterCriticalSection(m)
#define HRR_MUTEX_UNLOCK(m) LeaveCriticalSection(m)
static uint64_t hrr_now_ns(void) {
  LARGE_INTEGER freq, count;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&count);
  return (uint64_t)(count.QuadPart * 1000000000LL / freq.QuadPart);
}
#else
#include <sys/stat.h>
#include <pthread.h>
#define HRR_MKDIR(p) mkdir(p, 0755)
#define HRR_SEP "/"
#define HRR_MUTEX pthread_mutex_t
#define HRR_MUTEX_INIT(m) pthread_mutex_init(m, NULL)
#define HRR_MUTEX_LOCK(m) pthread_mutex_lock(m)
#define HRR_MUTEX_UNLOCK(m) pthread_mutex_unlock(m)
static uint64_t hrr_now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
#endif

/* Event types */
#define EVT_MALLOC        0x0001
#define EVT_FREE          0x0002
#define EVT_MEMCPY        0x0003
#define EVT_MEMSET        0x0004
#define EVT_MODULE_LOAD   0x0010
#define EVT_MODULE_UNLOAD 0x0011
#define EVT_KERNEL_LAUNCH 0x0020
#define EVT_DEVICE_SYNC   0x0050
#define EVT_STREAM_SYNC   0x0032

#define HRR_MAGIC   0x52524845
#define HRR_VERSION 1

#pragma pack(push, 1)
typedef struct {
  uint32_t magic;
  uint16_t version;
  uint16_t event_type;
  uint64_t sequence_id;
  uint64_t timestamp_ns;
  uint32_t stream_id;
  uint16_t device_id;
  uint16_t payload_length;
} event_header_t;
#pragma pack(pop)

/* Hash type (FNV-1a placeholder, same as in-tree) */
typedef struct { uint64_t lo, hi; } hash128_t;

static hash128_t hash_buffer(const void* data, size_t len) {
  uint64_t h1 = 0xcbf29ce484222325ULL, h2 = 0x100000001b3ULL;
  const uint8_t* p = (const uint8_t*)data;
  for (size_t i = 0; i < len; i++) {
    h1 ^= p[i]; h1 *= 0x100000001b3ULL;
    h2 ^= p[i]; h2 *= 0xcbf29ce484222325ULL;
  }
  return (hash128_t){h1, h2};
}

/* ---- Global state ---- */

#define HRR_STRINGIFY_INNER(x) #x
#define HRR_STRINGIFY(x) HRR_STRINGIFY_INNER(x)

/* Emit the "no metadata" warning.  Called when a kernel launch reaches the
 * recorder but the code-object parse didn't produce arg metadata (either the
 * module was never registered, or MAX_MODULES was hit and the entry was
 * silently dropped). */
#define HRR_WARN_NO_METADATA(kname) do { \
  int _total = 0; \
  for (int _mi = 0; _mi < g.num_modules; _mi++) _total += g.modules[_mi].num_kernels; \
  fprintf(stderr, "[HRR] WARNING: no metadata for kernel '%s'\n" \
          "[HRR]   0 args recorded (replay will fail). " \
          "%d / %d modules registered, %d kernels parsed total.\n" \
          "[HRR]   %s\n", \
          (kname), g.num_modules, MAX_MODULES, _total, \
          (g.num_modules >= MAX_MODULES) \
            ? "Module table FULL (MAX_MODULES=" HRR_STRINGIFY(MAX_MODULES) \
              "). Bump MAX_MODULES in hrr_trace_writer.c and rebuild." \
            : "Set HRR_VERBOSE=1 and re-run to see per-module details."); \
  if (g.verbose) { \
    for (int _mi = 0; _mi < g.num_modules; _mi++) { \
      fprintf(stderr, "[HRR]   module[%d]: handle=0x%llx %d kernels", \
              _mi, (unsigned long long)g.modules[_mi].handle, \
              g.modules[_mi].num_kernels); \
      if (g.modules[_mi].num_kernels > 0) \
        fprintf(stderr, " (first: '%.80s'%s)", \
                g.modules[_mi].kernels[0].name, \
                strlen(g.modules[_mi].kernels[0].name) > 80 ? "..." : ""); \
      fprintf(stderr, "\n"); \
    } \
  } \
} while (0)

#define MAX_ALLOCS 65536
#define MAX_MODULES 4096
#define MAX_CO_KERNELS_INITIAL 1024
/* Large enough to hold fat-binary registrations from all loaded HIP libraries
 * (rocBLAS alone contributes ~40 K entries via __hipRegisterFunction) plus
 * hipModuleGetFunction entries for runtime-loaded kernels (hipBLASLt). */
#define MAX_FUNC_ENTRIES 65536
/* Kernel names are stored as heap pointers (arbitrary length) so Tensile /
 * hipBLASLt names of 300-500 characters are not truncated. */

#define MAX_KERNEL_SUMMARIES 4096
#define MAX_SUMMARY_ARGS     64
#define MAX_SUMMARY_ARG_DATA 16

typedef struct {
  uint8_t  kind;   /* 0=scalar, 1=pointer, 2=hidden */
  uint16_t size;
  uint8_t  data[MAX_SUMMARY_ARG_DATA];
} kernel_summary_arg_t;

typedef struct {
  uint64_t event_index;
  char     name[512];
  uint32_t grid[3], block[3], shared;
  int      num_args;
  kernel_summary_arg_t args[MAX_SUMMARY_ARGS];
} kernel_summary_t;

typedef struct {
  uintptr_t ptr;
  uint64_t handle;
  size_t size;
} alloc_entry_t;

typedef struct {
  uintptr_t module;
  uint64_t handle;
  const void* image;
  size_t image_size;
  uint64_t image_hash_lo;  /* stored at load time so we don't need live image ptr later */
  uint64_t image_hash_hi;
  /* Parsed kernel metadata for this module (dynamically allocated) */
  hrr_kernel_meta_t* kernels;
  int num_kernels;
} module_entry_t;

typedef struct {
  uintptr_t handle;
  uint64_t module_handle;  /* handle of the owning hipModule_t (from g.next_mod_handle) */
  char* name;              /* heap-allocated; full length so Tensile names are not truncated */
} func_entry_t;

static struct {
  int active;
  int mode;  /* 0=timeline, 1=inputs, 2=full */
  char output_dir[512];
  char kernel_filter[256];
  size_t max_blob_mb;
  size_t max_snap_mb;  /* per-output-snapshot cap; 0 = inherit max_blob_mb */

  FILE* events_file;
  HRR_MUTEX mu;
  uint64_t seq_id;
  uint64_t next_handle;
  uint64_t next_mod_handle;

  alloc_entry_t allocs[MAX_ALLOCS];
  int num_allocs;

  module_entry_t modules[MAX_MODULES];
  int num_modules;

  /* Function handle → kernel name mapping (always active, not gated by HRR_RECORD) */
  func_entry_t funcs[MAX_FUNC_ENTRIES];
  int num_funcs;

  size_t blob_count;
  int verbose;  /* HRR_VERBOSE=1 — print diagnostic messages to stderr */

  int user_output;      /* 1 if HRR_OUTPUT was explicitly set */
  int shape_captured;
  char shape[128];      /* e.g. "16x16x16" — extracted from first kernel's scalar args */

  /* Device callbacks for full-mode output snapshot capture */
  hrr_device_sync_fn device_sync;
  hrr_memcpy_fn      memcpy_fn;
  hrr_memset_fn      memset_fn;

  kernel_summary_t* kernel_summaries;
  int num_kernel_summaries;
} g;

static void hash_hex(hash128_t h, char buf[33]) {
  snprintf(buf, 33, "%016llx%016llx",
           (unsigned long long)h.lo, (unsigned long long)h.hi);
}

/* Extract dimension-like scalar args from the first kernel launch to build a
 * shape string like "16x16x16".  Heuristic: collect 4-byte scalar values in
 * [2, 1M] that appear BEFORE the first pointer arg (pointer args separate
 * dimension scalars from stride/control scalars in Tensile GEMM layout). */
static void capture_shape(const hrr_kernel_meta_t* meta,
                          void** kernel_args, const void* packed_buf,
                          size_t packed_size) {
  if (!meta || g.shape_captured) return;
  g.shape_captured = 1;

  uint32_t dims[8];
  int ndims = 0;

  for (uint32_t i = 0; i < meta->num_args && ndims < 8; i++) {
    const hrr_arg_desc_t* ad = &meta->args[i];
    if (ad->kind == HRR_ARG_GLOBAL_BUFFER) break;  /* stop at first pointer */
    if (ad->kind == HRR_ARG_HIDDEN) continue;
    if (ad->size != 4) continue;

    const void* src = NULL;
    if (kernel_args && kernel_args[i]) {
      src = kernel_args[i];
    } else if (packed_buf && (size_t)(ad->offset + 4) <= packed_size) {
      src = (const char*)packed_buf + ad->offset;
    }
    if (!src) continue;

    uint32_t val;
    memcpy(&val, src, 4);
    if (val >= 2 && val <= 1048576) {
      dims[ndims++] = val;
    }
  }

  if (ndims > 0) {
    char* p = g.shape;
    for (int i = 0; i < ndims; i++) {
      if (i > 0) *p++ = 'x';
      p += snprintf(p, (size_t)(g.shape + sizeof(g.shape) - p), "%u", dims[i]);
    }
  }
}

static void write_blob(const void* data, size_t len, hash128_t* out_hash) {
  *out_hash = hash_buffer(data, len);
  char hex[33];
  hash_hex(*out_hash, hex);

  char subdir[600], path[640];
  snprintf(subdir, sizeof(subdir), "%s" HRR_SEP "blobs" HRR_SEP "%.2s",
           g.output_dir, hex);
  HRR_MKDIR(subdir);

  snprintf(path, sizeof(path), "%s" HRR_SEP "%s.blob", subdir, hex);
  FILE* f = fopen(path, "rb");
  if (f) { fclose(f); return; }  /* already exists */

  f = fopen(path, "wb");
  if (f) { fwrite(data, 1, len, f); fclose(f); g.blob_count++; }
}

static void write_event(uint16_t type, uint32_t stream_id,
                        const void* payload, uint16_t payload_len) {
  event_header_t hdr;
  hdr.magic = HRR_MAGIC;
  hdr.version = HRR_VERSION;
  hdr.event_type = type;
  hdr.sequence_id = g.seq_id++;
  hdr.timestamp_ns = hrr_now_ns();
  hdr.stream_id = stream_id;
  hdr.device_id = 0;
  hdr.payload_length = payload_len;

  HRR_MUTEX_LOCK(&g.mu);
  if (g.events_file) {
    fwrite(&hdr, sizeof(hdr), 1, g.events_file);
    if (payload && payload_len > 0)
      fwrite(payload, 1, payload_len, g.events_file);
  }
  HRR_MUTEX_UNLOCK(&g.mu);
}

static alloc_entry_t* find_alloc(uintptr_t ptr) {
  for (int i = 0; i < g.num_allocs; i++) {
    if (g.allocs[i].ptr == ptr) return &g.allocs[i];
  }
  return NULL;
}

/* Range-aware lookup for kernel arg recording.
 * Returns the actual GPU pointer value as the handle (matching the MALLOC
 * event format).  The replay's translate_ptr does a range search keyed by
 * GPU base address, so emitting the sub-alloc GPU address directly lets it
 * compute the correct byte offset into the base allocation.
 *
 * Returns 1 and sets *out_handle on success, 0 if the pointer is not tracked. */
static int find_alloc_handle(uintptr_t ptr, uint64_t* out_handle) {
  /* Exact match */
  for (int i = 0; i < g.num_allocs; i++) {
    if (g.allocs[i].ptr == ptr) {
      *out_handle = (uint64_t)ptr;
      return 1;
    }
  }
  /* Range match: ptr points into the interior of a known allocation */
  for (int i = 0; i < g.num_allocs; i++) {
    uintptr_t base = g.allocs[i].ptr;
    size_t    sz   = g.allocs[i].size;
    if (ptr > base && ptr < base + sz) {
      /* Emit the actual sub-alloc GPU address.  The replay's translate_ptr
       * range-searches by GPU base address, so it computes:
       *   live_ptr + (sub_alloc_addr - base_addr)  which is the correct offset. */
      *out_handle = (uint64_t)ptr;
      return 1;
    }
  }
  return 0;
}

static module_entry_t* find_module(uintptr_t mod) {
  for (int i = 0; i < g.num_modules; i++) {
    if (g.modules[i].module == mod) return &g.modules[i];
  }
  return NULL;
}

/* ---- Public API ---- */

/* Tracks whether g.mu has been initialized. Lives outside g so it survives
 * the memset(&g, 0, ...) in hrr_writer_init(). */
static int g_mu_initialized = 0;

void hrr_early_init(void) {
  if (!g_mu_initialized) {
    HRR_MUTEX_INIT(&g.mu);
    g_mu_initialized = 1;
  }
}

int hrr_writer_init(void) {
  /* Preserve function entries registered before this call via
   * __hipRegisterFunction (fat-binary DLL global constructors fire before
   * the first real HIP API call that triggers hrr_writer_init).
   * func_entry_t.name is a heap pointer so memcpy copies the pointer value;
   * the underlying strings survive the memset below and remain valid after
   * the restore. */
  int saved_num_funcs = g.num_funcs;
  func_entry_t* saved_funcs = NULL;
  if (saved_num_funcs > 0) {
    saved_funcs = (func_entry_t*)malloc((size_t)saved_num_funcs * sizeof(func_entry_t));
    if (saved_funcs)
      memcpy(saved_funcs, g.funcs, (size_t)saved_num_funcs * sizeof(func_entry_t));
    else
      saved_num_funcs = 0;  /* allocation failed — lose early registrations rather than crash */
  }

  memset(&g, 0, sizeof(g));
  g.next_handle = 1;
  g.next_mod_handle = 1;

  /* (Re-)initialize the mutex — safe to call on a zeroed CRITICAL_SECTION. */
  HRR_MUTEX_INIT(&g.mu);
  g_mu_initialized = 1;

  /* Restore pre-registered kernel names. */
  if (saved_num_funcs > 0 && saved_funcs) {
    memcpy(g.funcs, saved_funcs, (size_t)saved_num_funcs * sizeof(func_entry_t));
    g.num_funcs = saved_num_funcs;
    free(saved_funcs);
  }

  const char* env = getenv("HRR_RECORD");
  if (!env || strcmp(env, "1") != 0) return 0;

  const char* out = getenv("HRR_OUTPUT");
  if (out && out[0]) {
    strncpy(g.output_dir, out, sizeof(g.output_dir) - 1);
    g.user_output = 1;
  } else {
    /* Temporary name; will be renamed at shutdown if shape is captured */
    snprintf(g.output_dir, sizeof(g.output_dir), "capture.hrr");
    g.user_output = 0;
  }

  const char* mode = getenv("HRR_MODE");
  if (mode) {
    if (strcmp(mode, "timeline") == 0) g.mode = 0;
    else if (strcmp(mode, "full") == 0) g.mode = 2;
    else g.mode = 1;
  } else {
    g.mode = 1;
  }

  const char* filter = getenv("HRR_KERNEL_FILTER");
  if (filter) strncpy(g.kernel_filter, filter, sizeof(g.kernel_filter) - 1);

  const char* max_blob = getenv("HRR_MAX_BLOB_MB");
  if (max_blob) g.max_blob_mb = (size_t)atol(max_blob);

  /* Per-output-snapshot cap.  Without this, snapshot capture treats every
   * pointer arg as "owns the rest of its containing allocation", which for
   * arena allocators (MIGraphX, rocBLAS workspaces, ONNX runtime, ...)
   * sweeps in many unrelated tensors and produces noisy false-positive
   * verification mismatches.  Default to 16 MB which is large enough for
   * most individual tensor outputs but tight enough to avoid arena tails. */
  g.max_snap_mb = 16;
  const char* max_snap = getenv("HRR_MAX_SNAP_MB");
  if (max_snap) g.max_snap_mb = (size_t)atol(max_snap);

  const char* verbose = getenv("HRR_VERBOSE");
  if (verbose && verbose[0] == '1') g.verbose = 1;

  HRR_MKDIR(g.output_dir);
  char tmp[600];
  snprintf(tmp, sizeof(tmp), "%s" HRR_SEP "blobs", g.output_dir);
  HRR_MKDIR(tmp);
  snprintf(tmp, sizeof(tmp), "%s" HRR_SEP "code_objects", g.output_dir);
  HRR_MKDIR(tmp);

  snprintf(tmp, sizeof(tmp), "%s" HRR_SEP "events.bin", g.output_dir);
  g.events_file = fopen(tmp, "wb");
  if (!g.events_file) {
    fprintf(stderr, "[HRR] Cannot open %s\n", tmp);
    return 0;
  }

  g.active = 1;
  g.kernel_summaries = (kernel_summary_t*)calloc(MAX_KERNEL_SUMMARIES,
                                                  sizeof(kernel_summary_t));
  fprintf(stderr, "[HRR] Recording to %s (mode=%s)\n",
          g.output_dir, mode ? mode : "inputs");
  return 1;
}

static void write_metadata_json(void) {
  char path[600];
  snprintf(path, sizeof(path), "%s" HRR_SEP "metadata.json", g.output_dir);
  FILE* f = fopen(path, "w");
  if (!f) return;

  const char* modes[] = {"timeline", "inputs", "full"};
  fprintf(f, "{\n  \"version\": 1,\n  \"capture_mode\": \"%s\",\n",
          modes[g.mode]);

  if (g.shape[0])
    fprintf(f, "  \"shape\": \"%s\",\n", g.shape);

  /* Allocations still live at shutdown */
  fprintf(f, "  \"allocations\": [\n");
  for (int i = 0; i < g.num_allocs; i++) {
    fprintf(f, "    { \"handle_hex\": \"%llx\", \"size\": %llu }%s\n",
            (unsigned long long)g.allocs[i].handle,
            (unsigned long long)g.allocs[i].size,
            (i < g.num_allocs - 1) ? "," : "");
  }
  fprintf(f, "  ],\n");

  /* Kernel summaries */
  fprintf(f, "  \"kernels\": [\n");
  for (int ki = 0; ki < g.num_kernel_summaries; ki++) {
    const kernel_summary_t* ks = &g.kernel_summaries[ki];
    fprintf(f, "    {\n");
    fprintf(f, "      \"event_index\": %llu,\n",
            (unsigned long long)ks->event_index);
    fprintf(f, "      \"name\": \"%s\",\n", ks->name);
    fprintf(f, "      \"grid\": [%u, %u, %u],\n",
            ks->grid[0], ks->grid[1], ks->grid[2]);
    fprintf(f, "      \"block\": [%u, %u, %u],\n",
            ks->block[0], ks->block[1], ks->block[2]);
    fprintf(f, "      \"shared_bytes\": %u,\n", ks->shared);

    fprintf(f, "      \"args\": [\n");
    for (int ai = 0; ai < ks->num_args; ai++) {
      const kernel_summary_arg_t* sa = &ks->args[ai];
      fprintf(f, "        { \"idx\": %d, \"kind\": %d, \"size\": %u",
              ai, sa->kind, sa->size);
      if (sa->kind == 1) {
        uint64_t handle = 0;
        if (sa->size >= 8) memcpy(&handle, sa->data, 8);
        fprintf(f, ", \"handle_hex\": \"%llx\"",
                (unsigned long long)handle);
      } else if (sa->kind == 0) {
        uint16_t nbytes = sa->size < MAX_SUMMARY_ARG_DATA
                          ? sa->size : MAX_SUMMARY_ARG_DATA;
        fprintf(f, ", \"hex\": \"");
        for (int bi = 0; bi < nbytes; bi++)
          fprintf(f, "%02x", sa->data[bi]);
        fprintf(f, "\"");
      }
      fprintf(f, " }%s\n", (ai < ks->num_args - 1) ? "," : "");
    }
    fprintf(f, "      ]\n");
    fprintf(f, "    }%s\n", (ki < g.num_kernel_summaries - 1) ? "," : "");
  }
  fprintf(f, "  ]\n}\n");

  fclose(f);
}

/* Capture a kernel launch summary for metadata.json.
 * Called after the event is written; event_index = g.seq_id - 1. */
static void record_kernel_summary(const char* kernel_name,
                                   uint32_t gx, uint32_t gy, uint32_t gz,
                                   uint32_t bx, uint32_t by, uint32_t bz,
                                   uint32_t shared_mem,
                                   const hrr_kernel_meta_t* meta,
                                   void** kernel_args,
                                   const void* packed_buf,
                                   size_t packed_size) {
  if (!g.kernel_summaries || g.num_kernel_summaries >= MAX_KERNEL_SUMMARIES)
    return;

  kernel_summary_t* ks = &g.kernel_summaries[g.num_kernel_summaries];
  memset(ks, 0, sizeof(*ks));

  ks->event_index = g.seq_id - 1;
  strncpy(ks->name, kernel_name ? kernel_name : "<unknown>",
          sizeof(ks->name) - 1);
  ks->grid[0] = gx; ks->grid[1] = gy; ks->grid[2] = gz;
  ks->block[0] = bx; ks->block[1] = by; ks->block[2] = bz;
  ks->shared = shared_mem;

  if (meta) {
    int n = (int)meta->num_args < MAX_SUMMARY_ARGS
            ? (int)meta->num_args : MAX_SUMMARY_ARGS;
    ks->num_args = n;
    for (int i = 0; i < n; i++) {
      const hrr_arg_desc_t* ad = &meta->args[i];
      kernel_summary_arg_t* sa = &ks->args[i];
      sa->kind = (uint8_t)ad->kind;

      if (ad->kind == HRR_ARG_GLOBAL_BUFFER) {
        sa->size = 8;
        const void* src = NULL;
        if (kernel_args && kernel_args[i])
          src = kernel_args[i];
        else if (packed_buf && ad->offset + 8 <= packed_size)
          src = (const char*)packed_buf + ad->offset;
        if (src) memcpy(sa->data, src, 8);
      } else {
        sa->size = ad->size;
        uint16_t copy_len = ad->size < MAX_SUMMARY_ARG_DATA
                            ? ad->size : MAX_SUMMARY_ARG_DATA;
        if (ad->kind != HRR_ARG_HIDDEN) {
          const void* src = NULL;
          if (kernel_args && kernel_args[i])
            src = kernel_args[i];
          else if (packed_buf && ad->offset + copy_len <= packed_size)
            src = (const char*)packed_buf + ad->offset;
          if (src) memcpy(sa->data, src, copy_len);
        }
      }
    }
  }

  g.num_kernel_summaries++;
}

void hrr_writer_shutdown(void) {
  if (!g.active) return;
  if (g.events_file) { fflush(g.events_file); fclose(g.events_file); }
  g.events_file = NULL;

  /* Rename directory to embed shape if HRR_OUTPUT was not set */
  if (!g.user_output && g.shape[0]) {
    char new_dir[512];
    snprintf(new_dir, sizeof(new_dir), "capture_%s.hrr", g.shape);
    if (rename(g.output_dir, new_dir) == 0) {
      strncpy(g.output_dir, new_dir, sizeof(g.output_dir) - 1);
      g.output_dir[sizeof(g.output_dir) - 1] = '\0';
    }
    /* rename fails if destination exists — keep original name */
  }

  /* Write manifest */
  char path[600];
  snprintf(path, sizeof(path), "%s" HRR_SEP "manifest.json", g.output_dir);
  FILE* f = fopen(path, "w");
  if (f) {
    const char* modes[] = {"timeline", "inputs", "full"};
    fprintf(f, "{\n  \"version\": 1,\n  \"format\": \"hrr-v1\",\n"
               "  \"capture_mode\": \"%s\",\n  \"event_count\": %llu,\n"
               "  \"blob_count\": %zu\n}\n",
            modes[g.mode], (unsigned long long)g.seq_id, g.blob_count);
    fclose(f);
  }
  write_metadata_json();

  fprintf(stderr, "[HRR] Recording complete: %llu events, %zu blobs, "
          "%d kernel summaries\n",
          (unsigned long long)g.seq_id, g.blob_count, g.num_kernel_summaries);
  free(g.kernel_summaries);
  g.kernel_summaries = NULL;
  g.active = 0;
}

int hrr_writer_enabled(void) { return g.active; }

static void hrr_record_alloc_internal(const void* ptr, size_t size,
                                      unsigned int flags, int zero_init) {
  if (!g.active) return;
  /* Use the actual GPU pointer value as the allocation handle.
   * This allows the replay's range-based translate_ptr to find sub-allocations
   * within a pool without needing a separate handle namespace. */
  uint64_t handle = (uint64_t)(uintptr_t)ptr;
  HRR_MUTEX_LOCK(&g.mu);
  if (g.num_allocs < MAX_ALLOCS) {
    g.allocs[g.num_allocs++] = (alloc_entry_t){(uintptr_t)ptr, handle, size};
  }
  HRR_MUTEX_UNLOCK(&g.mu);

  /* In full mode the writer captures output snapshots that may include the
   * tail of an arena allocation (we don't know how much of the buffer the
   * kernel actually wrote).  hipMalloc returns uninitialized device memory,
   * so any byte the kernel doesn't touch contains garbage values that vary
   * run-to-run.  The replayer zero-inits on hipMalloc; if we don't match
   * that here, --verify will compare arena garbage on the recorder against
   * zeros on the replayer and report wholesale false-positive mismatches.
   * Skip the zero-init in non-full modes to avoid changing observable
   * behaviour for capture sessions that don't snapshot outputs.
   * Skip also for host-mapped allocations — hipMemset on a host pointer
   * would clobber data the host has either already filled or is about to
   * fill, and the readback path captures inputs explicitly. */
  if (zero_init && g.mode == 2 && ptr && size > 0 && g.memset_fn) {
    (void)g.memset_fn((void*)(uintptr_t)ptr, 0, size);
  }

#pragma pack(push,1)
  struct { uint64_t h; uint64_t s; uint32_t f; } pl = {handle, size, flags};
#pragma pack(pop)
  write_event(EVT_MALLOC, 0, &pl, sizeof(pl));
}

void hrr_record_malloc(const void* ptr, size_t size, unsigned int flags) {
  hrr_record_alloc_internal(ptr, size, flags, /*zero_init=*/1);
}

void hrr_record_host_alloc(const void* ptr, size_t size, unsigned int flags) {
  hrr_record_alloc_internal(ptr, size, flags, /*zero_init=*/0);
}

void hrr_record_external_launch_state(void) {
  if (!g.active || !g.device_sync || !g.memcpy_fn) return;

  /* Ensure the kernel has finished before we read GPU state. */
  g.device_sync();

  /* Snapshot every tracked allocation as an H2D blob.  We can't tell which
   * one(s) the untracked kernel actually wrote without parsing its args, so
   * we cover all of them; the resulting MEMCPY events on replay will simply
   * overwrite each live allocation with its post-launch bytes.  Bounded by
   * HRR_MAX_BLOB_MB. */
  int n;
  HRR_MUTEX_LOCK(&g.mu);
  n = g.num_allocs;
  HRR_MUTEX_UNLOCK(&g.mu);

  for (int i = 0; i < n; i++) {
    uintptr_t base;
    size_t    sz;
    HRR_MUTEX_LOCK(&g.mu);
    if (i >= g.num_allocs) { HRR_MUTEX_UNLOCK(&g.mu); break; }
    base = g.allocs[i].ptr;
    sz   = g.allocs[i].size;
    HRR_MUTEX_UNLOCK(&g.mu);
    if (sz == 0) continue;
    if (g.max_blob_mb > 0 && sz > g.max_blob_mb * 1024 * 1024) continue;

    void* cpu_buf = malloc(sz);
    if (!cpu_buf) continue;

    int err = g.memcpy_fn(cpu_buf, (const void*)base, sz, 2 /* D2H */);
    if (err != 0) { free(cpu_buf); continue; }

    /* Emit an H2D MEMCPY whose dst is the allocation address — the replay
     * translates dst via translate_ptr(dst_addr), so it lands on the live
     * replica of this allocation. */
    hrr_record_memcpy((void*)base, cpu_buf, sz,
                      1 /* hipMemcpyHostToDevice */, NULL);
    free(cpu_buf);
  }
}

void hrr_record_free(const void* ptr) {
  if (!g.active) return;
  HRR_MUTEX_LOCK(&g.mu);
  alloc_entry_t* e = find_alloc((uintptr_t)ptr);
  if (e) *e = g.allocs[--g.num_allocs];
  HRR_MUTEX_UNLOCK(&g.mu);
  /* Emit the GPU address as the handle — matches the MALLOC format. */
  uint64_t handle = (uint64_t)(uintptr_t)ptr;
  write_event(EVT_FREE, 0, &handle, sizeof(handle));
}

void hrr_record_memcpy(void* dst, const void* src, size_t size,
                       unsigned int kind, const void* stream) {
  if (!g.active) return;
  hash128_t blob_hash = {0, 0};
  if (g.mode > 0 && kind == 1 && src && size > 0) {
    if (g.max_blob_mb == 0 || size <= g.max_blob_mb * 1024 * 1024)
      write_blob(src, size, &blob_hash);
  }
#pragma pack(push,1)
  struct { uint64_t d,s,sz; uint32_t k; uint64_t hl,hh; } pl = {
    (uint64_t)(uintptr_t)dst, (uint64_t)(uintptr_t)src, size, kind,
    blob_hash.lo, blob_hash.hi};
#pragma pack(pop)
  write_event(EVT_MEMCPY, (uint32_t)(uintptr_t)stream, &pl, sizeof(pl));
}

void hrr_record_memset(void* dst, int value, size_t size, const void* stream) {
  if (!g.active) return;
#pragma pack(push,1)
  struct { uint64_t d; uint32_t v; uint64_t s; } pl = {
    (uint64_t)(uintptr_t)dst, (uint32_t)value, size};
#pragma pack(pop)
  write_event(EVT_MEMSET, (uint32_t)(uintptr_t)stream, &pl, sizeof(pl));
}

void hrr_record_module_load(void* module, const void* image, size_t image_size) {
  if (!g.active) return;
  uint64_t mod_handle;
  hash128_t h = {0, 0};
  if (image && image_size > 0)
    h = hash_buffer(image, image_size);

  HRR_MUTEX_LOCK(&g.mu);
  mod_handle = g.next_mod_handle++;
  if (g.num_modules < MAX_MODULES) {
    module_entry_t* me = &g.modules[g.num_modules++];
    me->module = (uintptr_t)module;
    me->handle = mod_handle;
    me->image = image;
    me->image_size = image_size;
    me->image_hash_lo = h.lo;
    me->image_hash_hi = h.hi;
    me->kernels = (hrr_kernel_meta_t*)calloc(MAX_CO_KERNELS_INITIAL,
                                              sizeof(hrr_kernel_meta_t));
    me->num_kernels = me->kernels
        ? hrr_parse_code_object(image, image_size,
                                me->kernels, MAX_CO_KERNELS_INITIAL)
        : 0;
    if (g.verbose) {
      fprintf(stderr, "[HRR] module_load: mod=%p handle=%llu image_size=%zu "
              "parsed %d kernels\n",
              module, (unsigned long long)mod_handle, image_size,
              me->num_kernels);
      for (int ki = 0; ki < me->num_kernels && ki < 3; ki++)
        fprintf(stderr, "[HRR]   kernel[%d]: '%s' (%u args)\n",
                ki, me->kernels[ki].name, me->kernels[ki].num_args);
    }
  }
  HRR_MUTEX_UNLOCK(&g.mu);

  /* Save code object */
  if (image && image_size > 0) {
    char hex[33]; hash_hex(h, hex);
    char path[640];
    snprintf(path, sizeof(path), "%s" HRR_SEP "code_objects" HRR_SEP "%s.hsaco",
             g.output_dir, hex);
    FILE* check = fopen(path, "rb");
    if (!check) {
      FILE* f = fopen(path, "wb");
      if (f) { fwrite(image, 1, image_size, f); fclose(f); }
    } else { fclose(check); }

#pragma pack(push,1)
    struct { uint64_t hl,hh,mh; } pl = {h.lo, h.hi, mod_handle};
#pragma pack(pop)
    write_event(EVT_MODULE_LOAD, 0, &pl, sizeof(pl));
  }
}

void hrr_record_module_unload(void* module) {
  if (!g.active) return;
  uint64_t handle = 0;
  HRR_MUTEX_LOCK(&g.mu);
  module_entry_t* me = find_module((uintptr_t)module);
  if (me) {
    handle = me->handle;
    free(me->kernels);
    *me = g.modules[--g.num_modules];
  }
  HRR_MUTEX_UNLOCK(&g.mu);
  write_event(EVT_MODULE_UNLOAD, 0, &handle, sizeof(handle));
}

/* ---- Full-mode output snapshot capture ----
 *
 * In full mode (g.mode == 2), after a kernel launch is recorded, the writer
 * synchronises the GPU and reads back every pointer-arg buffer.  Each readback
 * is saved as a blob and a snapshot record (direction=1, "output") is appended
 * to the kernel-launch event payload.  The replay's --verify flag then
 * compares these blobs against the live GPU state after re-executing the
 * kernel.
 *
 * The caller (proxy / interposer) MUST call the real kernel launch BEFORE
 * calling hrr_record_kernel_launch so that the GPU work is already queued. */

#define MAX_SNAP_PTRS 64

typedef struct {
  uint64_t ptr_handle;
  uint64_t offset;
  uint64_t length;
  uint64_t hash_lo;
  uint64_t hash_hi;
  uint8_t  direction;  /* 1 = output */
} snap_record_t;

/* Synchronise the GPU, read back each tracked pointer, save blobs, and
 * fill snap_out[].  Returns the number of valid snapshot records. */
static int capture_output_snapshots(const uint64_t* ptr_handles, int num_ptrs,
                                    snap_record_t* snap_out) {
  if (!g.device_sync || !g.memcpy_fn || num_ptrs == 0) return 0;

  g.device_sync();

  int count = 0;
  for (int i = 0; i < num_ptrs && count < MAX_SNAP_PTRS; i++) {
    uint64_t handle = ptr_handles[i];
    if (handle == 0) continue;

    uintptr_t ptr = (uintptr_t)handle;
    size_t snap_size = 0;

    HRR_MUTEX_LOCK(&g.mu);
    for (int j = 0; j < g.num_allocs; j++) {
      uintptr_t base = g.allocs[j].ptr;
      size_t    sz   = g.allocs[j].size;
      if (ptr >= base && ptr < base + sz) {
        snap_size = sz - (size_t)(ptr - base);
        break;
      }
    }
    HRR_MUTEX_UNLOCK(&g.mu);

    if (snap_size == 0) continue;
    /* Apply the snapshot-specific cap first (default 16 MB, tunable via
     * HRR_MAX_SNAP_MB), then the global blob cap if set.  Snapshots get
     * a tighter default than blobs because arena allocators make the
     * "remainder of allocation" heuristic over-capture aggressively. */
    if (g.max_snap_mb > 0 && snap_size > g.max_snap_mb * 1024 * 1024)
      snap_size = g.max_snap_mb * 1024 * 1024;
    if (g.max_blob_mb > 0 && snap_size > g.max_blob_mb * 1024 * 1024)
      snap_size = g.max_blob_mb * 1024 * 1024;

    void* cpu_buf = malloc(snap_size);
    if (!cpu_buf) continue;

    int err = g.memcpy_fn(cpu_buf, (const void*)ptr, snap_size, 2);
    if (err != 0) { free(cpu_buf); continue; }

    hash128_t blob_hash;
    write_blob(cpu_buf, snap_size, &blob_hash);
    free(cpu_buf);

    snap_out[count].ptr_handle = handle;
    snap_out[count].offset     = 0;
    snap_out[count].length     = (uint64_t)snap_size;
    snap_out[count].hash_lo    = blob_hash.lo;
    snap_out[count].hash_hi    = blob_hash.hi;
    snap_out[count].direction  = 1;
    count++;
  }

  if (g.verbose && count > 0) {
    size_t total = 0;
    for (int i = 0; i < count; i++) total += (size_t)snap_out[i].length;
    fprintf(stderr, "[HRR] Captured %d output snapshots (%zu bytes total)\n",
            count, total);
  }

  return count;
}

void hrr_record_kernel_launch(const char* kernel_name,
                              uint64_t co_hash_lo, uint64_t co_hash_hi,
                              uint32_t gx, uint32_t gy, uint32_t gz,
                              uint32_t bx, uint32_t by, uint32_t bz,
                              uint32_t shared_mem,
                              const void* stream,
                              void** kernel_args) {
  if (!g.active) return;
  if (g.kernel_filter[0] != '\0' && kernel_name) {
    size_t flen = strlen(g.kernel_filter);
    if (g.kernel_filter[flen-1] == '*') {
      if (strncmp(kernel_name, g.kernel_filter, flen-1) != 0) return;
    } else {
      if (strcmp(kernel_name, g.kernel_filter) != 0) return;
    }
  }

  /* Find kernel metadata for arg introspection */
  const hrr_kernel_meta_t* meta = NULL;
  if (kernel_name) {
    HRR_MUTEX_LOCK(&g.mu);
    for (int i = 0; i < g.num_modules && !meta; i++) {
      meta = hrr_find_kernel(g.modules[i].kernels,
                             g.modules[i].num_kernels, kernel_name);
    }
    HRR_MUTEX_UNLOCK(&g.mu);
  }

  if (meta && !g.shape_captured) {
    capture_shape(meta, kernel_args, NULL, 0);
  }

  /* Build payload */
  const char* name = kernel_name ? kernel_name : "<unknown>";
  uint16_t name_len = (uint16_t)strlen(name);
  uint16_t num_args = meta ? (uint16_t)meta->num_args : 0;
  uint16_t num_snaps = 0;

  if (!meta && kernel_name) HRR_WARN_NO_METADATA(kernel_name);

  /* Collect unique pointer-arg handles for output snapshot capture */
  uint64_t ptr_handles[MAX_SNAP_PTRS];
  int num_ptr_handles = 0;

  /* co_hash (16 bytes) added to payload after name */
  size_t pl_size = 2 + name_len + 16 + 12 + 12 + 4 + 2 + 2;
  /* Add args */
  for (uint32_t i = 0; i < num_args; i++) {
    pl_size += 1 + 2 + (meta->args[i].kind == HRR_ARG_GLOBAL_BUFFER ? 8 :
                         meta->args[i].size);
  }

  uint8_t* pl = (uint8_t*)malloc(pl_size);
  if (!pl) return;
  uint8_t* p = pl;

  memcpy(p, &name_len, 2); p += 2;
  memcpy(p, name, name_len); p += name_len;
  memcpy(p, &co_hash_lo, 8); p += 8;
  memcpy(p, &co_hash_hi, 8); p += 8;
  memcpy(p, &gx, 4); p += 4;
  memcpy(p, &gy, 4); p += 4;
  memcpy(p, &gz, 4); p += 4;
  memcpy(p, &bx, 4); p += 4;
  memcpy(p, &by, 4); p += 4;
  memcpy(p, &bz, 4); p += 4;
  memcpy(p, &shared_mem, 4); p += 4;
  memcpy(p, &num_args, 2); p += 2;
  /* num_snaps placeholder — patched below if snapshots are captured */
  uint8_t* num_snaps_ptr = p;
  memcpy(p, &num_snaps, 2); p += 2;

  for (uint32_t i = 0; i < num_args; i++) {
    const hrr_arg_desc_t* ad = &meta->args[i];
    uint8_t vk = (uint8_t)ad->kind;
    *p++ = vk;

    if (ad->kind == HRR_ARG_GLOBAL_BUFFER) {
      uint16_t sz = 8;
      memcpy(p, &sz, 2); p += 2;
      if (kernel_args && kernel_args[i]) {
        uint64_t dev_ptr = 0;
        memcpy(&dev_ptr, kernel_args[i], 8);
        uint64_t handle = 0;
        HRR_MUTEX_LOCK(&g.mu);
        find_alloc_handle((uintptr_t)dev_ptr, &handle);
        HRR_MUTEX_UNLOCK(&g.mu);
        if (!handle && dev_ptr != 0) {
          if (g.verbose) {
            fprintf(stderr, "[HRR] kernel '%s' arg[%u]: ptr=0x%llx untracked, "
                    "registering synthetic alloc (1MB)\n",
                    name, i, (unsigned long long)dev_ptr);
          }
          hrr_record_malloc((const void*)(uintptr_t)dev_ptr, 1024*1024, 0);
          HRR_MUTEX_LOCK(&g.mu);
          find_alloc_handle((uintptr_t)dev_ptr, &handle);
          HRR_MUTEX_UNLOCK(&g.mu);
        }
        memcpy(p, &handle, 8);

        /* Track unique pointer handles for output snapshot */
        if (handle != 0 && num_ptr_handles < MAX_SNAP_PTRS) {
          int dup = 0;
          for (int k = 0; k < num_ptr_handles; k++) {
            if (ptr_handles[k] == handle) { dup = 1; break; }
          }
          if (!dup) ptr_handles[num_ptr_handles++] = handle;
        }
      } else {
        memset(p, 0, 8);
      }
      p += 8;
    } else {
      uint16_t sz = ad->size;
      memcpy(p, &sz, 2); p += 2;
      if (kernel_args && kernel_args[i] && ad->kind != HRR_ARG_HIDDEN) {
        memcpy(p, kernel_args[i], ad->size);
      } else {
        memset(p, 0, ad->size);
      }
      p += ad->size;
    }
  }

  /* Full mode: capture output snapshots and append to payload */
  if (g.mode == 2 && num_ptr_handles > 0) {
    snap_record_t snaps[MAX_SNAP_PTRS];
    int snap_count = capture_output_snapshots(ptr_handles, num_ptr_handles,
                                              snaps);
    if (snap_count > 0) {
      size_t snap_data_size = (size_t)snap_count * 41;
      pl = (uint8_t*)realloc(pl, pl_size + snap_data_size);
      if (pl) {
        /* Recalculate num_snaps_ptr after realloc */
        num_snaps_ptr = pl + (2 + name_len + 16 + 12 + 12 + 4 + 2);
        num_snaps = (uint16_t)snap_count;
        memcpy(num_snaps_ptr, &num_snaps, 2);

        uint8_t* sp = pl + pl_size;
        for (int i = 0; i < snap_count; i++) {
          memcpy(sp, &snaps[i].ptr_handle, 8); sp += 8;
          memcpy(sp, &snaps[i].offset, 8);     sp += 8;
          memcpy(sp, &snaps[i].length, 8);     sp += 8;
          memcpy(sp, &snaps[i].hash_lo, 8);    sp += 8;
          memcpy(sp, &snaps[i].hash_hi, 8);    sp += 8;
          *sp++ = snaps[i].direction;
        }
        pl_size += snap_data_size;
      }
    }
  }

  uint16_t pl_len = (uint16_t)(pl_size > 65535 ? 65535 : pl_size);
  write_event(EVT_KERNEL_LAUNCH, (uint32_t)(uintptr_t)stream, pl, pl_len);
  free(pl);

  record_kernel_summary(kernel_name, gx, gy, gz, bx, by, bz, shared_mem,
                        meta, kernel_args, NULL, 0);
}

void hrr_record_kernel_launch_packed(const char* kernel_name,
                                      uint64_t co_hash_lo, uint64_t co_hash_hi,
                                      uint32_t gx, uint32_t gy, uint32_t gz,
                                      uint32_t bx, uint32_t by, uint32_t bz,
                                      uint32_t shared_mem,
                                      const void* stream,
                                      const void* packed_buf,
                                      size_t packed_size) {
  if (!g.active) return;
  if (g.kernel_filter[0] != '\0' && kernel_name) {
    size_t flen = strlen(g.kernel_filter);
    if (g.kernel_filter[flen-1] == '*') {
      if (strncmp(kernel_name, g.kernel_filter, flen-1) != 0) return;
    } else {
      if (strcmp(kernel_name, g.kernel_filter) != 0) return;
    }
  }

  /* Find kernel metadata for arg introspection */
  const hrr_kernel_meta_t* meta = NULL;
  if (kernel_name) {
    HRR_MUTEX_LOCK(&g.mu);
    for (int i = 0; i < g.num_modules && !meta; i++) {
      meta = hrr_find_kernel(g.modules[i].kernels,
                             g.modules[i].num_kernels, kernel_name);
    }
    HRR_MUTEX_UNLOCK(&g.mu);
  }

  if (meta && !g.shape_captured) {
    capture_shape(meta, NULL, packed_buf, packed_size);
  }

  const char* name = kernel_name ? kernel_name : "<unknown>";
  uint16_t name_len = (uint16_t)strlen(name);
  uint16_t num_args = meta ? (uint16_t)meta->num_args : 0;
  uint16_t num_snaps = 0;

  if (!meta && kernel_name) HRR_WARN_NO_METADATA(kernel_name);

  /* Collect unique pointer-arg handles for output snapshot capture */
  uint64_t ptr_handles[MAX_SNAP_PTRS];
  int num_ptr_handles = 0;

  /* co_hash (16 bytes) added to payload after name */
  size_t pl_size = 2 + name_len + 16 + 12 + 12 + 4 + 2 + 2;
  for (uint32_t i = 0; i < num_args; i++) {
    pl_size += 1 + 2 + (meta->args[i].kind == HRR_ARG_GLOBAL_BUFFER ? 8 :
                         meta->args[i].size);
  }

  uint8_t* pl = (uint8_t*)malloc(pl_size);
  if (!pl) return;
  uint8_t* p = pl;

  memcpy(p, &name_len, 2); p += 2;
  memcpy(p, name, name_len); p += name_len;
  memcpy(p, &co_hash_lo, 8); p += 8;
  memcpy(p, &co_hash_hi, 8); p += 8;
  memcpy(p, &gx, 4); p += 4;
  memcpy(p, &gy, 4); p += 4;
  memcpy(p, &gz, 4); p += 4;
  memcpy(p, &bx, 4); p += 4;
  memcpy(p, &by, 4); p += 4;
  memcpy(p, &bz, 4); p += 4;
  memcpy(p, &shared_mem, 4); p += 4;
  memcpy(p, &num_args, 2); p += 2;
  uint8_t* num_snaps_ptr = p;
  memcpy(p, &num_snaps, 2); p += 2;

  for (uint32_t i = 0; i < num_args; i++) {
    const hrr_arg_desc_t* ad = &meta->args[i];
    uint8_t vk = (uint8_t)ad->kind;
    *p++ = vk;

    const char* arg_data = packed_buf ?
        (const char*)packed_buf + ad->offset : NULL;
    int arg_in_bounds = packed_buf &&
        (size_t)(ad->offset + (ad->kind == HRR_ARG_GLOBAL_BUFFER ? 8 : ad->size))
        <= packed_size;

    if (ad->kind == HRR_ARG_GLOBAL_BUFFER) {
      uint16_t sz = 8;
      memcpy(p, &sz, 2); p += 2;
      if (arg_in_bounds) {
        uint64_t dev_ptr = 0;
        memcpy(&dev_ptr, arg_data, 8);
        uint64_t handle = 0;
        HRR_MUTEX_LOCK(&g.mu);
        int found = find_alloc_handle((uintptr_t)dev_ptr, &handle);
        HRR_MUTEX_UNLOCK(&g.mu);
        if (!found && dev_ptr != 0) {
          if (g.verbose) {
            fprintf(stderr, "[HRR] kernel '%s' arg[%u]: ptr=0x%llx untracked, "
                    "registering synthetic alloc (1MB)\n",
                    name, i, (unsigned long long)dev_ptr);
          }
          hrr_record_malloc((const void*)(uintptr_t)dev_ptr, 1024*1024, 0);
          HRR_MUTEX_LOCK(&g.mu);
          find_alloc_handle((uintptr_t)dev_ptr, &handle);
          HRR_MUTEX_UNLOCK(&g.mu);
        }
        memcpy(p, &handle, 8);

        if (handle != 0 && num_ptr_handles < MAX_SNAP_PTRS) {
          int dup = 0;
          for (int k = 0; k < num_ptr_handles; k++) {
            if (ptr_handles[k] == handle) { dup = 1; break; }
          }
          if (!dup) ptr_handles[num_ptr_handles++] = handle;
        }
      } else {
        memset(p, 0, 8);
      }
      p += 8;
    } else {
      uint16_t sz = ad->size;
      memcpy(p, &sz, 2); p += 2;
      if (arg_in_bounds && ad->kind != HRR_ARG_HIDDEN) {
        memcpy(p, arg_data, ad->size);
      } else {
        memset(p, 0, ad->size);
      }
      p += ad->size;
    }
  }

  /* Full mode: capture output snapshots and append to payload */
  if (g.mode == 2 && num_ptr_handles > 0) {
    snap_record_t snaps[MAX_SNAP_PTRS];
    int snap_count = capture_output_snapshots(ptr_handles, num_ptr_handles,
                                              snaps);
    if (snap_count > 0) {
      size_t snap_data_size = (size_t)snap_count * 41;
      pl = (uint8_t*)realloc(pl, pl_size + snap_data_size);
      if (pl) {
        num_snaps_ptr = pl + (2 + name_len + 16 + 12 + 12 + 4 + 2);
        num_snaps = (uint16_t)snap_count;
        memcpy(num_snaps_ptr, &num_snaps, 2);

        uint8_t* sp = pl + pl_size;
        for (int i = 0; i < snap_count; i++) {
          memcpy(sp, &snaps[i].ptr_handle, 8); sp += 8;
          memcpy(sp, &snaps[i].offset, 8);     sp += 8;
          memcpy(sp, &snaps[i].length, 8);     sp += 8;
          memcpy(sp, &snaps[i].hash_lo, 8);    sp += 8;
          memcpy(sp, &snaps[i].hash_hi, 8);    sp += 8;
          *sp++ = snaps[i].direction;
        }
        pl_size += snap_data_size;
      }
    }
  }

  uint16_t pl_len = (uint16_t)(pl_size > 65535 ? 65535 : pl_size);
  write_event(EVT_KERNEL_LAUNCH, (uint32_t)(uintptr_t)stream, pl, pl_len);
  free(pl);

  record_kernel_summary(kernel_name, gx, gy, gz, bx, by, bz, shared_mem,
                        meta, NULL, packed_buf, packed_size);
}

void hrr_record_device_sync(void) {
  if (!g.active) return;
  write_event(EVT_DEVICE_SYNC, 0, NULL, 0);
}

void hrr_record_stream_sync(const void* stream) {
  if (!g.active) return;
  write_event(EVT_STREAM_SYNC, (uint32_t)(uintptr_t)stream, NULL, 0);
}

void hrr_set_device_ops(hrr_device_sync_fn sync_fn, hrr_memcpy_fn memcpy_fn) {
  g.device_sync = sync_fn;
  g.memcpy_fn   = memcpy_fn;
}

void hrr_set_memset_op(hrr_memset_fn memset_fn) {
  g.memset_fn = memset_fn;
}

int hrr_capture_mode(void) { return g.mode; }

void hrr_register_function(const void* func_handle, const void* module_handle,
                            const char* kernel_name) {
  if (!func_handle || !kernel_name) return;
  HRR_MUTEX_LOCK(&g.mu);
  /* Find the sequential module handle from our module table */
  uint64_t mod_h = 0;
  uintptr_t raw_mod = (uintptr_t)module_handle;
  for (int i = 0; i < g.num_modules; i++) {
    if (g.modules[i].module == raw_mod) { mod_h = g.modules[i].handle; break; }
  }
  /* Check if already registered (update in place) */
  uintptr_t h = (uintptr_t)func_handle;
  for (int i = 0; i < g.num_funcs; i++) {
    if (g.funcs[i].handle == h) {
      free(g.funcs[i].name);
#ifdef _WIN32
      g.funcs[i].name = _strdup(kernel_name);
#else
      g.funcs[i].name = strdup(kernel_name);
#endif
      g.funcs[i].module_handle = mod_h;
      HRR_MUTEX_UNLOCK(&g.mu);
      return;
    }
  }
  if (g.num_funcs < MAX_FUNC_ENTRIES) {
    g.funcs[g.num_funcs].handle = h;
    g.funcs[g.num_funcs].module_handle = mod_h;
#ifdef _WIN32
    g.funcs[g.num_funcs].name = _strdup(kernel_name);
#else
    g.funcs[g.num_funcs].name = strdup(kernel_name);
#endif
    g.num_funcs++;
  }
  HRR_MUTEX_UNLOCK(&g.mu);
}

const char* hrr_lookup_function_name(const void* func_handle) {
  if (!func_handle) return NULL;
  uintptr_t h = (uintptr_t)func_handle;
  HRR_MUTEX_LOCK(&g.mu);
  for (int i = 0; i < g.num_funcs; i++) {
    if (g.funcs[i].handle == h) {
      const char* name = g.funcs[i].name;
      HRR_MUTEX_UNLOCK(&g.mu);
      return name;
    }
  }
  HRR_MUTEX_UNLOCK(&g.mu);
  return NULL;
}

int hrr_lookup_function_co_hash(const void* func_handle,
                                 uint64_t* hash_lo, uint64_t* hash_hi) {
  *hash_lo = 0; *hash_hi = 0;
  if (!func_handle) return 0;
  uintptr_t h = (uintptr_t)func_handle;
  HRR_MUTEX_LOCK(&g.mu);
  /* Find the function entry */
  uint64_t mod_h = 0;
  int found = 0;
  for (int i = 0; i < g.num_funcs; i++) {
    if (g.funcs[i].handle == h) { mod_h = g.funcs[i].module_handle; found = 1; break; }
  }
  if (found && mod_h != 0) {
    /* Find the module by its sequential handle and return its stored hash */
    for (int i = 0; i < g.num_modules; i++) {
      if (g.modules[i].handle == mod_h) {
        *hash_lo = g.modules[i].image_hash_lo;
        *hash_hi = g.modules[i].image_hash_hi;
        HRR_MUTEX_UNLOCK(&g.mu);
        return (*hash_lo != 0 || *hash_hi != 0) ? 1 : 0;
      }
    }
  }
  HRR_MUTEX_UNLOCK(&g.mu);
  return 0;
}
