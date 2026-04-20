/* Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License */

/* HRR LD_PRELOAD interposer for Linux.
 *
 * Build: gcc -shared -fPIC -o libhrr_record.so \
 *          hrr_interposer_linux.c hrr_trace_writer.c hrr_code_object.c \
 *          -ldl -lpthread
 *
 * Usage: HRR_RECORD=1 LD_PRELOAD=./libhrr_record.so <hip-application>
 *
 * Intercepts HIP API calls via dlsym(RTLD_NEXT) forwarding. */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hrr_trace_writer.h"

/* HIP types (avoid pulling in hip headers) */
typedef int hipError_t;
typedef void* hipModule_t;
typedef void* hipFunction_t;
typedef void* hipStream_t;
typedef void* hipEvent_t;
typedef unsigned int hipMemcpyKind;

/* Real function pointers */
static hipError_t (*real_hipMalloc)(void**, size_t) = NULL;
static hipError_t (*real_hipExtMallocWithFlags)(void**, size_t, unsigned int) = NULL;
static hipError_t (*real_hipMallocManaged)(void**, size_t, unsigned int) = NULL;
static hipError_t (*real_hipMallocAsync)(void**, size_t, hipStream_t) = NULL;
static hipError_t (*real_hipMallocFromPoolAsync)(void**, size_t, void*, hipStream_t) = NULL;
static hipError_t (*real_hipFreeAsync)(void*, hipStream_t) = NULL;
static hipError_t (*real_hipFree)(void*) = NULL;
static hipError_t (*real_hipMemcpy)(void*, const void*, size_t, hipMemcpyKind) = NULL;
static hipError_t (*real_hipMemcpyAsync)(void*, const void*, size_t,
                                         hipMemcpyKind, hipStream_t) = NULL;
static hipError_t (*real_hipMemcpyWithStream)(void*, const void*, size_t,
                                              hipMemcpyKind, hipStream_t) = NULL;
static hipError_t (*real_hipMemcpyHtoD)(void*, const void*, size_t) = NULL;
static hipError_t (*real_hipMemcpyHtoDAsync)(void*, const void*, size_t,
                                              hipStream_t) = NULL;
static hipError_t (*real_hipMemcpyDtoH)(void*, const void*, size_t) = NULL;
static hipError_t (*real_hipMemcpyDtoD)(void*, const void*, size_t) = NULL;
static hipError_t (*real_hipMemcpyDtoDAsync)(void*, const void*, size_t,
                                              hipStream_t) = NULL;
static hipError_t (*real_hipMemset)(void*, int, size_t) = NULL;
static hipError_t (*real_hipMemsetAsync)(void*, int, size_t, hipStream_t) = NULL;
static hipError_t (*real_hipMemsetD8)(void*, unsigned char, size_t) = NULL;
static hipError_t (*real_hipMemsetD8Async)(void*, unsigned char, size_t, hipStream_t) = NULL;
static hipError_t (*real_hipMemsetD16)(void*, unsigned short, size_t) = NULL;
static hipError_t (*real_hipMemsetD16Async)(void*, unsigned short, size_t, hipStream_t) = NULL;
static hipError_t (*real_hipMemsetD32)(void*, int, size_t) = NULL;
static hipError_t (*real_hipMemsetD32Async)(void*, int, size_t, hipStream_t) = NULL;
static hipError_t (*real_hipMemcpy2D)(void*, size_t, const void*, size_t, size_t,
                                       size_t, hipMemcpyKind) = NULL;
static hipError_t (*real_hipMemcpy2DAsync)(void*, size_t, const void*, size_t, size_t,
                                            size_t, hipMemcpyKind, hipStream_t) = NULL;
static hipError_t (*real_hipHostMalloc)(void**, size_t, unsigned int) = NULL;
static hipError_t (*real_hipHostFree)(void*) = NULL;
static hipError_t (*real_hipHostRegister)(void*, size_t, unsigned int) = NULL;
static hipError_t (*real_hipHostUnregister)(void*) = NULL;
static hipError_t (*real_hipHostGetDevicePointer)(void**, void*, unsigned int) = NULL;
static hipError_t (*real_hipModuleLoad)(hipModule_t*, const char*) = NULL;
static hipError_t (*real_hipModuleLoadData)(hipModule_t*, const void*) = NULL;
static hipError_t (*real_hipModuleUnload)(hipModule_t) = NULL;
static hipError_t (*real_hipModuleLaunchKernel)(hipFunction_t, unsigned, unsigned,
    unsigned, unsigned, unsigned, unsigned, unsigned, hipStream_t,
    void**, void**) = NULL;
static hipError_t (*real_hipModuleLaunchCooperativeKernel)(hipFunction_t,
    unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
    hipStream_t, void**) = NULL;
static hipError_t (*real_hipModuleGetFunction)(hipFunction_t*, hipModule_t,
    const char*) = NULL;
static hipError_t (*real_hipDeviceSynchronize)(void) = NULL;
static hipError_t (*real_hipStreamSynchronize)(hipStream_t) = NULL;
static hipError_t (*real_hipInit)(unsigned int) = NULL;

/* Explicit handle to the real HIP library.
 * RTLD_NEXT fails during early init if libamdhip64 isn't in the chain yet
 * (e.g. called from another library's constructor). We fall back to
 * dlopen("libamdhip64.so.N", RTLD_NOLOAD) to get the real functions without
 * pulling in a second copy.  Try versioned names first (ROCm 7 = .so.7,
 * ROCm 6 = .so.6) then the bare soname. */
static void* s_hip_lib = NULL;

static void* hrr_load_hip_sym(const char* name) {
  void* sym = dlsym(RTLD_NEXT, name);
  if (sym) return sym;

  /* RTLD_NEXT returned NULL — libamdhip64 not yet in chain. Try explicit
   * handle to whichever version is already mapped in this process.
   * RTLD_NOLOAD: only find it if already mapped; don't pull in a new copy. */
  if (!s_hip_lib) {
    static const char* const candidates[] = {
      "libamdhip64.so.7",
      "libamdhip64.so.6",
      "libamdhip64.so",
      NULL
    };
    for (int i = 0; candidates[i] && !s_hip_lib; ++i)
      s_hip_lib = dlopen(candidates[i], RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
  }
  if (s_hip_lib) return dlsym(s_hip_lib, name);
  return NULL;
}

#define LOAD_SYM(name) \
  if (!real_##name) real_##name = hrr_load_hip_sym(#name)

/* Forward a call to a real HIP function, returning hipErrorUnknown if the
 * symbol could not be resolved (prevents null-pointer dereference). */
#define FORWARD_OR_ERROR(name, args) \
  do { if (!real_##name) return -1; return real_##name args; } while(0)

static int g_initialized = 0;
static int g_device_ops_set = 0;

/* Resolve and register the real device ops used by the writer for full-mode
 * output snapshot capture (hipDeviceSynchronize + hipMemcpy D2H readback).
 *
 * This MUST be retried on every interception call until it succeeds: when
 * ensure_init() runs from the LD_PRELOAD constructor, libamdhip64 is not
 * yet in the dlsym chain and both LOAD_SYM lookups return NULL.  Without
 * retry, g.device_sync / g.memcpy_fn stay NULL forever and
 * capture_output_snapshots() silently returns 0 — which manifests as
 * "Verification: 0 passed, 0 failed" on replay even with HRR_MODE=full. */
static void try_register_device_ops(void) {
  if (g_device_ops_set || !hrr_writer_enabled()) return;
  LOAD_SYM(hipDeviceSynchronize);
  LOAD_SYM(hipMemcpy);
  LOAD_SYM(hipMemset);
  if (real_hipDeviceSynchronize && real_hipMemcpy) {
    hrr_set_device_ops((hrr_device_sync_fn)real_hipDeviceSynchronize,
                       (hrr_memcpy_fn)real_hipMemcpy);
    /* hipMemset is optional — only used in full mode to zero-init buffers
     * on hipMalloc so recorder and replayer see identical initial state. */
    if (real_hipMemset) {
      hrr_set_memset_op((hrr_memset_fn)real_hipMemset);
    }
    g_device_ops_set = 1;
  }
}

static void ensure_init(void) {
  if (!g_initialized) {
    g_initialized = 1;
    hrr_writer_init();
    if (hrr_writer_enabled()) atexit(hrr_writer_shutdown);
  }
  try_register_device_ops();
}

/* ---- Interposed functions ---- */

hipError_t hipInit(unsigned int flags) {
  LOAD_SYM(hipInit);
  ensure_init();
  FORWARD_OR_ERROR(hipInit, (flags));
}

hipError_t hipMalloc(void** ptr, size_t size) {
  LOAD_SYM(hipMalloc);
  ensure_init();
  if (!real_hipMalloc) return -1;
  hipError_t ret = real_hipMalloc(ptr, size);
  if (ret == 0 && hrr_writer_enabled()) {
    hrr_record_malloc(*ptr, size, 0);
  }
  return ret;
}

hipError_t hipExtMallocWithFlags(void** ptr, size_t sizeBytes, unsigned int flags) {
  LOAD_SYM(hipExtMallocWithFlags);
  ensure_init();
  if (!real_hipExtMallocWithFlags) return -1;
  hipError_t ret = real_hipExtMallocWithFlags(ptr, sizeBytes, flags);
  if (ret == 0 && hrr_writer_enabled()) {
    hrr_record_malloc(*ptr, sizeBytes, flags);
  }
  return ret;
}

hipError_t hipMallocManaged(void** ptr, size_t size, unsigned int flags) {
  LOAD_SYM(hipMallocManaged);
  ensure_init();
  if (!real_hipMallocManaged) return -1;
  hipError_t ret = real_hipMallocManaged(ptr, size, flags);
  if (ret == 0 && hrr_writer_enabled()) {
    hrr_record_malloc(*ptr, size, flags);
  }
  return ret;
}

hipError_t hipMallocAsync(void** ptr, size_t size, hipStream_t stream) {
  LOAD_SYM(hipMallocAsync);
  ensure_init();
  if (!real_hipMallocAsync) return -1;
  hipError_t ret = real_hipMallocAsync(ptr, size, stream);
  if (ret == 0 && hrr_writer_enabled()) {
    hrr_record_malloc(*ptr, size, 0);
  }
  return ret;
}

hipError_t hipMallocFromPoolAsync(void** ptr, size_t size, void* mem_pool,
                                   hipStream_t stream) {
  LOAD_SYM(hipMallocFromPoolAsync);
  ensure_init();
  if (!real_hipMallocFromPoolAsync) return -1;
  hipError_t ret = real_hipMallocFromPoolAsync(ptr, size, mem_pool, stream);
  if (ret == 0 && hrr_writer_enabled()) {
    hrr_record_malloc(*ptr, size, 0);
  }
  return ret;
}

hipError_t hipFreeAsync(void* ptr, hipStream_t stream) {
  LOAD_SYM(hipFreeAsync);
  if (hrr_writer_enabled()) {
    hrr_record_free(ptr);
  }
  FORWARD_OR_ERROR(hipFreeAsync, (ptr, stream));
}

hipError_t hipFree(void* ptr) {
  LOAD_SYM(hipFree);
  if (hrr_writer_enabled()) {
    hrr_record_free(ptr);
  }
  FORWARD_OR_ERROR(hipFree, (ptr));
}

hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes,
                     hipMemcpyKind kind) {
  LOAD_SYM(hipMemcpy);
  if (hrr_writer_enabled()) {
    hrr_record_memcpy(dst, src, sizeBytes, (unsigned int)kind, NULL);
  }
  FORWARD_OR_ERROR(hipMemcpy, (dst, src, sizeBytes, kind));
}

/* hipMemcpyAsync / hipMemcpyWithStream — async variants are the dominant
 * H2D path in ROCm libraries (MIGraphX, rocBLAS, MIOpen).  Without
 * intercepting them, H2D uploads are silently dropped from the trace and
 * downstream kernels read uninitialized device memory, often page-faulting
 * on a value the kernel interprets as a pointer.
 *
 * For H2D the source host buffer is captured immediately; this is safe
 * because the host pages are valid at call time even though the GPU copy
 * has not yet completed.  For D2D we record the event so replay can
 * re-issue the copy between translated live pointers.  D2H carries no
 * data needed for replay. */
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                          hipMemcpyKind kind, hipStream_t stream) {
  LOAD_SYM(hipMemcpyAsync);
  if (hrr_writer_enabled() && (kind == 1 /* H2D */ || kind == 3 /* D2D */)) {
    hrr_record_memcpy(dst, src, sizeBytes, (unsigned int)kind, stream);
  }
  FORWARD_OR_ERROR(hipMemcpyAsync, (dst, src, sizeBytes, kind, stream));
}

hipError_t hipMemcpyWithStream(void* dst, const void* src, size_t sizeBytes,
                               hipMemcpyKind kind, hipStream_t stream) {
  LOAD_SYM(hipMemcpyWithStream);
  if (hrr_writer_enabled() && (kind == 1 /* H2D */ || kind == 3 /* D2D */)) {
    hrr_record_memcpy(dst, src, sizeBytes, (unsigned int)kind, stream);
  }
  FORWARD_OR_ERROR(hipMemcpyWithStream, (dst, src, sizeBytes, kind, stream));
}

/* hipMemcpyHtoD / hipMemcpyDtoH — explicit-direction variants used by MIGraphX */
hipError_t hipMemcpyHtoD(void* dst, const void* src, size_t sizeBytes) {
  LOAD_SYM(hipMemcpyHtoD);
  if (hrr_writer_enabled()) {
    hrr_record_memcpy(dst, src, sizeBytes, 1 /* hipMemcpyHostToDevice */, NULL);
  }
  FORWARD_OR_ERROR(hipMemcpyHtoD, (dst, src, sizeBytes));
}

hipError_t hipMemcpyHtoDAsync(void* dst, const void* src, size_t sizeBytes,
                              hipStream_t stream) {
  LOAD_SYM(hipMemcpyHtoDAsync);
  if (hrr_writer_enabled()) {
    hrr_record_memcpy(dst, src, sizeBytes, 1 /* hipMemcpyHostToDevice */, stream);
  }
  FORWARD_OR_ERROR(hipMemcpyHtoDAsync, (dst, src, sizeBytes, stream));
}

hipError_t hipMemcpyDtoH(void* dst, const void* src, size_t sizeBytes) {
  LOAD_SYM(hipMemcpyDtoH);
  /* DtoH copies don't need blob capture (no GPU→CPU data needed for replay) */
  FORWARD_OR_ERROR(hipMemcpyDtoH, (dst, src, sizeBytes));
}

hipError_t hipMemcpyDtoD(void* dst, const void* src, size_t sizeBytes) {
  LOAD_SYM(hipMemcpyDtoD);
  if (hrr_writer_enabled()) {
    hrr_record_memcpy(dst, src, sizeBytes, 3 /* hipMemcpyDeviceToDevice */, NULL);
  }
  FORWARD_OR_ERROR(hipMemcpyDtoD, (dst, src, sizeBytes));
}

hipError_t hipMemcpyDtoDAsync(void* dst, const void* src, size_t sizeBytes,
                              hipStream_t stream) {
  LOAD_SYM(hipMemcpyDtoDAsync);
  if (hrr_writer_enabled()) {
    hrr_record_memcpy(dst, src, sizeBytes, 3 /* hipMemcpyDeviceToDevice */, stream);
  }
  FORWARD_OR_ERROR(hipMemcpyDtoDAsync, (dst, src, sizeBytes, stream));
}

/* Host-mapped memory APIs.
 *
 * MIGraphX (and other inference frameworks) skip explicit hipMemcpy for the
 * model input by pinning the user's CPU buffer with hipHostRegister and
 * obtaining a device-accessible alias via hipHostGetDevicePointer.  The GPU
 * kernel then reads the host bytes directly via that alias.  Without
 * intercepting these APIs, the recorder never sees the input bytes at all:
 * the kernel arg points to an "untracked" address (which find_alloc_handle
 * registers as a 1MB synthetic alloc full of zeros) and on replay the kernel
 * executes against that zero buffer, producing constant or zero outputs and
 * cascading verification failures.
 *
 * Treat each pinned host region as a malloc, and synthesize an H2D memcpy
 * carrying the current host bytes so replay restores them into the
 * corresponding live device alloc before the first kernel runs.  When
 * hipHostGetDevicePointer returns an alias different from the host pointer,
 * register both addresses (some HIP configurations return a dedicated
 * device-mapped address rather than a 1:1 alias). */
hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags) {
  LOAD_SYM(hipHostMalloc);
  ensure_init();
  if (!real_hipHostMalloc) return -1;
  hipError_t ret = real_hipHostMalloc(ptr, size, flags);
  if (ret == 0 && hrr_writer_enabled() && ptr && *ptr) {
    /* Buffer is fresh / not yet filled — record the alloc only.  Any later
     * data the host writes into this buffer will be observed via the
     * H2D snapshot we synthesise at hipHostUnregister/Free time, OR via
     * the kernel's own input snapshot when it reads the buffer. */
    hrr_record_host_alloc(*ptr, size, flags);
  }
  return ret;
}

hipError_t hipHostFree(void* ptr) {
  LOAD_SYM(hipHostFree);
  if (hrr_writer_enabled()) {
    hrr_record_free(ptr);
  }
  FORWARD_OR_ERROR(hipHostFree, (ptr));
}

hipError_t hipHostRegister(void* ptr, size_t size, unsigned int flags) {
  LOAD_SYM(hipHostRegister);
  ensure_init();
  if (!real_hipHostRegister) return -1;
  hipError_t ret = real_hipHostRegister(ptr, size, flags);
  if (ret == 0 && hrr_writer_enabled() && ptr) {
    /* Pin the existing buffer.  At this point the host bytes are already
     * the meaningful payload (caller filled them before pinning), so capture
     * them now as a synthetic H2D blob keyed on the host address — the same
     * address the kernel will receive as its device pointer. */
    hrr_record_host_alloc(ptr, size, flags);
    hrr_record_memcpy(ptr, ptr, size, 1 /* hipMemcpyHostToDevice */, NULL);
  }
  return ret;
}

hipError_t hipHostUnregister(void* ptr) {
  LOAD_SYM(hipHostUnregister);
  if (hrr_writer_enabled()) {
    hrr_record_free(ptr);
  }
  FORWARD_OR_ERROR(hipHostUnregister, (ptr));
}

hipError_t hipHostGetDevicePointer(void** dev_ptr, void* host_ptr,
                                   unsigned int flags) {
  LOAD_SYM(hipHostGetDevicePointer);
  if (!real_hipHostGetDevicePointer) return -1;
  hipError_t ret = real_hipHostGetDevicePointer(dev_ptr, host_ptr, flags);
  if (ret == 0 && hrr_writer_enabled() && dev_ptr && *dev_ptr &&
      *dev_ptr != host_ptr) {
    /* Distinct device alias — register it so kernel args using *dev_ptr also
     * resolve.  Size is unknown here; reuse a generous 1 MB placeholder which
     * is the same fallback hrr_record_kernel_launch uses for untracked args. */
    hrr_record_host_alloc(*dev_ptr, 1024 * 1024, 0);
  }
  return ret;
}

hipError_t hipMemset(void* dst, int value, size_t count) {
  LOAD_SYM(hipMemset);
  if (hrr_writer_enabled()) {
    hrr_record_memset(dst, value, count, NULL);
  }
  FORWARD_OR_ERROR(hipMemset, (dst, value, count));
}

/* Async / typed memset variants.
 *
 * MIOpen, rocBLAS, hipBLASLt, and MIGraphX itself use hipMemsetAsync to
 * zero workspace buffers and intermediate tensors before launching kernels
 * that read-modify-write them.  Without intercepting these, replay sees
 * stale (uninitialized) memory at those addresses, kernels read garbage,
 * and downstream verification fails with NaN/Inf cascades.  This was the
 * Linux-specific divergence vs Windows (which captures these via the
 * proxy DLL): MIOpen ships only on Linux, so the missing memset path was
 * only exposed in the Linux capture pipeline.
 *
 * The typed variants (D8/D16/D32) splat a fixed pattern of that width.
 * Convert them to a byte-equivalent for hrr_record_memset by repeating
 * only when the pattern is byte-uniform; otherwise we fall through to a
 * raw record using value & 0xFF, which loses information but is still
 * better than missing the event entirely.  In practice MIOpen / rocBLAS
 * use these only with value=0. */
hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes,
                          hipStream_t stream) {
  LOAD_SYM(hipMemsetAsync);
  if (hrr_writer_enabled()) {
    hrr_record_memset(dst, value, sizeBytes, stream);
  }
  FORWARD_OR_ERROR(hipMemsetAsync, (dst, value, sizeBytes, stream));
}

hipError_t hipMemsetD8(void* dst, unsigned char value, size_t count) {
  LOAD_SYM(hipMemsetD8);
  if (hrr_writer_enabled()) {
    hrr_record_memset(dst, (int)value, count, NULL);
  }
  FORWARD_OR_ERROR(hipMemsetD8, (dst, value, count));
}

hipError_t hipMemsetD8Async(void* dst, unsigned char value, size_t count,
                            hipStream_t stream) {
  LOAD_SYM(hipMemsetD8Async);
  if (hrr_writer_enabled()) {
    hrr_record_memset(dst, (int)value, count, stream);
  }
  FORWARD_OR_ERROR(hipMemsetD8Async, (dst, value, count, stream));
}

hipError_t hipMemsetD16(void* dst, unsigned short value, size_t count) {
  LOAD_SYM(hipMemsetD16);
  if (hrr_writer_enabled()) {
    /* count is element count, not bytes — most callers use value=0 so the
     * byte-extended pattern is identical (0x00). */
    unsigned char b = (unsigned char)(value & 0xFF);
    hrr_record_memset(dst, (int)b, count * 2, NULL);
  }
  FORWARD_OR_ERROR(hipMemsetD16, (dst, value, count));
}

hipError_t hipMemsetD16Async(void* dst, unsigned short value, size_t count,
                             hipStream_t stream) {
  LOAD_SYM(hipMemsetD16Async);
  if (hrr_writer_enabled()) {
    unsigned char b = (unsigned char)(value & 0xFF);
    hrr_record_memset(dst, (int)b, count * 2, stream);
  }
  FORWARD_OR_ERROR(hipMemsetD16Async, (dst, value, count, stream));
}

hipError_t hipMemsetD32(void* dst, int value, size_t count) {
  LOAD_SYM(hipMemsetD32);
  if (hrr_writer_enabled()) {
    unsigned char b = (unsigned char)(value & 0xFF);
    hrr_record_memset(dst, (int)b, count * 4, NULL);
  }
  FORWARD_OR_ERROR(hipMemsetD32, (dst, value, count));
}

hipError_t hipMemsetD32Async(void* dst, int value, size_t count,
                             hipStream_t stream) {
  LOAD_SYM(hipMemsetD32Async);
  if (hrr_writer_enabled()) {
    unsigned char b = (unsigned char)(value & 0xFF);
    hrr_record_memset(dst, (int)b, count * 4, stream);
  }
  FORWARD_OR_ERROR(hipMemsetD32Async, (dst, value, count, stream));
}

/* 2D memcpy — used by rocBLAS for sub-matrix transfers (leading-dimension
 * stride != width).  When pitch == width we collapse to a single 1D
 * memcpy; otherwise record one H2D event per row so the replay can
 * faithfully recreate the strided layout in the destination buffer. */
static void record_memcpy_2d(void* dst, size_t dpitch, const void* src,
                             size_t spitch, size_t width, size_t height,
                             hipMemcpyKind kind, hipStream_t stream) {
  if (!hrr_writer_enabled()) return;
  if (kind != 1 /* H2D */ && kind != 3 /* D2D */) return;
  if (height == 0 || width == 0) return;
  if (dpitch == width && spitch == width) {
    hrr_record_memcpy(dst, src, width * height, (unsigned int)kind, stream);
    return;
  }
  for (size_t r = 0; r < height; ++r) {
    void* drow = (char*)dst + r * dpitch;
    const void* srow = (const char*)src + r * spitch;
    hrr_record_memcpy(drow, srow, width, (unsigned int)kind, stream);
  }
}

hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
                       size_t width, size_t height, hipMemcpyKind kind) {
  LOAD_SYM(hipMemcpy2D);
  record_memcpy_2d(dst, dpitch, src, spitch, width, height, kind, NULL);
  FORWARD_OR_ERROR(hipMemcpy2D, (dst, dpitch, src, spitch, width, height, kind));
}

hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src,
                            size_t spitch, size_t width, size_t height,
                            hipMemcpyKind kind, hipStream_t stream) {
  LOAD_SYM(hipMemcpy2DAsync);
  record_memcpy_2d(dst, dpitch, src, spitch, width, height, kind, stream);
  FORWARD_OR_ERROR(hipMemcpy2DAsync,
                   (dst, dpitch, src, spitch, width, height, kind, stream));
}

/* Compute the true byte extent of an ELF64 binary by walking section headers.
 * The naive formula (e_shoff + e_shentsz * e_shnum) only covers the section
 * header table; section data follows after it in AMDGPU code objects. */
static size_t hrr_elf64_true_size(const unsigned char* p, size_t max_readable) {
  if (max_readable < 64) return 0;
  if (p[0] != 0x7f || p[1] != 'E' || p[2] != 'L' || p[3] != 'F') return 0;
  if (p[4] != 2) return 0;  /* ELFCLASS64 only */

  uint64_t e_shoff;
  uint16_t e_shentsz, e_shnum;
  memcpy(&e_shoff,  p + 40, 8);
  memcpy(&e_shentsz, p + 58, 2);
  memcpy(&e_shnum,   p + 60, 2);

  if (e_shentsz < 64 || e_shnum == 0 || e_shoff == 0) return 0;

  uint64_t end = e_shoff + (uint64_t)e_shentsz * e_shnum;

  for (uint16_t i = 0; i < e_shnum; i++) {
    uint64_t sh_base = e_shoff + (uint64_t)i * e_shentsz;
    if (sh_base + 64 > (uint64_t)max_readable) break;
    uint32_t sh_type;
    uint64_t sh_offset, sh_size;
    memcpy(&sh_type,   p + sh_base + 4,  4);
    memcpy(&sh_offset, p + sh_base + 24, 8);
    memcpy(&sh_size,   p + sh_base + 32, 8);
    if (sh_type == 8 || sh_offset == 0 || sh_size == 0) continue;  /* SHT_NOBITS */
    uint64_t section_end = sh_offset + sh_size;
    if (section_end > end) end = section_end;
  }

  if (end > (uint64_t)max_readable) end = (uint64_t)max_readable;
  return (size_t)end;
}

/* hipModuleLoad — load a code object from a file path.
 * MIGraphX loads its MLIR-compiled kernels via this path (from the code object
 * cache, typically ~/.cache/migraphx/).  Without intercepting this call those
 * kernels are never captured and replay fails to find them. */
hipError_t hipModuleLoad(hipModule_t* module, const char* fname) {
  LOAD_SYM(hipModuleLoad);
  if (!real_hipModuleLoad) return -1;
  hipError_t ret = real_hipModuleLoad(module, fname);
  if (ret == 0 && hrr_writer_enabled() && module && *module && fname) {
    FILE* f = fopen(fname, "rb");
    if (f) {
      fseek(f, 0, SEEK_END);
      long fsz = ftell(f);
      rewind(f);
      if (fsz > 0) {
        void* buf = malloc((size_t)fsz);
        if (buf && fread(buf, 1, (size_t)fsz, f) == (size_t)fsz) {
          const unsigned char* p = (const unsigned char*)buf;
          if (p[0] == 0x7f && p[1] == 'E' && p[2] == 'L' && p[3] == 'F') {
            size_t sz = hrr_elf64_true_size(p, (size_t)fsz);
            if (sz == 0) sz = (size_t)fsz;  /* fallback: use file size */
            hrr_record_module_load(*module, buf, sz);
          } else {
            /* Non-ELF (COB/CCOB) — capture raw; writer will save as-is */
            hrr_record_module_load(*module, buf, (size_t)fsz);
          }
        }
        free(buf);
      }
      fclose(f);
    }
  }
  return ret;
}

hipError_t hipModuleLoadData(hipModule_t* module, const void* image) {
  LOAD_SYM(hipModuleLoadData);
  if (!real_hipModuleLoadData) return -1;
  hipError_t ret = real_hipModuleLoadData(module, image);
  if (ret == 0 && hrr_writer_enabled() && module && *module && image) {
    const unsigned char* p = (const unsigned char*)image;
    size_t image_size = hrr_elf64_true_size(p, (size_t)-1);
    if (image_size > 0) {
      hrr_record_module_load(*module, image, image_size);
    }
  }
  return ret;
}

hipError_t hipModuleUnload(hipModule_t module) {
  LOAD_SYM(hipModuleUnload);
  if (hrr_writer_enabled()) {
    hrr_record_module_unload(module);
  }
  FORWARD_OR_ERROR(hipModuleUnload, (module));
}

hipError_t hipModuleGetFunction(hipFunction_t* hfunc, hipModule_t hmod,
                                const char* name) {
  LOAD_SYM(hipModuleGetFunction);
  if (!real_hipModuleGetFunction) return -1;
  hipError_t ret = real_hipModuleGetFunction(hfunc, hmod, name);
  /* Always register the handle→(module,name) mapping regardless of recording state.
   * module handle is used to identify the code object for precise replay. */
  if (ret == 0 && hfunc && *hfunc && name) {
    hrr_register_function(*hfunc, hmod, name);
  }
  return ret;
}

hipError_t hipModuleLaunchKernel(hipFunction_t f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, hipStream_t hStream,
    void** kernelParams, void** extra) {
  LOAD_SYM(hipModuleLaunchKernel);
  /* Ensure full-mode output snapshot capture has its device ops, even if
   * the app never went through any of the malloc-family entry points. */
  try_register_device_ops();

  /* Launch first so full-mode snapshot capture can sync + readback */
  if (!real_hipModuleLaunchKernel) return -1;
  int ret = real_hipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
      blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream,
      kernelParams, extra);

  if (hrr_writer_enabled()) {
    const char* kname = hrr_lookup_function_name(f);
    uint64_t co_lo = 0, co_hi = 0;
    hrr_lookup_function_co_hash(f, &co_lo, &co_hi);
    hrr_record_kernel_launch(kname, co_lo, co_hi,
                             gridDimX, gridDimY, gridDimZ,
                             blockDimX, blockDimY, blockDimZ,
                             sharedMemBytes, hStream, kernelParams);
  }

  return ret;
}

/* hipModuleLaunchCooperativeKernel — used by MIOpen for some convolution
 * implementations and any kernel that needs cooperative groups grid sync.
 *
 * MIGraphX delegates conv/pool layers to MIOpen on Linux; without this
 * interception, MIOpen's internal kernel launches between two MIGraphX
 * MLIR kernels are silently dropped from the trace.  On replay the device
 * memory those internal kernels were supposed to populate stays at its
 * prior value (typically zero from the initial memset of the workspace
 * pool), so the next MIGraphX kernel reads zeros for its input and emits
 * a constant or zero output — which is exactly the "got=0, exp=2.179"
 * verification mismatch we were chasing.  Windows doesn't ship MIOpen so
 * this path is Linux-specific.
 *
 * Same arg-recording path as hipModuleLaunchKernel — only the dispatch
 * function differs.  The cooperative variant has no `extra` parameter. */
hipError_t hipModuleLaunchCooperativeKernel(hipFunction_t f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, hipStream_t hStream,
    void** kernelParams) {
  LOAD_SYM(hipModuleLaunchCooperativeKernel);
  try_register_device_ops();

  if (!real_hipModuleLaunchCooperativeKernel) return -1;
  int ret = real_hipModuleLaunchCooperativeKernel(f,
      gridDimX, gridDimY, gridDimZ,
      blockDimX, blockDimY, blockDimZ,
      sharedMemBytes, hStream, kernelParams);

  if (hrr_writer_enabled()) {
    const char* kname = hrr_lookup_function_name(f);
    uint64_t co_lo = 0, co_hi = 0;
    hrr_lookup_function_co_hash(f, &co_lo, &co_hi);
    hrr_record_kernel_launch(kname, co_lo, co_hi,
                             gridDimX, gridDimY, gridDimZ,
                             blockDimX, blockDimY, blockDimZ,
                             sharedMemBytes, hStream, kernelParams);
  }

  return ret;
}

/* Note: hipLaunchKernel (<<<>>> / hipLaunchKernelGGL path) is handled in
 * hrr_interposer_cxx.cpp because its ABI uses dim3 (a C++ struct).  The
 * C interposer here covers the C-linkage launch entry points that MIGraphX
 * and the ROCm libraries it links use most heavily. */

hipError_t hipDeviceSynchronize(void) {
  LOAD_SYM(hipDeviceSynchronize);
  if (!real_hipDeviceSynchronize) return -1;
  hipError_t ret = real_hipDeviceSynchronize();
  if (hrr_writer_enabled()) {
    hrr_record_device_sync();
  }
  return ret;
}

hipError_t hipStreamSynchronize(hipStream_t stream) {
  LOAD_SYM(hipStreamSynchronize);
  if (!real_hipStreamSynchronize) return -1;
  hipError_t ret = real_hipStreamSynchronize(stream);
  if (hrr_writer_enabled()) {
    hrr_record_stream_sync(stream);
  }
  return ret;
}

/* Constructor/destructor for shared library lifecycle */
__attribute__((constructor))
static void hrr_lib_init(void) {
  /* Initialize early so recording is active before any HIP calls */
  ensure_init();
}

__attribute__((destructor))
static void hrr_lib_fini(void) {
  hrr_writer_shutdown();
}

/* Bind every interposed symbol to the same version libamdhip64 exports.
 * Without this, callers linked against e.g. hipModuleLaunchKernel@hip_4.2
 * resolve directly to libamdhip64 and skip our hooks (LD_PRELOAD only
 * intercepts unversioned references). */
#include "hrr_symver.h"
