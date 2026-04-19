/* Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License */
#pragma once

/* Shared trace writer for out-of-tree HRR recording.
 * Plain C, used by both LD_PRELOAD interposer and Windows proxy DLL. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize the mutex only — safe to call from DllMain / early DLL init,
 * before HRR_* env vars are read.  Must be called before any
 * hrr_register_function() calls (i.e. before DLL global constructors that
 * invoke __hipRegisterFunction run).  Idempotent. */
void hrr_early_init(void);

/* Initialize the trace writer. Reads HRR_* env vars.
 * Returns 1 if recording is enabled, 0 otherwise. */
int hrr_writer_init(void);

/* Flush and close the trace writer. */
void hrr_writer_shutdown(void);

/* Is recording active? */
int hrr_writer_enabled(void);

/* Record a malloc event */
void hrr_record_malloc(const void* ptr, size_t size, unsigned int flags);

/* Record a free event */
void hrr_record_free(const void* ptr);

/* Record a memcpy event. For H2D (kind=1), src data is captured. */
void hrr_record_memcpy(void* dst, const void* src, size_t size,
                       unsigned int kind, const void* stream);

/* Record a memset event */
void hrr_record_memset(void* dst, int value, size_t size, const void* stream);

/* Record a module load event with code object image */
void hrr_record_module_load(void* module, const void* image, size_t image_size);

/* Record a module unload event */
void hrr_record_module_unload(void* module);

/* Record a kernel launch event.
 * co_hash_lo/hi: XXH3-128 hash of the code object the function belongs to
 *   (0,0 if unknown — replay will fall back to searching all modules).
 * kernel_name: mangled kernel function name */
void hrr_record_kernel_launch(const char* kernel_name,
                              uint64_t co_hash_lo, uint64_t co_hash_hi,
                              uint32_t gx, uint32_t gy, uint32_t gz,
                              uint32_t bx, uint32_t by, uint32_t bz,
                              uint32_t shared_mem,
                              const void* stream,
                              void** kernel_args);

/* Record a kernel launch using the packed kernarg buffer (hipExtModuleLaunchKernel
 * 'extra' path). co_hash_lo/hi identifies the owning code object. */
void hrr_record_kernel_launch_packed(const char* kernel_name,
                                      uint64_t co_hash_lo, uint64_t co_hash_hi,
                                      uint32_t gx, uint32_t gy, uint32_t gz,
                                      uint32_t bx, uint32_t by, uint32_t bz,
                                      uint32_t shared_mem,
                                      const void* stream,
                                      const void* packed_buf,
                                      size_t packed_size);

/* Record a device synchronize event */
void hrr_record_device_sync(void);

/* Record a stream synchronize event */
void hrr_record_stream_sync(const void* stream);

/* Function handle → (module, kernel name) registry.
 * Called from hipModuleGetFunction hook.  module_handle is the hipModule_t
 * that owns the function; it is used to recover the code object hash for the
 * KERNEL_LAUNCH event so replay can select the exact code object. */
void hrr_register_function(const void* func_handle, const void* module_handle,
                            const char* kernel_name);

/* Look up a kernel name by its hipFunction_t handle.
 * Returns the registered name, or NULL if unknown. */
const char* hrr_lookup_function_name(const void* func_handle);

/* Look up the code-object hash for a function.
 * Fills hash_lo/hash_hi; returns 1 on success, 0 if not found (both set to 0). */
int hrr_lookup_function_co_hash(const void* func_handle,
                                 uint64_t* hash_lo, uint64_t* hash_hi);

/* Device operation callbacks for full-mode output snapshot capture.
 * In full mode (HRR_MODE=full), the writer calls these to synchronize the GPU
 * and read back output buffers after each kernel launch.  Must point to the
 * REAL HIP functions (not the proxy wrappers) to avoid re-entrancy.
 *   sync_fn:   hipDeviceSynchronize()
 *   memcpy_fn: hipMemcpy(dst, src, size, kind)  — called with kind=2 (D2H)
 *   memset_fn: hipMemset(dst, value, size)      — used to zero-init buffers
 *              on hipMalloc so that record-time and replay-time see the same
 *              initial GPU memory (replay zero-inits its own allocations).
 *              May be NULL; if NULL, zero-init is skipped. */
typedef int (*hrr_device_sync_fn)(void);
typedef int (*hrr_memcpy_fn)(void* dst, const void* src, size_t size,
                             unsigned int kind);
typedef int (*hrr_memset_fn)(void* dst, int value, size_t size);
void hrr_set_device_ops(hrr_device_sync_fn sync_fn, hrr_memcpy_fn memcpy_fn);
void hrr_set_memset_op(hrr_memset_fn memset_fn);

/* Returns the current capture mode (0=timeline, 1=inputs, 2=full). */
int hrr_capture_mode(void);

#ifdef __cplusplus
}
#endif
