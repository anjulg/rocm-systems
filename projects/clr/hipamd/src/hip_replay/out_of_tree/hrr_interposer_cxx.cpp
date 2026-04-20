/* Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License */

/* C++ extension for HRR LD_PRELOAD interposer.
 *
 * Intercepts C++ linkage HIP APIs that cannot be safely wrapped from plain C.
 * Currently: hipExtModuleLaunchKernel (mangled C++ function).
 *
 * Compiled alongside hrr_interposer_linux.c with g++. */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <cstddef>
#include <cstdint>

/* Auto-generated header (built at CMake configure time from the local
 * libamdhip64.so).  Provides:
 *   - HRR_VER_<symbol>  string-literal version macros
 *   - HRR_SYMVER(name)  helper that emits .symver name,name@HRR_VER_name
 * The macros below are the only thing this TU uses from it; the full
 * list of HRR_SYMVER() lines for C-linkage symbols is consumed from
 * hrr_interposer_linux.c instead. */
#include "hrr_symver.h"

/* Minimal HIP opaque struct forward-declarations.
 * These match the real HIP types and produce the correct Itanium C++ ABI
 * mangling so our function overrides the right symbol. */
struct ihipModuleSymbol_t;
struct ihipStream_t;
struct ihipEvent_t;

typedef int hipError_t;
typedef ihipModuleSymbol_t* hipFunction_t;
typedef ihipStream_t*       hipStream_t;
typedef ihipEvent_t*        hipEvent_t;

/* C recording functions from hrr_trace_writer.c */
extern "C" {
  int  hrr_writer_enabled(void);
  void hrr_record_kernel_launch(const char* kernel_name,
                                 uint64_t co_hash_lo, uint64_t co_hash_hi,
                                 uint32_t gx, uint32_t gy, uint32_t gz,
                                 uint32_t bx, uint32_t by, uint32_t bz,
                                 uint32_t shared_mem,
                                 const void* stream,
                                 void** kernel_args);
  void hrr_record_kernel_launch_packed(const char* kernel_name,
                                        uint64_t co_hash_lo, uint64_t co_hash_hi,
                                        uint32_t gx, uint32_t gy, uint32_t gz,
                                        uint32_t bx, uint32_t by, uint32_t bz,
                                        uint32_t shared_mem,
                                        const void* stream,
                                        const void* packed_buf,
                                        size_t packed_size);
  const char* hrr_lookup_function_name(const void* func_handle);
  int hrr_lookup_function_co_hash(const void* func_handle,
                                   uint64_t* hash_lo, uint64_t* hash_hi);
  void hrr_record_external_launch_state(void);
}

/* dim3 binary layout matches HIP's struct {uint32_t x, y, z;} so we can
 * intercept hipLaunchKernel from C++ — the symbol's mangled name uses
 * "5dim3", which we get for free with a struct named dim3.  Note: HIP's
 * dim3 has a default constructor with default args (1,1,1); a POD struct
 * here is layout-compatible for ABI purposes (passed by value as 12 bytes
 * on x86_64). */
struct dim3 { uint32_t x; uint32_t y; uint32_t z; };

/* Real function pointer — looked up via dlsym by mangled name to avoid
 * any ambiguity. */
static hipError_t (*real_hipExtModuleLaunchKernel)(
    hipFunction_t f,
    unsigned int globalWorkSizeX, unsigned int globalWorkSizeY,
    unsigned int globalWorkSizeZ,
    unsigned int localWorkSizeX, unsigned int localWorkSizeY,
    unsigned int localWorkSizeZ,
    unsigned long sharedMemBytes,
    hipStream_t hStream,
    void** kernelParams, void** extra,
    hipEvent_t startEvent, hipEvent_t stopEvent,
    unsigned int flags) = nullptr;

static const char kExtLaunchSym[] =
    "_Z24hipExtModuleLaunchKernelP18ihipModuleSymbol_t"
    "jjjjjjmP12ihipStream_tPPvS4_P11ihipEvent_tS6_j";

/* Override of hipExtModuleLaunchKernel.
 *
 * IMPORTANT — this MUST be C linkage (extern "C").  ROCm's
 * hip_runtime_api.h declares hipExtModuleLaunchKernel inside an
 * extern "C" block, so MIOpen, rocBLAS, hipBLASLt, MIGraphX, and
 * rocroller all link against the unmangled C symbol
 * "hipExtModuleLaunchKernel@hip_4.2".  A C++ definition here would
 * produce the Itanium-mangled name and intercept nothing — exactly
 * the regression that left every MIOpen convolution out of the
 * recorded trace and produced the consistent ~2.25x scaling factor
 * in replay verification.  libamdhip64 happens to export the C++
 * mangled symbol too as a back-compat alias, but no real workload
 * uses it.  Verified with `nm -D --undefined-only` over MIOpen,
 * rocBLAS, hipBLASLt, rocroller, and migraphx_gpu — they all
 * reference the C-mangled form exclusively.
 *
 * The function still lives in this .cpp because its prototype takes
 * dim3 and HIP opaque types that are easier to spell in C++. */
extern "C" hipError_t hipExtModuleLaunchKernel(
    hipFunction_t f,
    unsigned int globalWorkSizeX, unsigned int globalWorkSizeY,
    unsigned int globalWorkSizeZ,
    unsigned int localWorkSizeX, unsigned int localWorkSizeY,
    unsigned int localWorkSizeZ,
    unsigned long sharedMemBytes,
    hipStream_t hStream,
    void** kernelParams, void** extra,
    hipEvent_t startEvent, hipEvent_t stopEvent,
    unsigned int flags) {

  if (!real_hipExtModuleLaunchKernel) {
    real_hipExtModuleLaunchKernel =
        reinterpret_cast<decltype(real_hipExtModuleLaunchKernel)>(
            dlsym(RTLD_NEXT, kExtLaunchSym));

    if (!real_hipExtModuleLaunchKernel) {
      /* RTLD_NEXT failed (early init order issue); try by name in default scope */
      real_hipExtModuleLaunchKernel =
          reinterpret_cast<decltype(real_hipExtModuleLaunchKernel)>(
              dlsym(RTLD_DEFAULT, kExtLaunchSym));
    }
  }

  if (!real_hipExtModuleLaunchKernel) return -1;

  /* CRITICAL: launch first, THEN record.
   *
   * In HRR_MODE=full, hrr_record_kernel_launch internally calls
   * capture_output_snapshots() which does hipDeviceSynchronize + D2H readback
   * to capture the post-launch buffer state.  If we record BEFORE launching,
   * the snapshot captures the PRE-launch state, which is then stored as the
   * kernel's "expected output".  On replay --verify, the live post-launch
   * state never matches that pre-launch snapshot, so every check fails with
   * cascading max_diff growth (each kernel's wrong "expected" output corrupts
   * the inputs that downstream kernels are checked against).
   *
   * The C interposer (hipModuleLaunchKernel) and Windows proxy already
   * launch-then-record; this path was the only outlier.  See the contract
   * comment in hrr_trace_writer.c above capture_output_snapshots(). */
  hipError_t ret = real_hipExtModuleLaunchKernel(
      f,
      globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
      localWorkSizeX, localWorkSizeY, localWorkSizeZ,
      sharedMemBytes, hStream,
      kernelParams, extra,
      startEvent, stopEvent, flags);

  if (hrr_writer_enabled()) {
    const void* fv = static_cast<const void*>(f);
    const char* kname = hrr_lookup_function_name(fv);
    uint64_t co_lo = 0, co_hi = 0;
    hrr_lookup_function_co_hash(fv, &co_lo, &co_hi);

    /* hipExtModuleLaunchKernel uses (globalWorkSize, localWorkSize) semantics,
     * while hipModuleLaunchKernel uses (numBlocks, blockDim). Normalize to
     * numBlocks here so the trace format is consistent with the C interposer. */
    uint32_t nbx = localWorkSizeX ? (globalWorkSizeX + localWorkSizeX - 1) / localWorkSizeX : 1;
    uint32_t nby = localWorkSizeY ? (globalWorkSizeY + localWorkSizeY - 1) / localWorkSizeY : 1;
    uint32_t nbz = localWorkSizeZ ? (globalWorkSizeZ + localWorkSizeZ - 1) / localWorkSizeZ : 1;

    if (kernelParams) {
      /* Standard kernelParams path: array of pointers to arg values */
      hrr_record_kernel_launch(kname, co_lo, co_hi,
                                nbx, nby, nbz,
                                localWorkSizeX, localWorkSizeY, localWorkSizeZ,
                                static_cast<uint32_t>(sharedMemBytes),
                                static_cast<const void*>(hStream), kernelParams);
    } else if (extra) {
      /* Packed kernarg buffer path (MIGraphX / hipExtModuleLaunchKernel).
       * extra = { HIP_LAUNCH_PARAM_BUFFER_POINTER, buf,
       *           HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
       *           HIP_LAUNCH_PARAM_END } */
      const void* packed_buf = nullptr;
      size_t packed_size = 0;
      for (int ei = 0; ; ei += 2) {
        if (extra[ei] == reinterpret_cast<void*>(0x01)) {
          packed_buf = extra[ei + 1];
        } else if (extra[ei] == reinterpret_cast<void*>(0x02)) {
          packed_size = *reinterpret_cast<const size_t*>(extra[ei + 1]);
        } else {
          break;  // HIP_LAUNCH_PARAM_END or unknown tag
        }
      }
      if (packed_buf) {
        hrr_record_kernel_launch_packed(kname, co_lo, co_hi,
                                        nbx, nby, nbz,
                                        localWorkSizeX, localWorkSizeY, localWorkSizeZ,
                                        static_cast<uint32_t>(sharedMemBytes),
                                        static_cast<const void*>(hStream),
                                        packed_buf, packed_size);
      }
    }
  }

  return ret;
}

/* hipLaunchKernel — used by HIP fat-binary stubs (e.g. MIGraphX's input
 * transpose kernel compiled with hipcc and launched via the <<<>>> syntax).
 *
 * The first argument is a host-side function-pointer "stub" baked into the
 * caller's binary by hipcc; its identity has no meaning in another process,
 * so we cannot replay this kernel by re-launching it.  Instead we capture
 * its EFFECT: after the real launch completes we synchronize the device and
 * snapshot the post-launch contents of every tracked allocation as a series
 * of synthetic H2D MEMCPY events.  On replay these MEMCPYs restore device
 * state without needing to re-execute the kernel.
 *
 * NOTE: HIP exports hipLaunchKernel as a plain (unmangled) C symbol — the
 * declaration sits inside an extern "C" block in hip_runtime_api.h, even
 * though dim3 itself is a C++ class.  We therefore intercept it with C
 * linkage and dlsym by the unmangled name.  The dim3 ABI (three uint32_t
 * passed by value) is identical between the C and C++ definitions. */
extern "C" {
static hipError_t (*real_hipLaunchKernel)(
    const void* function_address,
    dim3 numBlocks,
    dim3 dimBlocks,
    void** args,
    size_t sharedMemBytes,
    hipStream_t stream) = nullptr;

hipError_t hipLaunchKernel(
    const void* function_address,
    dim3 numBlocks,
    dim3 dimBlocks,
    void** args,
    size_t sharedMemBytes,
    hipStream_t stream) {

  if (!real_hipLaunchKernel) {
    real_hipLaunchKernel =
        reinterpret_cast<decltype(real_hipLaunchKernel)>(
            dlsym(RTLD_NEXT, "hipLaunchKernel"));
    if (!real_hipLaunchKernel) {
      real_hipLaunchKernel =
          reinterpret_cast<decltype(real_hipLaunchKernel)>(
              dlsym(RTLD_DEFAULT, "hipLaunchKernel"));
    }
  }

  if (!real_hipLaunchKernel) return -1;

  hipError_t ret = real_hipLaunchKernel(
      function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream);

  if (ret == 0 && hrr_writer_enabled()) {
    hrr_record_external_launch_state();
  }

  return ret;
}

/* Versioned-alias the two C-linkage launch entry points defined above
 * to whatever node libamdhip64 ships them at on this build host
 * (hipLaunchKernel@hip_4.2 and hipExtModuleLaunchKernel@hip_4.2 today;
 * auto-detected at configure time so a future ROCm rebump is picked
 * up without source edits).
 *
 * Same rationale as the C interposer: callers linked against the
 * versioned reference would otherwise bypass LD_PRELOAD and go
 * straight to the real implementation.  Single "@" adds a versioned
 * alias without conflicting with the existing unversioned definition.
 *
 * HRR_VER_<name> macros come from the auto-generated hrr_symver.h. */
HRR_SYMVER(hipLaunchKernel);
}  /* extern "C" */

extern "C" {
HRR_SYMVER(hipExtModuleLaunchKernel);
}

/* libmigraphx_gpu links against the C++-mangled form of the same API
 * ("_Z24hipExtModuleLaunchKernel...j@hip_4.2") because the MIGraphX
 * build forward-declares hipExtModuleLaunchKernel as a plain C++
 * function (not inside extern "C").  Export a second symbol with the
 * Itanium-mangled name as an alias of the C-linkage definition above,
 * then versioned-alias it to the same node so MIGraphX's reference
 * resolves to us as well.  __attribute__((alias)) requires the target
 * to be defined in the same translation unit, which is true here.
 *
 * The mangled string can't be a macro identifier, so we spell it out
 * verbatim and only pull the version from the generated header. */
extern "C" hipError_t hrr_hipExtModuleLaunchKernel_mangled(
    hipFunction_t, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int,
    unsigned long, hipStream_t, void**, void**,
    hipEvent_t, hipEvent_t, unsigned int)
    __asm__("_Z24hipExtModuleLaunchKernelP18ihipModuleSymbol_t"
            "jjjjjjmP12ihipStream_tPPvS4_P11ihipEvent_tS6_j")
    __attribute__((alias("hipExtModuleLaunchKernel")));

__asm__(".symver _Z24hipExtModuleLaunchKernelP18ihipModuleSymbol_t"
        "jjjjjjmP12ihipStream_tPPvS4_P11ihipEvent_tS6_j,"
        "_Z24hipExtModuleLaunchKernelP18ihipModuleSymbol_t"
        "jjjjjjmP12ihipStream_tPPvS4_P11ihipEvent_tS6_j@"
        HRR_VER_hipExtModuleLaunchKernel_mangled);
