# cmake/DeviceCompileDefs.cmake
#
# Single source of truth for compile definitions needed by device code.
# Consumed by both:
#   - src/CMakeLists.txt      (via target_compile_definitions on the rccl target)
#   - cmake/DeviceLinker.cmake (via raw -D flags on add_custom_command)
#
# Populates: RCCL_DEVICE_COMPILE_DEFS  (list of NAME or NAME=VALUE entries,
#            without the -D prefix).

set(RCCL_DEVICE_COMPILE_DEFS "")

if(COLLTRACE)
  list(APPEND RCCL_DEVICE_COMPILE_DEFS ENABLE_COLLTRACE)
endif()
if(FAULT_INJECTION)
  list(APPEND RCCL_DEVICE_COMPILE_DEFS ENABLE_FAULT_INJECTION)
endif()
if(LL128_ENABLED)
  list(APPEND RCCL_DEVICE_COMPILE_DEFS ENABLE_LL128)
endif()
if(HIP_CONTIGUOUS_MEMORY)
  list(APPEND RCCL_DEVICE_COMPILE_DEFS HIP_CONTIGUOUS_MEMORY)
endif()
if(ENABLE_WARP_SPEED)
  list(APPEND RCCL_DEVICE_COMPILE_DEFS ENABLE_WARP_SPEED)
endif()
if(PROFILE)
  list(APPEND RCCL_DEVICE_COMPILE_DEFS ENABLE_PROFILING)
endif()

list(APPEND RCCL_DEVICE_COMPILE_DEFS
  FMT_HEADER_ONLY=1
  NCCL_MAJOR=${NCCL_MAJOR}
  NCCL_MINOR=${NCCL_MINOR}
  NCCL_PATCH=${NCCL_PATCH}
  NCCL_VERSION_CODE=${NCCL_VERSION}
  ROCM_VERSION=${ROCM_VERSION}
  __HIP_PLATFORM_AMD__=1
)

if("${hip_version_string}" VERSION_GREATER_EQUAL "5.7.31920")
  list(APPEND RCCL_DEVICE_COMPILE_DEFS HIP_UNCACHED_MEMORY)
endif()
if(HIP_HOST_UNCACHED_MEMORY)
  list(APPEND RCCL_DEVICE_COMPILE_DEFS HIP_HOST_UNCACHED_MEMORY)
endif()
