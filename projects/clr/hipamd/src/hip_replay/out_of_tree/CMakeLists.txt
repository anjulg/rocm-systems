cmake_minimum_required(VERSION 3.21)
project(hip_replay_out_of_tree LANGUAGES C CXX)

# All static libs linked into shared objects need PIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# Out-of-tree HIP Record & Replay
#
# Builds:
#   Linux:   libhrr_record.so  (LD_PRELOAD interposer)
#   Windows: amdhip64_7.dll    (proxy DLL)
#   Both:    hrr-replay, hrr-bench, hrr-info tools
# ============================================================================

# --- Shared code object parser (C, no dependencies) ---
add_library(hrr_code_object STATIC hrr_code_object.c)
target_include_directories(hrr_code_object PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# --- Shared trace writer (C, no dependencies) ---
add_library(hrr_trace_writer STATIC hrr_trace_writer.c)
target_link_libraries(hrr_trace_writer PRIVATE hrr_code_object)
target_include_directories(hrr_trace_writer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(NOT WIN32)
  target_link_libraries(hrr_trace_writer PRIVATE pthread)
endif()

# --- Platform-specific recording library ---
if(WIN32)
  add_library(amdhip64_7 SHARED hrr_proxy_win.c)
  target_link_libraries(amdhip64_7 PRIVATE hrr_trace_writer hrr_code_object)
  set_target_properties(amdhip64_7 PROPERTIES
    OUTPUT_NAME "amdhip64_7"
    PREFIX "")

  # ---------------------------------------------------------------------------
  # .def file for full symbol forwarding — three ways to supply it:
  #
  #  A) Pre-generated (split build/deploy):
  #       On the remote machine (where ROCm is installed):
  #         python gen_proxy_exports.py C:\ROCm\bin\amdhip64_7.dll amdhip64_7.def
  #       Copy amdhip64_7.def to the dev machine, then:
  #         cmake -B build -S . -DHRR_DEF_FILE=C:\path\to\amdhip64_7.def
  #
  #  B) Generate locally (dev machine has a copy of the real DLL):
  #         cmake -B build -S . -DHRR_REAL_HIP_DLL=C:\ROCm\bin\amdhip64_7.dll
  #
  #  C) Neither set → minimal mode: only the ~10 proxied symbols are exported.
  #       Apps calling other HIP APIs will crash with STATUS_ENTRYPOINT_NOT_FOUND.
  # ---------------------------------------------------------------------------
  set(HRR_DEF_FILE "" CACHE FILEPATH
      "Pre-generated amdhip64_7.def (use when building on a different machine from where ROCm is installed)")
  set(HRR_REAL_HIP_DLL "" CACHE FILEPATH
      "Path to the real amdhip64_7.dll — .def is auto-generated via gen_proxy_exports.py")

  # Helper: given a proxy .def path, create a stub import lib for
  # amdhip64_7_orig.dll and link the proxy against it.
  # MSVC requires the import lib to emit PE forwarder entries; lib.exe can
  # create one from the companion .def with no actual DLL on the build machine.
  macro(_hrr_link_forwarder_def _def_file_path)
    set(_orig_def  "${CMAKE_CURRENT_BINARY_DIR}/amdhip64_7_orig.def")
    set(_orig_lib  "${CMAKE_CURRENT_BINARY_DIR}/amdhip64_7_orig.lib")

    find_program(_LIB_EXE lib HINTS
      "$ENV{VSINSTALLDIR}/VC/Tools/MSVC"
      "$ENV{VCToolsInstallDir}/bin/Hostx64/x64"
      "$ENV{ProgramFiles}/Microsoft Visual Studio"
      PATH_SUFFIXES "bin/Hostx64/x64" "bin/HostX64/x64"
    )
    if(NOT _LIB_EXE)
      # Fall back: lib.exe is usually on PATH inside a VS Developer Prompt
      set(_LIB_EXE lib)
    endif()

    # lib.exe requires the companion amdhip64_7_orig.def next to the proxy def.
    # gen_proxy_exports.py writes it there automatically.
    if(NOT EXISTS "${_orig_def}")
      message(FATAL_ERROR
        "[HRR] ${_orig_def} not found.\n"
        "      Re-run gen_proxy_exports.py — it now writes both .def files.")
    endif()

    # Each COMMAND item is a separate argument — CMake handles shell quoting
    # automatically.  Do NOT embed extra quotes inside the argument strings or
    # lib.exe will receive literal quote characters in the path.
    execute_process(
      COMMAND "${_LIB_EXE}" /nologo /machine:x64
              "/def:${_orig_def}"
              "/out:${_orig_lib}"
      RESULT_VARIABLE _lib_result
    )
    if(NOT _lib_result EQUAL 0)
      message(FATAL_ERROR "[HRR] lib.exe failed creating stub import lib (exit ${_lib_result})")
    endif()

    # Copy the macro argument into a plain variable before use — cmake_path and
    # generator expressions cannot dereference macro formal parameters directly.
    set(_proxy_def "${_def_file_path}")
    target_link_options(amdhip64_7 PRIVATE "/DEF:${_proxy_def}")
    target_link_libraries(amdhip64_7 PRIVATE "${_orig_lib}")
  endmacro()

  if(HRR_DEF_FILE)
    if(NOT EXISTS "${HRR_DEF_FILE}")
      message(FATAL_ERROR "[HRR] HRR_DEF_FILE does not exist: ${HRR_DEF_FILE}")
    endif()
    # Copy the pre-generated def (and its companion) into the build dir so
    # _hrr_link_forwarder_def can find amdhip64_7_orig.def next to it.
    get_filename_component(_def_dir "${HRR_DEF_FILE}" DIRECTORY)
    set(_def_file "${CMAKE_CURRENT_BINARY_DIR}/amdhip64_7.def")
    configure_file("${HRR_DEF_FILE}" "${_def_file}" COPYONLY)
    configure_file("${_def_dir}/amdhip64_7_orig.def"
                   "${CMAKE_CURRENT_BINARY_DIR}/amdhip64_7_orig.def" COPYONLY)
    message(STATUS "[HRR] Using pre-generated .def: ${HRR_DEF_FILE}")
    _hrr_link_forwarder_def("${_def_file}")

  elseif(HRR_REAL_HIP_DLL)
    if(NOT EXISTS "${HRR_REAL_HIP_DLL}")
      message(FATAL_ERROR "[HRR] HRR_REAL_HIP_DLL does not exist: ${HRR_REAL_HIP_DLL}")
    endif()

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(NOT Python3_FOUND)
      message(FATAL_ERROR "[HRR] Python3 not found — needed to run gen_proxy_exports.py")
    endif()

    set(_def_file "${CMAKE_CURRENT_BINARY_DIR}/amdhip64_7.def")

    execute_process(
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/gen_proxy_exports.py"
              "${HRR_REAL_HIP_DLL}"
              "${_def_file}"
      RESULT_VARIABLE _def_result
    )
    if(NOT _def_result EQUAL 0)
      message(FATAL_ERROR "[HRR] gen_proxy_exports.py failed (exit ${_def_result})")
    endif()

    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
      "${HRR_REAL_HIP_DLL}"
      "${CMAKE_CURRENT_SOURCE_DIR}/gen_proxy_exports.py"
    )

    message(STATUS "[HRR] Generated .def from ${HRR_REAL_HIP_DLL}")
    _hrr_link_forwarder_def("${_def_file}")

  else()
    message(STATUS "[HRR] Proxy DLL in minimal mode — set HRR_DEF_FILE or HRR_REAL_HIP_DLL for full forwarding")
  endif()

else()
  # Linux LD_PRELOAD interposer
  # Include the C++ extension to intercept hipExtModuleLaunchKernel (MIGraphX path)
  add_library(hrr_record SHARED hrr_interposer_linux.c hrr_interposer_cxx.cpp)
  target_link_libraries(hrr_record PRIVATE hrr_trace_writer hrr_code_object
                        dl pthread)
  set_target_properties(hrr_record PROPERTIES
    OUTPUT_NAME "hrr_record"
    PREFIX "lib")

  # ---------------------------------------------------------------------------
  # Auto-generate hrr_symver.h + hrr_record.ver from the actual
  # libamdhip64.so on this build host.
  #
  # libamdhip64 exports its public API as VERSIONED symbols
  # (hipMalloc@hip_4.2, hipModuleLaunchCooperativeKernel@hip_5.5, ...).
  # An LD_PRELOAD library that exports only the unversioned name is
  # silently bypassed by the dynamic linker — calls fall through to
  # libamdhip64 and skip our recorder.  We therefore have to emit a
  # .symver alias for every intercepted API at the EXACT version the
  # local libamdhip64 ships.
  #
  # Hardcoding versions in source worked but would break if a future
  # ROCm release republished an existing symbol at a new node (analogous
  # to glibc's memcpy@GLIBC_2.14 ABI break).  Generating from the real
  # library means we always match whatever ROCm is installed.
  #
  # Override location of the source library with -DHRR_HIP_LIB=/path
  # if the auto-discovery picks the wrong one (e.g. multiple ROCm
  # installs side by side).
  # ---------------------------------------------------------------------------
  set(HRR_HIP_LIB "" CACHE FILEPATH
      "Path to libamdhip64.so used to extract symbol versions for the LD_PRELOAD interposer (auto-detected if empty)")

  if(NOT HRR_HIP_LIB)
    # Try to resolve the actual library file behind the hip::host
    # imported target.  IMPORTED_LOCATION may be unset for hip::host
    # itself (it's an INTERFACE target on some ROCm builds), so walk
    # the link interface looking for a libamdhip64*.so file.
    find_package(hip QUIET)
    if(TARGET hip::host)
      get_target_property(_hip_lib hip::host IMPORTED_LOCATION)
      if(NOT _hip_lib)
        get_target_property(_hip_link hip::host INTERFACE_LINK_LIBRARIES)
        foreach(_l ${_hip_link})
          if(_l MATCHES "libamdhip64")
            set(_hip_lib "${_l}")
            break()
          endif()
        endforeach()
      endif()
      if(_hip_lib AND EXISTS "${_hip_lib}")
        set(HRR_HIP_LIB "${_hip_lib}")
      endif()
    endif()
  endif()

  # Final fallback: probe the conventional install path so the build
  # still works in environments where hip-config.cmake isn't installed.
  if(NOT HRR_HIP_LIB)
    foreach(_candidate
        "$ENV{ROCM_PATH}/lib/libamdhip64.so"
        "/opt/rocm/lib/libamdhip64.so"
        "/opt/rocm/lib/libamdhip64.so.7")
      if(EXISTS "${_candidate}")
        set(HRR_HIP_LIB "${_candidate}")
        break()
      endif()
    endforeach()
  endif()

  if(NOT HRR_HIP_LIB)
    message(FATAL_ERROR
      "[HRR] could not locate libamdhip64.so to extract symbol versions.\n"
      "      Pass -DHRR_HIP_LIB=/path/to/libamdhip64.so to cmake, or set\n"
      "      ROCM_PATH so /opt/rocm/lib/libamdhip64.so resolves.")
  endif()

  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  set(_hrr_symver_h    "${CMAKE_CURRENT_BINARY_DIR}/hrr_symver.h")
  set(_hrr_verscript   "${CMAKE_CURRENT_BINARY_DIR}/hrr_record.ver")
  set(_hrr_gen_script  "${CMAKE_CURRENT_SOURCE_DIR}/gen_symver.py")

  # Run at configure time so the generated header is available for
  # the C/C++ compile.  CONFIGURE_DEPENDS re-runs the generator if
  # the script or the underlying libamdhip64 changes (catches ROCm
  # upgrades that would shift symbol versions).
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${_hrr_gen_script}"
            "${HRR_HIP_LIB}"
            --out-header    "${_hrr_symver_h}"
            --out-verscript "${_hrr_verscript}"
    RESULT_VARIABLE _gen_result
    OUTPUT_VARIABLE _gen_output
    ERROR_VARIABLE  _gen_output
  )
  if(NOT _gen_result EQUAL 0)
    message(FATAL_ERROR "[HRR] gen_symver.py failed (exit ${_gen_result}):\n${_gen_output}")
  endif()
  message(STATUS "[HRR] Generated symbol-version table from ${HRR_HIP_LIB}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_hrr_gen_script}"
    "${HRR_HIP_LIB}"
  )

  # Generated header lives in the build dir; expose it on the include
  # path for both the C and C++ TUs that make up libhrr_record.
  target_include_directories(hrr_record PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

  # Pass the generated version script so .symver directives in
  # hrr_symver.h can attach versioned aliases to our exported
  # functions.  Without this, callers linked against versioned HIP
  # references bind straight to libamdhip64 and skip our LD_PRELOAD
  # interception, leaving MIOpen/MIGraphX kernel launches and memsets
  # out of the trace.
  target_link_options(hrr_record PRIVATE
    "LINKER:--version-script=${_hrr_verscript}")
  set_property(TARGET hrr_record APPEND PROPERTY LINK_DEPENDS
    "${_hrr_verscript}")
endif()

# --- Tools (require HIP + shared sources in parent directory) ---
# When deployed standalone (out_of_tree/ only), the ../hrr_reader.cpp etc. won't
# exist. In that case, skip building the tools here — use HRR_BENCH_PATH to point
# at a pre-built hrr-bench for the integration test.
find_package(hip QUIET)
set(_hrr_parent_src "${CMAKE_CURRENT_SOURCE_DIR}/..")
if(hip_FOUND AND EXISTS "${_hrr_parent_src}/hrr_reader.cpp")
  # Archive reader (shared by tools)
  add_library(hrr_reader STATIC ${_hrr_parent_src}/hrr_reader.cpp)
  target_include_directories(hrr_reader PUBLIC ${_hrr_parent_src})
  target_compile_features(hrr_reader PUBLIC cxx_std_17)

  add_executable(hrr-replay ${_hrr_parent_src}/hrr_replay.cpp)
  target_link_libraries(hrr-replay PRIVATE hrr_reader hip::host)
  target_compile_features(hrr-replay PRIVATE cxx_std_17)

  add_executable(hrr-bench ${_hrr_parent_src}/hrr_bench.cpp)
  target_link_libraries(hrr-bench PRIVATE hrr_reader hip::host)
  target_compile_features(hrr-bench PRIVATE cxx_std_17)

  add_executable(hrr-info ${_hrr_parent_src}/hrr_info.cpp)
  target_link_libraries(hrr-info PRIVATE hrr_reader)
  target_compile_features(hrr-info PRIVATE cxx_std_17)

  install(TARGETS hrr-replay hrr-bench hrr-info RUNTIME DESTINATION bin)
elseif(NOT hip_FOUND)
  message(STATUS "[HRR] HIP not found - building recording library only (no tools)")
else()
  message(STATUS "[HRR] Standalone deploy (no parent sources) - building recording library only (no tools)")
endif()

# Install recording library
if(WIN32)
  install(TARGETS amdhip64_7 RUNTIME DESTINATION bin)
else()
  install(TARGETS hrr_record LIBRARY DESTINATION lib)
endif()

# ============================================================================
# MIGraphX / ONNX replay integration tests
#
# Run with: ctest --test-dir <build_dir> -L migraphx -V
#
# The test skips gracefully (exit 0) when migraphx, onnx, or the built
# binaries are not available — it only fails on actual GPU faults or
# unexpected hrr-bench errors.
#
# To point at an hrr-bench built elsewhere:
#   cmake -DHRR_BENCH_PATH=/path/to/hrr-bench ...
# ============================================================================
find_package(Python3 COMPONENTS Interpreter QUIET)

if(Python3_FOUND)
  enable_testing()

  # Prefer the locally-built hrr-bench; fall back to an explicit path.
  if(TARGET hrr-bench)
    set(_hrr_bench "$<TARGET_FILE:hrr-bench>")
  elseif(DEFINED HRR_BENCH_PATH AND EXISTS "${HRR_BENCH_PATH}")
    set(_hrr_bench "${HRR_BENCH_PATH}")
  else()
    set(_hrr_bench "hrr-bench-NOT-FOUND")
  endif()

  if(WIN32)
    set(_libhrr "$<TARGET_FILE:amdhip64_7>")
  else()
    set(_libhrr "$<TARGET_FILE:hrr_record>")
  endif()

  add_test(
    NAME hrr_migraphx_onnx_replay
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test_migraphx_replay.py
      --hrr-bench     ${_hrr_bench}
      --libhrr        ${_libhrr}
      --record-script ${CMAKE_CURRENT_SOURCE_DIR}/record_model.py
      --iterations    20
      --warmup        5
  )
  set_tests_properties(hrr_migraphx_onnx_replay PROPERTIES
    TIMEOUT 600
    LABELS  "hrr;migraphx;integration"
  )
endif()
