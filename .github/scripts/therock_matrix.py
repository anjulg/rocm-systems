"""
This dictionary is used to map specific file directory changes to the corresponding build flag and tests
"""
subtree_to_project_map = {
    "projects/amdsmi": "core",
    "projects/aqlprofile": "profiler",
    "projects/clr": "runtimes",
    "projects/cuid": "rdc",
    "projects/hip": "runtimes",
    "projects/hip-tests": "runtimes",
    "projects/hipother": "runtimes",
    "projects/rdc": "dc_tools",
    "projects/rocdbgapi": "debug_tools-dbgapi",
    # "projects/rocdecode": "media-libs",
    # "projects/rocjpeg": "media-libs",
    "projects/rocm-core": "core",
    "projects/rocminfo": "core",
    "projects/rocm-smi-lib": "core",
    "projects/rocprofiler": "profiler",
    "projects/rocprofiler-compute": "profiler",
    "projects/rocprofiler-register": "profiler",
    "projects/rocprofiler-sdk": "profiler",
    "projects/rocprofiler-systems": "profiler",
    "projects/rocr-debug-agent": "debug_tools-debug-agent",
    "projects/hotswap": "runtimes",
    "projects/rocr-runtime": "runtimes",
    "projects/rocshmem": "rocshmem",
    "projects/roctracer": "profiler",
    "shared/amdgpu-windows-interop": "runtimes",
}

# Common Python executable options for all builds.
common_python_options = [
    "-DTHEROCK_SHARED_PYTHON_EXECUTABLES=/opt/python-shared/cp310-cp310/bin/python3;/opt/python-shared/cp311-cp311/bin/python3;/opt/python-shared/cp312-cp312/bin/python3;/opt/python-shared/cp313-cp313/bin/python3;/opt/python-shared/cp314-cp314/bin/python3",
    "-DTHEROCK_DIST_PYTHON_EXECUTABLES=/opt/python/cp310-cp310/bin/python;/opt/python/cp311-cp311/bin/python;/opt/python/cp312-cp312/bin/python;/opt/python/cp313-cp313/bin/python",
]

project_map = {
    "core": {
        "cmake_options": ["-DTHEROCK_ENABLE_CORE=ON", "-DTHEROCK_ENABLE_ALL=OFF"]
        + common_python_options,
        "projects_to_test": "",  # will run sanity test to cover rocminfo and amdsmi
    },
    "dc_tools": {
        "cmake_options": ["-DTHEROCK_ENABLE_ALL=OFF", "-DTHEROCK_ENABLE_DC_TOOLS=ON"]
        + common_python_options,
        "projects_to_test": "",  # rdc-tests is not built by TheRock build system - TBD
    },
    # dbgapi changes need to exercise both ROCgdb and debug agent.
    "debug_tools-dbgapi": {
        "cmake_options": [
            "-DTHEROCK_ENABLE_ALL=OFF",
            "-DTHEROCK_ENABLE_DEBUG_TOOLS=ON",
        ]
        + common_python_options,
        "projects_to_test": "rocr-debug-agent, rocgdb",
    },
    # debug agent changes don't have to exercise ROCgdb.
    "debug_tools-debug-agent": {
        "cmake_options": [
            "-DTHEROCK_ENABLE_ALL=OFF",
            "-DTHEROCK_ENABLE_DEBUG_TOOLS=ON",
        ]
        + common_python_options,
        "projects_to_test": "rocr-debug-agent",
    },
    # media libs to be enabled in following PR
    # "media-libs": {
    #     "cmake_options": ["-DTHEROCK_ENABLE_ALL=OFF", "-DTHEROCK_ENABLE_PROFILER=ON", "-DTHEROCK_ENABLE_MEDIA_LIBS=ON"],
    #     "projects_to_test": "", # "rocdecode-tests, rocjpeg-tests",
    # },
    "profiler": {
        "cmake_options": ["-DTHEROCK_ENABLE_ALL=ON"] + common_python_options,
        "projects_to_test": "aqlprofile, rocprofiler-compute, rocprofiler-sdk, rocprofiler-systems",
    },
    "rocshmem": {
        "cmake_options": ["-DTHEROCK_ENABLE_ALL=OFF", "-DTHEROCK_ENABLE_ROCSHMEM=ON"]
        + common_python_options,
        "projects_to_test": "",  # rocshmem testing to be enabled in a future PR
    },
    "runtimes": {
        "cmake_options": ["-DTHEROCK_ENABLE_ALL=ON"] + common_python_options,
        "projects_to_test": "hip-tests, rocrtst",
    },
    "all": {
        "cmake_options": ["-DTHEROCK_ENABLE_ALL=ON"] + common_python_options,
        "projects_to_test": "hip-tests, rocrtst, aqlprofile, rocprofiler-compute, rocprofiler-sdk, rocprofiler-systems, rocr-debug-agent, rocgdb",
    },
}

# Subtrees that should only trigger Windows CI, not Linux CI.
# Note: Linux-only subtrees (e.g. projects/rocshmem) have no explicit list —
# any subtree absent from trigger_windows_ci_for_subtrees_paths will
# automatically skip Windows CI.
windows_only_subtrees = {
    "shared/amdgpu-windows-interop",
}

# Paths matching any of these patterns will trigger Windows CI.
# Subtrees not represented here are treated as Linux-only.
trigger_windows_ci_for_subtrees_paths = [
    "projects/clr/*",
    "projects/hip/*",
    "projects/hip-tests/*",
    "projects/rocr-runtime/*",
    "shared/amdgpu-windows-interop/**",
    ".github/*/therock*",
]
