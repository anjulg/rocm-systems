cmake_minimum_required(VERSION 3.28)
project(mirage LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------
option(MIRAGE_BUILD_TESTS "Build mirage tests" ON)

# --------------------------------------------------------------------------
# Compiler warnings
# --------------------------------------------------------------------------
add_library(mirage_warnings INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mirage_warnings INTERFACE
        -Wall -Wextra -Wpedantic -Werror)
elseif(MSVC)
    target_compile_options(mirage_warnings INTERFACE /W4 /WX)
endif()

# --------------------------------------------------------------------------
# Library
# --------------------------------------------------------------------------
add_subdirectory(lib)

# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
if(MIRAGE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
