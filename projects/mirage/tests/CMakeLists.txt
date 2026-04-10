include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# --------------------------------------------------------------------------
# Helper: add a mirage test executable and register with CTest
# --------------------------------------------------------------------------
function(mirage_add_test TEST_NAME)
    add_executable(${TEST_NAME} ${ARGN})
    target_link_libraries(${TEST_NAME} PRIVATE
        mirage
        mirage_warnings
        GTest::gtest_main
    )
    gtest_discover_tests(${TEST_NAME})
endfunction()

# --------------------------------------------------------------------------
# Unit tests
# --------------------------------------------------------------------------
mirage_add_test(mirage_types_test          unit/types_test.cpp)
mirage_add_test(mirage_time_utils_test     unit/time_utils_test.cpp)
mirage_add_test(mirage_profile_registry_test unit/profile_registry_test.cpp)
mirage_add_test(mirage_session_manager_test unit/session_manager_test.cpp)
mirage_add_test(mirage_container_test      unit/container_test.cpp)
mirage_add_test(mirage_simulator_test      unit/simulator_test.cpp)
mirage_add_test(mirage_daemon_test         unit/daemon_test.cpp)

# --------------------------------------------------------------------------
# Integration tests
# --------------------------------------------------------------------------
mirage_add_test(mirage_daemon_simulator_integration_test
    integration/daemon_simulator_test.cpp)
mirage_add_test(mirage_session_lifecycle_test
    integration/session_lifecycle_test.cpp)
mirage_add_test(mirage_full_workflow_test
    integration/full_workflow_test.cpp)
