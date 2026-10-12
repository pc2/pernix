find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

set(PERNIX_TEST_REQUIRED_BACKENDS "" CACHE STRING "Comma-separated backends required by the compiled test suite")

set(PERNIX_TEST_SOURCE_FILES
        deterministic_correctness_tests.cpp
        fallback_tests.cpp
        pack_tables_tests.cpp
        simd_tests.cpp
        unpack_tables_tests.cpp
        codec_matrix_tests.cpp
)

add_executable(pernix_tests ${PERNIX_TEST_SOURCE_FILES})
target_link_libraries(pernix_tests PRIVATE pernix GTest::gtest_main)
target_compile_features(pernix_tests PRIVATE cxx_std_20)
target_compile_definitions(pernix_tests PRIVATE PERNIX_TEST_BLOCK_SIZE=64)
target_include_directories(pernix_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_executable(pernix_header_only_tests header_only_tests.cpp codec_matrix_tests.cpp)
target_link_libraries(pernix_header_only_tests PRIVATE pernix_header_only GTest::gtest_main)
target_compile_features(pernix_header_only_tests PRIVATE cxx_std_20)
target_compile_definitions(pernix_header_only_tests PRIVATE PERNIX_TEST_BLOCK_SIZE=64)
target_include_directories(pernix_header_only_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_executable(pernix_c_api_roundtrip c_api_roundtrip.c)
target_link_libraries(pernix_c_api_roundtrip PRIVATE pernix)
target_compile_features(pernix_c_api_roundtrip PRIVATE c_std_11)
target_include_directories(pernix_c_api_roundtrip PRIVATE "${PROJECT_SOURCE_DIR}/include")
add_test(NAME pernix_c_api_roundtrip COMMAND pernix_c_api_roundtrip)
set_tests_properties(pernix_c_api_roundtrip PROPERTIES LABELS "contract;c-api")

add_executable(pernix_c_api_validation c_api_validation.c)
target_link_libraries(pernix_c_api_validation PRIVATE pernix)
target_compile_features(pernix_c_api_validation PRIVATE c_std_11)
target_include_directories(pernix_c_api_validation PRIVATE "${PROJECT_SOURCE_DIR}/include")
add_test(NAME pernix_c_api_validation COMMAND pernix_c_api_validation)
set_tests_properties(pernix_c_api_validation PROPERTIES LABELS "contract;c-api")

add_executable(pernix_c_api_matrix c_api_matrix.c)
target_link_libraries(pernix_c_api_matrix PRIVATE pernix)
target_compile_features(pernix_c_api_matrix PRIVATE c_std_11)
target_include_directories(pernix_c_api_matrix PRIVATE "${PROJECT_SOURCE_DIR}/include")
add_test(NAME pernix_c_api_matrix COMMAND pernix_c_api_matrix)
set_tests_properties(pernix_c_api_matrix PROPERTIES LABELS "contract;c-api;matrix")

if (PERNIX_ENABLE_INSTALL_CONSUMER_TESTS)
    add_test(NAME pernix_install_consumer
            COMMAND ${CMAKE_COMMAND}
            -DPERNIX_BUILD_DIR=${PROJECT_BINARY_DIR}
            -DPERNIX_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/install_consumer
            -DPERNIX_CONSUMER_BUILD_DIR=${PROJECT_BINARY_DIR}/install-consumer-build
            -DPERNIX_INSTALL_PREFIX=${PROJECT_BINARY_DIR}/install-consumer-prefix
            -DPERNIX_TEST_CONFIG=$<CONFIG>
            -DPERNIX_C_COMPILER=${CMAKE_C_COMPILER}
            -DPERNIX_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/install_consumer/run.cmake
    )
    set_tests_properties(pernix_install_consumer PROPERTIES LABELS "integration;install")
endif ()

set(PERNIX_TEST_TARGETS
        pernix
        pernix_tests
        pernix_header_only_tests
        pernix_c_api_roundtrip
        pernix_c_api_validation
        pernix_c_api_matrix
)

set(PERNIX_REQUIRED_BACKEND_PROPERTIES)
if (PERNIX_TEST_REQUIRED_BACKENDS)
    list(APPEND PERNIX_REQUIRED_BACKEND_PROPERTIES
            ENVIRONMENT "PERNIX_TEST_REQUIRED_BACKENDS=${PERNIX_TEST_REQUIRED_BACKENDS}")
endif ()

gtest_discover_tests(pernix_tests
        DISCOVERY_MODE PRE_TEST
        EXTRA_ARGS --gtest_color=yes
        NO_PRETTY_TYPES
        PROPERTIES LABELS "unit;compiled" ${PERNIX_REQUIRED_BACKEND_PROPERTIES}
)

gtest_discover_tests(pernix_header_only_tests
        DISCOVERY_MODE PRE_TEST
        EXTRA_ARGS --gtest_color=yes
        PROPERTIES LABELS "unit;header_only"
)

if (PERNIX_ENABLE_CODE_COVERAGE)
    if (NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" OR NOT CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR "PERNIX_ENABLE_CODE_COVERAGE requires GCC or Clang for both C and C++")
    endif ()

    find_program(LCOV_PROGRAM lcov REQUIRED)
    find_program(GENHTML_PROGRAM genhtml REQUIRED)

    foreach (TEST_TARGET IN LISTS PERNIX_TEST_TARGETS)
        if (TARGET ${TEST_TARGET})
            if (NOT "${TEST_TARGET}" STREQUAL "pernix_header_only_tests")
                target_compile_options(${TEST_TARGET} PRIVATE -g -O0 --coverage)
                target_link_options(${TEST_TARGET} PRIVATE --coverage)
            endif ()
        endif ()
    endforeach ()

    target_compile_options(pernix_header_only INTERFACE -g -O0 --coverage)
    target_link_options(pernix_header_only INTERFACE --coverage)

    add_custom_target(coverage
            COMMAND ${LCOV_PROGRAM} --directory ${PROJECT_BINARY_DIR} --zerocounters -q
            COMMAND ${CMAKE_CTEST_COMMAND} --test-dir ${PROJECT_BINARY_DIR} --output-on-failure --no-tests=error
            COMMAND ${LCOV_PROGRAM} --directory ${PROJECT_BINARY_DIR} --ignore-errors inconsistent --capture --output-file coverage.info -q
            COMMAND ${LCOV_PROGRAM} --remove coverage.info "/usr/*" "${PROJECT_SOURCE_DIR}/tests/*"
                    --output-file coverage.info.cleaned -q
            COMMAND ${GENHTML_PROGRAM} coverage.info.cleaned --output-directory coverage_report -q
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
            DEPENDS ${PERNIX_TEST_TARGETS}
            COMMENT "Generating Pernix code coverage report"
            VERBATIM
    )
endif ()
