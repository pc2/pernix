find_package(PkgConfig)
pkg_search_module(GTEST REQUIRED gtest)

file(GLOB PERNIX_TEST_SOURCE_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
list(FILTER PERNIX_TEST_SOURCE_FILES EXCLUDE REGEX ".*/header_only_tests\\.cpp$")

set(PERNIX_TEST_BLOCK_SIZES 64 128 256 512 CACHE STRING "Block sizes to build as separate test executables")
set(PERNIX_TEST_TARGETS)

foreach (BLOCK_SIZE IN LISTS PERNIX_TEST_BLOCK_SIZES)
    set(TEST_TARGET pernix_tests_${BLOCK_SIZE})
    list(APPEND PERNIX_TEST_TARGETS ${TEST_TARGET})

    add_executable(${TEST_TARGET})
    target_sources(${TEST_TARGET} PRIVATE ${PERNIX_TEST_SOURCE_FILES})
    target_link_libraries(${TEST_TARGET} PRIVATE pernix ${GTEST_LDFLAGS})
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_20)
    target_compile_options(${TEST_TARGET} PRIVATE ${GTEST_CFLAGS})
    target_compile_definitions(${TEST_TARGET} PRIVATE PERNIX_TEST_BLOCK_SIZE=${BLOCK_SIZE})
    target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    set(HEADER_ONLY_TARGET pernix_header_only_tests_${BLOCK_SIZE})
    list(APPEND PERNIX_TEST_TARGETS ${HEADER_ONLY_TARGET})
    add_executable(${HEADER_ONLY_TARGET} header_only_tests.cpp main.cpp)
    target_link_libraries(${HEADER_ONLY_TARGET} PRIVATE pernix_header_only ${GTEST_LDFLAGS})
    target_compile_features(${HEADER_ONLY_TARGET} PRIVATE cxx_std_20)
    target_compile_options(${HEADER_ONLY_TARGET} PRIVATE ${GTEST_CFLAGS})
    target_compile_definitions(${HEADER_ONLY_TARGET} PRIVATE PERNIX_TEST_BLOCK_SIZE=${BLOCK_SIZE})
    target_include_directories(${HEADER_ONLY_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endforeach ()

add_custom_target(pernix_tests DEPENDS ${PERNIX_TEST_TARGETS})

add_executable(pernix_c_api_roundtrip c_api_roundtrip.c)
target_link_libraries(pernix_c_api_roundtrip PRIVATE pernix)
target_include_directories(pernix_c_api_roundtrip PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME pernix_c_api_roundtrip COMMAND pernix_c_api_roundtrip)

add_executable(pernix_c_api_validation c_api_validation.c)
target_link_libraries(pernix_c_api_validation PRIVATE pernix)
target_include_directories(pernix_c_api_validation PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME pernix_c_api_validation COMMAND pernix_c_api_validation)

include(GoogleTest)
foreach (BLOCK_SIZE IN LISTS PERNIX_TEST_BLOCK_SIZES)
    gtest_discover_tests(pernix_tests_${BLOCK_SIZE}
            DISCOVERY_MODE PRE_TEST
            EXTRA_ARGS --gtest_color=yes
            NO_PRETTY_TYPES
            PROPERTIES LABELS "unit;block_size_${BLOCK_SIZE}"
    )
    gtest_discover_tests(pernix_header_only_tests_${BLOCK_SIZE}
            DISCOVERY_MODE PRE_TEST
            EXTRA_ARGS --gtest_color=yes
            PROPERTIES LABELS "unit;header_only;block_size_${BLOCK_SIZE}"
    )
endforeach ()

if (PERNIX_ENABLE_CODE_COVERAGE)
    message(STATUS "Code coverage enabled for tests")

    foreach (TEST_TARGET IN LISTS PERNIX_TEST_TARGETS)
        target_compile_options(${TEST_TARGET} PRIVATE -g -O0 --coverage)
        target_link_options(${TEST_TARGET} PRIVATE --coverage)
    endforeach ()

    find_program(LCOV_PROGRAM lcov)
    find_program(GENHTML_PROGRAM genhtml)
    if (NOT LCOV_PROGRAM OR NOT GENHTML_PROGRAM)
        message(FATAL_ERROR "Code coverage requires both lcov and genhtml")
    endif ()

    add_custom_target(coverage
            COMMAND ${LCOV_PROGRAM} --directory ${CMAKE_BINARY_DIR} --zerocounters -q
            COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
            COMMAND ${LCOV_PROGRAM} --directory ${CMAKE_BINARY_DIR} --ignore-errors inconsistent --capture --output-file coverage.info -q
            COMMAND ${LCOV_PROGRAM} --remove coverage.info /usr/* ${CMAKE_SOURCE_DIR}/tests/* --output-file coverage.info.cleaned -q
            COMMAND ${GENHTML_PROGRAM} coverage.info.cleaned --output-directory coverage_report -q
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            DEPENDS ${PERNIX_TEST_TARGETS}
            COMMENT "Generating code coverage report"
            VERBATIM
    )
endif ()
