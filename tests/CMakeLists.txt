find_package(PkgConfig)
pkg_search_module(GTEST REQUIRED gtest)

file(GLOB_RECURSE
        SOURCE_FILES
        *.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
)

set_source_files_properties(
        ${SOURCE_FILES}
        PROPERTIES LANGUAGE CXX
)

add_executable(libcompression_tests)
target_sources(libcompression_tests PRIVATE ${SOURCE_FILES})
target_link_libraries(libcompression_tests PRIVATE libcompression ${GTEST_LDFLAGS})
target_compile_options(libcompression_tests PRIVATE ${GTEST_CFLAGS})
target_include_directories(libcompression_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

check_cxx_compiler_flag(-03 PRIVATE HAS_OPTIMIZE3_FLAG)
if (HAS_OPTIMIZE3_FLAG)
    target_compile_options(libcompression_tests PRIVATE -O3)
endif ()

include(GoogleTest)
gtest_discover_tests(libcompression_tests)

if (LIBCOMPRESSION_ENABLE_CODE_COVERAGE)
    message(STATUS "Code coverage enabled for tests")

    # set compile and link options for code coverage
    target_compile_options(libcompression_tests PRIVATE -g -O0 --coverage)
    target_link_options(libcompression_tests PRIVATE --coverage)

    # find lcov and genhtml
    find_program(LCOV_PROGRAM lcov)
    find_program(GENHTML_PROGRAM genhtml)

    add_custom_target(coverage
            COMMAND ${LCOV_PROGRAM} --directory ${CMAKE_BINARY_DIR} --zerocounters -q
            COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
            COMMAND ${LCOV_PROGRAM} --directory ${CMAKE_BINARY_DIR} --ignore-errors inconsistent --capture --output-file coverage.info -q
            COMMAND ${LCOV_PROGRAM} --remove coverage.info /usr/* ${CMAKE_SOURCE_DIR}/tests/* --output-file coverage.info.cleaned -q
            COMMAND ${GENHTML_PROGRAM} coverage.info.cleaned --output-directory coverage_report -q
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            DEPENDS libcompression_tests
            COMMENT "Generating code coverage report"
            VERBATIM
    )
endif ()