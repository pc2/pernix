find_package(PkgConfig)
pkg_search_module(GTEST REQUIRED gtest)
include(CheckCXXCompilerFlag)

file(GLOB_RECURSE
        SOURCE_FILES
        CONFIGURE_DEPENDS
        *.cpp
)

file(GLOB_RECURSE
        HEADER_FILES
        CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
)

add_executable(pernix_tests)
target_sources(pernix_tests PRIVATE ${SOURCE_FILES} ${HEADER_FILES})
target_link_libraries(pernix_tests PRIVATE pernix ${GTEST_LDFLAGS})
target_compile_features(pernix_tests PRIVATE cxx_std_20)
target_compile_options(pernix_tests PRIVATE ${GTEST_CFLAGS})
target_include_directories(pernix_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# check_cxx_compiler_flag(-O3 HAS_OPTIMIZE3_FLAG)
# if (HAS_OPTIMIZE3_FLAG)
#     target_compile_options(pernix_tests PRIVATE -O3)
# endif ()

include(GoogleTest)
gtest_discover_tests(pernix_tests
        DISCOVERY_MODE PRE_TEST
        EXTRA_ARGS --gtest_color=yes
        NO_PRETTY_TYPES
        PROPERTIES LABELS unit
)

if (PERNIX_ENABLE_CODE_COVERAGE)
    message(STATUS "Code coverage enabled for tests")

    # set compile and link options for code coverage
    target_compile_options(pernix_tests PRIVATE -g -O0 --coverage)
    target_link_options(pernix_tests PRIVATE --coverage)

    # find lcov and genhtml
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
            DEPENDS pernix_tests
            COMMENT "Generating code coverage report"
            VERBATIM
    )
endif ()
