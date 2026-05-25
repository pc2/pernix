find_package(PkgConfig)
pkg_search_module(GTEST REQUIRED gtest)
include(CheckCXXCompilerFlag)

file(GLOB
        PERNIX_ROOT_TEST_SOURCES
        CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

file(GLOB_RECURSE
        PERNIX_FALLBACK_TEST_SOURCES
        CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/fallback/*.cpp
)

set(SOURCE_FILES ${PERNIX_ROOT_TEST_SOURCES} ${PERNIX_FALLBACK_TEST_SOURCES})

if (PERNIX_SELECTED_ARCH_BACKEND STREQUAL "X86")
    file(GLOB_RECURSE
            PERNIX_X86_TEST_SOURCES
            CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/x86/*.cpp
    )
    list(APPEND SOURCE_FILES ${PERNIX_X86_TEST_SOURCES})
elseif (PERNIX_SELECTED_ARCH_BACKEND STREQUAL "ARM64_NEON")
    file(GLOB_RECURSE
            PERNIX_ARM64_NEON_TEST_SOURCES
            CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/arm64/neon/*.cpp
    )
    list(APPEND SOURCE_FILES ${PERNIX_ARM64_NEON_TEST_SOURCES})
elseif (PERNIX_SELECTED_ARCH_BACKEND STREQUAL "ARM64_SVE")
    file(GLOB_RECURSE
            PERNIX_ARM64_SVE_TEST_SOURCES
            CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/arm64/sve/*.cpp
    )
    list(APPEND SOURCE_FILES ${PERNIX_ARM64_SVE_TEST_SOURCES})
elseif (PERNIX_SELECTED_ARCH_BACKEND STREQUAL "ARM64_SVE2")
    file(GLOB_RECURSE
            PERNIX_ARM64_SVE2_TEST_SOURCES
            CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/arm64/sve2/*.cpp
    )
    list(APPEND SOURCE_FILES ${PERNIX_ARM64_SVE2_TEST_SOURCES})
endif ()

file(GLOB_RECURSE
        HEADER_FILES
        CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
)

set(PERNIX_TEST_BLOCK_SIZES 64 128 256 512 CACHE STRING "Block sizes to build as separate test executables")
set(PERNIX_TEST_TARGETS)

foreach (BLOCK_SIZE IN LISTS PERNIX_TEST_BLOCK_SIZES)
    set(TEST_TARGET pernix_tests_${BLOCK_SIZE})
    list(APPEND PERNIX_TEST_TARGETS ${TEST_TARGET})

    add_executable(${TEST_TARGET})
    target_sources(${TEST_TARGET} PRIVATE ${SOURCE_FILES} ${HEADER_FILES})
    target_link_libraries(${TEST_TARGET} PRIVATE pernix ${GTEST_LDFLAGS})
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_20)
    target_compile_options(${TEST_TARGET} PRIVATE ${GTEST_CFLAGS})
    target_compile_definitions(${TEST_TARGET} PRIVATE PERNIX_TEST_BLOCK_SIZE=${BLOCK_SIZE})
    target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endforeach ()

add_custom_target(pernix_tests DEPENDS ${PERNIX_TEST_TARGETS})

# check_cxx_compiler_flag(-O3 HAS_OPTIMIZE3_FLAG)
# if (HAS_OPTIMIZE3_FLAG)
#     target_compile_options(pernix_tests PRIVATE -O3)
# endif ()

include(GoogleTest)
foreach (BLOCK_SIZE IN LISTS PERNIX_TEST_BLOCK_SIZES)
    gtest_discover_tests(pernix_tests_${BLOCK_SIZE}
            DISCOVERY_MODE PRE_TEST
            EXTRA_ARGS --gtest_color=yes
            NO_PRETTY_TYPES
            PROPERTIES LABELS "unit;block_size_${BLOCK_SIZE}"
    )
endforeach ()

if (PERNIX_ENABLE_CODE_COVERAGE)
    message(STATUS "Code coverage enabled for tests")

    # set compile and link options for code coverage
    foreach (TEST_TARGET IN LISTS PERNIX_TEST_TARGETS)
        target_compile_options(${TEST_TARGET} PRIVATE -g -O0 --coverage)
        target_link_options(${TEST_TARGET} PRIVATE --coverage)
    endforeach ()

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
            DEPENDS ${PERNIX_TEST_TARGETS}
            COMMENT "Generating code coverage report"
            VERBATIM
    )
endif ()
