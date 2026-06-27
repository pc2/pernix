cmake_minimum_required(VERSION 3.24)
project(pernix VERSION 0.1.0 LANGUAGES CXX C)

option(PERNIX_ENABLE_TESTS "Enable tests for pernix" on)
option(PERNIX_ENABLE_CODE_COVERAGE "Enable code coverage for pernix (requires lcov/genhtml)" off)
option(PERNIX_ENABLE_LTO "Enable Link Time Optimization (LTO)" off)
option(PERNIX_ENABLE_INSTALL "Enable installation of pernix" on)
option(PERNIX_ENABLE_DOXYGEN "Build documentation with Doxygen." off)
option(PERNIX_INSTALL_DOCS "Install documentation with pernix" off)

option(PERNIX_ENABLE_X86_BMI2 "Build x86 BMI2 backend" ON)
option(PERNIX_ENABLE_X86_AVX2 "Build x86 AVX2 backend" ON)
option(PERNIX_ENABLE_X86_AVX512VBMI "Build x86 AVX512-VBMI backend" ON)
option(PERNIX_ENABLE_ARM64_NEON "Build arm64 NEON backend" ON)
option(PERNIX_ENABLE_ARM64_SVE2 "Build arm64 SVE2 backend" ON)
option(PERNIX_ENABLE_FALLBACK_STDPAR "Build fallback stdpar skeleton backend" OFF)
option(PERNIX_ENABLE_FALLBACK_SIMD "Build fallback simd skeleton backend" OFF)
option(PERNIX_ENABLE_FALLBACK_STDPAR_NVCC "Enable NVCC-oriented stdpar skeleton wiring" OFF)

option(PERNIX_USE_SIMDE "Use SIMDe library for portable SIMD support" off)

option(PERNIX_ENABLE_FORTRAN_BINDINGS "Build Fortran bindings for pernix" off)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

#[===[
set(PERNIX_BUNDLE_SIMDE_FOR_INSTALL OFF)
if (PERNIX_USE_SIMDE)
    if (PERNIX_SIMDE_PROVIDER STREQUAL "AUTO" OR PERNIX_SIMDE_PROVIDER STREQUAL "PACKAGE")
        find_package(simde CONFIG QUIET)
    endif ()

    if (NOT TARGET simde::simde AND (PERNIX_SIMDE_PROVIDER STREQUAL "AUTO" OR PERNIX_SIMDE_PROVIDER STREQUAL "FETCH"))
        include(FetchContent)
        set(SIMDE_TEST_CMAKE_PACKAGING OFF CACHE BOOL "Test SIMDe CMake packaging" FORCE)
        FetchContent_Declare(
                simde
                GIT_REPOSITORY https://github.com/simd-everywhere/simde.git
                GIT_TAG f3e8262173b7089db9a9d57a9ecef8dd07ad9c97
                GIT_PROGRESS TRUE
                EXCLUDE_FROM_ALL
        )
        FetchContent_MakeAvailable(simde)
        set(PERNIX_BUNDLE_SIMDE_FOR_INSTALL ON)
    endif ()

    if (NOT TARGET simde::simde AND DEFINED simde_SOURCE_DIR AND EXISTS "${simde_SOURCE_DIR}/simde")
        add_library(simde::simde INTERFACE IMPORTED GLOBAL)
        target_include_directories(simde::simde INTERFACE "${simde_SOURCE_DIR}")
    endif ()

    if (NOT TARGET simde::simde)
        message(FATAL_ERROR "PERNIX_USE_SIMDE is enabled, but simde::simde was not found. Set PERNIX_SIMDE_PROVIDER=FETCH or install SIMDe's CMake package.")
    endif ()
endif ()
]===]

include(CTest)
include(GitVersion)
get_git_version(GIT_VERSION)

if ("${GIT_VERSION}" STREQUAL "v0.0.0")
    set(VERSION "${pernix_VERSION}")
else ()
    set(VERSION "${GIT_VERSION}")
endif ()

string(STRIP ${VERSION} VERSION)
if (VERSION MATCHES v[^-]*-)
    string(REGEX REPLACE "v([^-]*)-([0-9]+)-.*" "\\1.\\2" NORMALIZED_VERSION ${VERSION})
else ()
    string(REGEX REPLACE "v(.*)" "\\1" NORMALIZED_VERSION ${VERSION})
endif ()
message(STATUS "Pernix version: ${VERSION}, normalized to ${NORMALIZED_VERSION}")

if (NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|Intel")
    message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif ()

include(CheckCXXCompilerFlag)
set(PERNIX_PRIVATE_COMPILE_OPTIONS)
foreach (PERNIX_CXX_FLAG
        -Wall
        -Wextra
        -Wshadow
        -Wfloat-equal
        -Wold-style-cast
        -Wconversion
        -fstrict-aliasing
        -Wno-ignored-attributes
)
    string(MAKE_C_IDENTIFIER "PERNIX_HAS_CXX_FLAG_${PERNIX_CXX_FLAG}" PERNIX_CXX_FLAG_VARIABLE)
    check_cxx_compiler_flag("${PERNIX_CXX_FLAG}" "${PERNIX_CXX_FLAG_VARIABLE}")
    if (${PERNIX_CXX_FLAG_VARIABLE})
        list(APPEND PERNIX_PRIVATE_COMPILE_OPTIONS "${PERNIX_CXX_FLAG}")
    else ()
        message(STATUS "Compiler flag not supported: ${PERNIX_CXX_FLAG}")
    endif ()
endforeach ()

if (PERNIX_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT PERNIX_IPO_SUPPORTED OUTPUT PERNIX_IPO_ERROR)
    if (NOT PERNIX_IPO_SUPPORTED)
        message(FATAL_ERROR "PERNIX_ENABLE_LTO is enabled, but IPO/LTO is not supported: ${PERNIX_IPO_ERROR}")
    endif ()

    check_cxx_compiler_flag("-Wno-lto-type-mismatch" PERNIX_HAS_CXX_FLAG_WNO_LTO_TYPE_MISMATCH)
    if (PERNIX_HAS_CXX_FLAG_WNO_LTO_TYPE_MISMATCH)
        list(APPEND PERNIX_PRIVATE_COMPILE_OPTIONS "-Wno-lto-type-mismatch")
    endif ()

    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        find_program(GCC_AR gcc-ar)
        if (GCC_AR)
            set(CMAKE_AR ${GCC_AR})
        endif ()
        find_program(GCC_RANLIB gcc-ranlib)
        if (GCC_RANLIB)
            set(CMAKE_RANLIB ${GCC_RANLIB})
        endif ()
    elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        find_program(LLVM_AR llvm-ar)
        if (LLVM_AR)
            set(CMAKE_AR ${LLVM_AR})
        endif ()
        find_program(LLVM_RANLIB llvm-ranlib)
        if (LLVM_RANLIB)
            set(CMAKE_RANLIB ${LLVM_RANLIB})
        endif ()
    endif ()
endif ()

set(PERNIX_TARGET_IS_X86 OFF)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i[3-6]86|i686)$")
    set(PERNIX_TARGET_IS_X86 ON)
endif ()

set(PERNIX_TARGET_IS_ARM64 OFF)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(PERNIX_TARGET_IS_ARM64 ON)
endif ()

add_subdirectory(src)

if (PERNIX_ENABLE_FORTRAN_BINDINGS)
    message(STATUS "Building Fortran bindings for pernix")
    include(CMakeAddFortranSubdirectory)
    cmake_add_fortran_subdirectory(bindings/fortran NO_EXTERNAL_INSTALL)
endif ()

if (PERNIX_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()
