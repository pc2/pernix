cmake_minimum_required(VERSION 3.30)
project(pernix VERSION 0.1.0 LANGUAGES CXX C)

option(PERNIX_ENABLE_TESTS "Enable tests for pernix" on)
option(PERNIX_ENABLE_CODE_COVERAGE "Enable code coverage for pernix (requires lcov/genhtml)" off)
option(PERNIX_ENABLE_LTO "Enable Link Time Optimization (LTO)" off)
option(PERNIX_ENABLE_INSTALL "Enable installation of pernix" on)
option(PERNIX_ENABLE_DOXYGEN "Build documentation with Doxygen." off)
option(PERNIX_INSTALL_DOCS "Install documentation with pernix" off)

option(PERNIX_DISABLE_BMI2 "Disable BMI2 optimizations" off)
option(PERNIX_DISABLE_AVX2 "Disable AVX2 optimizations" off)
option(PERNIX_DISABLE_AVX512 "Disable AVX512 optimizations" off)

option(PERNIX_USE_SIMDE "Use SIMDe library for portable SIMD support" off)
set(PERNIX_ARCH_BACKEND "AUTO" CACHE STRING "Pernix architecture backend (AUTO, FALLBACK, X86, ARM64_NEON, ARM64_SVE, ARM64_SVE2)")
set_property(CACHE PERNIX_ARCH_BACKEND PROPERTY STRINGS AUTO FALLBACK X86 ARM64_NEON ARM64_SVE ARM64_SVE2)

option(PERNIX_ENABLE_FORTRAN_BINDINGS "Build Fortran bindings for pernix" off)

string(TOUPPER "${PERNIX_ARCH_BACKEND}" PERNIX_ARCH_BACKEND)
set(PERNIX_VALID_ARCH_BACKENDS AUTO FALLBACK X86 ARM64_NEON ARM64_SVE ARM64_SVE2)
if (NOT PERNIX_ARCH_BACKEND IN_LIST PERNIX_VALID_ARCH_BACKENDS)
    message(FATAL_ERROR "Unsupported PERNIX_ARCH_BACKEND='${PERNIX_ARCH_BACKEND}'. Expected one of: ${PERNIX_VALID_ARCH_BACKENDS}")
endif ()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

if (PERNIX_USE_SIMDE)
    include(FetchContent)
    FetchContent_Declare(
            simde
            GIT_REPOSITORY https://github.com/simd-everywhere/simde.git
            GIT_TAG master
            GIT_SHALLOW TRUE
            GIT_PROGRESS TRUE
            EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(simde)

    if (NOT TARGET simde::simde AND DEFINED simde_SOURCE_DIR AND EXISTS "${simde_SOURCE_DIR}/simde")
        add_library(simde::simde INTERFACE IMPORTED GLOBAL)
        target_include_directories(simde::simde INTERFACE "${simde_SOURCE_DIR}")
    endif ()
endif ()

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

set(BENCHMARK_CXX_STANDARD 20)

set(CMAKE_CXX_STANDARD ${BENCHMARK_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)

include(AddCXXCompilerFlag)
if (MSVC)
    message(FATAL_ERROR "MSVC compiler is not supported")
else ()
    add_cxx_compiler_flag(-Wall)
    add_cxx_compiler_flag(-Wextra)
    add_cxx_compiler_flag(-Wshadow)
    add_cxx_compiler_flag(-Wfloat-equal)
    add_cxx_compiler_flag(-Wold-style-cast)
    add_cxx_compiler_flag(-Wconversion)
    add_cxx_compiler_flag(-fstrict-aliasing)
    add_cxx_compiler_flag(-Wno-ignored-attributes)

    if (PERNIX_ENABLE_LTO)
        add_cxx_compiler_flag(-flto=auto)
        add_cxx_compiler_flag(-Wno-lto-type-mismatch)
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
endif ()

include_directories(${PROJECT_SOURCE_DIR}/include)

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
