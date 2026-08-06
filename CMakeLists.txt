cmake_minimum_required(VERSION 3.25)
project(pernix VERSION 0.1.0 LANGUAGES CXX C)

option(BUILD_SHARED_LIBS "Build pernix as a shared library by default" ON)
option(PERNIX_ENABLE_TESTS "Build pernix unit and integration tests" ON)
option(PERNIX_ENABLE_CODE_COVERAGE "Enable code coverage for pernix tests (requires lcov/genhtml)" OFF)
option(PERNIX_ENABLE_LTO "Enable interprocedural optimization/link-time optimization" OFF)
option(PERNIX_ENABLE_SANITIZER_ADDRESS "Build pernix targets with AddressSanitizer instrumentation" OFF)
option(PERNIX_ENABLE_SANITIZER_UNDEFINED "Build pernix targets with UndefinedBehaviorSanitizer instrumentation" OFF)
option(PERNIX_ENABLE_INSTALL "Install pernix library, headers, and package metadata" ON)
option(PERNIX_ENABLE_DOXYGEN "Build API documentation with Doxygen" OFF)
option(PERNIX_INSTALL_DOCS "Install generated Doxygen documentation; requires PERNIX_ENABLE_DOXYGEN=ON" OFF)
option(PERNIX_ENABLE_EXAMPLES "Build pernix example programs" OFF)
option(PERNIX_ENABLE_INSTALL_CONSUMER_TESTS "Add CTest checks for consuming an installed pernix package" ON)

option(PERNIX_ENABLE_X86_BMI2 "Build x86 BMI2 backend when targeting x86" ON)
option(PERNIX_ENABLE_X86_AVX2 "Build x86 AVX2 backend when targeting x86" ON)
option(PERNIX_ENABLE_X86_AVX512VBMI "Build x86 AVX512-VBMI backend when targeting x86" ON)
option(PERNIX_ENABLE_ARM64_NEON "Build arm64 NEON backend when targeting arm64" ON)
option(PERNIX_ENABLE_ARM64_SVE2 "Build arm64 SVE2 backend when targeting arm64" ON)
set(PERNIX_ENABLE_FALLBACK_STDPAR "OFF" CACHE STRING "Enable the stdpar fallback backend: OFF, AUTO, or ON. TBB is optional and enables parallel standard execution policies when found.")
set_property(CACHE PERNIX_ENABLE_FALLBACK_STDPAR PROPERTY STRINGS AUTO ON OFF)
set(PERNIX_ENABLE_FALLBACK_SIMD "AUTO" CACHE STRING "Enable the C++26 std::simd fallback backend: OFF, AUTO, or ON.")
set_property(CACHE PERNIX_ENABLE_FALLBACK_SIMD PROPERTY STRINGS AUTO ON OFF)

option(PERNIX_USE_SIMDE "Use SIMDe library for portable SIMD support" OFF)

option(PERNIX_ENABLE_FORTRAN_BINDINGS "Build optional Fortran bindings and round-trip test" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
set(PERNIX_BUNDLE_SIMDE_FOR_INSTALL OFF)

string(TOUPPER "${PERNIX_ENABLE_FALLBACK_STDPAR}" PERNIX_ENABLE_FALLBACK_STDPAR)
set(PERNIX_ENABLE_FALLBACK_STDPAR "${PERNIX_ENABLE_FALLBACK_STDPAR}" CACHE STRING "Enable the stdpar fallback backend: OFF, AUTO, or ON. TBB is optional and enables parallel standard execution policies when found." FORCE)
if (NOT PERNIX_ENABLE_FALLBACK_STDPAR MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "PERNIX_ENABLE_FALLBACK_STDPAR must be AUTO, ON, or OFF")
endif ()

string(TOUPPER "${PERNIX_ENABLE_FALLBACK_SIMD}" PERNIX_ENABLE_FALLBACK_SIMD)
set(PERNIX_ENABLE_FALLBACK_SIMD "${PERNIX_ENABLE_FALLBACK_SIMD}" CACHE STRING "Enable the C++26 std::simd fallback backend: OFF, AUTO, or ON." FORCE)
if (NOT PERNIX_ENABLE_FALLBACK_SIMD MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "PERNIX_ENABLE_FALLBACK_SIMD must be AUTO, ON, or OFF")
endif ()

set(PERNIX_BUILD_FALLBACK_SIMD OFF)
if (NOT PERNIX_ENABLE_FALLBACK_SIMD STREQUAL "OFF")
    include(CheckCXXSourceCompiles)
    if ("cxx_std_26" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
        if (DEFINED CMAKE_CXX_STANDARD)
            set(PERNIX_SAVED_CMAKE_CXX_STANDARD "${CMAKE_CXX_STANDARD}")
            set(PERNIX_CXX_STANDARD_WAS_DEFINED ON)
        endif ()
        if (DEFINED CMAKE_CXX_STANDARD_REQUIRED)
            set(PERNIX_SAVED_CMAKE_CXX_STANDARD_REQUIRED "${CMAKE_CXX_STANDARD_REQUIRED}")
            set(PERNIX_CXX_STANDARD_REQUIRED_WAS_DEFINED ON)
        endif ()
        set(CMAKE_CXX_STANDARD 26)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)
        check_cxx_source_compiles([=[
            #include <simd>
            #include <version>

            #if !defined(__cpp_lib_simd)
            #error "The standard library does not provide C++26 std::simd"
            #endif

            int main() {
                std::simd<int> values{};
                return static_cast<int>(values.size() == 0);
            }
        ]=] PERNIX_HAS_CXX26_STDSIMD)
        if (PERNIX_CXX_STANDARD_WAS_DEFINED)
            set(CMAKE_CXX_STANDARD "${PERNIX_SAVED_CMAKE_CXX_STANDARD}")
        else ()
            unset(CMAKE_CXX_STANDARD)
        endif ()
        if (PERNIX_CXX_STANDARD_REQUIRED_WAS_DEFINED)
            set(CMAKE_CXX_STANDARD_REQUIRED "${PERNIX_SAVED_CMAKE_CXX_STANDARD_REQUIRED}")
        else ()
            unset(CMAKE_CXX_STANDARD_REQUIRED)
        endif ()
    else ()
        set(PERNIX_HAS_CXX26_STDSIMD OFF)
    endif ()

    if (PERNIX_HAS_CXX26_STDSIMD)
        set(PERNIX_BUILD_FALLBACK_SIMD ON)
        message(STATUS "PERNIX fallback SIMD backend enabled (C++26 std::simd available)")
    elseif (PERNIX_ENABLE_FALLBACK_SIMD STREQUAL "ON")
        message(FATAL_ERROR "PERNIX_ENABLE_FALLBACK_SIMD=ON requires compiler and standard-library support for C++26 std::simd")
    else ()
        message(STATUS "PERNIX fallback SIMD backend disabled (C++26 std::simd unavailable)")
    endif ()
endif ()

if (PERNIX_INSTALL_DOCS AND NOT PERNIX_ENABLE_DOXYGEN)
    message(FATAL_ERROR "PERNIX_INSTALL_DOCS requires PERNIX_ENABLE_DOXYGEN=ON")
endif ()

if (PERNIX_ENABLE_INSTALL_CONSUMER_TESTS AND PERNIX_ENABLE_TESTS AND NOT PERNIX_ENABLE_INSTALL)
    message(FATAL_ERROR "PERNIX_ENABLE_INSTALL_CONSUMER_TESTS requires PERNIX_ENABLE_INSTALL=ON")
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

if (NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|Intel|NVHPC")
    message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif ()

include(CheckCXXCompilerFlag)
set(PERNIX_PRIVATE_COMPILE_OPTIONS)
foreach (PERNIX_CXX_FLAG
        -Wall
        -Wextra
        -Wpedantic
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

set(PERNIX_SANITIZER_OPTIONS)
if (PERNIX_ENABLE_SANITIZER_ADDRESS OR PERNIX_ENABLE_SANITIZER_UNDEFINED)
    if (NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
        message(FATAL_ERROR "PERNIX sanitizer options are supported only with GCC/Clang-like compilers")
    endif ()
endif ()

if (PERNIX_ENABLE_SANITIZER_ADDRESS)
    list(APPEND PERNIX_SANITIZER_OPTIONS "-fsanitize=address")
endif ()

if (PERNIX_ENABLE_SANITIZER_UNDEFINED)
    list(APPEND PERNIX_SANITIZER_OPTIONS "-fsanitize=undefined")
endif ()

if (PERNIX_SANITIZER_OPTIONS)
    add_compile_options(${PERNIX_SANITIZER_OPTIONS})
    add_link_options(${PERNIX_SANITIZER_OPTIONS})
endif ()

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
    enable_language(Fortran)
    add_subdirectory(bindings/fortran)
endif ()

if (PERNIX_ENABLE_EXAMPLES)
    add_subdirectory(examples)
endif ()

if (PERNIX_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()
