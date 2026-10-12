include_guard(GLOBAL)

# Main build options
option(BUILD_SHARED_LIBS "Build pernix as shared library" ON)
option(BUILD_TESTING "Build pernix tests" ON)
option(PERNIX_ENABLE_EXAMPLES "Build pernix examples" OFF)
option(PERNIX_ENABLE_INSTALL "Enable installation of pernix" ON)
option(PERNIX_ENABLE_FORTRAN_BINDINGS "Build pernix Fortran bindings" OFF)


# Test and dev tools
option(PERNIX_ENABLE_INSTALL_CONSUMER_TESTS "Enable installation of consumer tests" OFF)
option(PERNIX_ENABLE_CODE_COVERAGE "Enable code coverage" OFF)
option(PERNIX_ENABLE_LTO "Enable link time optimization" OFF)
option(PERNIX_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" OFF)
option(PERNIX_ENABLE_SANITIZER_UNDEFINED "Enable undefined behavior sanitizer" OFF)

# Documentation
option(PERNIX_ENABLE_DOCS "Enable documentation generation with Doxygen" OFF)
option(PERNIX_INSTALL_DOCS "Enable installation of documentation" OFF)

# Architecture and platform specific options
option(PERNIX_ENABLE_X86_64 "Enable x86_64 architecture specific optimizations" ON)
option(PERNIX_ENABLE_X86_64_AVX2 "Enable x86_64 AVX2 specific optimizations" ON)
option(PERNIX_ENABLE_X86_64_BMI2 "Enable x86_64 BMI2 specific optimizations" ON)
option(PERNIX_ENABLE_X86_64_AVX512 "Enable x86_64 AVX512 specific optimizations" OFF)

option(PERNIX_ENABLE_ARM64 "Enable ARM64 architecture specific optimizations" ON)
option(PERNIX_ENABLE_ARM64_NEON "Enable ARM64 NEON specific optimizations" ON)
# option(PERNIX_ENABLE_ARM64_SVE "Enable ARM64 SVE specific optimizations" OFF)
option(PERNIX_ENABLE_ARM64_SVE2 "Enable ARM64 SVE2 specific optimizations" OFF)
#LATER
option(PERNIX_ENABLE_CUDA "Enable CUDA support" OFF)

if (PERNIX_INSTALL_DOCS AND NOT PERNIX_ENABLE_DOCS)
    message(FATAL_ERROR "PERNIX_INSTALL_DOCS requires PERNIX_ENABLE_DOCS to be ON")
endif ()

if (PERNIX_INSTALL_DOCS AND NOT PERNIX_ENABLE_INSTALL)
    message(FATAL_ERROR "PERNIX_INSTALL_DOCS requires PERNIX_ENABLE_INSTALL to be ON")
endif ()

if (PERNIX_ENABLE_INSTALL_CONSUMER_TESTS AND NOT PERNIX_ENABLE_INSTALL)
    message(FATAL_ERROR "PERNIX_ENABLE_INSTALL_CONSUMER_TESTS requires PERNIX_ENABLE_INSTALL to be ON")
endif ()

if (PERNIX_ENABLE_INSTALL_CONSUMER_TESTS AND NOT BUILD_TESTING)
    message(FATAL_ERROR "PERNIX_ENABLE_INSTALL_CONSUMER_TESTS requires BUILD_TESTING to be ON")
endif ()

if (PERNIX_ENABLE_CODE_COVERAGE AND NOT BUILD_TESTING)
    message(FATAL_ERROR "PERNIX_ENABLE_CODE_COVERAGE requires BUILD_TESTING to be ON")
endif ()
