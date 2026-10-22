cmake_minimum_required(VERSION 3.25)
project(pernix VERSION 0.1.0 LANGUAGES C CXX)

include(cmake/PernixOptions.cmake)
include(cmake/PernixCompilerOptions.cmake)
include(cmake/PernixVersion.cmake)

include(GNUInstallDirs)

if (PROJECT_IS_TOP_LEVEL)
    include(CTest)
endif ()

if (PERNIX_ENABLE_FORTRAN_BINDINGS)
    include(CheckLanguage)
    check_language(Fortran)

    if (NOT CMAKE_Fortran_COMPILER)
        message(FATAL_ERROR "Fortran compiler not found, but Fortran bindings are enabled")
    endif ()

    enable_language(Fortran)
endif ()

if (PERNIX_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if (NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "CUDA compiler not found, but CUDA backend is enabled")
    endif ()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    if (NOT CUDAToolkit_FOUND)
        message(FATAL_ERROR "CUDAToolkit not found, but CUDA backend is enabled")
    endif ()

    if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0)
        message(FATAL_ERROR "CUDA compiler version must be at least 12.0, but found ${CMAKE_CUDA_COMPILER_VERSION}")
    endif ()

    if (NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif ()

    message(STATUS "PERNIX CUDA toolkit: ${CUDAToolkit_VERSION}")
    message(STATUS "PERNIX CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif ()

add_subdirectory(src)

if (PERNIX_ENABLE_EXAMPLES)
    add_subdirectory(examples)
endif ()

if (PERNIX_ENABLE_FORTRAN_BINDINGS)
    add_subdirectory(bindings/fortran)
endif ()

if (BUILD_TESTING AND PROJECT_IS_TOP_LEVEL)
    add_subdirectory(tests)
endif ()

if (PERNIX_ENABLE_DOCS)
    find_package(Doxygen REQUIRED)

    set(DOXYGEN_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/docs")
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_GENERATE_MAN NO)
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${PROJECT_SOURCE_DIR}/README.md")

    doxygen_add_docs(pernix_docs
            "${PROJECT_SOURCE_DIR}/include"
            "${PROJECT_SOURCE_DIR}/README.md"
            COMMENT "Generating Pernix API documentation")
endif ()

if (PERNIX_ENABLE_INSTALL)
    include(cmake/PernixInstall.cmake)
endif ()
