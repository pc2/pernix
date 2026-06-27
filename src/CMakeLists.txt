include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(pernix SHARED)
add_library(pernix_header_only INTERFACE)
add_library(pernix::pernix_header_only ALIAS pernix_header_only)

target_sources(pernix PRIVATE pernix.cpp)
target_compile_features(pernix PUBLIC cxx_std_20)
target_compile_features(pernix_header_only INTERFACE cxx_std_20)
target_compile_options(pernix PRIVATE ${PERNIX_PRIVATE_COMPILE_OPTIONS})

set_target_properties(pernix PROPERTIES
        OUTPUT_NAME "pernix"
        VERSION ${NORMALIZED_VERSION}
        LINKER_LANGUAGE CXX
)

target_include_directories(pernix
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_include_directories(pernix_header_only
        INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(pernix PRIVATE PERNIX_BUILD_LIB=1)

if (PERNIX_ENABLE_LTO)
    set_target_properties(pernix PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif ()

if (PERNIX_USE_SIMDE)
    target_compile_definitions(pernix PUBLIC PERNIX_USE_SIMDE=1)
    target_compile_definitions(pernix_header_only INTERFACE PERNIX_USE_SIMDE=1)
endif ()

if (PERNIX_ENABLE_FALLBACK_STDPAR)
    target_compile_definitions(pernix PUBLIC PERNIX_BUILD_FALLBACK_STDPAR=1)
    target_compile_definitions(pernix_header_only INTERFACE PERNIX_BUILD_FALLBACK_STDPAR=1)
endif ()

if (PERNIX_ENABLE_FALLBACK_SIMD)
    target_compile_definitions(pernix PUBLIC PERNIX_BUILD_FALLBACK_SIMD=1)
    target_compile_definitions(pernix_header_only INTERFACE PERNIX_BUILD_FALLBACK_SIMD=1)
endif ()

if (PERNIX_ENABLE_FALLBACK_STDPAR_NVCC)
    target_compile_definitions(pernix PUBLIC PERNIX_FALLBACK_STDPAR_USE_NVCC=1)
    target_compile_definitions(pernix_header_only INTERFACE PERNIX_FALLBACK_STDPAR_USE_NVCC=1)
endif ()

if (PERNIX_ENABLE_X86_BMI2 AND PERNIX_TARGET_IS_X86)
    target_compile_definitions(pernix PRIVATE PERNIX_BUILD_X86_BMI2=1)
endif ()

if (PERNIX_ENABLE_X86_AVX2 AND PERNIX_TARGET_IS_X86)
    target_compile_definitions(pernix PRIVATE PERNIX_BUILD_X86_AVX2=1)
endif ()

if (PERNIX_ENABLE_X86_AVX512VBMI AND PERNIX_TARGET_IS_X86)
    target_compile_definitions(pernix PRIVATE PERNIX_BUILD_X86_AVX512_VBMI=1)
endif ()

if (PERNIX_ENABLE_ARM64_NEON AND PERNIX_TARGET_IS_ARM64)
    target_compile_definitions(pernix PRIVATE PERNIX_BUILD_ARM64_NEON=1)
endif ()

if (PERNIX_ENABLE_ARM64_SVE2 AND PERNIX_TARGET_IS_ARM64)
    target_compile_definitions(pernix PRIVATE PERNIX_BUILD_ARM64_SVE2=1)
endif ()

if (PERNIX_TARGET_IS_X86)
    if (MSVC)
        if (PERNIX_ENABLE_X86_AVX512VBMI)
            target_compile_options(pernix PRIVATE /arch:AVX512)
        elseif (PERNIX_ENABLE_X86_AVX2 OR PERNIX_ENABLE_X86_BMI2)
            target_compile_options(pernix PRIVATE /arch:AVX2)
        endif ()
    else ()
        if (PERNIX_ENABLE_X86_AVX512VBMI)
            target_compile_options(pernix PRIVATE -mavx -mbmi -mbmi2 -mavx2 -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512cd -mavx512vbmi)
        elseif (PERNIX_ENABLE_X86_BMI2)
            target_compile_options(pernix PRIVATE -mbmi -mbmi2 -mavx2)
        elseif (PERNIX_ENABLE_X86_AVX2)
            target_compile_options(pernix PRIVATE -mavx2)
        endif ()
    endif ()
endif ()

if (PERNIX_TARGET_IS_ARM64)
    if (PERNIX_ENABLE_ARM64_SVE2)
        target_compile_options(pernix PRIVATE -march=armv8.2-a+simd+sve2)
    elseif (PERNIX_ENABLE_ARM64_NEON)
        target_compile_options(pernix PRIVATE -march=armv8-a+simd)
    endif ()
endif ()

if (PERNIX_DISABLE_BMI2)
    target_compile_definitions(pernix PUBLIC PERNIX_DISABLE_BMI2=1)
    target_compile_definitions(pernix_header_only INTERFACE PERNIX_DISABLE_BMI2=1)
endif ()

if (PERNIX_DISABLE_AVX2)
    target_compile_definitions(pernix PUBLIC PERNIX_DISABLE_AVX2=1)
    target_compile_definitions(pernix_header_only INTERFACE PERNIX_DISABLE_AVX2=1)
endif ()

if (PERNIX_DISABLE_AVX512)
    target_compile_definitions(pernix PUBLIC PERNIX_DISABLE_AVX512=1)
    target_compile_definitions(pernix_header_only INTERFACE PERNIX_DISABLE_AVX512=1)
endif ()

configure_file("${PROJECT_SOURCE_DIR}/cmake/pernix.pc.in" "${PROJECT_BINARY_DIR}/pernix.pc" @ONLY)

if (PERNIX_ENABLE_INSTALL)
    install(TARGETS pernix pernix_header_only
            EXPORT pernixTargets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/pernix"
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING PATTERN "*.*h"
    )

    if (PERNIX_USE_SIMDE AND PERNIX_BUNDLE_SIMDE_FOR_INSTALL)
        install(DIRECTORY "${simde_SOURCE_DIR}/simde"
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
                FILES_MATCHING PATTERN "*.h"
        )
    endif ()

    configure_package_config_file(
            "${PROJECT_SOURCE_DIR}/cmake/pernixConfig.cmake.in"
            "${PROJECT_BINARY_DIR}/pernixConfig.cmake"
            INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/pernix"
    )
    write_basic_package_version_file(
            "${PROJECT_BINARY_DIR}/pernixConfigVersion.cmake"
            VERSION ${NORMALIZED_VERSION}
            COMPATIBILITY SameMajorVersion
    )
    install(
            FILES
            "${PROJECT_BINARY_DIR}/pernixConfig.cmake"
            "${PROJECT_BINARY_DIR}/pernixConfigVersion.cmake"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/pernix"
    )
    install(
            EXPORT pernixTargets
            FILE pernixTargets.cmake
            NAMESPACE pernix::
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/pernix"
    )
    install(FILES "${PROJECT_BINARY_DIR}/pernix.pc"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
endif ()

if (PERNIX_ENABLE_DOXYGEN)
    message(STATUS "Doxygen documentation enabled")
    find_package(Doxygen REQUIRED)
    set(DOXYGEN_QUIET YES)
    set(DOXYGEN_RECURSIVE YES)
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_GENERATE_MAN NO)
    set(DOXYGEN_MARKDOWN_SUPPORT YES)
    set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
    set(DOXYGEN_EXTRACT_PACKAGE YES)
    set(DOXYGEN_EXTRACT_STATIC YES)
    set(DOXYGEN_SHOW_INCLUDE_FILES YES)
    set(DOXYGEN_BINARY_TOC YES)
    set(DOXYGEN_TOC_EXPAND YES)
    doxygen_add_docs(pernix_doxygen
            include
            src
            ALL
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            COMMENT "Generating documentation with Doxygen."
    )
    if (PERNIX_ENABLE_INSTALL AND PERNIX_INSTALL_DOCS)
        install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/html/"
                DESTINATION ${CMAKE_INSTALL_DOCDIR})
    endif ()
endif ()
