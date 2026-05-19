include(GNUInstallDirs)

file(GLOB_RECURSE
        PERNIX_COMMON_SOURCES
        ./fallback/*.cpp
        ./pernix.cpp
        ${PROJECT_SOURCE_DIR}/include/pernix/*.h
)

set(PERNIX_SOURCES ${PERNIX_COMMON_SOURCES})

set(PERNIX_SELECTED_ARCH_BACKEND "${PERNIX_ARCH_BACKEND}")
if (PERNIX_SELECTED_ARCH_BACKEND STREQUAL "AUTO")
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i[3-6]86|i686)$")
        set(PERNIX_SELECTED_ARCH_BACKEND "X86")
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
        set(PERNIX_SELECTED_ARCH_BACKEND "ARM64_NEON")
    else ()
        set(PERNIX_SELECTED_ARCH_BACKEND "FALLBACK")
    endif ()
endif ()
message(STATUS "Pernix architecture backend: ${PERNIX_SELECTED_ARCH_BACKEND}")

if (PERNIX_SELECTED_ARCH_BACKEND STREQUAL "X86")
    file(GLOB_RECURSE
            PERNIX_X86_SOURCES
            ./x86/*.cpp
            ${PROJECT_SOURCE_DIR}/include/pernix/x86/*.h
    )
    list(APPEND PERNIX_SOURCES ${PERNIX_X86_SOURCES})
elseif (PERNIX_SELECTED_ARCH_BACKEND STREQUAL "ARM64_NEON")
    file(GLOB_RECURSE
            PERNIX_ARM64_NEON_SOURCES
            ./arm64/neon/*.cpp
            ${PROJECT_SOURCE_DIR}/include/pernix/arm64/neon/*.h
    )
    list(APPEND PERNIX_SOURCES ${PERNIX_ARM64_NEON_SOURCES})
elseif (PERNIX_SELECTED_ARCH_BACKEND STREQUAL "ARM64_SVE")
    file(GLOB_RECURSE
            PERNIX_ARM64_SVE_SOURCES
            ./arm64/sve/*.cpp
            ${PROJECT_SOURCE_DIR}/include/pernix/arm64/sve/*.h
    )
    list(APPEND PERNIX_SOURCES ${PERNIX_ARM64_SVE_SOURCES})
elseif (PERNIX_SELECTED_ARCH_BACKEND STREQUAL "ARM64_SVE2")
    file(GLOB_RECURSE
            PERNIX_ARM64_SVE2_SOURCES
            ./arm64/sve2/*.cpp
            ${PROJECT_SOURCE_DIR}/include/pernix/arm64/sve2/*.h
    )
    list(APPEND PERNIX_SOURCES ${PERNIX_ARM64_SVE2_SOURCES})
endif ()

add_library(pernix SHARED ${PERNIX_SOURCES})
set_target_properties(pernix PROPERTIES
        OUTPUT_NAME "pernix"
        VERSION ${NORMALIZED_VERSION}
)
target_include_directories(pernix PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

if (PERNIX_USE_SIMDE)
    target_link_libraries(pernix PUBLIC simde::simde)
    target_compile_definitions(pernix PUBLIC PERNIX_USE_SIMDE=1)
endif ()

target_compile_definitions(pernix PUBLIC "PERNIX_BACKEND_${PERNIX_SELECTED_ARCH_BACKEND}=1")

if (PERNIX_DISABLE_BMI2)
    target_compile_definitions(pernix PUBLIC PERNIX_DISABLE_BMI2=1)
endif ()
if (PERNIX_DISABLE_AVX2)
    target_compile_definitions(pernix PUBLIC PERNIX_DISABLE_AVX2=1)
endif ()
if (PERNIX_DISABLE_AVX512)
    target_compile_definitions(pernix PUBLIC PERNIX_DISABLE_AVX512=1)
endif ()

set_target_properties(pernix PROPERTIES LINKER_LANGUAGE CXX)

configure_file(
        "${PROJECT_SOURCE_DIR}/cmake/pernix.pc.in" "${PROJECT_BINARY_DIR}/pernix.pc" @ONLY
)

if (PERNIX_ENABLE_INSTALL)
    install(TARGETS pernix
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/pernix ${PROJECT_BINARY_DIR}/include/pernix
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING PATTERN "*.*h"
    )

    install(
            FILES ${PROJECT_BINARY_DIR}/pernix.pc
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
    )

    add_custom_target(uninstall COMMAND xargs rm -vf < ${PROJECT_BINARY_DIR}/install_manifest.txt)
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
    # set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "index.md")
    doxygen_add_docs(pernix_doxygen
            #            docs
            include
            src
            ALL
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            COMMENT "Generating documentation with Doxygen."
    )
    if (BENCHMARK_ENABLE_INSTALL AND BENCHMARK_INSTALL_DOCS)
        install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/html/"
                DESTINATION ${CMAKE_INSTALL_DOCDIR})
    endif ()
endif ()
