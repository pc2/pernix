include(GNUInstallDirs)

file(GLOB_RECURSE
        PERNIX_SOURCES
        ./*.cpp
        ${PROJECT_SOURCE_DIR}/include/pernix/*.h
)

add_library(pernix SHARED ${PERNIX_SOURCES})
set_target_properties(pernix PROPERTIES
        OUTPUT_NAME "pernix"
        VERSION ${NORMALIZED_VERSION}
)
target_include_directories(pernix PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
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