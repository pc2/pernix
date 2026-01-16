file(GLOB_RECURSE
        LIBCOMPRESSION_SOURCES
        *.cpp
        ${PROJECT_SOURCE_DIR}/include/libcompression/*.h
)

add_library(libcompression STATIC ${LIBCOMPRESSION_SOURCES})
set_target_properties(libcompression PROPERTIES
        OUTPUT_NAME "libcompression"
        VERSION ${NORMALIZED_VERSION}
)
target_include_directories(libcompression PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

if (LIBCOMPRESSION_ENABLE_DOXYGEN)
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
    doxygen_add_docs(libcompression_docs
            #             docs
            include
            src
            ALL
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            COMMENT "Generating documentation with Doxygen."
    )
endif ()