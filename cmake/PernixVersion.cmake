include_guard(GLOBAL)

find_package(Git QUIET)

set(PERNIX_VERSION "${PROJECT_VERSION}")
set(PERNIX_VERSION_FULL "${PROJECT_VERSION}")

if (GIT_EXECUTABLE)
    execute_process(
            COMMAND "${GIT_EXECUTABLE}" describe
            --tags
            --match "v[0-9]*.[0-9]*.[0-9]*"
            --abbrev=8
            --dirty
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            RESULT_VARIABLE PERNIX_GIT_RESULT
            OUTPUT_VARIABLE PERNIX_GIT_VERSION
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if (PERNIX_GIT_RESULT EQUAL 0
            AND NOT PERNIX_GIT_VERSION STREQUAL "")
        set(PERNIX_VERSION_FULL "${PERNIX_GIT_VERSION}")
    endif ()
endif ()

message(STATUS "Pernix version: ${PERNIX_VERSION_FULL} (package version ${PERNIX_VERSION})")
