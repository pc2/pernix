find_package(Git)

function(get_git_version VAR)
    if (GIT_EXECUTABLE)
        execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --match "v[0-9]*.[0-9]*.[0-9]*" --abbrev=8 --dirt WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} RESULT_VARIABLE status OUTPUT_VARIABLE GIT_VERSION ERROR_QUIET)

        if (status)
            set(GIT_VERSION "v0.0.0")
        endif ()
    else ()
        set(GIT_VERSION "v0.0.0")
    endif ()

    set(${VAR} "${GIT_VERSION}" PARENT_SCOPE)
endfunction()