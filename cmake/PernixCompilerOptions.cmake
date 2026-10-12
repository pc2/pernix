include_guard(GLOBAL)

include(CheckCXXCompilerFlag)
include(CheckIPOSupported)

if (MSVC)
    set(PERNIX_PRIVATE_COMPILE_OPTIONS /W4 /permissive-)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|Intel|NVHPC")
    set(PERNIX_PRIVATE_COMPILE_OPTIONS)

    foreach (PERNIX_CXX_FLAG
            -Wall
            -Wextra
            -Wpedantic
            -Wshadow
            -Wfloat-equal
            -Wold-style-cast
            -Wconversion
            -fstrict-aliasing)
        string(MAKE_C_IDENTIFIER "PERNIX_HAS_CXX_FLAG_${PERNIX_CXX_FLAG}" PERNIX_CXX_FLAG_VARIABLE)
        check_cxx_compiler_flag("${PERNIX_CXX_FLAG}" "${PERNIX_CXX_FLAG_VARIABLE}")

        if (${PERNIX_CXX_FLAG_VARIABLE})
            list(APPEND PERNIX_PRIVATE_COMPILE_OPTIONS "${PERNIX_CXX_FLAG}")
        endif ()
    endforeach ()

    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        list(APPEND PERNIX_PRIVATE_COMPILE_OPTIONS -Wno-ignored-attributes)
    endif ()
else ()
    message(FATAL_ERROR "Unsupported C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
endif ()

set(PERNIX_SANITIZER_OPTIONS)
if (PERNIX_ENABLE_SANITIZER_ADDRESS)
    list(APPEND PERNIX_SANITIZER_OPTIONS -fsanitize=address)
endif ()
if (PERNIX_ENABLE_SANITIZER_UNDEFINED)
    list(APPEND PERNIX_SANITIZER_OPTIONS -fsanitize=undefined)
endif ()

if (PERNIX_SANITIZER_OPTIONS)
    if (NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
        message(FATAL_ERROR "Pernix sanitizers require a GCC- or Clang-compatible compiler")
    endif ()

    add_compile_options(${PERNIX_SANITIZER_OPTIONS})
    add_link_options(${PERNIX_SANITIZER_OPTIONS})
endif ()

if (PERNIX_ENABLE_LTO)
    check_ipo_supported(RESULT PERNIX_IPO_SUPPORTED OUTPUT PERNIX_IPO_ERROR)

    if (NOT PERNIX_IPO_SUPPORTED)
        message(FATAL_ERROR "LTO is enabled but unsupported: ${PERNIX_IPO_ERROR}")
    endif ()

    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif ()
