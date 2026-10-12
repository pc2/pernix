function(pernix_add_x86_backend option_name definition msvc_options other_options)
    if (NOT ${${option_name}})
        return()
    endif ()

    set(PERNIX_X86_BACKEND_SOURCES ${ARGN})
    target_sources(pernix PRIVATE ${PERNIX_X86_BACKEND_SOURCES})
    target_compile_definitions(pernix PRIVATE "${definition}=1")

    if (MSVC)
        set(PERNIX_X86_COMPILE_OPTIONS "${msvc_options}")
    else ()
        set(PERNIX_X86_COMPILE_OPTIONS "${other_options}")
    endif ()

    set_source_files_properties(${PERNIX_X86_BACKEND_SOURCES}
            TARGET_DIRECTORY pernix
            PROPERTIES COMPILE_OPTIONS "${PERNIX_X86_COMPILE_OPTIONS}")
endfunction()

pernix_add_x86_backend(
        PERNIX_ENABLE_X86_64_BMI2
        PERNIX_BUILD_X86_BMI2
        "/arch:AVX2"
        "-mavx2;-mbmi;-mbmi2"
        "${CMAKE_CURRENT_SOURCE_DIR}/bmi2/bmi2_compression.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/bmi2/bmi2_decompression.cpp")

pernix_add_x86_backend(
        PERNIX_ENABLE_X86_64_AVX2
        PERNIX_BUILD_X86_AVX2
        "/arch:AVX2"
        "-mavx2"
        "${CMAKE_CURRENT_SOURCE_DIR}/avx2/avx2_compression.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/avx2/avx2_decompression.cpp")

pernix_add_x86_backend(
        PERNIX_ENABLE_X86_64_AVX512
        PERNIX_BUILD_X86_AVX512_VBMI
        "/arch:AVX512"
        "-mavx;-mavx2;-mbmi;-mbmi2;-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512cd;-mavx512vbmi"
        "${CMAKE_CURRENT_SOURCE_DIR}/avx512vbmi/avx512vbmi_compression.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/avx512vbmi/avx512vbmi_decompression.cpp")
