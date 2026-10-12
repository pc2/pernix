if (PERNIX_ENABLE_ARM64_NEON)
    set(PERNIX_ARM64_NEON_SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/neon/compression.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/neon/decompression.cpp"
    )

    target_sources(pernix PRIVATE ${PERNIX_ARM64_NEON_SOURCES})
    target_compile_definitions(pernix PRIVATE PERNIX_BUILD_ARM64_NEON=1)

    if (NOT MSVC)
        set_source_files_properties(${PERNIX_ARM64_NEON_SOURCES}
                TARGET_DIRECTORY pernix
                PROPERTIES COMPILE_OPTIONS "-march=armv8-a+simd")
    endif ()
endif ()

if (PERNIX_ENABLE_ARM64_SVE2)
    if (MSVC)
        message(FATAL_ERROR "The ARM64 SVE2 backend requires a compiler that supports -march=armv8.2-a+simd+sve2")
    endif ()

    set(PERNIX_ARM64_SVE2_SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/sve2/compression.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/sve2/decompression.cpp"
    )

    target_sources(pernix PRIVATE ${PERNIX_ARM64_SVE2_SOURCES})
    target_compile_definitions(pernix PRIVATE PERNIX_BUILD_ARM64_SVE2=1)

    set_source_files_properties(${PERNIX_ARM64_SVE2_SOURCES}
            TARGET_DIRECTORY pernix
            PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+simd+sve2")
endif ()
