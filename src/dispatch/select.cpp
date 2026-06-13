#include <pernix/dispatch/select.h>
#include <pernix/dispatch/cpu_features.h>

namespace pernix::internal {
Kernel<KernelBlockF32Func> select_compress_block_f32(Backend backend, uint8_t bit_width, uint32_t block_size) {
    switch (backend) {
        case Backend::Auto:
            return select_auto_compress_block_f32(bit_width, block_size);
        case Backend::Fallback:
            return select_fallback_compress_block_f32(bit_width, block_size);
#if defined(PERNIX_BUILD_X86_AVX512_VBMI)
        case Backend::X86Avx512Vbmi:
            return select_avx512vbmi_compress_block_f32(bit_width, block_size);
#endif
#if defined(PERNIX_BUILD_X86_AVX2)
        case Backend::X86Avx2:
            return select_avx2_compress_block_f32(bit_width, block_size);
#endif
#if defined(PERNIX_BUILD_X86_BMI2)
        case Backend::X86Bmi2:
            return select_bmi2_compress_block_f32(bit_width, block_size);
#endif
#if defined(PERNIX_BUILD_ARM64_NEON)
        case Backend::Arm64Neon:
            return select_neon_compress_block_f32(bit_width, block_size);
#endif
#if defined(PERNIX_BUILD_ARM64_SVE2)
        case Backend::Arm64Sve:
            return select_sve2_compress_block_f32(bit_width, block_size);
#endif
        default:
            return {"invalid_backend", nullptr};
    }
}

Kernel<KernelBlocksF32Func> select_compress_blocks_f32(Backend backend, uint8_t bit_width, uint32_t block_size) {
    switch (backend) {
        case Backend::Auto:
            return select_auto_compress_blocks_f32(bit_width, block_size);
        case Backend::Fallback:
            return select_fallback_compress_blocks_f32(bit_width, block_size);
#if defined(PERNIX_BUILD_X86_AVX512_VBMI)
        case Backend::X86Avx512Vbmi:
            return select_avx512vbmi_compress_blocks_f32(bit_width, block_size);
#endif
#if defined(PERNIX_BUILD_X86_AVX2)
        case Backend::X86Avx2:
            return select_avx2_compress_blocks_f32(bit_width, block_size);
#endif
#if defined(PERNIX_BUILD_X86_BMI2)
        case Backend::X86Bmi2:
            return select_bmi2_compress_blocks_f32(bit_width, block_size);
#endif
#if defined(PERNIX_BUILD_ARM64_NEON)
        case Backend::Arm64Neon:
            return select_neon_compress_blocks_f32(bit_width, block_size);
#endif
#if defined(PERNIX_BUILD_ARM64_SVE2)
        case Backend::Arm64Sve:
            return select_sve2_compress_blocks_f32(bit_width, block_size);
#endif
        default:
            return {"invalid_backend", nullptr};
    }
}

Kernel<KernelBlockF64Func> select_compress_block_f64(Backend backend, uint8_t bit_width, uint32_t block_size) {
    switch (backend) {
        case Backend::Auto:
            return select_auto_compress_block_f64(bit_width, block_size);
        case Backend::Fallback:
            return select_fallback_compress_block_f64(bit_width, block_size);
#if defined(PERNIX_BUILD_X86_AVX512_VBMI)
        case Backend::X86Avx512Vbmi:
            return select_avx512vbmi_compress_block_f64(bit_width, block_size);
#endif
#if defined(PERNIX_BUILD_X86_AVX2)
        case Backend::X86Avx2:
            return select_avx2_compress_block_f64(bit_width, block_size);
#endif
#if defined(PERNIX_BUILD_X86_BMI2)
        case Backend::X86Bmi2:
            return select_bmi2_compress_block_f64(bit_width, block_size);
#endif
#if defined(PERNIX_BUILD_ARM64_NEON)
        case Backend::Arm64Neon:
            return select_neon_compress_block_f64(bit_width, block_size);
#endif
#if defined(PERNIX_BUILD_ARM64_SVE2)
        case Backend::Arm64Sve:
            return select_sve2_compress_block_f64(bit_width, block_size);
#endif
        default:
            return {"invalid_backend", nullptr};
    }
}

Kernel<KernelBlocksF64Func> select_compress_blocks_f64(Backend backend, uint8_t bit_width, uint32_t block_size) {
    switch (backend) {
        case Backend::Auto:
            return select_auto_compress_blocks_f64(bit_width, block_size);
        case Backend::Fallback:
            return select_fallback_compress_blocks_f64(bit_width, block_size);
#if defined(PERNIX_BUILD_X86_AVX512_VBMI)
        case Backend::X86Avx512Vbmi:
            return select_avx512vbmi_compress_blocks_f64(bit_width, block_size);
#endif
#if defined(PERNIX_BUILD_X86_AVX2)
        case Backend::X86Avx2:
            return select_avx2_compress_blocks_f64(bit_width, block_size);
#endif
#if defined(PERNIX_BUILD_X86_BMI2)
        case Backend::X86Bmi2:
            return select_bmi2_compress_blocks_f64(bit_width, block_size);
#endif
#if defined(PERNIX_BUILD_ARM64_NEON)
        case Backend::Arm64Neon:
            return select_neon_compress_blocks_f64(bit_width, block_size);
#endif
#if defined(PERNIX_BUILD_ARM64_SVE2)
        case Backend::Arm64Sve:
            return select_sve2_compress_blocks_f64(bit_width, block_size);
#endif
        default:
            return {"invalid_backend", nullptr};
    }
}

Kernel<KernelBlockF32Func> select_decompress_block_f32(Backend backend, uint8_t bit_width, uint32_t block_size, bool sign_values) {
    switch (backend) {
        case Backend::Auto:
            return select_auto_decompress_block_f32(bit_width, block_size, sign_values);
        case Backend::Fallback:
            return select_fallback_decompress_block_f32(bit_width, block_size, sign_values);
#if defined(PERNIX_BUILD_X86_AVX512_VBMI)
        case Backend::X86Avx512Vbmi:
            return select_avx512vbmi_decompress_block_f32(bit_width, block_size, sign_values);
#endif
#if defined(PERNIX_BUILD_X86_AVX2)
        case Backend::X86Avx2:
            return select_avx2_decompress_block_f32(bit_width, block_size, sign_values);
#endif
#if defined(PERNIX_BUILD_X86_BMI2)
        case Backend::X86Bmi2:
            return select_bmi2_decompress_block_f32(bit_width, block_size, sign_values);
#endif
#if defined(PERNIX_BUILD_ARM64_NEON)
        case Backend::Arm64Neon:
            return select_neon_decompress_block_f32(bit_width, block_size, sign_values);
#endif
#if defined(PERNIX_BUILD_ARM64_SVE2)
        case Backend::Arm64Sve:
            return select_sve2_decompress_block_f32(bit_width, block_size, sign_values);
#endif
        default:
            return {"invalid_backend", nullptr};
    }
}

Kernel<KernelBlocksF32Func> select_decompress_blocks_f32(Backend backend, uint8_t bit_width, uint32_t block_size, bool sign_values) {
    switch (backend) {
        case Backend::Auto:
            return select_auto_decompress_blocks_f32(bit_width, block_size, sign_values);
        case Backend::Fallback:
            return select_fallback_decompress_blocks_f32(bit_width, block_size, sign_values);
#if defined(PERNIX_BUILD_X86_AVX512_VBMI)
        case Backend::X86Avx512Vbmi:
            return select_avx512vbmi_decompress_blocks_f32(bit_width, block_size, sign_values);
#endif
#if defined(PERNIX_BUILD_X86_AVX2)
        case Backend::X86Avx2:
            return select_avx2_decompress_blocks_f32(bit_width, block_size, sign_values);
#endif
#if defined(PERNIX_BUILD_X86_BMI2)
        case Backend::X86Bmi2:
            return select_bmi2_decompress_blocks_f32(bit_width, block_size, sign_values);
#endif
#if defined(PERNIX_BUILD_ARM64_NEON)
        case Backend::Arm64Neon:
            return select_neon_decompress_blocks_f32(bit_width, block_size, sign_values);
#endif
#if defined(PERNIX_BUILD_ARM64_SVE2)
        case Backend::Arm64Sve:
            return select_sve2_decompress_blocks_f32(bit_width, block_size, sign_values);
#endif
        default:
            return {"invalid_backend", nullptr};
    }
}

Kernel<KernelBlockF64Func> select_decompress_block_f64(Backend backend, uint8_t bit_width, uint32_t block_size, bool sign_values) {
    switch (backend) {
        case Backend::Auto:
            return select_auto_decompress_block_f64(bit_width, block_size, sign_values);
        case Backend::Fallback:
            return select_fallback_decompress_block_f64(bit_width, block_size, sign_values);
#if defined(PERNIX_BUILD_X86_AVX512_VBMI)
        case Backend::X86Avx512Vbmi:
            return select_avx512vbmi_decompress_block_f64(bit_width, block_size, sign_values);
#endif
#if defined(PERNIX_BUILD_X86_AVX2)
        case Backend::X86Avx2:
            return select_avx2_decompress_block_f64(bit_width, block_size, sign_values);
#endif
#if defined(PERNIX_BUILD_X86_BMI2)
        case Backend::X86Bmi2:
            return select_bmi2_decompress_block_f64(bit_width, block_size, sign_values);
#endif
#if defined(PERNIX_BUILD_ARM64_NEON)
        case Backend::Arm64Neon:
            return select_neon_decompress_block_f64(bit_width, block_size, sign_values);
#endif
#if defined(PERNIX_BUILD_ARM64_SVE2)
        case Backend::Arm64Sve:
            return select_sve2_decompress_block_f64(bit_width, block_size, sign_values);
#endif
        default:
            return {"invalid_backend", nullptr};
    }
}

Kernel<KernelBlocksF64Func> select_decompress_blocks_f64(Backend backend, uint8_t bit_width, uint32_t block_size, bool sign_values) {
    switch (backend) {
        case Backend::Auto:
            return select_auto_decompress_blocks_f64(bit_width, block_size, sign_values);
        case Backend::Fallback:
            return select_fallback_decompress_blocks_f64(bit_width, block_size, sign_values);
#if defined(PERNIX_BUILD_X86_AVX512_VBMI)
        case Backend::X86Avx512Vbmi:
            return select_avx512vbmi_decompress_blocks_f64(bit_width, block_size, sign_values);
#endif
#if defined(PERNIX_BUILD_X86_AVX2)
        case Backend::X86Avx2:
            return select_avx2_decompress_blocks_f64(bit_width, block_size, sign_values);
#endif
#if defined(PERNIX_BUILD_X86_BMI2)
        case Backend::X86Bmi2:
            return select_bmi2_decompress_blocks_f64(bit_width, block_size, sign_values);
#endif
#if defined(PERNIX_BUILD_ARM64_NEON)
        case Backend::Arm64Neon:
            return select_neon_decompress_blocks_f64(bit_width, block_size, sign_values);
#endif
#if defined(PERNIX_BUILD_ARM64_SVE2)
        case Backend::Arm64Sve:
            return select_sve2_decompress_blocks_f64(bit_width, block_size, sign_values);
#endif
        default:
            return {"invalid_backend", nullptr};
    }
}

Kernel<KernelBlockF32Func> select_auto_compress_block_f32(uint8_t bit_width, uint32_t block_size) {
#if defined(PERNIX_BUILD_X86_AVX512_VBMI) || defined(PERNIX_BUILD_X86_AVX2)  || defined(PERNIX_BUILD_X86_BMI2) || defined(PERNIX_BUILD_ARM64_NEON) || defined(PERNIX_BUILD_ARM64_SVE2)
    const CpuFeatures features = get_cached_cpu_features();
#endif

#if defined(PERNIX_BUILD_X86_AVX512_VBMI)
    if (
        features.avx512f &&
        features.avx512dq &&
        features.avx512bw &&
        features.avx512vl &&
        features.avx512vbmi
    ) {
        if (auto kernel = select_avx512vbmi_compress_block_f32(bit_width, block_size)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_X86_AVX2)
    if (features.avx2) {
        if (auto kernel = select_avx2_compress_block_f32(bit_width, block_size)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_X86_BMI2)
    if (features.bmi2) {
        if (auto kernel = select_bmi2_compress_block_f32(bit_width, block_size)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_ARM64_NEON)
    if (features.neon) {
        if (auto kernel = select_neon_compress_block_f32(bit_width, block_size)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_ARM64_SVE2)
    if (features.sve) {
        if (auto kernel = select_sve2_compress_block_f32(bit_width, block_size)) {
            return kernel;
        }
    }
#endif

    return select_fallback_compress_block_f32(bit_width, block_size);
}

Kernel<KernelBlocksF32Func> select_auto_compress_blocks_f32(uint8_t bit_width, uint32_t block_size) {
#if defined(PERNIX_BUILD_X86_AVX512_VBMI) || defined(PERNIX_BUILD_X86_AVX2)  || defined(PERNIX_BUILD_X86_BMI2) || defined(PERNIX_BUILD_ARM64_NEON) || defined(PERNIX_BUILD_ARM64_SVE2)
    const CpuFeatures features = get_cached_cpu_features();
#endif

#if defined(PERNIX_BUILD_X86_AVX512_VBMI)
    if (
        features.avx512f &&
        features.avx512dq &&
        features.avx512bw &&
        features.avx512vl &&
        features.avx512vbmi
    ) {
        if (auto kernel = select_avx512vbmi_compress_blocks_f32(bit_width, block_size)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_X86_AVX2)
    if (features.avx2) {
        if (auto kernel = select_avx2_compress_blocks_f32(bit_width, block_size)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_X86_BMI2)
    if (features.bmi2) {
        if (auto kernel = select_bmi2_compress_blocks_f32(bit_width, block_size)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_ARM64_NEON)
    if (features.neon) {
        if (auto kernel = select_neon_compress_blocks_f32(bit_width, block_size)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_ARM64_SVE2)
    if (features.sve) {
        if (auto kernel = select_sve2_compress_blocks_f32(bit_width, block_size)) {
            return kernel;
        }
    }
#endif

    return select_fallback_compress_blocks_f32(bit_width, block_size);
}

Kernel<KernelBlockF64Func> select_auto_compress_block_f64(uint8_t bit_width, uint32_t block_size) {
#if defined(PERNIX_BUILD_X86_AVX512_VBMI) || defined(PERNIX_BUILD_X86_AVX2)  || defined(PERNIX_BUILD_X86_BMI2) || defined(PERNIX_BUILD_ARM64_NEON) || defined(PERNIX_BUILD_ARM64_SVE2)
    const CpuFeatures features = get_cached_cpu_features();
#endif

#if defined(PERNIX_BUILD_X86_AVX512_VBMI)
    if (
        features.avx512f &&
        features.avx512dq &&
        features.avx512bw &&
        features.avx512vl &&
        features.avx512vbmi
    ) {
        if (auto kernel = select_avx512vbmi_compress_block_f64(bit_width, block_size)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_X86_AVX2)
    if (features.avx2) {
        if (auto kernel = select_avx2_compress_block_f64(bit_width, block_size)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_X86_BMI2)
    if (features.bmi2) {
        if (auto kernel = select_bmi2_compress_block_f64(bit_width, block_size)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_ARM64_NEON)
    if (features.neon) {
        if (auto kernel = select_neon_compress_block_f64(bit_width, block_size)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_ARM64_SVE2)
    if (features.sve) {
        if (auto kernel = select_sve2_compress_block_f64(bit_width, block_size)) {
            return kernel;
        }
    }
#endif

    return select_fallback_compress_block_f64(bit_width, block_size);
}

Kernel<KernelBlocksF64Func> select_auto_compress_blocks_f64(uint8_t bit_width, uint32_t block_size) {
#if defined(PERNIX_BUILD_X86_AVX512_VBMI) || defined(PERNIX_BUILD_X86_AVX2)  || defined(PERNIX_BUILD_X86_BMI2) || defined(PERNIX_BUILD_ARM64_NEON) || defined(PERNIX_BUILD_ARM64_SVE2)
    const CpuFeatures features = get_cached_cpu_features();
#endif

#if defined(PERNIX_BUILD_X86_AVX512_VBMI)
    if (
        features.avx512f &&
        features.avx512dq &&
        features.avx512bw &&
        features.avx512vl &&
        features.avx512vbmi
    ) {
        if (auto kernel = select_avx512vbmi_compress_blocks_f64(bit_width, block_size)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_X86_AVX2)
    if (features.avx2) {
        if (auto kernel = select_avx2_compress_blocks_f64(bit_width, block_size)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_X86_BMI2)
    if (features.bmi2) {
        if (auto kernel = select_bmi2_compress_blocks_f64(bit_width, block_size)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_ARM64_NEON)
    if (features.neon) {
        if (auto kernel = select_neon_compress_blocks_f64(bit_width, block_size)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_ARM64_SVE2)
    if (features.sve) {
        if (auto kernel = select_sve2_compress_blocks_f64(bit_width, block_size)) {
            return kernel;
        }
    }
#endif

    return select_fallback_compress_blocks_f64(bit_width, block_size);
}

Kernel<KernelBlockF32Func> select_auto_decompress_block_f32(uint8_t bit_width, uint32_t block_size, bool sign_values) {
#if defined(PERNIX_BUILD_X86_AVX512_VBMI) || defined(PERNIX_BUILD_X86_AVX2)  || defined(PERNIX_BUILD_X86_BMI2) || defined(PERNIX_BUILD_ARM64_NEON) || defined(PERNIX_BUILD_ARM64_SVE2)
    const CpuFeatures features = get_cached_cpu_features();
#endif

#if defined(PERNIX_BUILD_X86_AVX512_VBMI)
    if (
        features.avx512f &&
        features.avx512dq &&
        features.avx512bw &&
        features.avx512vl &&
        features.avx512vbmi
    ) {
        if (auto kernel = select_avx512vbmi_decompress_block_f32(bit_width, block_size, sign_values)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_X86_AVX2)
    if (features.avx2) {
        if (auto kernel = select_avx2_decompress_block_f32(bit_width, block_size, sign_values)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_X86_BMI2)
    if (features.bmi2) {
        if (auto kernel = select_bmi2_decompress_block_f32(bit_width, block_size, sign_values)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_ARM64_NEON)
    if (features.neon) {
        if (auto kernel = select_neon_decompress_block_f32(bit_width, block_size, sign_values)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_ARM64_SVE2)
    if (features.sve) {
        if (auto kernel = select_sve2_decompress_block_f32(bit_width, block_size, sign_values)) {
            return kernel;
        }
    }
#endif

    return select_fallback_decompress_block_f32(bit_width, block_size, sign_values);
}

Kernel<KernelBlocksF32Func> select_auto_decompress_blocks_f32(uint8_t bit_width, uint32_t block_size, bool sign_values) {
#if defined(PERNIX_BUILD_X86_AVX512_VBMI) || defined(PERNIX_BUILD_X86_AVX2)  || defined(PERNIX_BUILD_X86_BMI2) || defined(PERNIX_BUILD_ARM64_NEON) || defined(PERNIX_BUILD_ARM64_SVE2)
    const CpuFeatures features = get_cached_cpu_features();
#endif

#if defined(PERNIX_BUILD_X86_AVX512_VBMI)
    if (
        features.avx512f &&
        features.avx512dq &&
        features.avx512bw &&
        features.avx512vl &&
        features.avx512vbmi
    ) {
        if (auto kernel = select_avx512vbmi_decompress_blocks_f32(bit_width, block_size, sign_values)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_X86_AVX2)
    if (features.avx2) {
        if (auto kernel = select_avx2_decompress_blocks_f32(bit_width, block_size, sign_values)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_X86_BMI2)
    if (features.bmi2) {
        if (auto kernel = select_bmi2_decompress_blocks_f32(bit_width, block_size, sign_values)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_ARM64_NEON)
    if (features.neon) {
        if (auto kernel = select_neon_decompress_blocks_f32(bit_width, block_size, sign_values)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_ARM64_SVE2)
    if (features.sve) {
        if (auto kernel = select_sve2_decompress_blocks_f32(bit_width, block_size, sign_values)) {
            return kernel;
        }
    }
#endif

    return select_fallback_decompress_blocks_f32(bit_width, block_size, sign_values);
}

Kernel<KernelBlockF64Func> select_auto_decompress_block_f64(uint8_t bit_width, uint32_t block_size, bool sign_values) {
#if defined(PERNIX_BUILD_X86_AVX512_VBMI) || defined(PERNIX_BUILD_X86_AVX2)  || defined(PERNIX_BUILD_X86_BMI2) || defined(PERNIX_BUILD_ARM64_NEON) || defined(PERNIX_BUILD_ARM64_SVE2)
    const CpuFeatures features = get_cached_cpu_features();
#endif

#if defined(PERNIX_BUILD_X86_AVX512_VBMI)
    if (
        features.avx512f &&
        features.avx512dq &&
        features.avx512bw &&
        features.avx512vl &&
        features.avx512vbmi
    ) {
        if (auto kernel = select_avx512vbmi_decompress_block_f64(bit_width, block_size, sign_values)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_X86_AVX2)
    if (features.avx2) {
        if (auto kernel = select_avx2_decompress_block_f64(bit_width, block_size, sign_values)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_X86_BMI2)
    if (features.bmi2) {
        if (auto kernel = select_bmi2_decompress_block_f64(bit_width, block_size, sign_values)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_ARM64_NEON)
    if (features.neon) {
        if (auto kernel = select_neon_decompress_block_f64(bit_width, block_size, sign_values)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_ARM64_SVE2)
    if (features.sve) {
        if (auto kernel = select_sve2_decompress_block_f64(bit_width, block_size, sign_values)) {
            return kernel;
        }
    }
#endif

    return select_fallback_decompress_block_f64(bit_width, block_size, sign_values);
}

Kernel<KernelBlocksF64Func> select_auto_decompress_blocks_f64(uint8_t bit_width, uint32_t block_size, bool sign_values) {
#if defined(PERNIX_BUILD_X86_AVX512_VBMI) || defined(PERNIX_BUILD_X86_AVX2)  || defined(PERNIX_BUILD_X86_BMI2) || defined(PERNIX_BUILD_ARM64_NEON) || defined(PERNIX_BUILD_ARM64_SVE2)
    const CpuFeatures features = get_cached_cpu_features();
#endif

#if defined(PERNIX_BUILD_X86_AVX512_VBMI)
    if (
        features.avx512f &&
        features.avx512dq &&
        features.avx512bw &&
        features.avx512vl &&
        features.avx512vbmi
    ) {
        if (auto kernel = select_avx512vbmi_decompress_blocks_f64(bit_width, block_size, sign_values)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_X86_AVX2)
    if (features.avx2) {
        if (auto kernel = select_avx2_decompress_blocks_f64(bit_width, block_size, sign_values)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_X86_BMI2)
    if (features.bmi2) {
        if (auto kernel = select_bmi2_decompress_blocks_f64(bit_width, block_size, sign_values)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_ARM64_NEON)
    if (features.neon) {
        if (auto kernel = select_neon_decompress_blocks_f64(bit_width, block_size, sign_values)) {
            return kernel;
        }
    }
#endif

#if defined(PERNIX_BUILD_ARM64_SVE2)
    if (features.sve) {
        if (auto kernel = select_sve2_decompress_blocks_f64(bit_width, block_size, sign_values)) {
            return kernel;
        }
    }
#endif

    return select_fallback_decompress_blocks_f64(bit_width, block_size, sign_values);
}
}
