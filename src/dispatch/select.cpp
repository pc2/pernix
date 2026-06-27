#include <pernix/dispatch/select.h>
#include <pernix/dispatch/cpu_features.h>

namespace pernix::internal {
    bool is_optional_fallback_backend(const Backend backend) {
        switch (backend) {
            case Backend::FallbackStdpar:
            case Backend::FallbackSimd:
                return true;
            default:
                return false;
        }
    }

    bool is_compiled_backend(const Backend backend) {
        switch (backend) {
            case Backend::Auto:
            case Backend::Fallback:
                return true;
#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
            case Backend::FallbackStdpar:
                return true;
#endif
#if defined(PERNIX_BUILD_FALLBACK_SIMD)
            case Backend::FallbackSimd:
                return true;
#endif
#if defined(PERNIX_BUILD_X86_AVX2)
            case Backend::X86Avx2:
                return true;
#endif
#if defined(PERNIX_BUILD_X86_BMI2)
            case Backend::X86Bmi2:
                return true;
#endif
#if defined(PERNIX_BUILD_X86_AVX512_VBMI)
            case Backend::X86Avx512Vbmi:
                return true;
#endif
#if defined(PERNIX_BUILD_ARM64_NEON)
            case Backend::Arm64Neon:
                return true;
#endif
#if defined(PERNIX_BUILD_ARM64_SVE2)
            case Backend::Arm64Sve:
                return true;
#endif
            default:
                return false;
        }
    }

    bool is_backend_supported_on_machine(const Backend backend) {
        const CpuFeatures features = get_cached_cpu_features();

        switch (backend) {
            case Backend::Auto:
            case Backend::Fallback:
            case Backend::FallbackStdpar:
            case Backend::FallbackSimd:
                return true;
            case Backend::X86Avx2:
                return features.avx2;
            case Backend::X86Bmi2:
                return features.avx2 && features.bmi2;
            case Backend::X86Avx512Vbmi:
                return features.avx512f && features.avx512dq && features.avx512bw && features.avx512vl &&
                       features.avx512vbmi;
            case Backend::Arm64Neon:
                return features.neon;
            case Backend::Arm64Sve:
                return features.sve2;
            default:
                return false;
        }
    }

    Kernel<KernelBlockF32Func> select_compress_block_f32(Backend backend, u8 bit_width, u32 block_size) {
        if (is_optional_fallback_backend(backend) && !is_compiled_backend(backend)) {
            return {"unsupported_implementation", nullptr};
        }
        if (backend != Backend::Auto && backend != Backend::Fallback && is_compiled_backend(backend) &&
            !is_backend_supported_on_machine(backend)) {
            return {"unsupported_implementation", nullptr};
        }
        switch (backend) {
            case Backend::Auto:
                return select_auto_compress_block_f32(bit_width, block_size);
            case Backend::Fallback:
                return select_fallback_compress_block_f32(bit_width, block_size);
#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
            case Backend::FallbackStdpar:
                return select_fallback_stdpar_compress_block_f32(bit_width, block_size);
#endif
#if defined(PERNIX_BUILD_FALLBACK_SIMD)
            case Backend::FallbackSimd:
                return select_fallback_simd_compress_block_f32(bit_width, block_size);
#endif
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
            case Backend::Arm64Sve: {
                if (get_cached_cpu_features().sve2) {
                    return select_sve2_compress_block_f32(bit_width, block_size);
                }
                return {"invalid_backend", nullptr};
            }
#endif
            default:
                return {"invalid_backend", nullptr};
        }
    }

    Kernel<KernelBlocksF32Func> select_compress_blocks_f32(Backend backend, u8 bit_width, u32 block_size) {
        if (is_optional_fallback_backend(backend) && !is_compiled_backend(backend)) {
            return {"unsupported_implementation", nullptr};
        }
        if (backend != Backend::Auto && backend != Backend::Fallback && is_compiled_backend(backend) &&
            !is_backend_supported_on_machine(backend)) {
            return {"unsupported_implementation", nullptr};
        }
        switch (backend) {
            case Backend::Auto:
                return select_auto_compress_blocks_f32(bit_width, block_size);
            case Backend::Fallback:
                return select_fallback_compress_blocks_f32(bit_width, block_size);
#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
            case Backend::FallbackStdpar:
                return select_fallback_stdpar_compress_blocks_f32(bit_width, block_size);
#endif
#if defined(PERNIX_BUILD_FALLBACK_SIMD)
            case Backend::FallbackSimd:
                return select_fallback_simd_compress_blocks_f32(bit_width, block_size);
#endif
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
            case Backend::Arm64Sve: {
                if (get_cached_cpu_features().sve2) {
                    return select_sve2_compress_blocks_f32(bit_width, block_size);
                }
                return {"invalid_backend", nullptr};
            }
#endif
            default:
                return {"invalid_backend", nullptr};
        }
    }

    Kernel<KernelBlockF64Func> select_compress_block_f64(Backend backend, u8 bit_width, u32 block_size) {
        if (is_optional_fallback_backend(backend) && !is_compiled_backend(backend)) {
            return {"unsupported_implementation", nullptr};
        }
        if (backend != Backend::Auto && backend != Backend::Fallback && is_compiled_backend(backend) &&
            !is_backend_supported_on_machine(backend)) {
            return {"unsupported_implementation", nullptr};
        }
        switch (backend) {
            case Backend::Auto:
                return select_auto_compress_block_f64(bit_width, block_size);
            case Backend::Fallback:
                return select_fallback_compress_block_f64(bit_width, block_size);
#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
            case Backend::FallbackStdpar:
                return select_fallback_stdpar_compress_block_f64(bit_width, block_size);
#endif
#if defined(PERNIX_BUILD_FALLBACK_SIMD)
            case Backend::FallbackSimd:
                return select_fallback_simd_compress_block_f64(bit_width, block_size);
#endif
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
            case Backend::Arm64Sve: {
                if (get_cached_cpu_features().sve2) {
                    return select_sve2_compress_block_f64(bit_width, block_size);
                }
                return {"invalid_backend", nullptr};
            }
#endif
            default:
                return {"invalid_backend", nullptr};
        }
    }

    Kernel<KernelBlocksF64Func> select_compress_blocks_f64(Backend backend, u8 bit_width, u32 block_size) {
        if (is_optional_fallback_backend(backend) && !is_compiled_backend(backend)) {
            return {"unsupported_implementation", nullptr};
        }
        if (backend != Backend::Auto && backend != Backend::Fallback && is_compiled_backend(backend) &&
            !is_backend_supported_on_machine(backend)) {
            return {"unsupported_implementation", nullptr};
        }
        switch (backend) {
            case Backend::Auto:
                return select_auto_compress_blocks_f64(bit_width, block_size);
            case Backend::Fallback:
                return select_fallback_compress_blocks_f64(bit_width, block_size);
#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
            case Backend::FallbackStdpar:
                return select_fallback_stdpar_compress_blocks_f64(bit_width, block_size);
#endif
#if defined(PERNIX_BUILD_FALLBACK_SIMD)
            case Backend::FallbackSimd:
                return select_fallback_simd_compress_blocks_f64(bit_width, block_size);
#endif
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
            case Backend::Arm64Sve: {
                if (get_cached_cpu_features().sve2) {
                    return select_sve2_compress_blocks_f64(bit_width, block_size);
                }
                return {"invalid_backend", nullptr};
            }
#endif
            default:
                return {"invalid_backend", nullptr};
        }
    }

    Kernel<KernelBlockF32Func> select_decompress_block_f32(Backend backend, u8 bit_width, u32 block_size,
                                                           bool sign_values) {
        if (is_optional_fallback_backend(backend) && !is_compiled_backend(backend)) {
            return {"unsupported_implementation", nullptr};
        }
        if (backend != Backend::Auto && backend != Backend::Fallback && is_compiled_backend(backend) &&
            !is_backend_supported_on_machine(backend)) {
            return {"unsupported_implementation", nullptr};
        }
        switch (backend) {
            case Backend::Auto:
                return select_auto_decompress_block_f32(bit_width, block_size, sign_values);
            case Backend::Fallback:
                return select_fallback_decompress_block_f32(bit_width, block_size, sign_values);
#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
            case Backend::FallbackStdpar:
                return select_fallback_stdpar_decompress_block_f32(bit_width, block_size, sign_values);
#endif
#if defined(PERNIX_BUILD_FALLBACK_SIMD)
            case Backend::FallbackSimd:
                return select_fallback_simd_decompress_block_f32(bit_width, block_size, sign_values);
#endif
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
            case Backend::Arm64Sve: {
                if (get_cached_cpu_features().sve2) {
                    return select_sve2_decompress_block_f32(bit_width, block_size, sign_values);
                }
                return {"invalid_backend", nullptr};
            }
#endif
            default:
                return {"invalid_backend", nullptr};
        }
    }

    Kernel<KernelBlocksF32Func> select_decompress_blocks_f32(Backend backend, u8 bit_width, u32 block_size,
                                                             bool sign_values) {
        if (is_optional_fallback_backend(backend) && !is_compiled_backend(backend)) {
            return {"unsupported_implementation", nullptr};
        }
        if (backend != Backend::Auto && backend != Backend::Fallback && is_compiled_backend(backend) &&
            !is_backend_supported_on_machine(backend)) {
            return {"unsupported_implementation", nullptr};
        }
        switch (backend) {
            case Backend::Auto:
                return select_auto_decompress_blocks_f32(bit_width, block_size, sign_values);
            case Backend::Fallback:
                return select_fallback_decompress_blocks_f32(bit_width, block_size, sign_values);
#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
            case Backend::FallbackStdpar:
                return select_fallback_stdpar_decompress_blocks_f32(bit_width, block_size, sign_values);
#endif
#if defined(PERNIX_BUILD_FALLBACK_SIMD)
            case Backend::FallbackSimd:
                return select_fallback_simd_decompress_blocks_f32(bit_width, block_size, sign_values);
#endif
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
            case Backend::Arm64Sve: {
                if (get_cached_cpu_features().sve2) {
                    return select_sve2_decompress_blocks_f32(bit_width, block_size, sign_values);
                }
                return {"invalid_backend", nullptr};
            }
#endif
            default:
                return {"invalid_backend", nullptr};
        }
    }

    Kernel<KernelBlockF64Func> select_decompress_block_f64(Backend backend, u8 bit_width, u32 block_size,
                                                           bool sign_values) {
        if (is_optional_fallback_backend(backend) && !is_compiled_backend(backend)) {
            return {"unsupported_implementation", nullptr};
        }
        if (backend != Backend::Auto && backend != Backend::Fallback && is_compiled_backend(backend) &&
            !is_backend_supported_on_machine(backend)) {
            return {"unsupported_implementation", nullptr};
        }
        switch (backend) {
            case Backend::Auto:
                return select_auto_decompress_block_f64(bit_width, block_size, sign_values);
            case Backend::Fallback:
                return select_fallback_decompress_block_f64(bit_width, block_size, sign_values);
#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
            case Backend::FallbackStdpar:
                return select_fallback_stdpar_decompress_block_f64(bit_width, block_size, sign_values);
#endif
#if defined(PERNIX_BUILD_FALLBACK_SIMD)
            case Backend::FallbackSimd:
                return select_fallback_simd_decompress_block_f64(bit_width, block_size, sign_values);
#endif
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
            case Backend::Arm64Sve: {
                if (get_cached_cpu_features().sve2) {
                    return select_sve2_decompress_block_f64(bit_width, block_size, sign_values);
                }
                return {"invalid_backend", nullptr};
            }
#endif
            default:
                return {"invalid_backend", nullptr};
        }
    }

    Kernel<KernelBlocksF64Func> select_decompress_blocks_f64(Backend backend, u8 bit_width, u32 block_size,
                                                             bool sign_values) {
        if (is_optional_fallback_backend(backend) && !is_compiled_backend(backend)) {
            return {"unsupported_implementation", nullptr};
        }
        if (backend != Backend::Auto && backend != Backend::Fallback && is_compiled_backend(backend) &&
            !is_backend_supported_on_machine(backend)) {
            return {"unsupported_implementation", nullptr};
        }
        switch (backend) {
            case Backend::Auto:
                return select_auto_decompress_blocks_f64(bit_width, block_size, sign_values);
            case Backend::Fallback:
                return select_fallback_decompress_blocks_f64(bit_width, block_size, sign_values);
#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
            case Backend::FallbackStdpar:
                return select_fallback_stdpar_decompress_blocks_f64(bit_width, block_size, sign_values);
#endif
#if defined(PERNIX_BUILD_FALLBACK_SIMD)
            case Backend::FallbackSimd:
                return select_fallback_simd_decompress_blocks_f64(bit_width, block_size, sign_values);
#endif
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
            case Backend::Arm64Sve: {
                if (get_cached_cpu_features().sve2) {
                    return select_sve2_decompress_blocks_f64(bit_width, block_size, sign_values);
                }
                return {"invalid_backend", nullptr};
            }
#endif
            default:
                return {"invalid_backend", nullptr};
        }
    }

    Kernel<KernelBlockF32Func> select_auto_compress_block_f32(u8 bit_width, u32 block_size) {
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
        if (features.sve2) {
            if (auto kernel = select_sve2_compress_block_f32(bit_width, block_size)) {
                return kernel;
            }
        }
#endif

        return select_fallback_compress_block_f32(bit_width, block_size);
    }

    Kernel<KernelBlocksF32Func> select_auto_compress_blocks_f32(u8 bit_width, u32 block_size) {
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
        if (features.sve2) {
            if (auto kernel = select_sve2_compress_blocks_f32(bit_width, block_size)) {
                return kernel;
            }
        }
#endif

        return select_fallback_compress_blocks_f32(bit_width, block_size);
    }

    Kernel<KernelBlockF64Func> select_auto_compress_block_f64(u8 bit_width, u32 block_size) {
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
        if (features.sve2) {
            if (auto kernel = select_sve2_compress_block_f64(bit_width, block_size)) {
                return kernel;
            }
        }
#endif

        return select_fallback_compress_block_f64(bit_width, block_size);
    }

    Kernel<KernelBlocksF64Func> select_auto_compress_blocks_f64(u8 bit_width, u32 block_size) {
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
        if (features.sve2) {
            if (auto kernel = select_sve2_compress_blocks_f64(bit_width, block_size)) {
                return kernel;
            }
        }
#endif

        return select_fallback_compress_blocks_f64(bit_width, block_size);
    }

    Kernel<KernelBlockF32Func> select_auto_decompress_block_f32(u8 bit_width, u32 block_size, bool sign_values) {
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
        if (features.sve2) {
            if (auto kernel = select_sve2_decompress_block_f32(bit_width, block_size, sign_values)) {
                return kernel;
            }
        }
#endif

        return select_fallback_decompress_block_f32(bit_width, block_size, sign_values);
    }

    Kernel<KernelBlocksF32Func> select_auto_decompress_blocks_f32(u8 bit_width, u32 block_size, bool sign_values) {
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
        if (features.sve2) {
            if (auto kernel = select_sve2_decompress_blocks_f32(bit_width, block_size, sign_values)) {
                return kernel;
            }
        }
#endif

        return select_fallback_decompress_blocks_f32(bit_width, block_size, sign_values);
    }

    Kernel<KernelBlockF64Func> select_auto_decompress_block_f64(u8 bit_width, u32 block_size, bool sign_values) {
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
        if (features.sve2) {
            if (auto kernel = select_sve2_decompress_block_f64(bit_width, block_size, sign_values)) {
                return kernel;
            }
        }
#endif

        return select_fallback_decompress_block_f64(bit_width, block_size, sign_values);
    }

    Kernel<KernelBlocksF64Func> select_auto_decompress_blocks_f64(u8 bit_width, u32 block_size, bool sign_values) {
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
        if (features.sve2) {
            if (auto kernel = select_sve2_decompress_blocks_f64(bit_width, block_size, sign_values)) {
                return kernel;
            }
        }
#endif

        return select_fallback_decompress_blocks_f64(bit_width, block_size, sign_values);
    }
}
