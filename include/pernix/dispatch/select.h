#ifndef PERNIX_SELECT_H
#define PERNIX_SELECT_H

#include <pernix/backend.hpp>
#include <pernix/dispatch/kernel.h>

#if !defined(PERNIX_BUILD_X86_AVX2) && !defined(PERNIX_DISABLE_AVX2)
#if defined(PERNIX_USE_SIMDE) || defined(__AVX2__)
#define PERNIX_BUILD_X86_AVX2 1
#endif
#endif

#if !defined(PERNIX_BUILD_X86_BMI2) && !defined(PERNIX_DISABLE_BMI2)
#if defined(PERNIX_USE_SIMDE) || (defined(__AVX2__) && defined(__BMI2__))
#define PERNIX_BUILD_X86_BMI2 1
#endif
#endif

#if !defined(PERNIX_BUILD_X86_AVX512_VBMI) && !defined(PERNIX_DISABLE_AVX512)
#if defined(PERNIX_USE_SIMDE) || (defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512VBMI__))
#define PERNIX_BUILD_X86_AVX512_VBMI 1
#endif
#endif

#if !defined(PERNIX_BUILD_ARM64_NEON)
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#define PERNIX_BUILD_ARM64_NEON 1
#endif
#endif

#if !defined(PERNIX_BUILD_ARM64_SVE2)
#if defined(__ARM_FEATURE_SVE2) || defined(PERNIX_USE_SIMDE)
#define PERNIX_BUILD_ARM64_SVE2 1
#endif
#endif

namespace pernix::internal {
    Kernel<KernelBlockF32Func> select_compress_block_f32(Backend backend, u8 bit_width, u32 block_size);

    Kernel<KernelBlocksF32Func> select_compress_blocks_f32(Backend backend, u8 bit_width, u32 block_size);

    Kernel<KernelBlockF64Func> select_compress_block_f64(Backend backend, u8 bit_width, u32 block_size);

    Kernel<KernelBlocksF64Func> select_compress_blocks_f64(Backend backend, u8 bit_width, u32 block_size);

    Kernel<KernelBlockF32Func> select_decompress_block_f32(Backend backend, u8 bit_width, u32 block_size,
                                                           bool sign_values);

    Kernel<KernelBlocksF32Func> select_decompress_blocks_f32(Backend backend, u8 bit_width, u32 block_size,
                                                             bool sign_values);

    Kernel<KernelBlockF64Func> select_decompress_block_f64(Backend backend, u8 bit_width, u32 block_size,
                                                           bool sign_values);

    Kernel<KernelBlocksF64Func> select_decompress_blocks_f64(Backend backend, u8 bit_width, u32 block_size,
                                                             bool sign_values);


    Kernel<KernelBlockF32Func> select_auto_compress_block_f32(u8 bit_width, u32 block_size);

    Kernel<KernelBlocksF32Func> select_auto_compress_blocks_f32(u8 bit_width, u32 block_size);

    Kernel<KernelBlockF64Func> select_auto_compress_block_f64(u8 bit_width, u32 block_size);

    Kernel<KernelBlocksF64Func> select_auto_compress_blocks_f64(u8 bit_width, u32 block_size);

    Kernel<KernelBlockF32Func> select_auto_decompress_block_f32(u8 bit_width, u32 block_size, bool sign_values);

    Kernel<KernelBlocksF32Func> select_auto_decompress_blocks_f32(u8 bit_width, u32 block_size, bool sign_values);

    Kernel<KernelBlockF64Func> select_auto_decompress_block_f64(u8 bit_width, u32 block_size, bool sign_values);

    Kernel<KernelBlocksF64Func> select_auto_decompress_blocks_f64(u8 bit_width, u32 block_size, bool sign_values);


    Kernel<KernelBlockF32Func> select_fallback_compress_block_f32(u8 bit_width, u32 block_size);

    Kernel<KernelBlocksF32Func> select_fallback_compress_blocks_f32(u8 bit_width, u32 block_size);

    Kernel<KernelBlockF64Func> select_fallback_compress_block_f64(u8 bit_width, u32 block_size);

    Kernel<KernelBlocksF64Func> select_fallback_compress_blocks_f64(u8 bit_width, u32 block_size);

    Kernel<KernelBlockF32Func>
    select_fallback_decompress_block_f32(u8 bit_width, u32 block_size, bool sign_values);

    Kernel<KernelBlocksF32Func> select_fallback_decompress_blocks_f32(u8 bit_width, u32 block_size,
                                                                      bool sign_values);

    Kernel<KernelBlockF64Func>
    select_fallback_decompress_block_f64(u8 bit_width, u32 block_size, bool sign_values);

    Kernel<KernelBlocksF64Func> select_fallback_decompress_blocks_f64(u8 bit_width, u32 block_size,
                                                                      bool sign_values);

    Kernel<KernelBlockF32Func> select_fallback_scalar_compress_block_f32(u8 bit_width, u32 block_size);

    Kernel<KernelBlocksF32Func> select_fallback_scalar_compress_blocks_f32(u8 bit_width, u32 block_size);

    Kernel<KernelBlockF64Func> select_fallback_scalar_compress_block_f64(u8 bit_width, u32 block_size);

    Kernel<KernelBlocksF64Func> select_fallback_scalar_compress_blocks_f64(u8 bit_width, u32 block_size);

    Kernel<KernelBlockF32Func>
    select_fallback_scalar_decompress_block_f32(u8 bit_width, u32 block_size, bool sign_values);

    Kernel<KernelBlocksF32Func> select_fallback_scalar_decompress_blocks_f32(u8 bit_width, u32 block_size,
                                                                             bool sign_values);

    Kernel<KernelBlockF64Func>
    select_fallback_scalar_decompress_block_f64(u8 bit_width, u32 block_size, bool sign_values);

    Kernel<KernelBlocksF64Func> select_fallback_scalar_decompress_blocks_f64(u8 bit_width, u32 block_size,
                                                                             bool sign_values);

    Kernel<KernelBlockF32Func> select_fallback_stdpar_compress_block_f32(u8 bit_width, u32 block_size);

    Kernel<KernelBlocksF32Func> select_fallback_stdpar_compress_blocks_f32(u8 bit_width, u32 block_size);

    Kernel<KernelBlockF64Func> select_fallback_stdpar_compress_block_f64(u8 bit_width, u32 block_size);

    Kernel<KernelBlocksF64Func> select_fallback_stdpar_compress_blocks_f64(u8 bit_width, u32 block_size);

    Kernel<KernelBlockF32Func>
    select_fallback_stdpar_decompress_block_f32(u8 bit_width, u32 block_size, bool sign_values);

    Kernel<KernelBlocksF32Func> select_fallback_stdpar_decompress_blocks_f32(u8 bit_width, u32 block_size,
                                                                             bool sign_values);

    Kernel<KernelBlockF64Func>
    select_fallback_stdpar_decompress_block_f64(u8 bit_width, u32 block_size, bool sign_values);

    Kernel<KernelBlocksF64Func> select_fallback_stdpar_decompress_blocks_f64(u8 bit_width, u32 block_size,
                                                                             bool sign_values);

    Kernel<KernelBlockF32Func> select_fallback_simd_compress_block_f32(u8 bit_width, u32 block_size);

    Kernel<KernelBlocksF32Func> select_fallback_simd_compress_blocks_f32(u8 bit_width, u32 block_size);

    Kernel<KernelBlockF64Func> select_fallback_simd_compress_block_f64(u8 bit_width, u32 block_size);

    Kernel<KernelBlocksF64Func> select_fallback_simd_compress_blocks_f64(u8 bit_width, u32 block_size);

    Kernel<KernelBlockF32Func>
    select_fallback_simd_decompress_block_f32(u8 bit_width, u32 block_size, bool sign_values);

    Kernel<KernelBlocksF32Func> select_fallback_simd_decompress_blocks_f32(u8 bit_width, u32 block_size,
                                                                           bool sign_values);

    Kernel<KernelBlockF64Func>
    select_fallback_simd_decompress_block_f64(u8 bit_width, u32 block_size, bool sign_values);

    Kernel<KernelBlocksF64Func> select_fallback_simd_decompress_blocks_f64(u8 bit_width, u32 block_size,
                                                                           bool sign_values);

#if defined(PERNIX_BUILD_X86_AVX2)

    Kernel<KernelBlockF32Func> select_avx2_compress_block_f32(u8 bit_width, u32 block_size);

    Kernel<KernelBlocksF32Func> select_avx2_compress_blocks_f32(u8 bit_width, u32 block_size);

    Kernel<KernelBlockF64Func> select_avx2_compress_block_f64(u8 bit_width, u32 block_size);

    Kernel<KernelBlocksF64Func> select_avx2_compress_blocks_f64(u8 bit_width, u32 block_size);

    Kernel<KernelBlockF32Func> select_avx2_decompress_block_f32(u8 bit_width, u32 block_size, bool sign_values);

    Kernel<KernelBlocksF32Func> select_avx2_decompress_blocks_f32(u8 bit_width, u32 block_size, bool sign_values);

    Kernel<KernelBlockF64Func> select_avx2_decompress_block_f64(u8 bit_width, u32 block_size, bool sign_values);

    Kernel<KernelBlocksF64Func> select_avx2_decompress_blocks_f64(u8 bit_width, u32 block_size, bool sign_values);

#endif

#if defined(PERNIX_BUILD_X86_BMI2)

    Kernel<KernelBlockF32Func> select_bmi2_compress_block_f32(u8 bit_width, u32 block_size);

    Kernel<KernelBlocksF32Func> select_bmi2_compress_blocks_f32(u8 bit_width, u32 block_size);

    Kernel<KernelBlockF64Func> select_bmi2_compress_block_f64(u8 bit_width, u32 block_size);

    Kernel<KernelBlocksF64Func> select_bmi2_compress_blocks_f64(u8 bit_width, u32 block_size);

    Kernel<KernelBlockF32Func> select_bmi2_decompress_block_f32(u8 bit_width, u32 block_size, bool sign_values);

    Kernel<KernelBlocksF32Func> select_bmi2_decompress_blocks_f32(u8 bit_width, u32 block_size, bool sign_values);

    Kernel<KernelBlockF64Func> select_bmi2_decompress_block_f64(u8 bit_width, u32 block_size, bool sign_values);

    Kernel<KernelBlocksF64Func> select_bmi2_decompress_blocks_f64(u8 bit_width, u32 block_size, bool sign_values);

#endif

#if defined(PERNIX_BUILD_X86_AVX512_VBMI)

    Kernel<KernelBlockF32Func> select_avx512vbmi_compress_block_f32(u8 bit_width, u32 block_size);

    Kernel<KernelBlocksF32Func> select_avx512vbmi_compress_blocks_f32(u8 bit_width, u32 block_size);

    Kernel<KernelBlockF64Func> select_avx512vbmi_compress_block_f64(u8 bit_width, u32 block_size);

    Kernel<KernelBlocksF64Func> select_avx512vbmi_compress_blocks_f64(u8 bit_width, u32 block_size);

    Kernel<KernelBlockF32Func> select_avx512vbmi_decompress_block_f32(u8 bit_width, u32 block_size,
                                                                      bool sign_values);

    Kernel<KernelBlocksF32Func> select_avx512vbmi_decompress_blocks_f32(u8 bit_width, u32 block_size,
                                                                        bool sign_values);

    Kernel<KernelBlockF64Func> select_avx512vbmi_decompress_block_f64(u8 bit_width, u32 block_size,
                                                                      bool sign_values);

    Kernel<KernelBlocksF64Func> select_avx512vbmi_decompress_blocks_f64(u8 bit_width, u32 block_size,
                                                                        bool sign_values);

#endif

#if defined(PERNIX_BUILD_ARM64_NEON)

    Kernel<KernelBlockF32Func> select_neon_compress_block_f32(u8 bit_width, u32 block_size);

    Kernel<KernelBlocksF32Func> select_neon_compress_blocks_f32(u8 bit_width, u32 block_size);

    Kernel<KernelBlockF64Func> select_neon_compress_block_f64(u8 bit_width, u32 block_size);

    Kernel<KernelBlocksF64Func> select_neon_compress_blocks_f64(u8 bit_width, u32 block_size);

    Kernel<KernelBlockF32Func> select_neon_decompress_block_f32(u8 bit_width, u32 block_size, bool sign_values);

    Kernel<KernelBlocksF32Func> select_neon_decompress_blocks_f32(u8 bit_width, u32 block_size, bool sign_values);

    Kernel<KernelBlockF64Func> select_neon_decompress_block_f64(u8 bit_width, u32 block_size, bool sign_values);

    Kernel<KernelBlocksF64Func> select_neon_decompress_blocks_f64(u8 bit_width, u32 block_size, bool sign_values);

#endif

#if defined(PERNIX_BUILD_ARM64_SVE2)

    Kernel<KernelBlockF32Func> select_sve2_compress_block_f32(u8 bit_width, u32 block_size);

    Kernel<KernelBlocksF32Func> select_sve2_compress_blocks_f32(u8 bit_width, u32 block_size);

    Kernel<KernelBlockF64Func> select_sve2_compress_block_f64(u8 bit_width, u32 block_size);

    Kernel<KernelBlocksF64Func> select_sve2_compress_blocks_f64(u8 bit_width, u32 block_size);

    Kernel<KernelBlockF32Func> select_sve2_decompress_block_f32(u8 bit_width, u32 block_size, bool sign_values);

    Kernel<KernelBlocksF32Func> select_sve2_decompress_blocks_f32(u8 bit_width, u32 block_size, bool sign_values);

    Kernel<KernelBlockF64Func> select_sve2_decompress_block_f64(u8 bit_width, u32 block_size, bool sign_values);

    Kernel<KernelBlocksF64Func> select_sve2_decompress_blocks_f64(u8 bit_width, u32 block_size, bool sign_values);

#endif
}

#if !defined(PERNIX_BUILD_LIB)
#include <pernix/dispatch/select_impl.h>
#endif

#endif //PERNIX_SELECT_H
