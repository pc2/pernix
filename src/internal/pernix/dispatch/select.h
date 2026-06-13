#ifndef PERNIX_SELECT_H
#define PERNIX_SELECT_H

#include <pernix/pernix.hpp>
#include <pernix/dispatch/kernel.h>

namespace pernix::internal {
Kernel<KernelBlockF32Func> select_compress_block_f32(Backend backend, uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlocksF32Func> select_compress_blocks_f32(Backend backend, uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlockF64Func> select_compress_block_f64(Backend backend, uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlocksF64Func> select_compress_blocks_f64(Backend backend, uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlockF32Func> select_decompress_block_f32(Backend backend, uint8_t bit_width, uint32_t block_size, bool sign_values);

Kernel<KernelBlocksF32Func> select_decompress_blocks_f32(Backend backend, uint8_t bit_width, uint32_t block_size, bool sign_values);

Kernel<KernelBlockF64Func> select_decompress_block_f64(Backend backend, uint8_t bit_width, uint32_t block_size, bool sign_values);

Kernel<KernelBlocksF64Func> select_decompress_blocks_f64(Backend backend, uint8_t bit_width, uint32_t block_size, bool sign_values);


Kernel<KernelBlockF32Func> select_auto_compress_block_f32(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlocksF32Func> select_auto_compress_blocks_f32(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlockF64Func> select_auto_compress_block_f64(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlocksF64Func> select_auto_compress_blocks_f64(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlockF32Func> select_auto_decompress_block_f32(uint8_t bit_width, uint32_t block_size, bool sign_values);

Kernel<KernelBlocksF32Func> select_auto_decompress_blocks_f32(uint8_t bit_width, uint32_t block_size, bool sign_values);

Kernel<KernelBlockF64Func> select_auto_decompress_block_f64(uint8_t bit_width, uint32_t block_size, bool sign_values);

Kernel<KernelBlocksF64Func> select_auto_decompress_blocks_f64(uint8_t bit_width, uint32_t block_size, bool sign_values);


Kernel<KernelBlockF32Func> select_fallback_compress_block_f32(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlocksF32Func> select_fallback_compress_blocks_f32(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlockF64Func> select_fallback_compress_block_f64(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlocksF64Func> select_fallback_compress_blocks_f64(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlockF32Func> select_fallback_decompress_block_f32(uint8_t bit_width, uint32_t block_size, bool sign_values);

Kernel<KernelBlocksF32Func> select_fallback_decompress_blocks_f32(uint8_t bit_width, uint32_t block_size, bool sign_values);

Kernel<KernelBlockF64Func> select_fallback_decompress_block_f64(uint8_t bit_width, uint32_t block_size, bool sign_values);

Kernel<KernelBlocksF64Func> select_fallback_decompress_blocks_f64(uint8_t bit_width, uint32_t block_size, bool sign_values);

#if defined(PERNIX_BUILD_X86_AVX2)

Kernel<KernelBlockF32Func> select_avx2_compress_block_f32(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlocksF32Func> select_avx2_compress_blocks_f32(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlockF64Func> select_avx2_compress_block_f64(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlocksF64Func> select_avx2_compress_blocks_f64(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlockF32Func> select_avx2_decompress_block_f32(uint8_t bit_width, uint32_t block_size, bool sign_values);

Kernel<KernelBlocksF32Func> select_avx2_decompress_blocks_f32(uint8_t bit_width, uint32_t block_size, bool sign_values);

Kernel<KernelBlockF64Func> select_avx2_decompress_block_f64(uint8_t bit_width, uint32_t block_size, bool sign_values);

Kernel<KernelBlocksF64Func> select_avx2_decompress_blocks_f64(uint8_t bit_width, uint32_t block_size, bool sign_values);

#endif

#if defined(PERNIX_BUILD_X86_BMI2)

Kernel<KernelBlockF32Func> select_bmi2_compress_block_f32(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlocksF32Func> select_bmi2_compress_blocks_f32(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlockF64Func> select_bmi2_compress_block_f64(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlocksF64Func> select_bmi2_compress_blocks_f64(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlockF32Func> select_bmi2_decompress_block_f32(uint8_t bit_width, uint32_t block_size, bool sign_values);

Kernel<KernelBlocksF32Func> select_bmi2_decompress_blocks_f32(uint8_t bit_width, uint32_t block_size, bool sign_values);

Kernel<KernelBlockF64Func> select_bmi2_decompress_block_f64(uint8_t bit_width, uint32_t block_size, bool sign_values);

Kernel<KernelBlocksF64Func> select_bmi2_decompress_blocks_f64(uint8_t bit_width, uint32_t block_size, bool sign_values);

#endif

#if defined(PERNIX_BUILD_X86_AVX512_VBMI)

Kernel<KernelBlockF32Func> select_avx512vbmi_compress_block_f32(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlocksF32Func> select_avx512vbmi_compress_blocks_f32(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlockF64Func> select_avx512vbmi_compress_block_f64(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlocksF64Func> select_avx512vbmi_compress_blocks_f64(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlockF32Func> select_avx512vbmi_decompress_block_f32(uint8_t bit_width, uint32_t block_size, bool sign_values);

Kernel<KernelBlocksF32Func> select_avx512vbmi_decompress_blocks_f32(uint8_t bit_width, uint32_t block_size, bool sign_values);

Kernel<KernelBlockF64Func> select_avx512vbmi_decompress_block_f64(uint8_t bit_width, uint32_t block_size, bool sign_values);

Kernel<KernelBlocksF64Func> select_avx512vbmi_decompress_blocks_f64(uint8_t bit_width, uint32_t block_size, bool sign_values);

#endif

#if defined(PERNIX_BUILD_ARM64_NEON)

Kernel<KernelBlockF32Func> select_neon_compress_block_f32(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlocksF32Func> select_neon_compress_blocks_f32(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlockF64Func> select_neon_compress_block_f64(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlocksF64Func> select_neon_compress_blocks_f64(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlockF32Func> select_neon_decompress_block_f32(uint8_t bit_width, uint32_t block_size, bool sign_values);

Kernel<KernelBlocksF32Func> select_neon_decompress_blocks_f32(uint8_t bit_width, uint32_t block_size, bool sign_values);

Kernel<KernelBlockF64Func> select_neon_decompress_block_f64(uint8_t bit_width, uint32_t block_size, bool sign_values);

Kernel<KernelBlocksF64Func> select_neon_decompress_blocks_f64(uint8_t bit_width, uint32_t block_size, bool sign_values);

#endif

#if defined(PERNIX_BUILD_ARM64_SVE2)

Kernel<KernelBlockF32Func> select_sve2_compress_block_f32(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlocksF32Func> select_sve2_compress_blocks_f32(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlockF64Func> select_sve2_compress_block_f64(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlocksF64Func> select_sve2_compress_blocks_f64(uint8_t bit_width, uint32_t block_size);

Kernel<KernelBlockF32Func> select_sve2_decompress_block_f32(uint8_t bit_width, uint32_t block_size, bool sign_values);

Kernel<KernelBlocksF32Func> select_sve2_decompress_blocks_f32(uint8_t bit_width, uint32_t block_size, bool sign_values);

Kernel<KernelBlockF64Func> select_sve2_decompress_block_f64(uint8_t bit_width, uint32_t block_size, bool sign_values);

Kernel<KernelBlocksF64Func> select_sve2_decompress_blocks_f64(uint8_t bit_width, uint32_t block_size, bool sign_values);

#endif
}

#endif //PERNIX_SELECT_H
