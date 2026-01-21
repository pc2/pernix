#ifndef PERNIX_DECOMPRESSION_H
#define PERNIX_DECOMPRESSION_H

#include <pernix/bitpacking/unpacking.h>
#include <pernix/helper.h>
#include <pernix/quantization/dequantization.h>

#include <cmath>
#include <cstdint>
#include <vector>

using namespace pernix::bitpacking;
using namespace pernix::quantization;

namespace pernix {
#ifdef PERNIX_AVX2_ENABLED

/**
 * @brief Decompress a single 512\-bit block using AVX2 and BMI2 instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 16).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 *
 * @param input pointer to the start of the compressed block.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 and BMI2 support.
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 16)
int mm256_decompress_block_bmi2(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = 512 / BIT_WIDTH;
    constexpr uint32_t iterations_8       = elements_per_block / 8;
    constexpr uint8_t remaining           = elements_per_block - iterations_8 * 8;

    const __m256 scale_v = _mm256_set1_ps(scale);
#pragma GCC unroll 4
    for (uint32_t iter = 0; iter < iterations_8; iter++) {
        const __m256i unpacked   = mm256_unpack_epi32_bmi2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m256 dequantized = mm256_dequantize_epi32(unpacked, scale_v);
        _mm256_storeu_ps(output, dequantized);
        input += BIT_WIDTH;
        output += 8;
    }

    constexpr __mmask8 remaining_mask = (1 << remaining) - 1;
    if constexpr (remaining > 0) {
        const __m256i unpacked   = mm256_unpack_epi32_bmi2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m256 dequantized = mm256_dequantize_epi32(unpacked, scale_v);
        _mm256_maskstore_ps(output, internal::mm256_convert_vmask_epi32(remaining_mask), dequantized);
    }

    return 0;
}

/**
 * @brief Decompress multiple 512\-bit blocks using AVX2 and BMI2 instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 16).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 *
 * @param input pointer to the start of the compressed data.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @param blocks number of 512-bit blocks to decompress.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 and BMI2 support.
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 16)
int mm256_decompress_blocks_bmi2(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output,
                                 const uint32_t blocks) {
    const uint8_t* block_input = input;
    float_t* block_output      = output;

    for (uint32_t block = 0; block < blocks; block++) {
        mm256_decompress_block_bmi2<BIT_WIDTH, SIGN_VALUES>(block_input, scale, block_output);
        block_input += (BIT_WIDTH * 512) / 8;
        block_output += 512 / BIT_WIDTH;
    }

    return 0;
}

/**
 * @brief Decompress a single 512-bit block using AVX2 and BMI2 instructions.
 *
 * @param bit_width bit width per value in the packed representation (1 to 16).
 * @param input pointer to the start of the compressed block.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 and BMI2 support.
 */
int mm256_decompress_block_bmi2(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output);

/**
 * @brief Decompress multiple 512-bit blocks using AVX2 and BMI2 instructions.
 *
 * @param bit_width bit width per value in the packed representation (1 to 16).
 * @param input pointer to the start of the compressed data.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @param blocks number of 512-bit blocks to decompress.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 and BMI2 support.
 */
int mm256_decompress_blocks_bmi2(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output,
                                 uint32_t blocks);

/**
 * @brief Decompress a single 512\-bit block using AVX2 instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 *
 * @param input pointer to the start of the compressed block.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 support.
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
int mm256_decompress_block_avx2(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = 512 / BIT_WIDTH;
    constexpr uint32_t iterations_8       = elements_per_block / 8;
    constexpr uint8_t remaining           = elements_per_block - iterations_8 * 8;

    const __m256 scale_v = _mm256_set1_ps(scale);
#pragma GCC unroll 4
    for (uint32_t iter = 0; iter < iterations_8; iter++) {
        const __m256i unpacked   = mm256_unpack_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m256 dequantized = mm256_dequantize_epi32(unpacked, scale_v);
        _mm256_storeu_ps(output, dequantized);
        input += BIT_WIDTH;
        output += 8;
    }

    constexpr __mmask8 remaining_mask = (1 << remaining) - 1;
    if constexpr (remaining > 0) {
        const __m256i unpacked   = mm256_unpack_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m256 dequantized = mm256_dequantize_epi32(unpacked, scale_v);
        _mm256_maskstore_ps(output, internal::mm256_convert_vmask_epi32(remaining_mask), dequantized);
    }

    return 0;
}

/**
 * @brief Decompress multiple 512\-bit blocks using AVX2 instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 *
 * @param input pointer to the start of the compressed data.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @param blocks number of 512-bit blocks to decompress.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 support.
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
int mm256_decompress_blocks_avx2(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output,
                                 const uint32_t blocks) {
    const uint8_t* block_input = input;
    float_t* block_output      = output;

    for (uint32_t block = 0; block < blocks; block++) {
        mm256_decompress_block_avx2<BIT_WIDTH, SIGN_VALUES>(block_input, scale, block_output);
        block_input += (BIT_WIDTH * 512) / 8;
        block_output += 512 / BIT_WIDTH;
    }

    return 0;
}

/**
 * @brief Decompress a single 512-bit block using AVX2 instructions.
 *
 * @param bit_width bit width per value in the packed representation (1 to 24).
 * @param input pointer to the start of the compressed block.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 support.
 */
int mm256_decompress_block_avx2(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output);

/**
 * @brief Decompress multiple 512-bit blocks using AVX2 instructions.
 *
 * @param bit_width bit width per value in the packed representation (1 to 24).
 * @param input pointer to the start of the compressed data.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @param blocks number of 512-bit blocks to decompress.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 support.
 */
int mm256_decompress_blocks_avx2(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output,
                                 uint32_t blocks);
#endif  // PERNIX_AVX2_ENABLED

#ifdef PERNIX_AVX512_VBMI_ENABLED

/**
 * @brief Decompress a single 512\-bit block using AVX-512 and AVX-512-VBMI instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 *
 * @param input pointer to the start of the compressed block.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX-512 and AVX-512-VBMI support.
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
__always_inline int mm512_decompress_block_avx512vbmi(const uint8_t* __restrict__ input, const float_t scale,
                                                      float_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = 512 / BIT_WIDTH;
    constexpr uint32_t iterations_16      = elements_per_block / 16;
    constexpr uint32_t iterations_8       = (elements_per_block % 16) / 8;
    constexpr uint32_t iterations_4       = (elements_per_block % 8) / 4;
    constexpr uint8_t remaining           = elements_per_block - iterations_16 * 16 - iterations_8 * 8 - iterations_4 * 4;

    const __m512 scale_v = _mm512_set1_ps(scale);
#pragma GCC unroll 4
    for (uint32_t iter = 0; iter < iterations_16; iter++) {
        const __m512i unpacked   = mm512_unpack_epi32_avx512vbmi<BIT_WIDTH, SIGN_VALUES>(input);
        const __m512 dequantized = mm512_dequantize_epi32(unpacked, scale_v);
        _mm512_storeu_ps(output, dequantized);
        input += 2 * BIT_WIDTH;
        output += 16;
    }

    if (iterations_8 > 0) {
        const __m256 scale_v256  = _mm256_set1_ps(scale);
        const __m256i unpacked   = mm256_unpack_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m256 dequantized = mm256_dequantize_epi32(unpacked, scale_v256);
        _mm256_storeu_ps(output, dequantized);
        input += BIT_WIDTH;
        output += 8;
    }

    if (iterations_4 > 0) {
        const __m128 scale_v128  = _mm_set1_ps(scale);
        const __m128i unpacked   = mm_unpack_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m128 dequantized = mm_dequantize_epi32(unpacked, scale_v128);
        _mm_storeu_ps(output, dequantized);
        input += BIT_WIDTH / 2;
        output += 4;
    }

    if (remaining > 0) {
        // const __m256 scale_v256 = _mm256_set1_ps(scale);
        // constexpr __mmask8 remaining_mask = (1 << remaining) - 1;
        // const __m256i unpacked = mm256_unpack_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(input);
        // const __m256 dequantized = mm256_dequantize_epi32(unpacked, scale_v256);
        // _mm256_mask_storeu_ps(output, remaining_mask, dequantized);

        //             const std::vector<int32_t> block_values = unpack_epi32_fallback<BIT_WIDTH, SIGN_VALUES>(input, remaining);
        //
        // #pragma GCC unroll 4
        //             for (uint32_t i = 0; i < remaining; i++) {
        //                 output[i] = dequantize_epi32(block_values[i], scale);
        //             }

        std::size_t idx            = 0;
        uint8_t bits_in_buffer     = 16;
        auto buffer                = static_cast<uint64_t>(input[idx++]);
        constexpr uint16_t bitmask = BIT_WIDTH == 16 ? std::numeric_limits<uint16_t>::max() : (1U << BIT_WIDTH) - 1U;

#pragma GCC unroll 3
        for (uint32_t i = 0; i < remaining; i++) {
            if (BIT_WIDTH > bits_in_buffer) {
                const auto next_value = static_cast<uint64_t>(input[idx++]) << bits_in_buffer;
                buffer |= next_value;
                bits_in_buffer += 16;
            }

            const auto raw_value = static_cast<uint16_t>(buffer & bitmask);
            if constexpr (SIGN_VALUES) {
                constexpr uint32_t shift = 32 - BIT_WIDTH;
                output[i]                = (static_cast<int32_t>(raw_value) << shift) >> shift;
            } else {
                output[i] = raw_value;
            }

            buffer >>= BIT_WIDTH;
            bits_in_buffer -= BIT_WIDTH;
        }
    }

    return 0;
}

/**
 * @brief Decompress multiple 512\-bit blocks using AVX-512 and AVX-512-VBMI instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 *
 * @param input pointer to the start of the compressed data.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @param blocks number of 512-bit blocks to decompress.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX-512 and AVX-512-VBMI support.
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
int mm512_decompress_blocks_avx512vbmi(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output,
                                       const uint32_t blocks) {
    const uint8_t* block_input = input;
    float_t* block_output      = output;

    for (uint32_t block = 0; block < blocks; block++) {
        mm512_decompress_block_avx512vbmi<BIT_WIDTH, SIGN_VALUES>(block_input, scale, block_output);
        block_input += (BIT_WIDTH * 512) / 8;
        block_output += 512 / BIT_WIDTH;
    }

    return 0;
}

/**
 * @brief Decompress a single 512-bit block using AVX-512 and AVX-512-VBMI instructions.
 *
 * @param bit_width bit width per value in the packed representation (1 to 24).
 * @param input pointer to the start of the compressed block.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX-512 and AVX-512-VBMI support.
 */
int mm512_decompress_block_avx512vbmi(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output);

/**
 * @brief Decompress multiple 512-bit blocks using AVX-512 and AVX-512-VBMI instructions.
 *
 * @param bit_width bit width per value in the packed representation (1 to 24).
 * @param input pointer to the start of the compressed data.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @param blocks number of 512-bit blocks to decompress.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX-512 and AVX-512-VBMI support.
 */
int mm512_decompress_blocks_avx512vbmi(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output,
                                       uint32_t blocks);

#endif  // PERNIX_AVX512_VBMI_ENABLED

/**
 * @brief Decompress a single 512\-bit block using fallback scalar implementation.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 * @param input pointer to the start of the compressed block.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @return int status code (0 for success).
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
int decompress_block_fallback(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = 512 / BIT_WIDTH;

    const std::vector<int32_t> block_values = unpack_epi32_fallback<BIT_WIDTH, SIGN_VALUES>(input, elements_per_block);

#pragma GCC unroll 512
    for (uint32_t i = 0; i < elements_per_block; i++) {
        output[i] = dequantize_epi32(block_values[i], scale);
    }

    return 0;
}

/**
 * @brief Decompress multiple 512\-bit blocks using fallback scalar implementation.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 * @param input pointer to the start of the compressed data.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @param blocks number of 512-bit blocks to decompress.
 * @return int status code (0 for success).
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
int decompress_blocks_fallback(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output,
                               const uint32_t blocks) {
    const uint8_t* block_input = input;
    float_t* block_output      = output;

    for (uint32_t block = 0; block < blocks; block++) {
        decompress_block_fallback<BIT_WIDTH, SIGN_VALUES>(block_input, scale, block_output);
        block_input += (BIT_WIDTH * 512) / 8;
        block_output += 512 / BIT_WIDTH;
    }

    return 0;
}

/**
 * @brief Decompress a single 512-bit block using fallback scalar implementation.
 *
 * @param bit_width bit width per value in the packed representation (1 to 24).
 * @param input pointer to the start of the compressed block.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @return int status code (0 for success).
 */
int decompress_block_fallback(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output);

/**
 * @brief Decompress multiple 512-bit blocks using fallback scalar implementation.
 *
 * @param bit_width bit width per value in the packed representation (1 to 24).
 * @param input pointer to the start of the compressed data.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @param blocks number of 512-bit blocks to decompress.
 * @return int status code (0 for success).
 */
int decompress_blocks_fallback(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output,
                               uint32_t blocks);

/**
 * @brief Decompress a single 512-bit block using the specified implementation.
 *
 * @param implementation the available implementation to use for decompression.
 * @param bit_width bit width per value in the packed representation (1 to 24).
 * @param input pointer to the start of the compressed block.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @param blocks number of 512-bit blocks to decompress.
 * @return int status code (0 for success).
 */
int decompress_blocks(AvailableImplementations implementation, uint8_t bit_width, const uint8_t* input, float_t scale, float_t* output,
                      uint32_t blocks);

/**
 * @brief Decompress multiple 512-bit blocks using the specified implementation.
 *
 * @param implementation the available implementation to use for decompression.
 * @param bit_width bit width per value in the packed representation (1 to 24).
 * @param input pointer to the start of the compressed data.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @return int status code (0 for success).
 */
int decompress_block(AvailableImplementations implementation, uint8_t bit_width, const uint8_t* input, float_t scale, float_t* output);
}  // namespace pernix

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
#ifdef PERNIX_AVX2_ENABLED
int mm256_decompress_block_bmi2_c(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output);

int mm256_decompress_blocks_bmi2_c(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output,
                                   uint32_t blocks);

int mm256_decompress_block_avx2_c(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output);

int mm256_decompress_blocks_avx2_c(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output,
                                   uint32_t blocks);
#endif  // PERNIX_AVX2_ENABLED

#ifdef PERNIX_AVX512_VBMI_ENABLED

int mm512_decompress_block_avx512vbmi_c(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output);

int mm512_decompress_blocks_avx512vbmi_c(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output,
                                         uint32_t blocks);

#endif  // PERNIX_AVX512_VBMI_ENABLED

int decompress_block_fallback_c(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output);

int decompress_blocks_fallback_c(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output,
                                 uint32_t blocks);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // PERNIX_DECOMPRESSION_H
