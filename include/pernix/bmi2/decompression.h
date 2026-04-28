#ifndef PERNIX_BMI2_DECOMPRESSION_H
#define PERNIX_BMI2_DECOMPRESSION_H

#include <pernix/avx2/decompression.h>
#include <pernix/simd_compat.h>

#include <cmath>
#include <limits>
#include <cstring>

namespace pernix {
namespace internal {
/**
 * @brief Sign-extend packed values after BMI2 expansion into 32-bit lanes.
 *
 * @tparam BIT_WIDTH original encoded bit width.
 * @param source register containing unpacked values.
 * @return __m128i sign-extended values.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
__m128i mm_sign_extend32(__m128i source) {
    if constexpr (BIT_WIDTH == 1) {
        // Keep 1-bit values as 0/1 to match fallback decoding semantics.
        return source;
    }

    constexpr uint16_t shift = 32 - BIT_WIDTH;
    source                   = _mm_slli_epi32(source, shift);
    return _mm_srai_epi32(source, shift);
}

/**
 * @brief Sign-extend packed values after BMI2 expansion into eight 32-bit lanes.
 *
 * @tparam BIT_WIDTH original encoded bit width.
 * @param source register containing unpacked values.
 * @return __m256i sign-extended values.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
__m256i mm256_sign_extend32(__m256i source) {
    if constexpr (BIT_WIDTH == 1) {
        // Keep 1-bit values as 0/1 to match fallback decoding semantics.
        return source;
    }

    constexpr uint16_t shift = 32 - BIT_WIDTH;
    source                   = _mm256_slli_epi32(source, shift);
    return _mm256_srai_epi32(source, shift);
}

/**
 * @brief Unpack four values from a BMI2-packed input buffer.
 *
 * @tparam BIT_WIDTH bit width per packed value.
 * @tparam SIGN_VALUES whether to sign-extend the unpacked values.
 * @param input pointer to the packed input buffer.
 * @return __m128i unpacked values.
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH > 0 && BIT_WIDTH <= 24)
__m128i mm_unpack_epi32_bmi2(const uint8_t* __restrict__ input) {
    constexpr uint32_t mask = BIT_WIDTH == 32 ? std::numeric_limits<uint32_t>::max() : (1ULL << BIT_WIDTH) - 1U;

    __m128i result;
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
        constexpr uint64_t pdep_mask = 0x0101010101010101ULL * mask;

        uint32_t temp_value;
        std::memcpy(&temp_value, input, sizeof(uint32_t));

        const int32_t value  = _pdep_u32(temp_value, static_cast<uint32_t>(pdep_mask));
        const __m128i source = _mm_insert_epi32(_mm_setzero_si128(), value, 0);

        result = _mm_cvtepi8_epi32(source);
    } else if constexpr (BIT_WIDTH == 16) {
        const __m128i source = _mm_loadu_si64(input);
        result               = _mm_cvtepi16_epi32(source);
    } else if constexpr (BIT_WIDTH > 8 && BIT_WIDTH <= 16) {
        constexpr uint64_t pdep_mask = 0x0001000100010001ULL * mask;

        uint64_t temp_value;
        std::memcpy(&temp_value, input, sizeof(uint64_t));

        const int64_t value  = _pdep_u64(temp_value, pdep_mask);
        const __m128i source = _mm_insert_epi64(_mm_setzero_si128(), value, 0);

        result = _mm_cvtepi16_epi32(source);
    } else {
        constexpr uint64_t pdep_mask = 0x0000000100000001ULL * mask;
        constexpr uint32_t shift1    = BIT_WIDTH * 2;
        constexpr uint32_t shift2    = 64 - shift1;

        alignas(16) uint64_t temp_values[2];
        std::memcpy(temp_values, input, 2 * sizeof(uint64_t));

        alignas(16) int64_t values[2];
        values[0] = _pdep_u64(temp_values[0], pdep_mask);
        values[1] = _pdep_u64((temp_values[0] >> shift1) | (temp_values[1] << shift2), pdep_mask);

        result = _mm_set_epi64x(values[1], values[0]);
    }

    if constexpr (SIGN_VALUES) {
        result = internal::mm_sign_extend32<BIT_WIDTH>(result);
    }
    return result;
}

/**
 * @brief Unpack eight values from a BMI2-packed input buffer.
 *
 * @tparam BIT_WIDTH bit width per packed value.
 * @tparam SIGN_VALUES whether to sign-extend the unpacked values.
 * @param input pointer to the packed input buffer.
 * @return __m256i unpacked values.
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
__m256i mm256_unpack_epi32_bmi2(const uint8_t* __restrict__ input) {
    constexpr uint32_t mask = BIT_WIDTH == 32 ? std::numeric_limits<uint32_t>::max() : (1ULL << BIT_WIDTH) - 1U;

    __m256i result;
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
        constexpr uint64_t pdep_mask = 0x0101010101010101ULL * mask;

        uint64_t temp_value;
        std::memcpy(&temp_value, input, sizeof(uint64_t));

        const int64_t value  = _pdep_u64(temp_value, pdep_mask);
        const __m128i source = _mm_insert_epi64(_mm_setzero_si128(), value, 0);

        result = _mm256_cvtepi8_epi32(source);
    } else if constexpr (BIT_WIDTH == 16) {
        const __m128i source = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input));
        result               = _mm256_cvtepi16_epi32(source);
    } else if constexpr (BIT_WIDTH > 8 && BIT_WIDTH <= 16) {
        constexpr uint64_t pdep_mask = 0x0001000100010001ULL * mask;
        constexpr uint64_t shift1    = BIT_WIDTH * 4;
        constexpr uint64_t shift2    = 64 - shift1;

        alignas(16) uint64_t temp_values[2];
        std::memcpy(temp_values, input, 2 * sizeof(uint64_t));

        alignas(16) int64_t values[2];
        values[0] = _pdep_u64(temp_values[0], pdep_mask);
        values[1] = _pdep_u64((temp_values[0] >> shift1) | (temp_values[1] << shift2), pdep_mask);

        const __m128i source = _mm_set_epi64x(values[1], values[0]);
        result               = _mm256_cvtepi16_epi32(source);
    } else {
        constexpr uint64_t pdep_mask = 0x0000000100000001ULL * mask;
        constexpr uint32_t shift1    = BIT_WIDTH * 2;
        constexpr uint32_t shift2    = 64 - shift1;

        alignas(16) uint64_t temp_values[4];
        std::memcpy(temp_values, input, 2 * sizeof(uint64_t));

        if constexpr ((BIT_WIDTH % 2) == 0) {
            std::memcpy(temp_values + 2, input + BIT_WIDTH / 2, 2 * sizeof(uint64_t));
        } else {
            constexpr uint32_t second_group_bit_offset  = BIT_WIDTH * 4;
            constexpr uint32_t second_group_byte_offset = second_group_bit_offset / 8;
            constexpr uint32_t second_group_shift       = second_group_bit_offset % 8;

            alignas(16) uint64_t raw_values[2];
            std::memcpy(raw_values, input + second_group_byte_offset, 2 * sizeof(uint64_t));

            temp_values[2] = (raw_values[0] >> second_group_shift) | (raw_values[1] << (64 - second_group_shift));
            temp_values[3] = raw_values[1] >> second_group_shift;
        }

        alignas(16) uint64_t values[4];
        values[0] = _pdep_u64((temp_values[0]), pdep_mask);
        values[1] = _pdep_u64((temp_values[0] >> shift1) | (temp_values[1] << shift2), pdep_mask);
        values[2] = _pdep_u64((temp_values[2]), pdep_mask);
        values[3] = _pdep_u64((temp_values[2] >> shift1) | (temp_values[3] << shift2), pdep_mask);

        result = _mm256_set_epi64x(static_cast<int64_t>(values[3]), static_cast<int64_t>(values[2]), static_cast<int64_t>(values[1]),
                                   static_cast<int64_t>(values[0]));
    }

    if constexpr (SIGN_VALUES) {
        result = internal::mm256_sign_extend32<BIT_WIDTH>(result);
    }
    return result;
}
} // namespace internal

/**
 * @brief Decompress a single 512\-bit block using AVX2 and BMI2 instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 16).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 * @tparam BLOCK_SIZE size of the block in bytes (must be a multiple of 32).
 *
 * @param input pointer to the start of the compressed block.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 and BMI2 support.
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_decompress_block_bmi2(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;
    constexpr uint32_t iterations_8       = elements_per_block / 8;
    constexpr uint8_t remaining           = elements_per_block - iterations_8 * 8;

    const __m256 scale_v = _mm256_set1_ps(scale);
#pragma GCC unroll 4
    for (uint32_t iter = 0; iter < iterations_8; iter++) {
        const __m256i unpacked   = internal::mm256_unpack_epi32_bmi2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m256 dequantized = internal::mm256_dequantize_epi32(unpacked, scale_v);
        _mm256_storeu_ps(output, dequantized);
        input  += BIT_WIDTH;
        output += 8;
    }

    constexpr __mmask8 remaining_mask = (1 << remaining) - 1;
    if constexpr (remaining > 0) {
        const __m256i unpacked   = internal::mm256_unpack_epi32_bmi2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m256 dequantized = internal::mm256_dequantize_epi32(unpacked, scale_v);
        _mm256_maskstore_ps(output, internal::mm256_convert_vmask_epi32(remaining_mask), dequantized);
    }

    return 0;
}

/**
 * @brief Decompress a single block to double values using AVX2 and BMI2 instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 16).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 * @tparam BLOCK_SIZE size of the block in bytes (must be a multiple of 32).
 *
 * @param input pointer to the start of the compressed block.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed double values will be stored.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 and BMI2 support.
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_decompress_block_bmi2(const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;
    constexpr uint32_t iterations_8       = elements_per_block / 8;
    constexpr uint8_t remaining           = elements_per_block - iterations_8 * 8;
    const __m256d scale_v                 = _mm256_set1_pd(scale);
#pragma GCC unroll 4
    for (uint32_t iter = 0; iter < iterations_8; iter++) {
        const __m256i unpacked = internal::mm256_unpack_epi32_bmi2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m256i extend1  = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(unpacked));
        const __m256i extend2  = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(unpacked, 1));

        const __m256d dequantized1 = internal::mm256_dequantize_epi64_pd(extend1, scale_v);
        const __m256d dequantized2 = internal::mm256_dequantize_epi64_pd(extend2, scale_v);

        _mm256_storeu_pd(output, dequantized1);
        _mm256_storeu_pd(output + 4, dequantized2);

        input  += BIT_WIDTH;
        output += 8;
    }

    constexpr __mmask8 remaining_mask = (1 << remaining) - 1;
    if constexpr (remaining > 0) {
        const __m256i unpacked = internal::mm256_unpack_epi32_bmi2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m256i extend1  = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(unpacked));
        const __m256i extend2  = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(unpacked, 1));

        const __m256d dequantized1 = internal::mm256_dequantize_epi64_pd(extend1, scale_v);
        const __m256d dequantized2 = internal::mm256_dequantize_epi64_pd(extend2, scale_v);

        constexpr auto mask_lo = static_cast<__mmask8>(remaining_mask & 0x0F);
        _mm256_maskstore_pd(output, internal::mm256_convert_vmask_epi64(mask_lo), dequantized1);

        if constexpr (remaining > 4) {
            constexpr auto mask_hi = static_cast<__mmask8>((remaining_mask >> 4) & 0x0F);
            _mm256_maskstore_pd(output + 4, internal::mm256_convert_vmask_epi64(mask_hi), dequantized2);
        }
    }

    return 0;
}

/**
 * @brief Decompress multiple 512\-bit blocks using AVX2 and BMI2 instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 * @tparam BLOCK_SIZE size of the block in bytes (must be a multiple of 32).
 *
 * @param input pointer to the start of the compressed data.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @param blocks number of 512-bit blocks to decompress.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 and BMI2 support.
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_decompress_blocks_bmi2(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output,
                                 const uint32_t blocks) {
    const uint8_t* block_input = input;
    float_t* block_output      = output;

    for (uint32_t block = 0; block < blocks; block++) {
        mm256_decompress_block_bmi2<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(block_input, scale, block_output);
        block_input  += BLOCK_SIZE;
        block_output += (BLOCK_SIZE * 8) / BIT_WIDTH;
    }

    return 0;
}

/**
 * @brief Decompress multiple blocks to double values using AVX2 and BMI2 instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 * @tparam BLOCK_SIZE size of the block in bytes (must be a multiple of 32).
 *
 * @param input pointer to the start of the compressed data.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed double values will be stored.
 * @param blocks number of blocks to decompress.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 and BMI2 support.
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_decompress_blocks_bmi2(const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output,
                                 const uint32_t blocks) {
    const uint8_t* block_input = input;
    double_t* block_output     = output;

    for (uint32_t block = 0; block < blocks; block++) {
        mm256_decompress_block_bmi2<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(block_input, scale, block_output);
        block_input  += BLOCK_SIZE;
        block_output += (BLOCK_SIZE * 8) / BIT_WIDTH;
    }

    return 0;
}
} // namespace pernix

#ifdef __cplusplus
namespace pernix {
extern "C" {
#endif

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
int mm256_decompress_block_f64_bmi2(uint8_t bit_width, const uint8_t* __restrict__ input, double_t scale, double_t* __restrict__ output);

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
int mm256_decompress_blocks_f64_bmi2(uint8_t bit_width, const uint8_t* __restrict__ input, double_t scale, double_t* __restrict__ output,
                                     uint32_t blocks);

#ifdef __cplusplus
}
} // namespace pernix
#endif

#endif  // PERNIX_BMI2_DECOMPRESSION_H