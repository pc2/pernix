#ifndef PERNIX_BMI2_DECOMPRESSION_H
#define PERNIX_BMI2_DECOMPRESSION_H

#include <pernix/simd_compat.h>
#include <pernix/x86/avx2/avx2_decompression.h>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>

namespace pernix {
namespace internal {
/**
 * @brief Sign-extend packed values after BMI2 expansion into 32-bit lanes.
 *
 * @tparam BIT_WIDTH original encoded bit width.
 * @param source register containing unpacked values.
 * @return __m128i sign-extended values.
 */
template <u8 BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
__m128i mm_sign_extend32(__m128i source) {
    if constexpr (BIT_WIDTH == 1) {
        // Keep 1-bit values as 0/1 to match fallback decoding semantics.
        return source;
    }

    constexpr u16 shift = 32 - BIT_WIDTH;
    source              = _mm_slli_epi32(source, shift);
    return _mm_srai_epi32(source, shift);
}

/**
 * @brief Sign-extend packed values after BMI2 expansion into eight 32-bit lanes.
 *
 * @tparam BIT_WIDTH original encoded bit width.
 * @param source register containing unpacked values.
 * @return __m256i sign-extended values.
 */
template <u8 BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
__m256i mm256_sign_extend32(__m256i source) {
    if constexpr (BIT_WIDTH == 1) {
        // Keep 1-bit values as 0/1 to match fallback decoding semantics.
        return source;
    }

    constexpr u16 shift = 32 - BIT_WIDTH;
    source              = _mm256_slli_epi32(source, shift);
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
template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH > 0 && BIT_WIDTH <= 24)
__m128i mm_unpack_epi32_bmi2(const u8* __restrict__ input) {
    constexpr u32 mask                 = BIT_WIDTH == 32 ? std::numeric_limits<u32>::max() : (1ULL << BIT_WIDTH) - 1U;
    constexpr std::size_t packed_bytes = (4 * BIT_WIDTH + 7) / 8;

    __m128i result;
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
        constexpr u64 pdep_mask = 0x0101010101010101ULL * mask;

        u32 temp_value = 0;
        std::memcpy(&temp_value, input, packed_bytes);

        const i32 value      = _pdep_u32(temp_value, static_cast<u32>(pdep_mask));
        const __m128i source = _mm_insert_epi32(_mm_setzero_si128(), value, 0);

        result = _mm_cvtepi8_epi32(source);
    } else if constexpr (BIT_WIDTH == 16) {
        const __m128i source = _mm_loadu_si64(input);
        result               = _mm_cvtepi16_epi32(source);
    } else if constexpr (BIT_WIDTH > 8 && BIT_WIDTH <= 16) {
        constexpr u64 pdep_mask = 0x0001000100010001ULL * mask;

        u64 temp_value = 0;
        std::memcpy(&temp_value, input, packed_bytes);

        const i64 value      = _pdep_u64(temp_value, pdep_mask);
        const __m128i source = _mm_insert_epi64(_mm_setzero_si128(), value, 0);

        result = _mm_cvtepi16_epi32(source);
    } else {
        constexpr u64 pdep_mask = 0x0000000100000001ULL * mask;
        constexpr u32 shift1    = BIT_WIDTH * 2;
        constexpr u32 shift2    = 64 - shift1;

        alignas(16) u64 temp_values[2]{};
        std::memcpy(temp_values, input, packed_bytes);

        alignas(16) i64 values[2];
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
template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
__m256i mm256_unpack_epi32_bmi2(const u8* __restrict__ input) {
    constexpr u32 mask                 = BIT_WIDTH == 32 ? std::numeric_limits<u32>::max() : (1ULL << BIT_WIDTH) - 1U;
    constexpr std::size_t packed_bytes = BIT_WIDTH;

    __m256i result;
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
        constexpr u64 pdep_mask = 0x0101010101010101ULL * mask;

        u64 temp_value = 0;
        std::memcpy(&temp_value, input, packed_bytes);

        const i64 value      = _pdep_u64(temp_value, pdep_mask);
        const __m128i source = _mm_insert_epi64(_mm_setzero_si128(), value, 0);

        result = _mm256_cvtepi8_epi32(source);
    } else if constexpr (BIT_WIDTH == 16) {
        const __m128i source = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input));
        result               = _mm256_cvtepi16_epi32(source);
    } else if constexpr (BIT_WIDTH > 8 && BIT_WIDTH <= 16) {
        constexpr u64 pdep_mask = 0x0001000100010001ULL * mask;
        constexpr u64 shift1    = BIT_WIDTH * 4;
        constexpr u64 shift2    = 64 - shift1;

        alignas(16) u64 temp_values[2]{};
        std::memcpy(temp_values, input, packed_bytes);

        alignas(16) i64 values[2];
        values[0] = _pdep_u64(temp_values[0], pdep_mask);
        values[1] = _pdep_u64((temp_values[0] >> shift1) | (temp_values[1] << shift2), pdep_mask);

        const __m128i source = _mm_set_epi64x(values[1], values[0]);
        result               = _mm256_cvtepi16_epi32(source);
    } else {
        constexpr u64 pdep_mask = 0x0000000100000001ULL * mask;
        constexpr u32 shift1    = BIT_WIDTH * 2;
        constexpr u32 shift2    = 64 - shift1;

        alignas(16) u64 temp_values[4]{};
        std::memcpy(temp_values, input, packed_bytes);

        if constexpr ((BIT_WIDTH % 2) == 0) {
            std::memcpy(temp_values + 2, input + BIT_WIDTH / 2, packed_bytes - BIT_WIDTH / 2);
        } else {
            constexpr u32 second_group_bit_offset  = BIT_WIDTH * 4;
            constexpr u32 second_group_byte_offset = second_group_bit_offset / 8;
            constexpr u32 second_group_shift       = second_group_bit_offset % 8;

            alignas(16) u64 raw_values[2]{};
            std::memcpy(raw_values, input + second_group_byte_offset, packed_bytes - second_group_byte_offset);

            temp_values[2] = (raw_values[0] >> second_group_shift) | (raw_values[1] << (64 - second_group_shift));
            temp_values[3] = raw_values[1] >> second_group_shift;
        }

        alignas(16) u64 values[4];
        values[0] = _pdep_u64((temp_values[0]), pdep_mask);
        values[1] = _pdep_u64((temp_values[0] >> shift1) | (temp_values[1] << shift2), pdep_mask);
        values[2] = _pdep_u64((temp_values[2]), pdep_mask);
        values[3] = _pdep_u64((temp_values[2] >> shift1) | (temp_values[3] << shift2), pdep_mask);

        result = _mm256_set_epi64x(static_cast<i64>(values[3]), static_cast<i64>(values[2]), static_cast<i64>(values[1]),
                                   static_cast<i64>(values[0]));
    }

    if constexpr (SIGN_VALUES) {
        result = internal::mm256_sign_extend32<BIT_WIDTH>(result);
    }
    return result;
}
}  // namespace internal

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
template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_decompress_block_bmi2(const void* __restrict__ input_ptr, const f32 scale, void* __restrict__ output_ptr) {
    const auto* input = static_cast<const u8*>(input_ptr);
    auto* output      = static_cast<f32*>(output_ptr);

    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;
    constexpr u32 iterations_8       = elements_per_block / 8;
    constexpr u8 remaining           = elements_per_block - iterations_8 * 8;

    const __m256 scale_v = _mm256_set1_ps(scale);
#pragma GCC unroll 4
    for (u32 iter = 0; iter < iterations_8; iter++) {
        const __m256i unpacked   = internal::mm256_unpack_epi32_bmi2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m256 dequantized = internal::mm256_dequantize_epi32(unpacked, scale_v);
        _mm256_storeu_ps(output, dequantized);
        input += BIT_WIDTH;
        output += 8;
    }

    if constexpr (remaining > 0) {
        const std::vector<i32> tail_values = internal::unpack_epi32_fallback<BIT_WIDTH, SIGN_VALUES>(input, remaining);
        for (u32 i = 0; i < remaining; i++) {
            output[i] = internal::dequantize_epi32(tail_values[i], scale);
        }
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
template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_decompress_block_bmi2(const void* __restrict__ input_ptr, const f64 scale, void* __restrict__ output_ptr) {
    const auto* input = static_cast<const u8*>(input_ptr);
    auto* output      = static_cast<f64*>(output_ptr);

    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;
    constexpr u32 iterations_8       = elements_per_block / 8;
    constexpr u8 remaining           = elements_per_block - iterations_8 * 8;
    const __m256d scale_v            = _mm256_set1_pd(scale);
#pragma GCC unroll 4
    for (u32 iter = 0; iter < iterations_8; iter++) {
        const __m256i unpacked = internal::mm256_unpack_epi32_bmi2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m256i extend1  = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(unpacked));
        const __m256i extend2  = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(unpacked, 1));

        const __m256d dequantized1 = internal::mm256_dequantize_epi64_pd(extend1, scale_v);
        const __m256d dequantized2 = internal::mm256_dequantize_epi64_pd(extend2, scale_v);

        _mm256_storeu_pd(output, dequantized1);
        _mm256_storeu_pd(output + 4, dequantized2);

        input += BIT_WIDTH;
        output += 8;
    }

    if constexpr (remaining > 0) {
        const std::vector<i32> tail_values = internal::unpack_epi32_fallback<BIT_WIDTH, SIGN_VALUES>(input, remaining);
        for (u32 i = 0; i < remaining; i++) {
            output[i] = internal::dequantize_epi64(tail_values[i], scale);
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
template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_decompress_blocks_bmi2(const void* __restrict__ input_ptr, const f32 scale, void* __restrict__ output_ptr, const u32 blocks) {
    const auto* input = static_cast<const u8*>(input_ptr);
    auto* output      = static_cast<f32*>(output_ptr);

    const u8* block_input = input;
    f32* block_output     = output;

    for (u32 block = 0; block < blocks; block++) {
        mm256_decompress_block_bmi2<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(block_input, scale, block_output);
        block_input += BLOCK_SIZE;
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
template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_decompress_blocks_bmi2(const void* __restrict__ input_ptr, const f64 scale, void* __restrict__ output_ptr, const u32 blocks) {
    const auto* input = static_cast<const u8*>(input_ptr);
    auto* output      = static_cast<double*>(output_ptr);

    const u8* block_input = input;
    f64* block_output     = output;

    for (u32 block = 0; block < blocks; block++) {
        mm256_decompress_block_bmi2<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(block_input, scale, block_output);
        block_input += BLOCK_SIZE;
        block_output += (BLOCK_SIZE * 8) / BIT_WIDTH;
    }

    return 0;
}
}  // namespace pernix

#endif  // PERNIX_BMI2_DECOMPRESSION_H
