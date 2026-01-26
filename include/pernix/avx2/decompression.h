#ifndef PERNIX_AVX2_DECOMPRESSION_H
#define PERNIX_AVX2_DECOMPRESSION_H

#include <immintrin.h>
#include <pernix/avx2/tables.h>

#include <cmath>
#include <cstdint>

namespace pernix {

namespace internal {
__always_inline __m256i mm256_convert_vmask_epi32(const __mmask8 mask8) {
    return _mm256_setr_epi32((mask8 & 0x1) ? -1 : 0, (mask8 & 0x2) ? -1 : 0, (mask8 & 0x4) ? -1 : 0, (mask8 & 0x8) ? -1 : 0,
                             (mask8 & 0x10) ? -1 : 0, (mask8 & 0x20) ? -1 : 0, (mask8 & 0x40) ? -1 : 0, (mask8 & 0x80) ? -1 : 0);
}

__always_inline __m128 mm_dequantize_epi32(const __m128i& input, const __m128& scale) {
    const __m128 converted = _mm_cvtepi32_ps(input);
    return _mm_mul_ps(converted, scale);
}

__always_inline __m256 mm256_dequantize_epi32(const __m256i& input, const __m256& scale) {
    const __m256 converted = _mm256_cvtepi32_ps(input);
    return _mm256_mul_ps(converted, scale);
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH == 8 || BIT_WIDTH == 16)
__m128i mm_unpack_aligned_epi32_avx2(const uint8_t* __restrict__ input) {
    if constexpr (BIT_WIDTH == 8) {
        const __m128i source = _mm_loadu_si32(input);
        if constexpr (SIGN_VALUES) {
            return _mm_cvtepi8_epi32(source);
        } else {
            return _mm_cvtepu8_epi32(source);
        }
    } else {
        const __m128i source = _mm_loadu_si64(input);
        if constexpr (SIGN_VALUES) {
            return _mm_cvtepi16_epi32(source);
        } else {
            return _mm_cvtepu16_epi32(source);
        }
    }
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 16)
__m128i mm_unpack_epi32_avx2(const uint8_t* __restrict__ input) {
    __m128i source;
    if constexpr (BIT_WIDTH <= 8) {
        source = _mm_loadu_si32(input);
    } else if constexpr (BIT_WIDTH <= 16) {
        source = _mm_loadu_si64(input);
    } else {
        source = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input));
    }
    const __m128i shuffled = _mm_shuffle_epi8(source, unpack_tables_avx2<BIT_WIDTH, __m128i>::get_shuffle());

    constexpr uint16_t shift = 32 - BIT_WIDTH;
    __m128i shifted          = _mm_sllv_epi32(shuffled, unpack_tables_avx2<BIT_WIDTH, __m128i>::get_shift());
    if constexpr (SIGN_VALUES) {
        shifted = _mm_srai_epi32(shifted, shift);
    } else {
        shifted = _mm_srli_epi32(shifted, shift);
    }

    return shifted;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH == 8 || BIT_WIDTH == 16)
__m256i mm256_unpack_aligned_epi32_avx2(const uint8_t* __restrict__ input) {
    if constexpr (BIT_WIDTH == 8) {
        const __m128i source = _mm_loadu_si64(input);
        if constexpr (SIGN_VALUES) {
            return _mm256_cvtepi8_epi32(source);
        } else {
            return _mm256_cvtepu8_epi32(source);
        }
    } else {
        const __m128i source = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input));
        if constexpr (SIGN_VALUES) {
            return _mm256_cvtepi16_epi32(source);
        } else {
            return _mm256_cvtepu16_epi32(source);
        }
    }
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 16)
__m256i mm256_unpack_epi32_avx2(const uint8_t* __restrict__ input) {
    __m256i source;
    if constexpr (BIT_WIDTH <= 8) {
        source = _mm256_castsi128_si256(_mm_loadu_si64(input));
    } else if constexpr (BIT_WIDTH <= 16) {
        source = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(input)));
    } else {
        source = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input));
    }
    const __m256i permuted = _mm256_permutevar8x32_epi32(source, unpack_tables_avx2<BIT_WIDTH, __m256i>::get_permute());
    const __m256i shuffled = _mm256_shuffle_epi8(permuted, unpack_tables_avx2<BIT_WIDTH, __m256i>::get_shuffle());

    constexpr uint16_t shift = 32 - BIT_WIDTH;
    __m256i shifted          = _mm256_sllv_epi32(shuffled, unpack_tables_avx2<BIT_WIDTH, __m256i>::get_shift());
    if constexpr (SIGN_VALUES) {
        shifted = _mm256_srai_epi32(shifted, shift);
    } else {
        shifted = _mm256_srli_epi32(shifted, shift);
    }

    return shifted;
}

}  // namespace internal

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
        const __m256i unpacked   = internal::mm256_unpack_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m256 dequantized = internal::mm256_dequantize_epi32(unpacked, scale_v);
        _mm256_storeu_ps(output, dequantized);
        input += BIT_WIDTH;
        output += 8;
    }

    constexpr __mmask8 remaining_mask = (1 << remaining) - 1;
    if constexpr (remaining > 0) {
        const __m256i unpacked   = internal::mm256_unpack_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m256 dequantized = internal::mm256_dequantize_epi32(unpacked, scale_v);
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
        block_input += 64;
        block_output += 512 / BIT_WIDTH;
    }

    return 0;
}
}  // namespace pernix

#ifdef __cplusplus
namespace pernix {
extern "C" {
#endif

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

#ifdef __cplusplus
}
}  // namespace pernix
#endif

#endif  // PERNIX_AVX2_DECOMPRESSION_H
