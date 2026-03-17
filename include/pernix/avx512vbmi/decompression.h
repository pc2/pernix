#ifndef PERNIX_AVX512VBMI_DECOMPRESSION_H
#define PERNIX_AVX512VBMI_DECOMPRESSION_H

#include <immintrin.h>
#include <pernix/avx2/decompression.h>
#include <pernix/avx512vbmi/tables.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <tuple>

namespace pernix {

namespace internal {
template <uint32_t LANES>
    requires(LANES <= 8)
consteval __mmask8 lane_mask_8() {
    if constexpr (LANES == 8) {
        return static_cast<__mmask8>(0xFF);
    } else {
        return static_cast<__mmask8>((1u << LANES) - 1u);
    }
}

/**
 * @brief Dequantize up to four integer values under a mask.
 */
__always_inline __m128 mm_maskz_dequantize_epi32(const __mmask8& mask, const __m128i& input, const __m128& scale) {
    const __m128 converted = _mm_maskz_cvtepi32_ps(mask, input);
    return _mm_maskz_mul_ps(mask, converted, scale);
}

/**
 * @brief Dequantize up to eight integer values under a mask.
 */
__always_inline __m256 mm256_maskz_dequantize_epi32(const __mmask8& mask, const __m256i& input, const __m256& scale) {
    const __m256 converted = _mm256_maskz_cvtepi32_ps(mask, input);
    return _mm256_maskz_mul_ps(mask, converted, scale);
}

/**
 * @brief Dequantize sixteen integer values to floats.
 */
__always_inline __m512 mm512_dequantize_epi32(const __m512i& input, const __m512& scale) {
    const __m512 converted = _mm512_cvtepi32_ps(input);
    return _mm512_mul_ps(converted, scale);
}

/**
 * @brief Dequantize up to sixteen integer values under a mask.
 */
__always_inline __m512 mm512_maskz_dequantize_epi32(const __mmask8& mask, const __m512i& input, const __m512& scale) {
    const __m512 converted = _mm512_maskz_cvtepi32_ps(mask, input);
    return _mm512_maskz_mul_ps(mask, converted, scale);
}

/**
 * @brief Unpack four values with the 128-bit VBMI shuffle path.
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
__always_inline __m128i mm_unpack_epi32_avx512vbmi_internal(const uint8_t* __restrict__ input) {
    if constexpr (BIT_WIDTH > 0 && BIT_WIDTH <= 8) {
        const __m128i source   = _mm_loadu_si32(input);
        const __m128i shuffled = _mm_permutexvar_epi8(unpack_tables_avx512_8<BIT_WIDTH, __m128i>::get_shuffle(), source);

        __m128i shifted = _mm_sllv_epi16(shuffled, unpack_tables_avx512_8<BIT_WIDTH, __m128i>::get_shift());

        constexpr uint16_t shift = 16 - BIT_WIDTH;
        if constexpr (SIGN_VALUES && BIT_WIDTH > 1) {
            shifted = _mm_srai_epi16(shifted, shift);
        } else {
            shifted = _mm_srli_epi16(shifted, shift);
        }

        return _mm_cvtepi16_epi32(shifted);
    } else {
        __m128i source         = _mm_loadu_si64(input);
        const __m128i shuffled = _mm_permutexvar_epi8(unpack_tables_avx512_24<BIT_WIDTH, __m128i>::get_shuffle(), source);

        constexpr uint16_t shift = 32 - BIT_WIDTH;
        __m128i shifted          = _mm_sllv_epi32(shuffled, unpack_tables_avx512_24<BIT_WIDTH, __m128i>::get_shift());
        if constexpr (SIGN_VALUES && BIT_WIDTH > 1) {
            shifted = _mm_srai_epi32(shifted, shift);
        } else {
            shifted = _mm_srli_epi32(shifted, shift);
        }

        return shifted;
    }
}

/**
 * @brief Unpack eight values with the 256-bit VBMI shuffle path.
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
__always_inline __m256i mm256_unpack_epi32_avx512vbmi_internal(const uint8_t* __restrict__ input) {
    if constexpr (BIT_WIDTH > 0 && BIT_WIDTH <= 8) {
        const __m128i source   = _mm_loadu_si64(input);
        const __m128i shuffled = _mm_permutexvar_epi8(unpack_tables_avx512_8<BIT_WIDTH, __m128i>::get_shuffle(), source);

        __m128i shifted = _mm_sllv_epi16(shuffled, unpack_tables_avx512_8<BIT_WIDTH, __m128i>::get_shift());

        constexpr uint16_t shift = 16 - BIT_WIDTH;
        if constexpr (SIGN_VALUES && BIT_WIDTH > 1) {
            shifted = _mm_srai_epi16(shifted, shift);
        } else {
            shifted = _mm_srli_epi16(shifted, shift);
        }

        return _mm256_cvtepi16_epi32(shifted);
    } else {
        __m256i source = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(input)));

        const __m256i shuffled = _mm256_permutexvar_epi8(unpack_tables_avx512_24<BIT_WIDTH, __m256i>::get_shuffle(), source);

        constexpr uint16_t shift = 32 - BIT_WIDTH;
        __m256i shifted          = _mm256_sllv_epi32(shuffled, unpack_tables_avx512_24<BIT_WIDTH, __m256i>::get_shift());
        if constexpr (SIGN_VALUES && BIT_WIDTH > 1) {
            shifted = _mm256_srai_epi32(shifted, shift);
        } else {
            shifted = _mm256_srli_epi32(shifted, shift);
        }

        return shifted;
    }
}

/**
 * @brief Unpack aligned 8-bit or 16-bit values directly into a 512-bit register.
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH == 8 || BIT_WIDTH == 16)
__always_inline __m512i mm512_unpack_aligned_epi32_avx512vbmi(const uint8_t* __restrict__ input) {
    if constexpr (BIT_WIDTH == 8) {
        const __m128i source = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input));
        if constexpr (SIGN_VALUES && BIT_WIDTH > 1) {
            return _mm512_cvtepi8_epi32(source);
        } else {
            return _mm512_cvtepu8_epi32(source);
        }
    } else {
        const __m256i source = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input));
        if constexpr (SIGN_VALUES && BIT_WIDTH > 1) {
            return _mm512_cvtepi16_epi32(source);
        } else {
            return _mm512_cvtepu16_epi32(source);
        }
    }
}

/**
 * @brief Unpack sixteen values with the 512-bit VBMI shuffle path.
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
__always_inline __m512i mm512_unpack_epi32_avx512vbmi_internal(const uint8_t* __restrict__ input) {
    if constexpr (BIT_WIDTH > 0 && BIT_WIDTH <= 8) {
        const __m256i source   = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(input)));
        const __m256i permuted = _mm256_permutexvar_epi8(unpack_tables_avx512_8<BIT_WIDTH, __m256i>::get_shuffle(), source);

        __m256i shifted = _mm256_sllv_epi16(permuted, unpack_tables_avx512_8<BIT_WIDTH, __m256i>::get_shift());

        constexpr uint16_t shift = 16 - BIT_WIDTH;
        if constexpr (SIGN_VALUES && BIT_WIDTH > 1) {
            shifted = _mm256_srai_epi16(shifted, shift);
        } else {
            shifted = _mm256_srli_epi16(shifted, shift);
        }

        return _mm512_cvtepi16_epi32(shifted);
    } else {
        __m512i source;
        if constexpr (BIT_WIDTH <= 16) {
            source = _mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input)));
        } else {
            source = _mm512_loadu_si512(input);
        }
        const __m512i permuted = _mm512_permutexvar_epi8(unpack_tables_avx512_24<BIT_WIDTH, __m512i>::get_shuffle(), source);

        __m512i shifted = _mm512_sllv_epi32(permuted, unpack_tables_avx512_24<BIT_WIDTH, __m512i>::get_shift());

        constexpr uint16_t shift = 32 - BIT_WIDTH;
        if constexpr (SIGN_VALUES && BIT_WIDTH > 1) {
            shifted = _mm512_srai_epi32(shifted, shift);
        } else {
            shifted = _mm512_srli_epi32(shifted, shift);
        }

        return shifted;
    }
}

/**
 * @brief Dispatch to the appropriate 128-bit VBMI unpacker.
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
__always_inline __m128i mm_unpack_epi32_avx512vbmi(const uint8_t* __restrict__ input) {
    if constexpr (BIT_WIDTH == 8 || BIT_WIDTH == 16) {
        return mm_unpack_aligned_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(input);
    } else {
        return internal::mm_unpack_epi32_avx512vbmi_internal<BIT_WIDTH, SIGN_VALUES>(input);
    }
}

/**
 * @brief Dispatch to the appropriate 256-bit VBMI unpacker.
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
__always_inline __m256i mm256_unpack_epi32_avx512vbmi(const uint8_t* __restrict__ input) {
    if constexpr (BIT_WIDTH == 8 || BIT_WIDTH == 16) {
        return mm256_unpack_aligned_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(input);
    } else {
        return internal::mm256_unpack_epi32_avx512vbmi_internal<BIT_WIDTH, SIGN_VALUES>(input);
    }
}

/**
 * @brief Dispatch to the appropriate 512-bit VBMI unpacker.
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
__always_inline __m512i mm512_unpack_epi32_avx512vbmi(const uint8_t* __restrict__ input) {
    if constexpr (BIT_WIDTH == 8 || BIT_WIDTH == 16) {
        return internal::mm512_unpack_aligned_epi32_avx512vbmi<BIT_WIDTH, SIGN_VALUES>(input);
    } else {
        return internal::mm512_unpack_epi32_avx512vbmi_internal<BIT_WIDTH, SIGN_VALUES>(input);
    }
}
}  // namespace internal

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
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_decompress_block_avx512vbmi(const uint8_t* __restrict__ input, const float_t scale,
                                                      float_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;
    constexpr uint32_t iterations_16      = elements_per_block / 16;
    constexpr uint32_t iterations_8       = (elements_per_block % 16) / 8;
    constexpr uint32_t iterations_4       = (elements_per_block % 8) / 4;
    constexpr uint8_t remaining           = elements_per_block - iterations_16 * 16 - iterations_8 * 8 - iterations_4 * 4;

    const __m512 scale_v   = _mm512_set1_ps(scale);
    const __m256 scale_v256 = _mm256_set1_ps(scale);
#pragma GCC unroll 4
    for (uint32_t iter = 0; iter < iterations_16; iter++) {
        const __m512i unpacked   = internal::mm512_unpack_epi32_avx512vbmi<BIT_WIDTH, SIGN_VALUES>(input);
        const __m512 dequantized = internal::mm512_dequantize_epi32(unpacked, scale_v);
        _mm512_storeu_ps(output, dequantized);
        input += 2 * BIT_WIDTH;
        output += 16;
    }

    if (iterations_8 > 0) {
        const __m256i unpacked   = internal::mm256_unpack_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m256 dequantized = internal::mm256_dequantize_epi32(unpacked, scale_v256);
        _mm256_storeu_ps(output, dequantized);
        input += BIT_WIDTH;
        output += 8;
    }

    constexpr uint32_t base_remaining = elements_per_block - iterations_16 * 16 - iterations_8 * 8 - iterations_4 * 4;
    constexpr bool use_vec4           = (BIT_WIDTH % 2 == 0) && (iterations_4 > 0);
    constexpr uint32_t tail_elements  = base_remaining + ((BIT_WIDTH % 2 == 0) ? 0 : (iterations_4 * 4));

    if constexpr (use_vec4) {
        const __m128 scale_v128  = _mm_set1_ps(scale);
        const __m128i unpacked   = internal::mm_unpack_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m128 dequantized = internal::mm_dequantize_epi32(unpacked, scale_v128);
        _mm_storeu_ps(output, dequantized);
        input += BIT_WIDTH / 2;  // valid only for even BIT_WIDTH
        output += 4;
    }

    if constexpr (tail_elements > 0) {
        constexpr __mmask8 tail_mask = internal::lane_mask_8<tail_elements>();
        constexpr uint32_t tail_bytes = (tail_elements * BIT_WIDTH + 7) / 8;

        alignas(16) uint8_t tail_buffer[32] = {};
        std::memcpy(tail_buffer, input, tail_bytes);

        const __m256i unpacked   = internal::mm256_unpack_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(tail_buffer);
        const __m256 dequantized = internal::mm256_maskz_dequantize_epi32(tail_mask, unpacked, scale_v256);
        _mm256_mask_storeu_ps(output, tail_mask, dequantized);
    }

    return 0;
}

/**
 * @brief Decompress a single block to double values using AVX-512 and AVX-512-VBMI instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 * @param input pointer to the start of the compressed block.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed double values will be stored.
 * @return int status code.
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_decompress_block_avx512vbmi(const uint8_t* __restrict__ input, const double_t scale,
                                                      double_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;
    constexpr uint32_t iterations_16      = elements_per_block / 16;
    constexpr uint32_t iterations_8       = (elements_per_block % 16) / 8;
    constexpr uint32_t iterations_4       = (elements_per_block % 8) / 4;
    const __m256d scale_v     = _mm256_set1_pd(scale);
    const __m512d scale_v512  = _mm512_set1_pd(scale);

#pragma GCC unroll 4
    for (uint32_t iter = 0; iter < iterations_16; iter++) {
        const __m512i unpacked = internal::mm512_unpack_epi32_avx512vbmi<BIT_WIDTH, SIGN_VALUES>(input);

        const __m256i unpacked_lo = _mm512_castsi512_si256(unpacked);
        const __m256i unpacked_hi = _mm512_extracti64x4_epi64(unpacked, 1);

        const __m256i extend1 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(unpacked_lo));
        const __m256i extend2 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(unpacked_lo, 1));
        const __m256i extend3 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(unpacked_hi));
        const __m256i extend4 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(unpacked_hi, 1));

        const __m256d dequantized1 = internal::mm256_dequantize_epi64_pd(extend1, scale_v);
        const __m256d dequantized2 = internal::mm256_dequantize_epi64_pd(extend2, scale_v);
        const __m256d dequantized3 = internal::mm256_dequantize_epi64_pd(extend3, scale_v);
        const __m256d dequantized4 = internal::mm256_dequantize_epi64_pd(extend4, scale_v);

        _mm256_storeu_pd(output, dequantized1);
        _mm256_storeu_pd(output + 4, dequantized2);
        _mm256_storeu_pd(output + 8, dequantized3);
        _mm256_storeu_pd(output + 12, dequantized4);

        input += 2 * BIT_WIDTH;
        output += 16;
    }

    if (iterations_8 > 0) {
        const __m256i unpacked = internal::mm256_unpack_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m256i extend1  = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(unpacked));
        const __m256i extend2  = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(unpacked, 1));

        const __m256d dequantized1 = internal::mm256_dequantize_epi64_pd(extend1, scale_v);
        const __m256d dequantized2 = internal::mm256_dequantize_epi64_pd(extend2, scale_v);

        _mm256_storeu_pd(output, dequantized1);
        _mm256_storeu_pd(output + 4, dequantized2);

        input += BIT_WIDTH;
        output += 8;
    }

    constexpr uint32_t base_remaining = elements_per_block - iterations_16 * 16 - iterations_8 * 8 - iterations_4 * 4;
    constexpr bool use_vec4           = (BIT_WIDTH % 2 == 0) && (iterations_4 > 0);
    constexpr uint32_t tail_elements  = base_remaining + ((BIT_WIDTH % 2 == 0) ? 0 : (iterations_4 * 4));

    if constexpr (use_vec4) {
        const __m128i unpacked    = internal::mm_unpack_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m256i extend      = _mm256_cvtepi32_epi64(unpacked);
        const __m256d dequantized = internal::mm256_dequantize_epi64_pd(extend, scale_v);

        _mm256_storeu_pd(output, dequantized);
        input += BIT_WIDTH / 2;  // valid only for even BIT_WIDTH
        output += 4;
    }

    if constexpr (tail_elements > 0) {
        constexpr __mmask8 tail_mask = internal::lane_mask_8<tail_elements>();
        constexpr uint32_t tail_bytes = (tail_elements * BIT_WIDTH + 7) / 8;

        alignas(16) uint8_t tail_buffer[32] = {};
        std::memcpy(tail_buffer, input, tail_bytes);

        const __m256i unpacked    = internal::mm256_unpack_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(tail_buffer);
        const __m512d converted   = _mm512_cvtepi32_pd(unpacked);
        const __m512d dequantized = _mm512_mul_pd(converted, scale_v512);
        _mm512_mask_storeu_pd(output, tail_mask, dequantized);
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
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm512_decompress_blocks_avx512vbmi(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output,
                                       const uint32_t blocks) {
    const uint8_t* block_input = input;
    float_t* block_output      = output;

    for (uint32_t block = 0; block < blocks; block++) {
        mm512_decompress_block_avx512vbmi<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(block_input, scale, block_output);
        block_input += BLOCK_SIZE;
        block_output += (BLOCK_SIZE * 8) / BIT_WIDTH;
    }

    return 0;
}

/**
 * @brief Decompress multiple blocks to double values using AVX-512 and AVX-512-VBMI instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 * @param input pointer to the start of the compressed data.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed double values will be stored.
 * @param blocks number of blocks to decompress.
 * @return int status code.
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm512_decompress_blocks_avx512vbmi(const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output,
                                       const uint32_t blocks) {
    const uint8_t* block_input = input;
    double_t* block_output     = output;

    for (uint32_t block = 0; block < blocks; block++) {
        mm512_decompress_block_avx512vbmi<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(block_input, scale, block_output);
        block_input += BLOCK_SIZE;
        block_output += (BLOCK_SIZE * 8) / BIT_WIDTH;
    }
    return 0;
}
}  // namespace pernix

#ifdef __cplusplus
namespace pernix {
extern "C" {
#endif
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
 * @brief Decompress a single 512-bit block using AVX-512 and AVX-512-VBMI instructions.
 *
 * @param bit_width bit width per value in the packed representation (1 to 24).
 * @param input pointer to the start of the compressed block.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed double values will be stored.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX-512 and AVX-512-VBMI support.
 */
int mm512_decompress_block_f64_avx512vbmi(uint8_t bit_width, const uint8_t* __restrict__ input, double_t scale,
                                          double_t* __restrict__ output);

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

/**
 * @brief Decompress multiple 512-bit blocks using AVX-512 and AVX-512-VBMI instructions.
 *
 * @param bit_width bit width per value in the packed representation (1 to 24).
 * @param input pointer to the start of the compressed data.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed double values will be stored.
 * @param blocks number of 512-bit blocks to decompress.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX-512 and AVX-512-VBMI support.
 */
int mm512_decompress_blocks_f64_avx512vbmi(uint8_t bit_width, const uint8_t* __restrict__ input, double_t scale,
                                           double_t* __restrict__ output, uint32_t blocks);

#ifdef __cplusplus
}
}  // namespace pernix
#endif

#endif  // PERNIX_AVX512VBMI_DECOMPRESSION_H
