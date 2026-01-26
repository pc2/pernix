#ifndef PERNIX_AVX512VBMI_DECOMPRESSION_H
#define PERNIX_AVX512VBMI_DECOMPRESSION_H

#include <immintrin.h>
#include <pernix/avx2/decompression.h>
#include <pernix/avx512vbmi/tables.h>

#include <cmath>
#include <cstdint>
#include <tuple>

namespace pernix {

namespace internal {
__always_inline __m128 mm_maskz_dequantize_epi32(const __mmask8& mask, const __m128i& input, const __m128& scale) {
    const __m128 converted = _mm_maskz_cvtepi32_ps(mask, input);
    return _mm_maskz_mul_ps(mask, converted, scale);
}

__always_inline __m256 mm256_maskz_dequantize_epi32(const __mmask8& mask, const __m256i& input, const __m256& scale) {
    const __m256 converted = _mm256_maskz_cvtepi32_ps(mask, input);
    return _mm256_maskz_mul_ps(mask, converted, scale);
}

__always_inline __m512 mm512_dequantize_epi32(const __m512i& input, const __m512& scale) {
    const __m512 converted = _mm512_cvtepi32_ps(input);
    return _mm512_mul_ps(converted, scale);
}

__always_inline __m512 mm512_maskz_dequantize_epi32(const __mmask8& mask, const __m512i& input, const __m512& scale) {
    const __m512 converted = _mm512_maskz_cvtepi32_ps(mask, input);
    return _mm512_maskz_mul_ps(mask, converted, scale);
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
__always_inline __m128i mm_unpack_epi32_avx512vbmi_internal(const uint8_t* __restrict__ input) {
    if constexpr (BIT_WIDTH > 0 && BIT_WIDTH <= 8) {
        const __m128i source   = _mm_loadu_si32(input);
        const __m128i shuffled = _mm_permutexvar_epi8(unpack_tables_avx512_8<BIT_WIDTH, __m128i>::get_shuffle(), source);

        __m128i shifted = _mm_sllv_epi16(shuffled, unpack_tables_avx512_8<BIT_WIDTH, __m128i>::get_shift());

        constexpr uint16_t shift = 16 - BIT_WIDTH;
        if constexpr (SIGN_VALUES) {
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
        if constexpr (SIGN_VALUES) {
            shifted = _mm_srai_epi32(shifted, shift);
        } else {
            shifted = _mm_srli_epi32(shifted, shift);
        }

        return shifted;
    }
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
__always_inline __m256i mm256_unpack_epi32_avx512vbmi_internal(const uint8_t* __restrict__ input) {
    if constexpr (BIT_WIDTH > 0 && BIT_WIDTH <= 8) {
        const __m128i source   = _mm_loadu_si64(input);
        const __m128i shuffled = _mm_permutexvar_epi8(unpack_tables_avx512_8<BIT_WIDTH, __m128i>::get_shuffle(), source);

        __m128i shifted = _mm_sllv_epi16(shuffled, unpack_tables_avx512_8<BIT_WIDTH, __m128i>::get_shift());

        constexpr uint16_t shift = 16 - BIT_WIDTH;
        if constexpr (SIGN_VALUES) {
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
        if constexpr (SIGN_VALUES) {
            shifted = _mm256_srai_epi32(shifted, shift);
        } else {
            shifted = _mm256_srli_epi32(shifted, shift);
        }

        return shifted;
    }
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH == 8 || BIT_WIDTH == 16)
__always_inline __m512i mm512_unpack_aligned_epi32_avx512vbmi(const uint8_t* __restrict__ input) {
    if constexpr (BIT_WIDTH == 8) {
        const __m128i source = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input));
        if constexpr (SIGN_VALUES) {
            return _mm512_cvtepi8_epi32(source);
        } else {
            return _mm512_cvtepu8_epi32(source);
        }
    } else {
        const __m256i source = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input));
        if constexpr (SIGN_VALUES) {
            return _mm512_cvtepi16_epi32(source);
        } else {
            return _mm512_cvtepu16_epi32(source);
        }
    }
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
__always_inline __m512i mm512_unpack_epi32_avx512vbmi_internal(const uint8_t* __restrict__ input) {
    if constexpr (BIT_WIDTH > 0 && BIT_WIDTH <= 8) {
        const __m256i source   = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(input)));
        const __m256i permuted = _mm256_permutexvar_epi8(unpack_tables_avx512_8<BIT_WIDTH, __m256i>::get_shuffle(), source);

        __m256i shifted = _mm256_sllv_epi16(permuted, unpack_tables_avx512_8<BIT_WIDTH, __m256i>::get_shift());

        constexpr uint16_t shift = 16 - BIT_WIDTH;
        if constexpr (SIGN_VALUES) {
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
        if constexpr (SIGN_VALUES) {
            shifted = _mm512_srai_epi32(shifted, shift);
        } else {
            shifted = _mm512_srli_epi32(shifted, shift);
        }

        return shifted;
    }
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
__always_inline __m128i mm_unpack_epi32_avx512vbmi(const uint8_t* __restrict__ input) {
    if constexpr (BIT_WIDTH == 8 || BIT_WIDTH == 16) {
        return mm_unpack_aligned_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(input);
    } else {
        return internal::mm_unpack_epi32_avx512vbmi_internal<BIT_WIDTH, SIGN_VALUES>(input);
    }
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
__always_inline __m256i mm256_unpack_epi32_avx512vbmi(const uint8_t* __restrict__ input) {
    if constexpr (BIT_WIDTH == 8 || BIT_WIDTH == 16) {
        return mm256_unpack_aligned_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(input);
    } else {
        return internal::mm256_unpack_epi32_avx512vbmi_internal<BIT_WIDTH, SIGN_VALUES>(input);
    }
}

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
        const __m512i unpacked   = internal::mm512_unpack_epi32_avx512vbmi<BIT_WIDTH, SIGN_VALUES>(input);
        const __m512 dequantized = internal::mm512_dequantize_epi32(unpacked, scale_v);
        _mm512_storeu_ps(output, dequantized);
        input += 2 * BIT_WIDTH;
        output += 16;
    }

    if (iterations_8 > 0) {
        const __m256 scale_v256  = _mm256_set1_ps(scale);
        const __m256i unpacked   = internal::mm256_unpack_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m256 dequantized = internal::mm256_dequantize_epi32(unpacked, scale_v256);
        _mm256_storeu_ps(output, dequantized);
        input += BIT_WIDTH;
        output += 8;
    }

    if (iterations_4 > 0) {
        const __m128 scale_v128  = _mm_set1_ps(scale);
        const __m128i unpacked   = internal::mm_unpack_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m128 dequantized = internal::mm_dequantize_epi32(unpacked, scale_v128);
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
                output[i]                = static_cast<float_t>((static_cast<int32_t>(raw_value) << shift) >> shift);
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

#ifdef __cplusplus
}
}  // namespace pernix
#endif

#endif  // PERNIX_AVX512VBMI_DECOMPRESSION_H
