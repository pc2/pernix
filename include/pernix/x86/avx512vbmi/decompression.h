#ifndef PERNIX_AVX512VBMI_DECOMPRESSION_H
#define PERNIX_AVX512VBMI_DECOMPRESSION_H

#include <pernix/simd_compat.h>
#include <pernix/x86/avx2/decompression.h>
#include <pernix/x86/avx512vbmi/compat.h>
#include <pernix/x86/avx512vbmi/unpacking.h>
#include <pernix/x86/utils.h>

#include <cmath>

using namespace pernix::x86::internal;

namespace pernix {
namespace internal {
/**
 * @brief Dequantize sixteen integer values to floats.
 */
[[gnu::always_inline]] inline __m512 mm512_dequantize_epi32(const __m512i& input, const __m512& scale) {
    const __m512 converted = _mm512_cvtepi32_ps(input);
    return _mm512_mul_ps(converted, scale);
}

[[gnu::always_inline]] inline __m512d mm512_dequantize_epi64(const __m512i& input, const __m512d& scale) {
    const __m512d converted = _mm512_cvtepi64_pd(input);
    return _mm512_mul_pd(converted, scale);
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8) && (BLOCK_SIZE % 32 == 0)
[[gnu::always_inline]] inline int mm512_decompress_block_avx512vbmi_1to8(const uint8_t* __restrict__ input, const float_t scale,
                                                                         float_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    const uint32_t iterations_64      = elements_per_block / 64;
    const uint32_t iterations_32      = (elements_per_block % 64) / 32;
    const uint32_t iterations_16      = (elements_per_block % 32) / 16;
    const uint32_t remaining_elements = elements_per_block - iterations_64 * 64 - iterations_32 * 32 - iterations_16 * 16;

    const __m512 scale_v = _mm512_set1_ps(scale);

    if constexpr (iterations_64 > 0) {
#pragma GCC unroll 8
        for (uint32_t i = 0; i < iterations_64; ++i) {
            const __m512i source   = mm512_loadu_elements_epi64(BIT_WIDTH, input);
            const __m512i unpacked = m512::mm512_unpack_epi8_avx512vbmi_1to8<BIT_WIDTH, SIGN_VALUES>(source);

            const __m512i converted1 = _mm512_cvtepi8_epi32(_mm512_castsi512_si128(unpacked));
            const __m512i converted2 = _mm512_cvtepi8_epi32(_mm512_extracti64x2_epi64(unpacked, 1));
            const __m512i converted3 = _mm512_cvtepi8_epi32(_mm512_extracti64x2_epi64(unpacked, 2));
            const __m512i converted4 = _mm512_cvtepi8_epi32(_mm512_extracti64x2_epi64(unpacked, 3));

            const __m512 dequantized1 = mm512_dequantize_epi32(converted1, scale_v);
            const __m512 dequantized2 = mm512_dequantize_epi32(converted2, scale_v);
            const __m512 dequantized3 = mm512_dequantize_epi32(converted3, scale_v);
            const __m512 dequantized4 = mm512_dequantize_epi32(converted4, scale_v);

            _mm512_storeu_ps(output, dequantized1);
            _mm512_storeu_ps(output + 16, dequantized2);
            _mm512_storeu_ps(output + 32, dequantized3);
            _mm512_storeu_ps(output + 48, dequantized4);

            output += 64;
            input += 8 * BIT_WIDTH;
        }
    }

    if constexpr (iterations_32 > 0) {
        const __m256i source   = mm256_loadu_elements_epi32(BIT_WIDTH, input);
        const __m256i unpacked = m256::mm256_unpack_epi8_avx512vbmi_1to8<BIT_WIDTH, SIGN_VALUES>(source);

        const __m512i converted1 = _mm512_cvtepi8_epi32(_mm256_castsi256_si128(unpacked));
        const __m512i converted2 = _mm512_cvtepi8_epi32(_mm256_extracti128_si256(unpacked, 1));

        const __m512 dequantized1 = mm512_dequantize_epi32(converted1, scale_v);
        const __m512 dequantized2 = mm512_dequantize_epi32(converted2, scale_v);

        _mm512_storeu_ps(output, dequantized1);
        _mm512_storeu_ps(output + 16, dequantized2);

        output += 32;
        input += 4 * BIT_WIDTH;
    }

    if constexpr (iterations_16 > 0) {
        const __m128i source   = mm_loadu_elements_epi16(BIT_WIDTH, input);
        const __m128i unpacked = m128::mm_unpack_epi8_avx512vbmi_1to8<BIT_WIDTH, SIGN_VALUES>(source);

        const __m512i converted = _mm512_cvtepi8_epi32(unpacked);

        const __m512 dequantized = mm512_dequantize_epi32(converted, scale_v);

        _mm512_storeu_ps(output, dequantized);

        output += 16;
        input += 2 * BIT_WIDTH;
    }

    if constexpr (remaining_elements > 0) {
        const __m128i source   = mm_loadu_elements_epi8(tail_bytes(BIT_WIDTH, remaining_elements), input);
        const __m128i unpacked = m128::mm_unpack_epi8_avx512vbmi_1to8<BIT_WIDTH, SIGN_VALUES>(source);

        const __m512i converted = _mm512_cvtepi8_epi32(unpacked);

        const __m512 dequantized = mm512_dequantize_epi32(converted, scale_v);

        mm512_storeu_elements_ps(output, remaining_elements, dequantized);
    }

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8) && (BLOCK_SIZE % 32 == 0)
[[gnu::always_inline]] inline int mm512_decompress_block_avx512vbmi_1to8(const uint8_t* __restrict__ input, const double_t scale,
                                                                         double_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    const uint32_t iterations_64      = elements_per_block / 64;
    const uint32_t iterations_32      = (elements_per_block % 64) / 32;
    const uint32_t iterations_16      = (elements_per_block % 32) / 16;
    const uint32_t remaining_elements = elements_per_block - iterations_64 * 64 - iterations_32 * 32 - iterations_16 * 16;

    const __m512d scale_v = _mm512_set1_pd(scale);

    if constexpr (iterations_64 > 0) {
#pragma GCC unroll 8
        for (uint32_t i = 0; i < iterations_64; ++i) {
            const __m512i source   = mm512_loadu_elements_epi64(BIT_WIDTH, input);
            const __m512i unpacked = m512::mm512_unpack_epi8_avx512vbmi_1to8<BIT_WIDTH, SIGN_VALUES>(source);

            const __m128i extracted1 = _mm512_castsi512_si128(unpacked);
            const __m128i extracted2 = _mm512_extracti64x2_epi64(unpacked, 1);
            const __m128i extracted3 = _mm512_extracti64x2_epi64(unpacked, 2);
            const __m128i extracted4 = _mm512_extracti64x2_epi64(unpacked, 3);

            const __m512i converted1 = _mm512_cvtepi8_epi64(extracted1);
            const __m512i converted2 = _mm512_cvtepi8_epi64(_mm_srli_si128(extracted1, 8));
            const __m512i converted3 = _mm512_cvtepi8_epi64(extracted2);
            const __m512i converted4 = _mm512_cvtepi8_epi64(_mm_srli_si128(extracted2, 8));
            const __m512i converted5 = _mm512_cvtepi8_epi64(extracted3);
            const __m512i converted6 = _mm512_cvtepi8_epi64(_mm_srli_si128(extracted3, 8));
            const __m512i converted7 = _mm512_cvtepi8_epi64(extracted4);
            const __m512i converted8 = _mm512_cvtepi8_epi64(_mm_srli_si128(extracted4, 8));

            const __m512d dequantized1 = mm512_dequantize_epi64(converted1, scale_v);
            const __m512d dequantized2 = mm512_dequantize_epi64(converted2, scale_v);
            const __m512d dequantized3 = mm512_dequantize_epi64(converted3, scale_v);
            const __m512d dequantized4 = mm512_dequantize_epi64(converted4, scale_v);
            const __m512d dequantized5 = mm512_dequantize_epi64(converted5, scale_v);
            const __m512d dequantized6 = mm512_dequantize_epi64(converted6, scale_v);
            const __m512d dequantized7 = mm512_dequantize_epi64(converted7, scale_v);
            const __m512d dequantized8 = mm512_dequantize_epi64(converted8, scale_v);

            _mm512_storeu_pd(output, dequantized1);
            _mm512_storeu_pd(output + 8, dequantized2);
            _mm512_storeu_pd(output + 16, dequantized3);
            _mm512_storeu_pd(output + 24, dequantized4);
            _mm512_storeu_pd(output + 32, dequantized5);
            _mm512_storeu_pd(output + 40, dequantized6);
            _mm512_storeu_pd(output + 48, dequantized7);
            _mm512_storeu_pd(output + 56, dequantized8);

            output += 64;
            input += 8 * BIT_WIDTH;
        }

        if constexpr (iterations_32 > 0) {
            const __m256i source   = mm256_loadu_elements_epi32(BIT_WIDTH, input);
            const __m256i unpacked = m256::mm256_unpack_epi8_avx512vbmi_1to8<BIT_WIDTH, SIGN_VALUES>(source);

            const __m128i extracted1 = _mm256_castsi256_si128(unpacked);
            const __m128i extracted2 = _mm256_extracti64x2_epi64(unpacked, 1);

            const __m512i converted1 = _mm512_cvtepi8_epi64(extracted1);
            const __m512i converted2 = _mm512_cvtepi8_epi64(_mm_srli_si128(extracted1, 8));
            const __m512i converted3 = _mm512_cvtepi8_epi64(extracted2);
            const __m512i converted4 = _mm512_cvtepi8_epi64(_mm_srli_si128(extracted2, 8));

            const __m512d dequantized1 = mm512_dequantize_epi64(converted1, scale_v);
            const __m512d dequantized2 = mm512_dequantize_epi64(converted2, scale_v);
            const __m512d dequantized3 = mm512_dequantize_epi64(converted3, scale_v);
            const __m512d dequantized4 = mm512_dequantize_epi64(converted4, scale_v);

            _mm512_storeu_pd(output, dequantized1);
            _mm512_storeu_pd(output + 8, dequantized2);
            _mm512_storeu_pd(output + 16, dequantized3);
            _mm512_storeu_pd(output + 24, dequantized4);

            output += 32;
            input += 4 * BIT_WIDTH;
        }

        if constexpr (iterations_16 > 0) {
            const __m128i source   = mm_loadu_elements_epi16(BIT_WIDTH, input);
            const __m128i unpacked = m128::mm_unpack_epi8_avx512vbmi_1to8<BIT_WIDTH, SIGN_VALUES>(source);

            const __m512i converted1 = _mm512_cvtepi8_epi64(unpacked);
            const __m512i converted2 = _mm512_cvtepi8_epi64(_mm_srli_si128(unpacked, 8));

            const __m512d dequantized1 = mm512_dequantize_epi64(converted1, scale_v);
            const __m512d dequantized2 = mm512_dequantize_epi64(converted2, scale_v);

            _mm512_storeu_pd(output, dequantized1);
            _mm512_storeu_pd(output + 8, dequantized2);

            output += 16;
            input += 2 * BIT_WIDTH;
        }

        if constexpr (remaining_elements > 0) {
            const __m128i source   = mm_loadu_elements_epi8(tail_bytes(BIT_WIDTH, remaining_elements), input);
            const __m128i unpacked = m128::mm_unpack_epi8_avx512vbmi_1to8<BIT_WIDTH, SIGN_VALUES>(source);

            const __m512i converted1 = _mm512_cvtepi8_epi64(unpacked);
            const __m512i converted2 = _mm512_cvtepi8_epi64(_mm_srli_si128(unpacked, 8));

            const __m512d dequantized1 = mm512_dequantize_epi64(converted1, scale_v);
            const __m512d dequantized2 = mm512_dequantize_epi64(converted2, scale_v);

            mm512_storeu_elements_pd(output, remaining_elements < 8 ? remaining_elements : 8, dequantized1);
            if constexpr (remaining_elements > 8) {
                mm512_storeu_elements_pd(output + 8, remaining_elements - 8, dequantized2);
            }
        }
    }

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16) && (BLOCK_SIZE % 32 == 0)
[[gnu::always_inline]] inline int mm512_decompress_block_avx512vbmi_9to16(const uint8_t* __restrict__ input, const float_t scale,
                                                                          float_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr uint32_t iterations_32      = elements_per_block / 32;
    constexpr uint32_t iterations_16      = (elements_per_block % 32) / 16;
    constexpr uint32_t iterations_8       = (elements_per_block % 16) / 8;
    constexpr uint32_t remaining_elements = elements_per_block - iterations_32 * 32 - iterations_16 * 16 - iterations_8 * 8;

    const __m512 scale_v    = _mm512_set1_ps(scale);
    const __m256 scale_v256 = _mm256_set1_ps(scale);

    if constexpr (iterations_32 > 0) {
#pragma GCC unroll 4
        for (uint32_t i = 0; i < iterations_32; ++i) {
            const __m512i source   = mm512_loadu_elements_epi32(BIT_WIDTH, input);
            const __m512i unpacked = m512::mm512_unpack_epi16_avx512vbmi_9to16<BIT_WIDTH, SIGN_VALUES>(source);

            const __m512i converted1 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(unpacked));
            const __m512i converted2 = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(unpacked, 1));

            const __m512 dequantized1 = mm512_dequantize_epi32(converted1, scale_v);
            const __m512 dequantized2 = mm512_dequantize_epi32(converted2, scale_v);

            _mm512_storeu_ps(output, dequantized1);
            _mm512_storeu_ps(output + 16, dequantized2);

            output += 32;
            input += 4 * BIT_WIDTH;
        }
    }

    if constexpr (iterations_16 > 0) {
        const __m256i source   = mm256_loadu_elements_epi16(BIT_WIDTH, input);
        const __m256i unpacked = m256::mm256_unpack_epi16_avx512vbmi_9to16<BIT_WIDTH, SIGN_VALUES>(source);

        const __m512i converted  = _mm512_cvtepi16_epi32(unpacked);
        const __m512 dequantized = mm512_dequantize_epi32(converted, scale_v);

        _mm512_storeu_ps(output, dequantized);

        output += 16;
        input += 2 * BIT_WIDTH;
    }

    if constexpr (iterations_8 > 0) {
        const __m128i source   = mm_loadu_elements_epi8(BIT_WIDTH, input);
        const __m128i unpacked = m128::mm_unpack_epi16_avx512vbmi_9to16<BIT_WIDTH, SIGN_VALUES>(source);

        const __m256i converted  = _mm256_cvtepi16_epi32(unpacked);
        const __m256 dequantized = mm256_dequantize_epi32(converted, scale_v256);

        _mm256_storeu_ps(output, dequantized);

        output += 8;
        input += BIT_WIDTH;
    }

    if constexpr (remaining_elements > 0) {
        const __m128i source   = mm_loadu_elements_epi8(tail_bytes(BIT_WIDTH, remaining_elements), input);
        const __m128i unpacked = m128::mm_unpack_epi16_avx512vbmi_9to16<BIT_WIDTH, SIGN_VALUES>(source);

        const __m256i converted  = _mm256_cvtepi16_epi32(unpacked);
        const __m256 dequantized = mm256_dequantize_epi32(converted, scale_v256);

        mm256_storeu_elements_ps(output, remaining_elements, dequantized);
    }

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16) && (BLOCK_SIZE % 32 == 0)
[[gnu::always_inline]] inline int mm512_decompress_block_avx512vbmi_9to16(const uint8_t* __restrict__ input, const double_t scale,
                                                                          double_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr uint32_t iterations_32      = elements_per_block / 32;
    constexpr uint32_t iterations_16      = (elements_per_block % 32) / 16;
    constexpr uint32_t iterations_8       = (elements_per_block % 16) / 8;
    constexpr uint32_t remaining_elements = elements_per_block - iterations_32 * 32 - iterations_16 * 16 - iterations_8 * 8;

    const __m512d scale_v = _mm512_set1_pd(scale);

    if constexpr (iterations_32 > 0) {
#pragma GCC unroll 4
        for (uint32_t i = 0; i < iterations_32; ++i) {
            const __m512i source   = mm512_loadu_elements_epi32(BIT_WIDTH, input);
            const __m512i unpacked = m512::mm512_unpack_epi16_avx512vbmi_9to16<BIT_WIDTH, SIGN_VALUES>(source);

            const __m512i converted1 = _mm512_cvtepi16_epi64(_mm512_castsi512_si128(unpacked));
            const __m512i converted2 = _mm512_cvtepi16_epi64(_mm512_extracti64x2_epi64(unpacked, 1));
            const __m512i converted3 = _mm512_cvtepi16_epi64(_mm512_extracti64x2_epi64(unpacked, 2));
            const __m512i converted4 = _mm512_cvtepi16_epi64(_mm512_extracti64x2_epi64(unpacked, 3));

            const __m512d dequantized1 = mm512_dequantize_epi64(converted1, scale_v);
            const __m512d dequantized2 = mm512_dequantize_epi64(converted2, scale_v);
            const __m512d dequantized3 = mm512_dequantize_epi64(converted3, scale_v);
            const __m512d dequantized4 = mm512_dequantize_epi64(converted4, scale_v);

            _mm512_storeu_pd(output, dequantized1);
            _mm512_storeu_pd(output + 8, dequantized2);
            _mm512_storeu_pd(output + 16, dequantized3);
            _mm512_storeu_pd(output + 24, dequantized4);

            output += 32;
            input += 4 * BIT_WIDTH;
        }
    }

    if constexpr (iterations_16 > 0) {
        const __m256i source   = mm256_loadu_elements_epi16(BIT_WIDTH, input);
        const __m256i unpacked = m256::mm256_unpack_epi16_avx512vbmi_9to16<BIT_WIDTH, SIGN_VALUES>(source);

        const __m512i converted1 = _mm512_cvtepi16_epi64(_mm256_castsi256_si128(unpacked));
        const __m512i converted2 = _mm512_cvtepi16_epi64(_mm256_extracti64x2_epi64(unpacked, 1));

        const __m512d dequantized1 = mm512_dequantize_epi64(converted1, scale_v);
        const __m512d dequantized2 = mm512_dequantize_epi64(converted2, scale_v);

        _mm512_storeu_pd(output, dequantized1);
        _mm512_storeu_pd(output + 8, dequantized2);

        output += 16;
        input += 2 * BIT_WIDTH;
    }

    if constexpr (iterations_8 > 0) {
        const __m128i source   = mm_loadu_elements_epi8(BIT_WIDTH, input);
        const __m128i unpacked = m128::mm_unpack_epi16_avx512vbmi_9to16<BIT_WIDTH, SIGN_VALUES>(source);

        const __m512i converted = _mm512_cvtepi16_epi64(unpacked);

        const __m512d dequantized = mm512_dequantize_epi64(converted, scale_v);

        _mm512_storeu_pd(output, dequantized);

        output += 8;
        input += BIT_WIDTH;
    }

    if constexpr (remaining_elements > 0) {
        const __m128i source   = mm_loadu_elements_epi8(tail_bytes(BIT_WIDTH, remaining_elements), input);
        const __m128i unpacked = m128::mm_unpack_epi16_avx512vbmi_9to16<BIT_WIDTH, SIGN_VALUES>(source);

        const __m512i converted = _mm512_cvtepi16_epi64(unpacked);

        const __m512d dequantized = mm512_dequantize_epi64(converted, scale_v);

        mm512_storeu_elements_pd(output, remaining_elements, dequantized);
    }

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
[[gnu::always_inline]] inline int mm512_decompress_block_avx512vbmi_17to24(const uint8_t* __restrict__ input, const float_t scale,
                                                                           float_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr uint32_t iterations_16      = elements_per_block / 16;
    constexpr uint32_t iterations_8       = (elements_per_block % 16) / 8;
    constexpr uint32_t remaining_elements = elements_per_block - iterations_16 * 16 - iterations_8 * 8;

    const __m512 scale_v    = _mm512_set1_ps(scale);
    const __m256 scale_v256 = _mm256_set1_ps(scale);

    if constexpr (iterations_16 > 0) {
#pragma GCC unroll 2
        for (uint32_t i = 0; i < iterations_16; ++i) {
            const __m512i source   = mm512_loadu_elements_epi16(BIT_WIDTH, input);
            const __m512i unpacked = m512::mm512_unpack_epi32_avx512vbmi_17to24<BIT_WIDTH, SIGN_VALUES>(source);

            const __m512 dequantized = mm512_dequantize_epi32(unpacked, scale_v);

            _mm512_storeu_ps(output, dequantized);

            output += 16;
            input += 2 * BIT_WIDTH;
        }
    }

    if constexpr (iterations_8 > 0) {
        const __m256i source   = mm256_loadu_elements_epi8(BIT_WIDTH, input);
        const __m256i unpacked = m256::mm256_unpack_epi32_avx512vbmi_17to24<BIT_WIDTH, SIGN_VALUES>(source);

        const __m256 dequantized = mm256_dequantize_epi32(unpacked, scale_v256);

        _mm256_storeu_ps(output, dequantized);

        output += 8;
        input += BIT_WIDTH;
    }

    if constexpr (remaining_elements > 0) {
        const __m256i source   = mm256_loadu_elements_epi8(tail_bytes(BIT_WIDTH, remaining_elements), input);
        const __m256i unpacked = m256::mm256_unpack_epi32_avx512vbmi_17to24<BIT_WIDTH, SIGN_VALUES>(source);

        const __m256 dequantized = mm256_dequantize_epi32(unpacked, scale_v256);

        mm256_storeu_elements_ps(output, remaining_elements, dequantized);
    }

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
[[gnu::always_inline]] inline int mm512_decompress_block_avx512vbmi_17to24(const uint8_t* __restrict__ input, const double_t scale,
                                                                           double_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr uint32_t iterations_16      = elements_per_block / 16;
    constexpr uint32_t iterations_8       = (elements_per_block % 16) / 8;
    constexpr uint32_t remaining_elements = elements_per_block - iterations_16 * 16 - iterations_8 * 8;

    const __m512d scale_v = _mm512_set1_pd(scale);

    if constexpr (iterations_16 > 0) {
#pragma GCC unroll 2
        for (uint32_t i = 0; i < iterations_16; ++i) {
            const __m512i source   = mm512_loadu_elements_epi16(BIT_WIDTH, input);
            const __m512i unpacked = m512::mm512_unpack_epi32_avx512vbmi_17to24<BIT_WIDTH, SIGN_VALUES>(source);

            const __m512i converted1 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(unpacked));
            const __m512i converted2 = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(unpacked, 1));

            const __m512d dequantized1 = mm512_dequantize_epi64(converted1, scale_v);
            const __m512d dequantized2 = mm512_dequantize_epi64(converted2, scale_v);

            _mm512_storeu_pd(output, dequantized1);
            _mm512_storeu_pd(output + 8, dequantized2);

            output += 16;
            input += 2 * BIT_WIDTH;
        }
    }

    if constexpr (iterations_8 > 0) {
        const __m256i source   = mm256_loadu_elements_epi8(BIT_WIDTH, input);
        const __m256i unpacked = m256::mm256_unpack_epi32_avx512vbmi_17to24<BIT_WIDTH, SIGN_VALUES>(source);

        const __m512i converted = _mm512_cvtepi32_epi64(unpacked);

        const __m512d dequantized = mm512_dequantize_epi64(converted, scale_v);

        _mm512_storeu_pd(output, dequantized);

        output += 8;
        input += BIT_WIDTH;
    }

    if constexpr (remaining_elements > 0) {
        const __m256i source   = mm256_loadu_elements_epi8(tail_bytes(BIT_WIDTH, remaining_elements), input);
        const __m256i unpacked = m256::mm256_unpack_epi32_avx512vbmi_17to24<BIT_WIDTH, SIGN_VALUES>(source);

        const __m512i converted = _mm512_cvtepi32_epi64(unpacked);

        const __m512d dequantized = mm512_dequantize_epi64(converted, scale_v);

        mm512_storeu_elements_pd(output, remaining_elements, dequantized);
    }

    return 0;
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
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
        return internal::mm512_decompress_block_avx512vbmi_1to8<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
        return internal::mm512_decompress_block_avx512vbmi_9to16<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH >= 17 && BIT_WIDTH <= 24) {
        return internal::mm512_decompress_block_avx512vbmi_17to24<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
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
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
        return internal::mm512_decompress_block_avx512vbmi_1to8<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
        return internal::mm512_decompress_block_avx512vbmi_9to16<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH >= 17 && BIT_WIDTH <= 24) {
        return internal::mm512_decompress_block_avx512vbmi_17to24<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
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

    for (uint32_t block = 0; block < blocks; ++block) {
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

    for (uint32_t block = 0; block < blocks; ++block) {
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
