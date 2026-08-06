#ifndef PERNIX_AVX512VBMI_DECOMPRESSION_H
#define PERNIX_AVX512VBMI_DECOMPRESSION_H

#include <pernix/simd_compat.h>
#include <pernix/x86/avx2/avx2_decompression.h>
#include <pernix/x86/avx512vbmi/compat.h>
#include <pernix/x86/avx512vbmi/unpacking.h>
#include <pernix/x86/utils.h>

#include <cstddef>
#include <iostream>

using namespace pernix::x86::internal;

namespace pernix {
namespace internal {
/**
 * @brief Dequantize sixteen integer values to floats.
 */
__always_inline __m512 mm512_dequantize_epi32(__m512i input, __m512 scale) {
    const __m512 converted = _mm512_cvtepi32_ps(input);
    return _mm512_mul_ps(converted, scale);
}

__always_inline __m512d mm512_dequantize_epi64(__m512i input, __m512d scale) {
    const __m512d converted = _mm512_cvtepi64_pd(input);
    return _mm512_mul_pd(converted, scale);
}

template <bool SIGN_VALUES>
__always_inline __m512i mm512_extend_epi8_epi32(const __m128i input) {
    if constexpr (SIGN_VALUES) {
        return _mm512_cvtepi8_epi32(input);
    } else {
        return _mm512_cvtepu8_epi32(input);
    }
}

template <bool SIGN_VALUES>
__always_inline __m512i mm512_extend_epi8_epi64(const __m128i input) {
    if constexpr (SIGN_VALUES) {
        return _mm512_cvtepi8_epi64(input);
    } else {
        return _mm512_cvtepu8_epi64(input);
    }
}

template <bool SIGN_VALUES>
__always_inline __m512i mm512_extend_epi16_epi32(const __m256i input) {
    if constexpr (SIGN_VALUES) {
        return _mm512_cvtepi16_epi32(input);
    } else {
        return _mm512_cvtepu16_epi32(input);
    }
}

template <bool SIGN_VALUES>
__always_inline __m512i mm512_extend_epi16_epi64(const __m128i input) {
    if constexpr (SIGN_VALUES) {
        return _mm512_cvtepi16_epi64(input);
    } else {
        return _mm512_cvtepu16_epi64(input);
    }
}

template <bool SIGN_VALUES>
__always_inline __m256i mm256_extend_epi16_epi32(const __m128i input) {
    if constexpr (SIGN_VALUES) {
        return _mm256_cvtepi16_epi32(input);
    } else {
        return _mm256_cvtepu16_epi32(input);
    }
}

template <bool SIGN_VALUES>
__always_inline void process_single(const __m128i extracted, const __m512 scale_v, f32* destination) {
    const __m512i converted  = mm512_extend_epi8_epi32<SIGN_VALUES>(extracted);
    const __m512 dequantized = mm512_dequantize_epi32(converted, scale_v);
    _mm512_storeu_ps(destination, dequantized);
}

template <bool SIGN_VALUES>
__always_inline void process_pair(const __m128i extracted, const __m512d scale_v, f64* destination) {
    const __m512i low  = mm512_extend_epi8_epi64<SIGN_VALUES>(extracted);
    const __m512i high = mm512_extend_epi8_epi64<SIGN_VALUES>(_mm_srli_si128(extracted, 8));

    const __m512d dequantized_low  = mm512_dequantize_epi64(low, scale_v);
    const __m512d dequantized_high = mm512_dequantize_epi64(high, scale_v);

    _mm512_storeu_pd(destination, dequantized_low);
    _mm512_storeu_pd(destination + 8, dequantized_high);
}

template <bool SIGN_VALUES>
__always_inline void process_single(const __m256i extracted, const __m512 scale_v, f32* destination) {
    const __m512i converted  = mm512_extend_epi16_epi32<SIGN_VALUES>(extracted);
    const __m512 dequantized = mm512_dequantize_epi32(converted, scale_v);
    _mm512_storeu_ps(destination, dequantized);
}

template <bool SIGN_VALUES>
__always_inline void process_single(const __m128i extracted, const __m512d scale_v, f64* destination) {
    const __m512i converted   = mm512_extend_epi16_epi64<SIGN_VALUES>(extracted);
    const __m512d dequantized = mm512_dequantize_epi64(converted, scale_v);
    _mm512_storeu_pd(destination, dequantized);
}

template <bool SIGN_VALUES>
__always_inline void process_pair(const __m256i extracted, const __m512d scale_v, f64* destination) {
    process_single<SIGN_VALUES>(_mm256_castsi256_si128(extracted), scale_v, destination);
    process_single<SIGN_VALUES>(_mm256_extracti128_si256(extracted, 1), scale_v, destination + 8);
}

__always_inline void process_single(const __m256i extracted, const __m512d scale_v, f64* destination) {
    const __m512i converted   = _mm512_cvtepi32_epi64(extracted);
    const __m512d dequantized = mm512_dequantize_epi64(converted, scale_v);
    _mm512_storeu_pd(destination, dequantized);
}

__always_inline void process_pair(const __m512i extracted, const __m512d scale_v, f64* destination) {
    process_single(_mm512_castsi512_si256(extracted), scale_v, destination);
    process_single(_mm512_extracti64x4_epi64(extracted, 1), scale_v, destination + 8);
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 7) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_decompress_block_avx512vbmi_1to7(const u8* __restrict__ input, const f32 scale, f32* __restrict__ output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    const u32 iterations_64      = elements_per_block / 64;
    const u32 iterations_32      = (elements_per_block % 64) / 32;
    const u32 iterations_16      = (elements_per_block % 32) / 16;
    const u32 remaining_elements = elements_per_block - (iterations_64 * 64) - (iterations_32 * 32) - (iterations_16 * 16);

    const __m512 scale_v = _mm512_set1_ps(scale);

    if constexpr (iterations_64 > 0) {
#pragma GCC unroll 8
        for (u32 i = 0; i < iterations_64; ++i) {
            const __m512i source   = mm512_loadu_elements_epi64(BIT_WIDTH, input);
            const __m512i unpacked = m512::mm512_unpack_epi8_avx512vbmi_1to7<BIT_WIDTH, SIGN_VALUES>(source);

            process_single<SIGN_VALUES>(_mm512_castsi512_si128(unpacked), scale_v, output);
            process_single<SIGN_VALUES>(_mm512_extracti64x2_epi64(unpacked, 1), scale_v, output + 16);
            process_single<SIGN_VALUES>(_mm512_extracti64x2_epi64(unpacked, 2), scale_v, output + 32);
            process_single<SIGN_VALUES>(_mm512_extracti64x2_epi64(unpacked, 3), scale_v, output + 48);

            output += 64;
            input += static_cast<ptrdiff_t>(8 * BIT_WIDTH);
        }
    }

    if constexpr (iterations_32 > 0) {
        const __m256i source   = mm256_loadu_elements_epi32(BIT_WIDTH, input);
        const __m256i unpacked = m256::mm256_unpack_epi8_avx512vbmi_1to8<BIT_WIDTH, SIGN_VALUES>(source);

        process_single<SIGN_VALUES>(_mm256_castsi256_si128(unpacked), scale_v, output);
        process_single<SIGN_VALUES>(_mm256_extracti128_si256(unpacked, 1), scale_v, output + 16);

        output += 32;
        input += static_cast<ptrdiff_t>(4 * BIT_WIDTH);
    }

    if constexpr (iterations_16 > 0) {
        const __m128i source   = mm_loadu_elements_epi16(BIT_WIDTH, input);
        const __m128i unpacked = m128::mm_unpack_epi8_avx512vbmi_1to8<BIT_WIDTH, SIGN_VALUES>(source);

        process_single<SIGN_VALUES>(unpacked, scale_v, output);

        output += 16;
        input += static_cast<ptrdiff_t>(2 * BIT_WIDTH);
    }

    if constexpr (remaining_elements > 0) {
        const __m128i source   = mm_loadu_elements_epi8(tail_bytes(BIT_WIDTH, remaining_elements), input);
        const __m128i unpacked = m128::mm_unpack_epi8_avx512vbmi_1to8<BIT_WIDTH, SIGN_VALUES>(source);

        const __m512i converted = mm512_extend_epi8_epi32<SIGN_VALUES>(unpacked);

        const __m512 dequantized = mm512_dequantize_epi32(converted, scale_v);

        mm512_storeu_elements_ps(output, remaining_elements, dequantized);
    }

    return 0;
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH == 8) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_decompress_block_avx512vbmi_8(const u8* __restrict__ input, const f32 scale, f32* __restrict__ output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    const u32 iterations = elements_per_block / 16;

    const __m512 scale_v = _mm512_set1_ps(scale);

#pragma GCC unroll 32
    for (u32 i = 0; i < iterations; ++i) {
        const __m128i source    = _mm_loadu_epi8(input);
        const __m512i converted = [&] {
            if constexpr (SIGN_VALUES) {
                return _mm512_cvtepi8_epi32(source);
            } else {
                return _mm512_cvtepu8_epi32(source);
            }
        }();
        const __m512 dequantized = mm512_dequantize_epi32(converted, scale_v);
        _mm512_storeu_ps(output, dequantized);

        input += 16;
        output += 16;
    }

    return 0;
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 7) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_decompress_block_avx512vbmi_1to7(const u8* __restrict__ input, const f64 scale, f64* __restrict__ output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    const u32 iterations_64      = elements_per_block / 64;
    const u32 iterations_32      = (elements_per_block % 64) / 32;
    const u32 iterations_16      = (elements_per_block % 32) / 16;
    const u32 remaining_elements = elements_per_block - (iterations_64 * 64) - (iterations_32 * 32) - (iterations_16 * 16);

    const __m512d scale_v = _mm512_set1_pd(scale);

    if constexpr (iterations_64 > 0) {
#pragma GCC unroll 8
        for (u32 i = 0; i < iterations_64; ++i) {
            const __m512i source   = mm512_loadu_elements_epi64(BIT_WIDTH, input);
            const __m512i unpacked = m512::mm512_unpack_epi8_avx512vbmi_1to7<BIT_WIDTH, SIGN_VALUES>(source);

            process_pair<SIGN_VALUES>(_mm512_castsi512_si128(unpacked), scale_v, output);
            process_pair<SIGN_VALUES>(_mm512_extracti64x2_epi64(unpacked, 1), scale_v, output + 16);
            process_pair<SIGN_VALUES>(_mm512_extracti64x2_epi64(unpacked, 2), scale_v, output + 32);
            process_pair<SIGN_VALUES>(_mm512_extracti64x2_epi64(unpacked, 3), scale_v, output + 48);

            output += 64;
            input += static_cast<ptrdiff_t>(8 * BIT_WIDTH);
        }
    }

    if constexpr (iterations_32 > 0) {
        const __m256i source   = mm256_loadu_elements_epi32(BIT_WIDTH, input);
        const __m256i unpacked = m256::mm256_unpack_epi8_avx512vbmi_1to8<BIT_WIDTH, SIGN_VALUES>(source);

        process_pair<SIGN_VALUES>(_mm256_castsi256_si128(unpacked), scale_v, output);
        process_pair<SIGN_VALUES>(_mm256_extracti128_si256(unpacked, 1), scale_v, output + 16);

        output += 32;
        input += static_cast<ptrdiff_t>(4 * BIT_WIDTH);
    }

    if constexpr (iterations_16 > 0) {
        const __m128i source   = mm_loadu_elements_epi16(BIT_WIDTH, input);
        const __m128i unpacked = m128::mm_unpack_epi8_avx512vbmi_1to8<BIT_WIDTH, SIGN_VALUES>(source);

        process_pair<SIGN_VALUES>(unpacked, scale_v, output);

        output += 16;
        input += static_cast<ptrdiff_t>(2 * BIT_WIDTH);
    }

    if constexpr (remaining_elements > 0) {
        const __m128i source   = mm_loadu_elements_epi8(tail_bytes(BIT_WIDTH, remaining_elements), input);
        const __m128i unpacked = m128::mm_unpack_epi8_avx512vbmi_1to8<BIT_WIDTH, SIGN_VALUES>(source);

        const __m512i converted_low   = mm512_extend_epi8_epi64<SIGN_VALUES>(unpacked);
        const __m512d dequantized_low = mm512_dequantize_epi64(converted_low, scale_v);
        mm512_storeu_elements_pd(output, remaining_elements < 8 ? remaining_elements : 8, dequantized_low);

        if constexpr (remaining_elements > 8) {
            const __m512i converted_high   = mm512_extend_epi8_epi64<SIGN_VALUES>(_mm_srli_si128(unpacked, 8));
            const __m512d dequantized_high = mm512_dequantize_epi64(converted_high, scale_v);
            mm512_storeu_elements_pd(output + 8, remaining_elements - 8, dequantized_high);
        }
    }

    return 0;
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH == 8) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_decompress_block_avx512vbmi_8(const u8* __restrict__ input, const f64 scale, f64* __restrict__ output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    const u32 iterations = elements_per_block / 8;

    const __m512d scale_v = _mm512_set1_pd(scale);

#pragma GCC unroll 32
    for (u32 i = 0; i < iterations; ++i) {
        const __m128i source    = _mm_loadu_si64(input);
        const __m512i converted = [&] {
            if constexpr (SIGN_VALUES) {
                return _mm512_cvtepi8_epi64(source);
            } else {
                return _mm512_cvtepu8_epi64(source);
            }
        }();
        const __m512d dequantized = mm512_dequantize_epi64(converted, scale_v);
        _mm512_storeu_pd(output, dequantized);

        input += 8;
        output += 8;
    }

    return 0;
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 15) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_decompress_block_avx512vbmi_9to15(const u8* __restrict__ input, const f32 scale, f32* __restrict__ output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr u32 iterations_32      = elements_per_block / 32;
    constexpr u32 iterations_16      = (elements_per_block % 32) / 16;
    constexpr u32 iterations_8       = (elements_per_block % 16) / 8;
    constexpr u32 remaining_elements = elements_per_block - (iterations_32 * 32) - (iterations_16 * 16) - (iterations_8 * 8);

    const __m512 scale_v    = _mm512_set1_ps(scale);
    const __m256 scale_v256 = _mm256_set1_ps(scale);

    if constexpr (iterations_32 > 0) {
#pragma GCC unroll 4
        for (u32 i = 0; i < iterations_32; ++i) {
            const __m512i source   = mm512_loadu_elements_epi32(BIT_WIDTH, input);
            const __m512i unpacked = m512::mm512_unpack_epi16_avx512vbmi_9to16<BIT_WIDTH, SIGN_VALUES>(source);

            process_single<SIGN_VALUES>(_mm512_castsi512_si256(unpacked), scale_v, output);
            process_single<SIGN_VALUES>(_mm512_extracti32x8_epi32(unpacked, 1), scale_v, output + 16);

            output += 32;
            input += static_cast<ptrdiff_t>(4 * BIT_WIDTH);
        }
    }

    if constexpr (iterations_16 > 0) {
        const __m256i source   = mm256_loadu_elements_epi16(BIT_WIDTH, input);
        const __m256i unpacked = m256::mm256_unpack_epi16_avx512vbmi_9to16<BIT_WIDTH, SIGN_VALUES>(source);

        process_single<SIGN_VALUES>(unpacked, scale_v, output);

        output += 16;
        input += static_cast<ptrdiff_t>(2 * BIT_WIDTH);
    }

    if constexpr (iterations_8 > 0) {
        const __m128i source   = mm_loadu_elements_epi8(BIT_WIDTH, input);
        const __m128i unpacked = m128::mm_unpack_epi16_avx512vbmi_9to16<BIT_WIDTH, SIGN_VALUES>(source);

        const __m256i converted  = mm256_extend_epi16_epi32<SIGN_VALUES>(unpacked);
        const __m256 dequantized = mm256_dequantize_epi32(converted, scale_v256);

        _mm256_storeu_ps(output, dequantized);

        output += 8;
        input += static_cast<ptrdiff_t>(BIT_WIDTH);
    }

    if constexpr (remaining_elements > 0) {
        const __m128i source   = mm_loadu_elements_epi8(tail_bytes(BIT_WIDTH, remaining_elements), input);
        const __m128i unpacked = m128::mm_unpack_epi16_avx512vbmi_9to16<BIT_WIDTH, SIGN_VALUES>(source);

        const __m256i converted  = mm256_extend_epi16_epi32<SIGN_VALUES>(unpacked);
        const __m256 dequantized = mm256_dequantize_epi32(converted, scale_v256);

        mm256_storeu_elements_ps(output, remaining_elements, dequantized);
    }

    return 0;
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH == 16) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_decompress_block_avx512vbmi_16(const u8* __restrict__ input, const f32 scale, f32* __restrict__ output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    const u32 iterations = elements_per_block / 16;

    const __m512 scale_v = _mm512_set1_ps(scale);

#pragma GCC unroll 16
    for (u32 i = 0; i < iterations; ++i) {
        const __m256i source    = _mm256_loadu_epi16(input);
        const __m512i converted = [&] {
            if constexpr (SIGN_VALUES) {
                return _mm512_cvtepi16_epi32(source);
            } else {
                return _mm512_cvtepu16_epi32(source);
            }
        }();
        const __m512 dequantized = mm512_dequantize_epi32(converted, scale_v);
        _mm512_storeu_ps(output, dequantized);

        input += 32;
        output += 16;
    }

    return 0;
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 15) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_decompress_block_avx512vbmi_9to15(const u8* __restrict__ input, const f64 scale, f64* __restrict__ output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr u32 iterations_32      = elements_per_block / 32;
    constexpr u32 iterations_16      = (elements_per_block % 32) / 16;
    constexpr u32 iterations_8       = (elements_per_block % 16) / 8;
    constexpr u32 remaining_elements = elements_per_block - iterations_32 * 32 - iterations_16 * 16 - iterations_8 * 8;

    const __m512d scale_v = _mm512_set1_pd(scale);

    if constexpr (iterations_32 > 0) {
#pragma GCC unroll 4
        for (u32 i = 0; i < iterations_32; ++i) {
            const __m512i source   = mm512_loadu_elements_epi32(BIT_WIDTH, input);
            const __m512i unpacked = m512::mm512_unpack_epi16_avx512vbmi_9to16<BIT_WIDTH, SIGN_VALUES>(source);

            process_pair<SIGN_VALUES>(_mm512_castsi512_si256(unpacked), scale_v, output);
            process_pair<SIGN_VALUES>(_mm512_extracti32x8_epi32(unpacked, 1), scale_v, output + 16);

            output += 32;
            input += static_cast<ptrdiff_t>(4 * BIT_WIDTH);
        }
    }

    if constexpr (iterations_16 > 0) {
        const __m256i source   = mm256_loadu_elements_epi16(BIT_WIDTH, input);
        const __m256i unpacked = m256::mm256_unpack_epi16_avx512vbmi_9to16<BIT_WIDTH, SIGN_VALUES>(source);

        process_pair<SIGN_VALUES>(unpacked, scale_v, output);

        output += 16;
        input += static_cast<ptrdiff_t>(2 * BIT_WIDTH);
    }

    if constexpr (iterations_8 > 0) {
        const __m128i source   = mm_loadu_elements_epi8(BIT_WIDTH, input);
        const __m128i unpacked = m128::mm_unpack_epi16_avx512vbmi_9to16<BIT_WIDTH, SIGN_VALUES>(source);

        const __m512i converted = mm512_extend_epi16_epi64<SIGN_VALUES>(unpacked);

        const __m512d dequantized = mm512_dequantize_epi64(converted, scale_v);

        _mm512_storeu_pd(output, dequantized);

        output += 8;
        input += static_cast<ptrdiff_t>(BIT_WIDTH);
    }

    if constexpr (remaining_elements > 0) {
        const __m128i source   = mm_loadu_elements_epi8(tail_bytes(BIT_WIDTH, remaining_elements), input);
        const __m128i unpacked = m128::mm_unpack_epi16_avx512vbmi_9to16<BIT_WIDTH, SIGN_VALUES>(source);

        const __m512i converted = mm512_extend_epi16_epi64<SIGN_VALUES>(unpacked);

        const __m512d dequantized = mm512_dequantize_epi64(converted, scale_v);

        mm512_storeu_elements_pd(output, remaining_elements, dequantized);
    }

    return 0;
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH == 16) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_decompress_block_avx512vbmi_16(const u8* __restrict__ input, const f64 scale, f64* __restrict__ output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    const u32 iterations = elements_per_block / 8;

    const __m512d scale_v = _mm512_set1_pd(scale);

#pragma GCC unroll 16
    for (u32 i = 0; i < iterations; ++i) {
        const __m128i source    = _mm_loadu_epi16(input);
        const __m512i converted = [&] {
            if constexpr (SIGN_VALUES) {
                return _mm512_cvtepi16_epi64(source);
            } else {
                return _mm512_cvtepu16_epi64(source);
            }
        }();
        const __m512d dequantized = mm512_dequantize_epi64(converted, scale_v);
        _mm512_storeu_pd(output, dequantized);

        input += 16;
        output += 8;
    }

    return 0;
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_decompress_block_avx512vbmi_17to24(const u8* __restrict__ input, const f32 scale, f32* __restrict__ output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr u32 iterations_16      = elements_per_block / 16;
    constexpr u32 iterations_8       = (elements_per_block % 16) / 8;
    constexpr u32 remaining_elements = elements_per_block - iterations_16 * 16 - iterations_8 * 8;

    const __m512 scale_v    = _mm512_set1_ps(scale);
    const __m256 scale_v256 = _mm256_set1_ps(scale);

    if constexpr (iterations_16 > 0) {
#pragma GCC unroll 2
        for (u32 i = 0; i < iterations_16; ++i) {
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

template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_decompress_block_avx512vbmi_17to24(const u8* __restrict__ input, const f64 scale, f64* __restrict__ output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr u32 iterations_16      = elements_per_block / 16;
    constexpr u32 iterations_8       = (elements_per_block % 16) / 8;
    constexpr u32 remaining_elements = elements_per_block - iterations_16 * 16 - iterations_8 * 8;

    const __m512d scale_v = _mm512_set1_pd(scale);

    if constexpr (iterations_16 > 0) {
#pragma GCC unroll 2
        for (u32 i = 0; i < iterations_16; ++i) {
            const __m512i source   = mm512_loadu_elements_epi16(BIT_WIDTH, input);
            const __m512i unpacked = m512::mm512_unpack_epi32_avx512vbmi_17to24<BIT_WIDTH, SIGN_VALUES>(source);

            process_pair(unpacked, scale_v, output);

            output += 16;
            input += static_cast<ptrdiff_t>(2 * BIT_WIDTH);
        }
    }

    if constexpr (iterations_8 > 0) {
        const __m256i source   = mm256_loadu_elements_epi8(BIT_WIDTH, input);
        const __m256i unpacked = m256::mm256_unpack_epi32_avx512vbmi_17to24<BIT_WIDTH, SIGN_VALUES>(source);

        process_single(unpacked, scale_v, output);

        output += 8;
        input += static_cast<ptrdiff_t>(BIT_WIDTH);
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
template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_decompress_block_avx512vbmi(const void* __restrict__ input_ptr, const f32 scale, void* __restrict__ output_ptr) {
    const auto* input = static_cast<const u8*>(input_ptr);
    auto* output      = static_cast<f32*>(output_ptr);

    if constexpr (BIT_WIDTH <= 7) {
        return internal::mm512_decompress_block_avx512vbmi_1to7<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH == 8) {
        return internal::mm512_decompress_block_avx512vbmi_8<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH <= 15) {
        return internal::mm512_decompress_block_avx512vbmi_9to15<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH == 16) {
        return internal::mm512_decompress_block_avx512vbmi_16<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH <= 24) {
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
template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_decompress_block_avx512vbmi(const void* __restrict__ input_ptr, const f64 scale, void* __restrict__ output_ptr) {
    const auto* input = static_cast<const u8*>(input_ptr);
    auto* output      = static_cast<f64*>(output_ptr);

    if constexpr (BIT_WIDTH <= 7) {
        return internal::mm512_decompress_block_avx512vbmi_1to7<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH == 8) {
        return internal::mm512_decompress_block_avx512vbmi_8<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH <= 15) {
        return internal::mm512_decompress_block_avx512vbmi_9to15<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH == 16) {
        return internal::mm512_decompress_block_avx512vbmi_16<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH <= 24) {
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
template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm512_decompress_blocks_avx512vbmi(const void* __restrict__ input_ptr, const f32 scale, void* __restrict__ output_ptr,
                                       const u32 blocks) {
    const auto* input = static_cast<const u8*>(input_ptr);
    auto* output      = static_cast<f32*>(output_ptr);

    const u8* block_input = input;
    f32* block_output     = output;

    for (u32 block = 0; block < blocks; ++block) {
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
template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm512_decompress_blocks_avx512vbmi(const void* __restrict__ input_ptr, const f64 scale, void* __restrict__ output_ptr,
                                       const u32 blocks) {
    const auto* input = static_cast<const u8*>(input_ptr);
    auto* output      = static_cast<f64*>(output_ptr);

    const u8* block_input = input;
    f64* block_output     = output;

    for (u32 block = 0; block < blocks; ++block) {
        mm512_decompress_block_avx512vbmi<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(block_input, scale, block_output);
        block_input += BLOCK_SIZE;
        block_output += (BLOCK_SIZE * 8) / BIT_WIDTH;
    }
    return 0;
}
}  // namespace pernix

#endif  // PERNIX_AVX512VBMI_DECOMPRESSION_H
