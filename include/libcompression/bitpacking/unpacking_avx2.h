#ifndef LIBCOMPRESSION_UNPACKING_AVX2_H
#define LIBCOMPRESSION_UNPACKING_AVX2_H

#include <libcompression/helper.h>

#ifdef LIBCOMPRESSION_AVX2_ENABLED

#include <immintrin.h>

#include <cstdint>

#include "unpacking_tables.h"

namespace libcompression::bitpacking {
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
    const __m128i shuffled = _mm_shuffle_epi8(source, internal::unpack_tables_avx2<BIT_WIDTH, __m128i>::get_shuffle());

    constexpr uint16_t shift = 32 - BIT_WIDTH;
    __m128i shifted          = _mm_sllv_epi32(shuffled, internal::unpack_tables_avx2<BIT_WIDTH, __m128i>::get_shift());
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
    const __m256i permuted = _mm256_permutevar8x32_epi32(source, internal::unpack_tables_avx2<BIT_WIDTH, __m256i>::get_permute());
    const __m256i shuffled = _mm256_shuffle_epi8(permuted, internal::unpack_tables_avx2<BIT_WIDTH, __m256i>::get_shuffle());

    constexpr uint16_t shift = 32 - BIT_WIDTH;
    __m256i shifted          = _mm256_sllv_epi32(shuffled, internal::unpack_tables_avx2<BIT_WIDTH, __m256i>::get_shift());
    if constexpr (SIGN_VALUES) {
        shifted = _mm256_srai_epi32(shifted, shift);
    } else {
        shifted = _mm256_srli_epi32(shifted, shift);
    }

    return shifted;
}

auto mm_unpack_aligned_epi32_avx2(uint8_t bit_width, const uint8_t* __restrict__ input) -> __m128i;

auto mm_unpack_epi32_avx2(uint8_t bit_width, const uint8_t* __restrict__ input) -> __m128i;

auto mm256_unpack_aligned_epi32_avx2(uint8_t bit_width, const uint8_t* __restrict__ input) -> __m256i;

auto mm256_unpack_epi32_avx2(uint8_t bit_width, const uint8_t* __restrict__ input) -> __m256i;
}  // namespace libcompression::bitpacking
#endif  // LIBCOMPRESSION_AVX2_ENABLED
#endif  // LIBCOMPRESSION_UNPACKING_AVX2_H
