#ifndef LIBCOMPRESSION_UNPACKING_AVX512_H
#define LIBCOMPRESSION_UNPACKING_AVX512_H

#include <libcompression/helper.h>

#ifdef LIBCOMPRESSION_AVX512_VBMI_ENABLED

#include <immintrin.h>
#include <cstdint>
#include <stdexcept>

#include "unpacking_tables.h"

namespace libcompression::bitpacking {
    template<uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
        requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
    __m128i mm_unpack_epi32_avx512vbmi(const uint8_t *__restrict__ input) {
        if constexpr (BIT_WIDTH > 0 && BIT_WIDTH <= 8) {
            const __m128i source = _mm_loadu_si32(input);
            const __m128i shuffled = _mm_permutexvar_epi8(
                internal::unpack_tables_avx512_8<BIT_WIDTH, __m128i>::get_shuffle(), source);

            __m128i shifted = _mm_sllv_epi16(
                shuffled, internal::unpack_tables_avx512_8<BIT_WIDTH, __m128i>::get_shift());

            constexpr uint16_t shift = 16 - BIT_WIDTH;
            if constexpr (SIGN_VALUES) {
                shifted = _mm_srai_epi16(shifted, shift);
            } else {
                shifted = _mm_srli_epi16(shifted, shift);
            }

            return _mm_cvtepi16_epi32(shifted);
        } else {
            __m128i source = _mm_loadu_si64(input);
            const __m128i shuffled = _mm_permutexvar_epi8(
                internal::unpack_tables_avx512_24<BIT_WIDTH, __m128i>::get_shuffle(), source);

            constexpr uint16_t shift = 32 - BIT_WIDTH;
            __m128i shifted = _mm_sllv_epi32(
                shuffled, internal::unpack_tables_avx512_24<BIT_WIDTH, __m128i>::get_shift());
            if constexpr (SIGN_VALUES) {
                shifted = _mm_srai_epi32(shifted, shift);
            } else {
                shifted = _mm_srli_epi32(shifted, shift);
            }

            return shifted;
        }
    }

    template<uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
        requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
    __m256i mm256_unpack_epi32_avx512vbmi(const uint8_t *__restrict__ input) {
        if constexpr (BIT_WIDTH > 0 && BIT_WIDTH <= 8) {
            const __m128i source = _mm_loadu_si64(input);
            const __m128i shuffled = _mm_permutexvar_epi8(
                internal::unpack_tables_avx512_8<BIT_WIDTH, __m128i>::get_shuffle(), source);

            __m128i shifted = _mm_sllv_epi16(
                shuffled, internal::unpack_tables_avx512_8<BIT_WIDTH, __m128i>::get_shift());

            constexpr uint16_t shift = 16 - BIT_WIDTH;
            if constexpr (SIGN_VALUES) {
                shifted = _mm_srai_epi16(shifted, shift);
            } else {
                shifted = _mm_srli_epi16(shifted, shift);
            }

            return _mm256_cvtepi16_epi32(shifted);
        } else {
            __m256i source = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(input)));

            const __m256i shuffled = _mm256_permutexvar_epi8(
                internal::unpack_tables_avx512_24<BIT_WIDTH, __m256i>::get_shuffle(), source);

            constexpr uint16_t shift = 32 - BIT_WIDTH;
            __m256i shifted = _mm256_sllv_epi32(
                shuffled, internal::unpack_tables_avx512_24<BIT_WIDTH, __m256i>::get_shift());
            if constexpr (SIGN_VALUES) {
                shifted = _mm256_srai_epi32(shifted, shift);
            } else {
                shifted = _mm256_srli_epi32(shifted, shift);
            }

            return shifted;
        }
    }

    template<uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
        requires(BIT_WIDTH == 8 || BIT_WIDTH == 16)
    __m512i mm512_unpack_aligned_epi32_avx512vbmi(const uint8_t *__restrict__ input) {
        if constexpr (BIT_WIDTH == 8) {
            const __m128i source = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input));
            if constexpr (SIGN_VALUES) {
                return _mm512_cvtepi8_epi32(source);
            } else {
                return _mm512_cvtepu8_epi32(source);
            }
        } else {
            const __m256i source = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input));
            if constexpr (SIGN_VALUES) {
                return _mm512_cvtepi16_epi32(source);
            } else {
                return _mm512_cvtepu16_epi32(source);
            }
        }
    }

    template<uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
        requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
    auto mm512_unpack_epi32_avx512vbmi(const uint8_t *__restrict__ input) -> __m512i {
        if constexpr (BIT_WIDTH > 0 && BIT_WIDTH <= 8) {
            const __m256i source = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(input)));
            const __m256i permuted = _mm256_permutexvar_epi8(
                internal::unpack_tables_avx512_8<BIT_WIDTH, __m256i>::get_shuffle(), source);

            __m256i shifted = _mm256_sllv_epi16(
                permuted, internal::unpack_tables_avx512_8<BIT_WIDTH, __m256i>::get_shift());

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
                source = _mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(input)));
            } else {
                source = _mm512_loadu_si512(input);
            }
            const __m512i permuted = _mm512_permutexvar_epi8(
                internal::unpack_tables_avx512_24<BIT_WIDTH, __m512i>::get_shuffle(), source);

            __m512i shifted = _mm512_sllv_epi32(
                permuted, internal::unpack_tables_avx512_24<BIT_WIDTH, __m512i>::get_shift());

            constexpr uint16_t shift = 32 - BIT_WIDTH;
            if constexpr (SIGN_VALUES) {
                shifted = _mm512_srai_epi32(shifted, shift);
            } else {
                shifted = _mm512_srli_epi32(shifted, shift);
            }

            return shifted;
        }
    }

    auto mm_unpack_epi32_avx512vbmi(const uint8_t bit_width, const uint8_t *__restrict__ input) -> __m128i;

    auto mm256_unpack_epi32_avx512vbmi(const uint8_t bit_width,
                                       const uint8_t *__restrict__ input) -> __m256i;

    auto mm512_unpack_aligned_epi32_avx512vbmi(const uint8_t bit_width,
                                               const uint8_t *__restrict__ input) -> __m512i;

    auto mm512_unpack_epi32_avx512vbmi(const uint8_t bit_width,
                                       const uint8_t *__restrict__ input) -> __m512i;
} // namespace libcompression::bitpacking
#endif  // LIBCOMPRESSION_AVX512_VBMI_ENABLED
#endif  // LIBCOMPRESSION_UNPACKING_AVX512_H
