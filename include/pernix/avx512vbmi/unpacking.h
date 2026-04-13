#ifndef PERNIX_AVX512VBMI_UNPACKING_H
#define PERNIX_AVX512VBMI_UNPACKING_H

#include <immintrin.h>
#include <pernix/avx512vbmi/tables.h>

namespace pernix::internal {

namespace m128 {
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8)
[[gnu::always_inline]] inline __m128i mm_unpack_epi8_avx512vbmi_1to8(const __m128i& input) {
    if (BIT_WIDTH <= 8) {
        return input;
    } else {
        return _mm_setzero_si128();
    }
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
[[gnu::always_inline]] inline __m128i mm_unpack_epi16_avx512vbmi_9to16(const __m128i& input) {
    if (BIT_WIDTH == 16) {
        return input;
    } else {
        using tables = unpack_tables_avx512_16_new<BIT_WIDTH, __m128i>;

        const __m128i permuted = _mm_permutexvar_epi8(tables::get_permute1(), input);

        __m128i shifted = _mm_srlv_epi16(permuted, tables::get_shift1());

        if constexpr (BIT_WIDTH == 11 || BIT_WIDTH == 13 || BIT_WIDTH == 14 || BIT_WIDTH == 15) {
            const __m128i permuted2 = _mm_permutexvar_epi8(tables::get_permute2(), input);
            const __m128i shifted2  = _mm_sllv_epi16(permuted2, tables::get_shift2());
            shifted                 = _mm_or_si128(shifted, shifted2);
        }

        constexpr uint32_t shift = 16 - BIT_WIDTH;
        shifted                  = _mm_slli_epi16(shifted, shift);
        if (SIGN_VALUES) {
            shifted = _mm_srai_epi16(shifted, shift);
        } else {
            shifted = _mm_srli_epi16(shifted, shift);
        }

        return shifted;
    }
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24)
[[gnu::always_inline]] inline __m128i mm_unpack_epi32_avx512vbmi_17to24(const __m128i& input) {
    using tables = unpack_tables_avx512_24<BIT_WIDTH, __m128i>;

    const __m128i permuted = _mm_permutexvar_epi8(tables::get_permute(), input);

    constexpr uint32_t shift = 32 - BIT_WIDTH;
    __m128i shifted          = _mm_sllv_epi32(permuted, tables::get_shift());
    if constexpr (SIGN_VALUES && BIT_WIDTH > 1) {
        shifted = _mm_srai_epi32(shifted, shift);
    } else {
        shifted = _mm_srli_epi32(shifted, shift);
    }

    return shifted;
}
}  // namespace m128

namespace m256 {

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8)
[[gnu::always_inline]] inline __m256i mm256_unpack_epi8_avx512vbmi_1to8(const __m256i& input) {
    if (BIT_WIDTH <= 8) {
        return input;
    } else {
        return _mm256_setzero_si256();
    }
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
[[gnu::always_inline]] inline __m256i mm256_unpack_epi16_avx512vbmi_9to16(const __m256i& input) {
    if (BIT_WIDTH == 16) {
        return input;
    } else {
        using tables = unpack_tables_avx512_16_new<BIT_WIDTH, __m256i>;

        const __m256i permuted = _mm256_permutexvar_epi8(tables::get_permute1(), input);

        __m256i shifted = _mm256_srlv_epi16(permuted, tables::get_shift1());

        if constexpr (BIT_WIDTH == 11 || BIT_WIDTH == 13 || BIT_WIDTH == 14 || BIT_WIDTH == 15) {
            const __m256i permuted2 = _mm256_permutexvar_epi8(tables::get_permute2(), input);
            const __m256i shifted2  = _mm256_sllv_epi16(permuted2, tables::get_shift2());
            shifted                 = _mm256_or_si256(shifted, shifted2);
        }

        constexpr uint32_t shift = 16 - BIT_WIDTH;
        shifted                  = _mm256_slli_epi16(shifted, shift);
        if (SIGN_VALUES) {
            shifted = _mm256_srai_epi16(shifted, shift);
        } else {
            shifted = _mm256_srli_epi16(shifted, shift);
        }

        return shifted;
    }
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24)
[[gnu::always_inline]] inline __m256i mm256_unpack_epi32_avx512vbmi_17to24(const __m256i& input) {
    using tables = unpack_tables_avx512_24<BIT_WIDTH, __m256i>;

    const __m256i permuted = _mm256_permutexvar_epi8(tables::get_permute(), input);

    constexpr uint32_t shift = 32 - BIT_WIDTH;
    __m256i shifted          = _mm256_sllv_epi32(permuted, tables::get_shift());
    if constexpr (SIGN_VALUES && BIT_WIDTH > 1) {
        shifted = _mm256_srai_epi32(shifted, shift);
    } else {
        shifted = _mm256_srli_epi32(shifted, shift);
    }

    return shifted;
}

}  // namespace m256

namespace m512 {

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8)
[[gnu::always_inline]] inline __m512i mm512_unpack_epi8_avx512vbmi_1to8(const __m512i& input) {
    if (BIT_WIDTH == 8) {
        return input;
    } else {
        if (BIT_WIDTH == 1) {
            const auto value       = static_cast<__mmask64>(_mm_cvtsi128_si64(_mm512_castsi512_si128(input)));
            const __m512i source   = _mm512_movm_epi8(value);
            const __m512i unpacked = _mm512_abs_epi8(source);
            return unpacked;
        } else if (BIT_WIDTH == 2) {
        }

        return _mm512_setzero_si512();
    }
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
[[gnu::always_inline]] inline __m512i mm512_unpack_epi16_avx512vbmi_9to16(const __m512i& input) {
    if (BIT_WIDTH == 16) {
        return input;
    } else {
        using tables = unpack_tables_avx512_16_new<BIT_WIDTH, __m512i>;

        const __m512i permuted = _mm512_permutexvar_epi8(tables::get_permute1(), input);
        __m512i shifted        = _mm512_srlv_epi16(permuted, tables::get_shift1());

        if constexpr (BIT_WIDTH == 11 || BIT_WIDTH == 13 || BIT_WIDTH == 14 || BIT_WIDTH == 15) {
            const __m512i permuted2 = _mm512_permutexvar_epi8(tables::get_permute2(), input);
            const __m512i shifted2  = _mm512_sllv_epi16(permuted2, tables::get_shift2());
            shifted                 = _mm512_or_si512(shifted, shifted2);
        }

        constexpr uint32_t shift = 16 - BIT_WIDTH;
        shifted                  = _mm512_slli_epi16(shifted, shift);
        if (SIGN_VALUES) {
            shifted = _mm512_srai_epi16(shifted, shift);
        } else {
            shifted = _mm512_srli_epi16(shifted, shift);
        }

        return shifted;
    }
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24)
[[gnu::always_inline]] inline __m512i mm512_unpack_epi32_avx512vbmi_17to24(const __m512i& input) {
    using tables = unpack_tables_avx512_24<BIT_WIDTH, __m512i>;

    const __m512i permuted = _mm512_permutexvar_epi8(tables::get_permute(), input);
    __m512i shifted        = _mm512_sllv_epi32(permuted, tables::get_shift());

    constexpr uint32_t shift = 32 - BIT_WIDTH;
    if constexpr (SIGN_VALUES) {
        shifted = _mm512_srai_epi32(shifted, shift);
    } else {
        shifted = _mm512_srli_epi32(shifted, shift);
    }

    return shifted;
}

}  // namespace m512
}  // namespace pernix::internal

#endif  // PERNIX_AVX512VBMI_UNPACKING_H
