#ifndef PERNIX_AVX512VBMI_UNPACKING_H
#define PERNIX_AVX512VBMI_UNPACKING_H

#include <immintrin.h>
#include <pernix/avx512vbmi/tables.h>

namespace pernix::internal {

namespace m128 {

constexpr __mmask16 kAlternateByteMask16 = 0xAAAAULL;

[[gnu::always_inline]] inline static __m128i _mm_srlv_epi8(const __m128i a, const __m128i count) {
    const __m128i mask      = _mm_set1_epi16(0x00ff);
    const __m128i low_half  = _mm_srlv_epi16(_mm_and_si128(mask, a), _mm_and_si128(mask, count));
    const __m128i high_half = _mm_srlv_epi16(a, _mm_srli_epi16(count, 8));
    return _mm_mask_blend_epi8(kAlternateByteMask16, low_half, high_half);
}

[[gnu::always_inline]] inline static __m128i _mm_sllv_epi8(const __m128i a, const __m128i count) {
    const __m128i mask      = _mm_set1_epi16(0xff00);
    const __m128i low_half  = _mm_sllv_epi16(a, _mm_andnot_si128(mask, count));
    const __m128i high_half = _mm_sllv_epi16(_mm_and_si128(mask, a), _mm_srli_epi16(count, 8));
    return _mm_mask_blend_epi8(kAlternateByteMask16, low_half, high_half);
}

[[gnu::always_inline]] inline static __m128i _mm_slli_epi8(const __m128i a, const int8_t imm8) {
    return _mm_sllv_epi8(a, _mm_set1_epi8(imm8));
}

[[gnu::always_inline]] inline static __m128i _mm_srli_epi8(const __m128i a, const int8_t imm8) {
    const __m128i lo_mask = _mm_set1_epi16(0x00ff);
    const __m128i hi_mask = _mm_set1_epi16(0xff00);

    const __m128i lo = _mm_srli_epi16(_mm_and_si128(a, lo_mask), imm8);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(a, imm8), hi_mask);

    return _mm_mask_blend_epi8(kAlternateByteMask16, lo, hi);
}

[[gnu::always_inline]] inline static __m128i _mm_srai_epi8(const __m128i a, const int8_t imm8) {
    const __m128i lo_mask = _mm_set1_epi16(0x00ff);
    const __m128i hi_mask = _mm_set1_epi16(0xff00);

    const __m128i hi = _mm_and_si128(_mm_srai_epi16(a, imm8), hi_mask);

    const __m128i lo_as_hi = _mm_slli_epi16(_mm_and_si128(a, lo_mask), 8);
    const __m128i lo       = _mm_and_si128(_mm_srli_epi16(_mm_srai_epi16(lo_as_hi, imm8), 8), lo_mask);

    return _mm_mask_blend_epi8(kAlternateByteMask16, lo, hi);
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8)
[[gnu::always_inline]] inline __m128i mm_unpack_epi8_avx512vbmi_1to8(const __m128i& input) {
    if constexpr (BIT_WIDTH == 8) {
        return input;
    } else {
        if constexpr (BIT_WIDTH == 1) {
            const auto value       = static_cast<__mmask64>(_mm_cvtsi128_si64(input));
            const __m128i source   = _mm_movm_epi8(value);
            const __m128i unpacked = _mm_abs_epi8(source);
            return unpacked;
        } else if constexpr (BIT_WIDTH == 2) {
            __m128i values_shift0       = input;
            __m128i values_shift2       = _mm_srli_epi16(values_shift0, 2);
            const __m128i values_shift4 = _mm_srli_epi16(values_shift0, 4);
            const __m128i values_shift6 = _mm_srli_epi16(values_shift0, 6);

            __m128i interleave_tmp = _mm_unpacklo_epi8(values_shift0, values_shift2);
            values_shift0          = _mm_unpackhi_epi8(values_shift0, values_shift2);
            values_shift0          = _mm_unpacklo_epi64(interleave_tmp, values_shift0);

            interleave_tmp = _mm_unpacklo_epi8(values_shift4, values_shift6);
            values_shift2  = _mm_unpackhi_epi8(values_shift4, values_shift6);
            values_shift2  = _mm_unpacklo_epi64(interleave_tmp, values_shift2);

            interleave_tmp = _mm_unpacklo_epi16(values_shift0, values_shift2);
            values_shift0  = _mm_unpackhi_epi16(values_shift0, values_shift2);
            values_shift0  = _mm_unpacklo_epi64(interleave_tmp, values_shift0);
            values_shift0  = _mm_shuffle_epi32(values_shift0, 0xD8);

            values_shift0 = _mm_and_si128(values_shift0, _mm_set1_epi16(0x0303));

            return values_shift0;
        } else if constexpr (BIT_WIDTH == 4) {
            __m128i values_shift0       = input;
            const __m128i values_shift4 = _mm_srli_epi16(values_shift0, 4);

            const __m128i interleave_tmp = _mm_unpacklo_epi8(values_shift0, values_shift4);
            values_shift0                = _mm_unpackhi_epi8(values_shift0, values_shift4);
            values_shift0                = _mm_unpacklo_epi64(interleave_tmp, values_shift0);
            values_shift0                = _mm_shuffle_epi32(values_shift0, 0xD8);

            values_shift0 = _mm_and_si128(values_shift0, _mm_set1_epi16(0x0F0F));

            return values_shift0;
        } else {
            using tables = unpack_tables_avx512_8<BIT_WIDTH, __m128i>;

            const __m128i permuted1 = _mm_permutexvar_epi8(tables::get_permute1(), input);
            const __m128i permuted2 = _mm_permutexvar_epi8(tables::get_permute2(), input);

            const __m128i shifted1 = _mm_srlv_epi8(permuted1, tables::get_shift1());
            const __m128i shifted2 = _mm_sllv_epi8(permuted2, tables::get_shift2());

            const __mmask16 spill_mask = _mm_cmpneq_epi8_mask(tables::get_shift2(), _mm_setzero_si128());
            __m128i combined           = _mm_or_si128(shifted1, _mm_maskz_mov_epi8(spill_mask, shifted2));

            constexpr uint32_t shift = 8 - BIT_WIDTH;
            combined                 = _mm_slli_epi8(combined, shift);
            if (SIGN_VALUES) {
                combined = _mm_srai_epi8(combined, shift);
            } else {
                combined = _mm_srli_epi8(combined, shift);
            }

            return combined;
        }
    }
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
[[gnu::always_inline]] inline __m128i mm_unpack_epi16_avx512vbmi_9to16(const __m128i& input) {
    if constexpr (BIT_WIDTH == 16) {
        return input;
    } else {
        using tables = unpack_tables_avx512_16<BIT_WIDTH, __m128i>;

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

constexpr __mmask32 kAlternateByteMask32 = 0xAAAAAAAAULL;

[[gnu::always_inline]] inline static __m256i _mm256_srlv_epi8(const __m256i a, const __m256i count) {
    const __m256i mask      = _mm256_set1_epi16(0x00ff);
    const __m256i low_half  = _mm256_srlv_epi16(_mm256_and_si256(mask, a), _mm256_and_si256(mask, count));
    const __m256i high_half = _mm256_srlv_epi16(a, _mm256_srli_epi16(count, 8));
    return _mm256_mask_blend_epi8(kAlternateByteMask32, low_half, high_half);
}

[[gnu::always_inline]] inline static __m256i _mm256_sllv_epi8(const __m256i a, const __m256i count) {
    const __m256i mask      = _mm256_set1_epi16(0xff00);
    const __m256i low_half  = _mm256_sllv_epi16(a, _mm256_andnot_si256(mask, count));
    const __m256i high_half = _mm256_sllv_epi16(_mm256_and_si256(mask, a), _mm256_srli_epi16(count, 8));
    return _mm256_mask_blend_epi8(kAlternateByteMask32, low_half, high_half);
}

[[gnu::always_inline]] inline static __m256i _mm256_slli_epi8(const __m256i a, const int8_t imm8) {
    return _mm256_sllv_epi8(a, _mm256_set1_epi8(imm8));
}

[[gnu::always_inline]] inline static __m256i _mm256_srli_epi8(const __m256i a, const int8_t imm8) {
    const __m256i lo_mask = _mm256_set1_epi16(0x00ff);
    const __m256i hi_mask = _mm256_set1_epi16(0xff00);

    const __m256i lo = _mm256_srli_epi16(_mm256_and_si256(a, lo_mask), imm8);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(a, imm8), hi_mask);

    return _mm256_mask_blend_epi8(kAlternateByteMask32, lo, hi);
}

[[gnu::always_inline]] inline static __m256i _mm256_srai_epi8(const __m256i a, const int8_t imm8) {
    const __m256i lo_mask = _mm256_set1_epi16(0x00ff);
    const __m256i hi_mask = _mm256_set1_epi16(0xff00);

    const __m256i hi = _mm256_and_si256(_mm256_srai_epi16(a, imm8), hi_mask);

    const __m256i lo_as_hi = _mm256_slli_epi16(_mm256_and_si256(a, lo_mask), 8);
    const __m256i lo       = _mm256_and_si256(_mm256_srli_epi16(_mm256_srai_epi16(lo_as_hi, imm8), 8), lo_mask);

    return _mm256_mask_blend_epi8(kAlternateByteMask32, lo, hi);
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8)
[[gnu::always_inline]] inline __m256i mm256_unpack_epi8_avx512vbmi_1to8(const __m256i& input) {
    if constexpr (BIT_WIDTH == 8) {
        return input;
    } else {
        if constexpr (BIT_WIDTH == 1) {
            const auto value       = static_cast<__mmask64>(_mm_cvtsi128_si64(_mm256_castsi256_si128(input)));
            const __m256i source   = _mm256_movm_epi8(value);
            const __m256i unpacked = _mm256_abs_epi8(source);
            return unpacked;
        } else if constexpr (BIT_WIDTH == 2) {
            __m256i values_shift0       = input;
            __m256i values_shift2       = _mm256_srli_epi16(values_shift0, 2);
            const __m256i values_shift4 = _mm256_srli_epi16(values_shift0, 4);
            const __m256i values_shift6 = _mm256_srli_epi16(values_shift0, 6);

            __m256i interleave_tmp = _mm256_unpacklo_epi8(values_shift0, values_shift2);
            values_shift0          = _mm256_unpackhi_epi8(values_shift0, values_shift2);
            values_shift0          = _mm256_shuffle_i64x2(interleave_tmp, values_shift0, 0b00000000);

            interleave_tmp = _mm256_unpacklo_epi8(values_shift4, values_shift6);
            values_shift2  = _mm256_unpackhi_epi8(values_shift4, values_shift6);
            values_shift2  = _mm256_shuffle_i64x2(interleave_tmp, values_shift2, 0b00000000);

            interleave_tmp = _mm256_unpacklo_epi16(values_shift0, values_shift2);
            values_shift0  = _mm256_unpackhi_epi16(values_shift0, values_shift2);
            values_shift0  = _mm256_shuffle_i64x2(interleave_tmp, values_shift0, 0x88);
            values_shift0  = _mm256_shuffle_i64x2(values_shift0, values_shift0, 0xD8);

            values_shift0 = _mm256_and_si256(values_shift0, _mm256_set1_epi16(0x0303));

            return values_shift0;
        } else if constexpr (BIT_WIDTH == 4) {
            __m256i values_shift0       = input;
            const __m256i values_shift4 = _mm256_srli_epi16(values_shift0, 4);

            __m256i interleave_tmp = _mm256_unpacklo_epi8(values_shift0, values_shift4);
            values_shift0          = _mm256_unpackhi_epi8(values_shift0, values_shift4);
            values_shift0          = _mm256_shuffle_i64x2(interleave_tmp, values_shift0, 0x44);
            values_shift0          = _mm256_shuffle_i64x2(values_shift0, values_shift0, 0xD8);

            values_shift0 = _mm256_and_si256(values_shift0, _mm256_set1_epi16(0x0F0F));

            return values_shift0;
        } else {
            using tables = unpack_tables_avx512_8<BIT_WIDTH, __m256i>;

            const __m256i permuted1 = _mm256_permutexvar_epi8(tables::get_permute1(), input);
            const __m256i permuted2 = _mm256_permutexvar_epi8(tables::get_permute2(), input);

            const __m256i shifted1 = _mm256_srlv_epi8(permuted1, tables::get_shift1());
            const __m256i shifted2 = _mm256_sllv_epi8(permuted2, tables::get_shift2());

            const __mmask32 spill_mask = _mm256_cmpneq_epi8_mask(tables::get_shift2(), _mm256_setzero_si256());
            __m256i combined           = _mm256_or_si256(shifted1, _mm256_maskz_mov_epi8(spill_mask, shifted2));

            constexpr uint32_t shift = 8 - BIT_WIDTH;
            combined                 = _mm256_slli_epi8(combined, shift);
            if (SIGN_VALUES) {
                combined = _mm256_srai_epi8(combined, shift);
            } else {
                combined = _mm256_srli_epi8(combined, shift);
            }

            return combined;
        }
    }
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
[[gnu::always_inline]] inline __m256i mm256_unpack_epi16_avx512vbmi_9to16(const __m256i& input) {
    if constexpr (BIT_WIDTH == 16) {
        return input;
    } else {
        using tables = unpack_tables_avx512_16<BIT_WIDTH, __m256i>;

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

constexpr __mmask64 kAlternateByteMask64 = 0xAAAAAAAAAAAAAAAAULL;

[[gnu::always_inline]] inline static __m512i _mm512_srlv_epi8(const __m512i a, const __m512i count) {
    const __m512i mask      = _mm512_set1_epi16(0x00ff);
    const __m512i low_half  = _mm512_srlv_epi16(_mm512_and_si512(mask, a), _mm512_and_si512(mask, count));
    const __m512i high_half = _mm512_srlv_epi16(a, _mm512_srli_epi16(count, 8));
    return _mm512_mask_blend_epi8(kAlternateByteMask64, low_half, high_half);
}

[[gnu::always_inline]] inline static __m512i _mm512_sllv_epi8(const __m512i a, const __m512i count) {
    const __m512i mask      = _mm512_set1_epi16(0xff00);
    const __m512i low_half  = _mm512_sllv_epi16(a, _mm512_andnot_si512(mask, count));
    const __m512i high_half = _mm512_sllv_epi16(_mm512_and_si512(mask, a), _mm512_srli_epi16(count, 8));
    return _mm512_mask_blend_epi8(kAlternateByteMask64, low_half, high_half);
}

[[gnu::always_inline]] inline static __m512i _mm512_slli_epi8(const __m512i a, const int8_t imm8) {
    return _mm512_sllv_epi8(a, _mm512_set1_epi8(imm8));
}

[[gnu::always_inline]] inline static __m512i _mm512_srli_epi8(const __m512i a, const int8_t imm8) {
    const __m512i lo_mask = _mm512_set1_epi16(0x00ff);
    const __m512i hi_mask = _mm512_set1_epi16(0xff00);

    const __m512i lo = _mm512_srli_epi16(_mm512_and_si512(a, lo_mask), imm8);
    const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(a, imm8), hi_mask);

    return _mm512_mask_blend_epi8(kAlternateByteMask64, lo, hi);
}

[[gnu::always_inline]] inline static __m512i _mm512_srai_epi8(const __m512i a, const int8_t imm8) {
    const __m512i lo_mask = _mm512_set1_epi16(0x00ff);
    const __m512i hi_mask = _mm512_set1_epi16(0xff00);

    const __m512i hi = _mm512_and_si512(_mm512_srai_epi16(a, imm8), hi_mask);

    const __m512i lo_as_hi = _mm512_slli_epi16(_mm512_and_si512(a, lo_mask), 8);
    const __m512i lo       = _mm512_and_si512(_mm512_srli_epi16(_mm512_srai_epi16(lo_as_hi, imm8), 8), lo_mask);

    return _mm512_mask_blend_epi8(kAlternateByteMask64, lo, hi);
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8)
[[gnu::always_inline]] inline __m512i mm512_unpack_epi8_avx512vbmi_1to8(const __m512i& input) {
    if constexpr (BIT_WIDTH == 8) {
        return input;
    } else {
        if constexpr (BIT_WIDTH == 1) {
            const auto value       = static_cast<__mmask64>(_mm_cvtsi128_si64(_mm512_castsi512_si128(input)));
            const __m512i source   = _mm512_movm_epi8(value);
            const __m512i unpacked = _mm512_abs_epi8(source);
            return unpacked;
        } else if constexpr (BIT_WIDTH == 2) {
            __m512i values_shift0       = input;
            __m512i values_shift2       = _mm512_srli_epi16(values_shift0, 2);
            const __m512i values_shift4 = _mm512_srli_epi16(values_shift0, 4);
            const __m512i values_shift6 = _mm512_srli_epi16(values_shift0, 6);

            __m512i interleave_tmp = _mm512_unpacklo_epi8(values_shift0, values_shift2);
            values_shift0          = _mm512_unpackhi_epi8(values_shift0, values_shift2);
            values_shift0          = _mm512_shuffle_i64x2(interleave_tmp, values_shift0, 0b00000000);

            interleave_tmp = _mm512_unpacklo_epi8(values_shift4, values_shift6);
            values_shift2  = _mm512_unpackhi_epi8(values_shift4, values_shift6);
            values_shift2  = _mm512_shuffle_i64x2(interleave_tmp, values_shift2, 0b00000000);

            interleave_tmp = _mm512_unpacklo_epi16(values_shift0, values_shift2);
            values_shift0  = _mm512_unpackhi_epi16(values_shift0, values_shift2);
            values_shift0  = _mm512_shuffle_i64x2(interleave_tmp, values_shift0, 0x88);
            values_shift0  = _mm512_shuffle_i64x2(values_shift0, values_shift0, 0xD8);

            values_shift0 = _mm512_and_si512(values_shift0, _mm512_set1_epi16(0x0303));

            return values_shift0;
        } else if constexpr (BIT_WIDTH == 4) {
            __m512i values_shift0       = input;
            const __m512i values_shift4 = _mm512_srli_epi16(values_shift0, 4);

            __m512i interleave_tmp = _mm512_unpacklo_epi8(values_shift0, values_shift4);
            values_shift0          = _mm512_unpackhi_epi8(values_shift0, values_shift4);
            values_shift0          = _mm512_shuffle_i64x2(interleave_tmp, values_shift0, 0x44);
            values_shift0          = _mm512_shuffle_i64x2(values_shift0, values_shift0, 0xD8);

            values_shift0 = _mm512_and_si512(values_shift0, _mm512_set1_epi16(0x0F0F));

            return values_shift0;
        } else {
            using tables = unpack_tables_avx512_8<BIT_WIDTH, __m512i>;

            const __m512i permuted1 = _mm512_permutexvar_epi8(tables::get_permute1(), input);
            const __m512i permuted2 = _mm512_permutexvar_epi8(tables::get_permute2(), input);

            const __m512i shifted1 = _mm512_srlv_epi8(permuted1, tables::get_shift1());
            const __m512i shifted2 = _mm512_sllv_epi8(permuted2, tables::get_shift2());

            const __mmask64 spill_mask = _mm512_cmpneq_epi8_mask(tables::get_shift2(), _mm512_setzero_si512());
            __m512i combined           = _mm512_or_si512(shifted1, _mm512_maskz_mov_epi8(spill_mask, shifted2));

            constexpr uint32_t shift = 8 - BIT_WIDTH;
            combined                 = _mm512_slli_epi8(combined, shift);
            if (SIGN_VALUES) {
                combined = _mm512_srai_epi8(combined, shift);
            } else {
                combined = _mm512_srli_epi8(combined, shift);
            }

            return combined;
        }
    }
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
[[gnu::always_inline]] inline __m512i mm512_unpack_epi16_avx512vbmi_9to16(const __m512i& input) {
    if constexpr (BIT_WIDTH == 16) {
        return input;
    } else {
        using tables = unpack_tables_avx512_16<BIT_WIDTH, __m512i>;

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
