#ifndef PERNIX_AVX512VBMI_UNPACKING_H
#define PERNIX_AVX512VBMI_UNPACKING_H

#include <pernix/detail/unpack_tables.h>
#include <pernix/simd_compat.h>
#include <pernix/x86/avx512vbmi/tables.h>

namespace pernix::internal {
namespace m128 {
constexpr __mmask16 kAlternateByteMask16 = 0xAAAAULL;

__always_inline static __m128i _mm_srlv_epi8(const __m128i a, const __m128i count) {
    const __m128i mask      = _mm_set1_epi16(0x00ff);
    const __m128i low_half  = _mm_srlv_epi16(_mm_and_si128(mask, a), _mm_and_si128(mask, count));
    const __m128i high_half = _mm_srlv_epi16(a, _mm_srli_epi16(count, 8));
    return _mm_mask_blend_epi8(kAlternateByteMask16, low_half, high_half);
}

__always_inline static __m128i _mm_sllv_epi8(const __m128i a, const __m128i count) {
    const __m128i mask      = _mm_set1_epi16(static_cast<i16>(0xff00u));
    const __m128i low_half  = _mm_sllv_epi16(a, _mm_andnot_si128(mask, count));
    const __m128i high_half = _mm_sllv_epi16(_mm_and_si128(mask, a), _mm_srli_epi16(count, 8));
    return _mm_mask_blend_epi8(kAlternateByteMask16, low_half, high_half);
}

__always_inline static __m128i _mm_slli_epi8(const __m128i a, const i8 imm8) {
    return _mm_sllv_epi8(a, _mm_set1_epi8(imm8));
}

__always_inline static __m128i _mm_srli_epi8(const __m128i a, const int imm8) {
    const __m128i lo_mask = _mm_set1_epi16(0x00ff);
    const __m128i hi_mask = _mm_set1_epi16(static_cast<i16>(0xff00u));
    const __m128i shift   = _mm_cvtsi32_si128(imm8);

    const __m128i lo = _mm_srl_epi16(_mm_and_si128(a, lo_mask), shift);
    const __m128i hi = _mm_and_si128(_mm_srl_epi16(a, shift), hi_mask);

    return _mm_mask_blend_epi8(kAlternateByteMask16, lo, hi);
}

__always_inline static __m128i _mm_srai_epi8(const __m128i a, const i8 imm8) {
    const __m128i lo_mask = _mm_set1_epi16(0x00ff);
    const __m128i hi_mask = _mm_set1_epi16(static_cast<i16>(0xff00u));
    const __m128i shift   = _mm_cvtsi32_si128(imm8);

    const __m128i hi = _mm_and_si128(_mm_sra_epi16(a, shift), hi_mask);

    const __m128i lo_as_hi = _mm_slli_epi16(_mm_and_si128(a, lo_mask), 8);
    const __m128i lo       = _mm_and_si128(_mm_srli_epi16(_mm_sra_epi16(lo_as_hi, shift), 8), lo_mask);

    return _mm_mask_blend_epi8(kAlternateByteMask16, lo, hi);
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8)
__always_inline __m128i mm_unpack_epi8_avx512vbmi_1to8(const __m128i& input) {
    if constexpr (BIT_WIDTH == 8) {
        return input;
    } else {
        using tables = detail::unpack_table<i8, BIT_WIDTH, sizeof(__m128i)>;

        const __m128i permuted    = _mm_permutexvar_epi8(load_table<__m128i>(tables::primary_permute), input);
        const __m128i right_shift = load_table<__m128i>(tables::right_shift_magnitude);
        __m128i combined          = _mm_srlv_epi8(permuted, right_shift);

        if constexpr (tables::has_spill) {
            const __m128i spill_permuted = _mm_permutexvar_epi8(load_table<__m128i>(tables::spill_permute), input);
            const __m128i spill_shift    = load_table<__m128i>(tables::left_shift_for_spill);
            const __m128i spill_values   = _mm_sllv_epi8(spill_permuted, spill_shift);
            const __mmask16 spill_mask   = _mm_cmpneq_epi8_mask(spill_shift, _mm_setzero_si128());
            combined                     = _mm_or_si128(combined, _mm_maskz_mov_epi8(spill_mask, spill_values));
        }

        constexpr u32 shift = 8 - BIT_WIDTH;
        combined            = _mm_slli_epi8(combined, shift);
        if constexpr (SIGN_VALUES && BIT_WIDTH > 1) {
            combined = _mm_srai_epi8(combined, shift);
        } else {
            combined = _mm_srli_epi8(combined, shift);
        }

        return combined;
    }
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
__always_inline __m128i mm_unpack_epi16_avx512vbmi_9to16(const __m128i& input) {
    if constexpr (BIT_WIDTH == 16) {
        return input;
    } else {
        using tables = detail::unpack_table<i16, BIT_WIDTH, sizeof(__m128i)>;

        const __m128i permuted = _mm_permutexvar_epi8(load_table<__m128i>(tables::primary_permute), input);

        const __m128i right_shift = load_table<__m128i>(tables::right_shift_magnitude);
        __m128i shifted           = _mm_srlv_epi16(permuted, right_shift);

        if constexpr (tables::has_spill) {
            const __m128i permuted2 = _mm_permutexvar_epi8(load_table<__m128i>(tables::spill_permute), input);
            const __m128i shifted2  = _mm_sllv_epi16(permuted2, load_table<__m128i>(tables::left_shift_for_spill));
            shifted                 = _mm_or_si128(shifted, shifted2);
        }

        constexpr u32 shift = 16 - BIT_WIDTH;
        shifted             = _mm_slli_epi16(shifted, shift);
        if (SIGN_VALUES) {
            shifted = _mm_srai_epi16(shifted, shift);
        } else {
            shifted = _mm_srli_epi16(shifted, shift);
        }

        return shifted;
    }
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24)
__always_inline __m128i mm_unpack_epi32_avx512vbmi_17to24(const __m128i& input) {
    using tables = detail::unpack_table<i32, BIT_WIDTH, sizeof(__m128i)>;

    const __m128i permuted    = _mm_permutexvar_epi8(load_table<__m128i>(tables::primary_permute), input);
    const __m128i right_shift = load_table<__m128i>(tables::right_shift_magnitude);

    constexpr u32 shift = 32 - BIT_WIDTH;
    __m128i shifted     = _mm_srlv_epi32(permuted, right_shift);
    shifted             = _mm_slli_epi32(shifted, shift);
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

__always_inline static __m256i _mm256_srlv_epi8(const __m256i a, const __m256i count) {
    const __m256i mask      = _mm256_set1_epi16(0x00ff);
    const __m256i low_half  = _mm256_srlv_epi16(_mm256_and_si256(mask, a), _mm256_and_si256(mask, count));
    const __m256i high_half = _mm256_srlv_epi16(a, _mm256_srli_epi16(count, 8));
    return _mm256_mask_blend_epi8(kAlternateByteMask32, low_half, high_half);
}

__always_inline static __m256i _mm256_sllv_epi8(const __m256i a, const __m256i count) {
    const __m256i mask      = _mm256_set1_epi16(static_cast<i16>(0xff00u));
    const __m256i low_half  = _mm256_sllv_epi16(a, _mm256_andnot_si256(mask, count));
    const __m256i high_half = _mm256_sllv_epi16(_mm256_and_si256(mask, a), _mm256_srli_epi16(count, 8));
    return _mm256_mask_blend_epi8(kAlternateByteMask32, low_half, high_half);
}

__always_inline static __m256i _mm256_slli_epi8(const __m256i a, const i8 imm8) {
    return _mm256_sllv_epi8(a, _mm256_set1_epi8(imm8));
}

__always_inline static __m256i _mm256_srli_epi8(const __m256i a, const i8 imm8) {
    const __m256i lo_mask = _mm256_set1_epi16(0x00ff);
    const __m256i hi_mask = _mm256_set1_epi16(static_cast<i16>(0xff00u));
    const __m128i shift   = _mm_cvtsi32_si128(imm8);

    const __m256i lo = _mm256_srl_epi16(_mm256_and_si256(a, lo_mask), shift);
    const __m256i hi = _mm256_and_si256(_mm256_srl_epi16(a, shift), hi_mask);

    return _mm256_mask_blend_epi8(kAlternateByteMask32, lo, hi);
}

__always_inline static __m256i _mm256_srai_epi8(const __m256i a, const i8 imm8) {
    const __m256i lo_mask = _mm256_set1_epi16(0x00ff);
    const __m256i hi_mask = _mm256_set1_epi16(static_cast<i16>(0xff00u));
    const __m128i shift   = _mm_cvtsi32_si128(imm8);

    const __m256i hi = _mm256_and_si256(_mm256_sra_epi16(a, shift), hi_mask);

    const __m256i lo_as_hi = _mm256_slli_epi16(_mm256_and_si256(a, lo_mask), 8);
    const __m256i lo       = _mm256_and_si256(_mm256_srli_epi16(_mm256_sra_epi16(lo_as_hi, shift), 8), lo_mask);

    return _mm256_mask_blend_epi8(kAlternateByteMask32, lo, hi);
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8)
__always_inline __m256i mm256_unpack_epi8_avx512vbmi_1to8(const __m256i& input) {
    if constexpr (BIT_WIDTH == 8) {
        return input;
    } else {
        using tables = detail::unpack_table<i8, BIT_WIDTH, sizeof(__m256i)>;

        const __m256i permuted    = _mm256_permutexvar_epi8(load_table<__m256i>(tables::primary_permute), input);
        const __m256i right_shift = load_table<__m256i>(tables::right_shift_magnitude);
        __m256i combined          = _mm256_srlv_epi8(permuted, right_shift);

        if constexpr (tables::has_spill) {
            const __m256i spill_permuted = _mm256_permutexvar_epi8(load_table<__m256i>(tables::spill_permute), input);
            const __m256i spill_shift    = load_table<__m256i>(tables::left_shift_for_spill);
            const __m256i spill_values   = _mm256_sllv_epi8(spill_permuted, spill_shift);
            const __mmask32 spill_mask   = _mm256_cmpneq_epi8_mask(spill_shift, _mm256_setzero_si256());
            combined                     = _mm256_or_si256(combined, _mm256_maskz_mov_epi8(spill_mask, spill_values));
        }

        constexpr u32 shift = 8 - BIT_WIDTH;
        combined            = _mm256_slli_epi8(combined, shift);
        if constexpr (SIGN_VALUES && BIT_WIDTH > 1) {
            combined = _mm256_srai_epi8(combined, shift);
        } else {
            combined = _mm256_srli_epi8(combined, shift);
        }

        return combined;
    }
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
__always_inline __m256i mm256_unpack_epi16_avx512vbmi_9to16(const __m256i& input) {
    if constexpr (BIT_WIDTH == 16) {
        return input;
    } else {
        using tables = detail::unpack_table<i16, BIT_WIDTH, sizeof(__m256i)>;

        const __m256i permuted = _mm256_permutexvar_epi8(load_table<__m256i>(tables::primary_permute), input);

        const __m256i right_shift = load_table<__m256i>(tables::right_shift_magnitude);
        __m256i shifted           = _mm256_srlv_epi16(permuted, right_shift);

        if constexpr (tables::has_spill) {
            const __m256i permuted2 = _mm256_permutexvar_epi8(load_table<__m256i>(tables::spill_permute), input);
            const __m256i shifted2  = _mm256_sllv_epi16(permuted2, load_table<__m256i>(tables::left_shift_for_spill));
            shifted                 = _mm256_or_si256(shifted, shifted2);
        }

        constexpr u32 shift = 16 - BIT_WIDTH;
        shifted             = _mm256_slli_epi16(shifted, shift);
        if (SIGN_VALUES) {
            shifted = _mm256_srai_epi16(shifted, shift);
        } else {
            shifted = _mm256_srli_epi16(shifted, shift);
        }

        return shifted;
    }
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24)
__always_inline __m256i mm256_unpack_epi32_avx512vbmi_17to24(const __m256i& input) {
    using tables = detail::unpack_table<i32, BIT_WIDTH, sizeof(__m256i)>;

    const __m256i permuted    = _mm256_permutexvar_epi8(load_table<__m256i>(tables::primary_permute), input);
    const __m256i right_shift = load_table<__m256i>(tables::right_shift_magnitude);

    constexpr u32 shift = 32 - BIT_WIDTH;
    __m256i shifted     = _mm256_srlv_epi32(permuted, right_shift);
    shifted             = _mm256_slli_epi32(shifted, shift);
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

__always_inline static __m512i _mm512_srlv_epi8(const __m512i a, const __m512i count) {
    const __m512i mask      = _mm512_set1_epi16(0x00ff);
    const __m512i low_half  = _mm512_srlv_epi16(_mm512_and_si512(mask, a), _mm512_and_si512(mask, count));
    const __m512i high_half = _mm512_srlv_epi16(a, _mm512_srli_epi16(count, 8));
    return _mm512_mask_blend_epi8(kAlternateByteMask64, low_half, high_half);
}

__always_inline static __m512i _mm512_sllv_epi8(const __m512i a, const __m512i count) {
    const __m512i mask      = _mm512_set1_epi16(static_cast<i16>(0xff00u));
    const __m512i low_half  = _mm512_sllv_epi16(a, _mm512_andnot_si512(mask, count));
    const __m512i high_half = _mm512_sllv_epi16(_mm512_and_si512(mask, a), _mm512_srli_epi16(count, 8));
    return _mm512_mask_blend_epi8(kAlternateByteMask64, low_half, high_half);
}

__always_inline static __m512i _mm512_slli_epi8(const __m512i a, const i8 imm8) {
    return _mm512_sllv_epi8(a, _mm512_set1_epi8(imm8));
}

__always_inline static __m512i _mm512_srli_epi8(const __m512i a, const i8 imm8) {
    const __m512i lo_mask = _mm512_set1_epi16(0x00ff);
    const __m512i hi_mask = _mm512_set1_epi16(static_cast<i16>(0xff00u));
    const __m128i shift   = _mm_cvtsi32_si128(imm8);

    const __m512i lo = _mm512_srl_epi16(_mm512_and_si512(a, lo_mask), shift);
    const __m512i hi = _mm512_and_si512(_mm512_srl_epi16(a, shift), hi_mask);

    return _mm512_mask_blend_epi8(kAlternateByteMask64, lo, hi);
}

__always_inline static __m512i _mm512_srai_epi8(const __m512i a, const i8 imm8) {
    const __m512i lo_mask = _mm512_set1_epi16(0x00ff);
    const __m512i hi_mask = _mm512_set1_epi16(static_cast<i16>(0xff00u));
    const __m128i shift   = _mm_cvtsi32_si128(imm8);

    const __m512i hi = _mm512_and_si512(_mm512_sra_epi16(a, shift), hi_mask);

    const __m512i lo_as_hi = _mm512_slli_epi16(_mm512_and_si512(a, lo_mask), 8);
    const __m512i lo       = _mm512_and_si512(_mm512_srli_epi16(_mm512_sra_epi16(lo_as_hi, shift), 8), lo_mask);

    return _mm512_mask_blend_epi8(kAlternateByteMask64, lo, hi);
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8)
__always_inline __m512i mm512_unpack_epi8_avx512vbmi_1to8(const __m512i& input) {
    if constexpr (BIT_WIDTH == 8) {
        return input;
    } else {
        using tables = detail::unpack_table<i8, BIT_WIDTH, sizeof(__m512i)>;

        const __m512i permuted    = _mm512_permutexvar_epi8(load_table<__m512i>(tables::primary_permute), input);
        const __m512i right_shift = load_table<__m512i>(tables::right_shift_magnitude);
        __m512i combined          = _mm512_srlv_epi8(permuted, right_shift);

        if constexpr (tables::has_spill) {
            const __m512i spill_permuted = _mm512_permutexvar_epi8(load_table<__m512i>(tables::spill_permute), input);
            const __m512i spill_shift    = load_table<__m512i>(tables::left_shift_for_spill);
            const __m512i spill_values   = _mm512_sllv_epi8(spill_permuted, spill_shift);
            const __mmask64 spill_mask   = _mm512_cmpneq_epi8_mask(spill_shift, _mm512_setzero_si512());
            combined                     = _mm512_or_si512(combined, _mm512_maskz_mov_epi8(spill_mask, spill_values));
        }

        constexpr u32 shift = 8 - BIT_WIDTH;
        combined            = _mm512_slli_epi8(combined, shift);
        if constexpr (SIGN_VALUES && BIT_WIDTH > 1) {
            combined = _mm512_srai_epi8(combined, shift);
        } else {
            combined = _mm512_srli_epi8(combined, shift);
        }

        return combined;
    }
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
__always_inline __m512i mm512_unpack_epi16_avx512vbmi_9to16(const __m512i& input) {
    if constexpr (BIT_WIDTH == 16) {
        return input;
    } else {
        using tables = detail::unpack_table<i16, BIT_WIDTH, sizeof(__m512i)>;

        const __m512i permuted    = _mm512_permutexvar_epi8(load_table<__m512i>(tables::primary_permute), input);
        const __m512i right_shift = load_table<__m512i>(tables::right_shift_magnitude);
        __m512i shifted           = _mm512_srlv_epi16(permuted, right_shift);

        if constexpr (tables::has_spill) {
            const __m512i permuted2 = _mm512_permutexvar_epi8(load_table<__m512i>(tables::spill_permute), input);
            const __m512i shifted2  = _mm512_sllv_epi16(permuted2, load_table<__m512i>(tables::left_shift_for_spill));
            shifted                 = _mm512_or_si512(shifted, shifted2);
        }

        constexpr u32 shift = 16 - BIT_WIDTH;
        shifted             = _mm512_slli_epi16(shifted, shift);
        if (SIGN_VALUES) {
            shifted = _mm512_srai_epi16(shifted, shift);
        } else {
            shifted = _mm512_srli_epi16(shifted, shift);
        }

        return shifted;
    }
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24)
__always_inline __m512i mm512_unpack_epi32_avx512vbmi_17to24(const __m512i& input) {
    using tables = detail::unpack_table<i32, BIT_WIDTH, sizeof(__m512i)>;

    const __m512i permuted    = _mm512_permutexvar_epi8(load_table<__m512i>(tables::primary_permute), input);
    const __m512i right_shift = load_table<__m512i>(tables::right_shift_magnitude);
    __m512i shifted           = _mm512_srlv_epi32(permuted, right_shift);

    constexpr u32 shift = 32 - BIT_WIDTH;
    shifted             = _mm512_slli_epi32(shifted, shift);
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
