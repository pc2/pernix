#ifndef PERNIX_BITPACKING_PACKING_AVX512_H
#define PERNIX_BITPACKING_PACKING_AVX512_H

#include <pernix/helper.h>

#ifdef PERNIX_AVX512_VBMI_ENABLED

#include <immintrin.h>
#include <pernix/bitpacking/packing_tables.h>

#include <cstdint>

namespace pernix::bitpacking {
namespace internal {
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
__always_inline auto mm256_pack_epi32_avx512vbmi_9to15(const __m256i& input) -> __m128i {
    using tables = internal::pack_tables_avx512<BIT_WIDTH, __m128i>;

    constexpr uint16_t bit_mask = (1u << BIT_WIDTH) - 1u;

    const __m128i packed = _mm256_cvtepi32_epi16(input);
    const __m128i maskv  = _mm_set1_epi16(static_cast<int16_t>(bit_mask));
    const __m128i masked = _mm_and_si128(packed, maskv);

    const __m128i p1 = tables::get_permute1();
    const __m128i p2 = tables::get_permute2();
    const __m128i s1 = tables::get_shift1();
    const __m128i s2 = tables::get_shift2();

    if constexpr (BIT_WIDTH == 12 || BIT_WIDTH == 14 || BIT_WIDTH == 15) {
        const __m128i permuted1 = _mm_permutexvar_epi16(p1, masked);
        const __m128i permuted2 = _mm_permutexvar_epi16(p2, masked);

        const __m128i shifted1 = _mm_sllv_epi16(permuted1, s1);
        const __m128i shifted2 = _mm_srlv_epi16(permuted2, s2);

        return _mm_or_si128(shifted1, shifted2);
    } else {
        const __m128i p3 = tables::get_permute3();
        const __m128i s3 = tables::get_shift3();

        const auto [mask1, mask2, mask3] = tables::get_permute_masks();

        const __m128i permuted1 = _mm_maskz_permutexvar_epi16(mask1, p1, masked);
        const __m128i permuted2 = _mm_maskz_permutexvar_epi16(mask2, p2, masked);
        const __m128i permuted3 = _mm_maskz_permutexvar_epi16(mask3, p3, masked);

        const __m128i shifted1 = _mm_sllv_epi16(permuted1, s1);
        const __m128i shifted2 = _mm_sllv_epi16(permuted2, s2);
        const __m128i shifted3 = _mm_srlv_epi16(permuted3, s3);

        return _mm_or_si128(_mm_or_si128(shifted1, shifted2), shifted3);
    }
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
__always_inline auto mm_pack_epi32_avx512vbmi_9to15(const __m128i& input) -> __m128i {
    return mm256_pack_epi32_avx512vbmi_9to15<BIT_WIDTH>(_mm256_castsi128_si256(input));
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
__always_inline auto mm512_pack_epi32_avx512vbmi_9to15(const __m512i& input) -> __m256i {
    using tables = internal::pack_tables_avx512<BIT_WIDTH, __m256i>;

    constexpr uint16_t bit_mask = (1u << BIT_WIDTH) - 1u;

    const __m256i converted = _mm512_cvtepi32_epi16(input);
    const __m256i maskv     = _mm256_set1_epi16(static_cast<int16_t>(bit_mask));
    const __m256i masked    = _mm256_and_si256(converted, maskv);

    if constexpr (BIT_WIDTH == 12 || BIT_WIDTH == 14 || BIT_WIDTH == 15) {
        const __m256i permuted1 = _mm256_permutexvar_epi16(tables::get_permute1(), masked);
        const __m256i permuted2 = _mm256_permutexvar_epi16(tables::get_permute2(), masked);

        const __m256i shifted1 = _mm256_sllv_epi16(permuted1, tables::get_shift1());
        const __m256i shifted2 = _mm256_srlv_epi16(permuted2, tables::get_shift2());

        return _mm256_or_si256(shifted1, shifted2);
    } else {
        const auto [mask1, mask2, mask3] = tables::get_permute_masks();

        const __m256i permuted1 = _mm256_maskz_permutexvar_epi16(mask1, tables::get_permute1(), masked);
        const __m256i permuted2 = _mm256_maskz_permutexvar_epi16(mask2, tables::get_permute2(), masked);
        const __m256i permuted3 = _mm256_maskz_permutexvar_epi16(mask3, tables::get_permute3(), masked);

        const __m256i shifted1 = _mm256_sllv_epi16(permuted1, tables::get_shift1());
        const __m256i shifted2 = _mm256_sllv_epi16(permuted2, tables::get_shift2());
        const __m256i shifted3 = _mm256_srlv_epi16(permuted3, tables::get_shift3());

        return _mm256_or_si256(_mm256_or_si256(shifted1, shifted2), shifted3);
    }
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
__always_inline auto mm1024_pack_epi32_avx512vbmi_9to15(const __m512i& input) -> __m512i {
    using tables                = internal::pack_tables_avx512<BIT_WIDTH, __m512i>;
    constexpr uint16_t bit_mask = (1u << BIT_WIDTH) - 1u;

    const __m512i maskv  = _mm512_set1_epi16(static_cast<int16_t>(bit_mask));
    const __m512i masked = _mm512_and_si512(input, maskv);

    if constexpr (BIT_WIDTH == 12 || BIT_WIDTH == 14 || BIT_WIDTH == 15) {
        const __m512i permuted1 = _mm512_permutexvar_epi16(tables::get_permute1(), masked);
        const __m512i permuted2 = _mm512_permutexvar_epi16(tables::get_permute2(), masked);

        const __m512i shifted1 = _mm512_sllv_epi16(permuted1, tables::get_shift1());
        const __m512i shifted2 = _mm512_srlv_epi16(permuted2, tables::get_shift2());

        return _mm512_or_si512(shifted1, shifted2);
    } else {
        const auto [mask1, mask2, mask3] = tables::get_permute_masks();

        const __m512i permuted1 = _mm512_maskz_permutexvar_epi16(mask1, tables::get_permute1(), masked);
        const __m512i permuted2 = _mm512_maskz_permutexvar_epi16(mask2, tables::get_permute2(), masked);
        const __m512i permuted3 = _mm512_maskz_permutexvar_epi16(mask3, tables::get_permute3(), masked);

        const __m512i shifted1 = _mm512_sllv_epi16(permuted1, tables::get_shift1());
        const __m512i shifted2 = _mm512_sllv_epi16(permuted2, tables::get_shift2());
        const __m512i shifted3 = _mm512_srlv_epi16(permuted3, tables::get_shift3());

        return _mm512_or_si512(_mm512_or_si512(shifted1, shifted2), shifted3);
    }
}
}  // namespace internal

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH == 8 || BIT_WIDTH == 16)
__always_inline auto mm_pack_aligned_epi32_avx512(const __m128i& input) -> __m128i {
    if constexpr (BIT_WIDTH == 8) {
        return _mm_cvtepi32_epi8(input);
    } else {
        return _mm_cvtepi32_epi16(input);
    }
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH == 8 || BIT_WIDTH == 16)
__always_inline auto mm256_pack_aligned_epi32_avx512(const __m256i& input) -> __m128i {
    if constexpr (BIT_WIDTH == 8) {
        return _mm256_cvtepi32_epi8(input);
    } else {
        return _mm256_cvtepi32_epi16(input);
    }
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH == 8 || BIT_WIDTH == 16)
__always_inline auto mm512_pack_aligned_epi32_avx512(const __m512i& input) -> __m256i {
    if constexpr (BIT_WIDTH == 8) {
        return _mm256_castsi128_si256(_mm512_cvtepi32_epi8(input));
    } else {
        return _mm512_cvtepi32_epi16(input);
    }
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH == 8 || BIT_WIDTH == 16)
__always_inline auto mm1024_pack_aligned_epi32_avx512(const __m512i& input) -> __m512i {
    if constexpr (BIT_WIDTH == 8) {
        return _mm512_castsi256_si512(_mm512_cvtepi16_epi8(input));
    } else {
        return input;
    }
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
__always_inline auto mm_pack_epi32_avx512vbmi(const __m128i& input) -> __m128i {
    if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 15) {
        return internal::mm_pack_epi32_avx512vbmi_9to15<BIT_WIDTH>(input);
    } else {
        return mm_pack_aligned_epi32_avx512<BIT_WIDTH>(input);
    }
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
__always_inline auto mm256_pack_epi32_avx512vbmi(const __m256i& input) -> __m128i {
    if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 15) {
        return internal::mm256_pack_epi32_avx512vbmi_9to15<BIT_WIDTH>(input);
    } else {
        return mm256_pack_aligned_epi32_avx512<BIT_WIDTH>(input);
    }
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
__always_inline auto mm512_pack_epi32_avx512vbmi(const __m512i& input) -> __m256i {
    if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 15) {
        return internal::mm512_pack_epi32_avx512vbmi_9to15<BIT_WIDTH>(input);
    } else {
        return mm512_pack_aligned_epi32_avx512<BIT_WIDTH>(input);
    }
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
__always_inline auto mm1024_pack_epi32_avx512vbmi(const __m512i& input1, const __m512i& input2) -> __m512i {
    const __m256i converted1 = _mm512_cvtepi32_epi16(input1);
    const __m256i converted2 = _mm512_cvtepi32_epi16(input2);

    __m512i packed = _mm512_castsi256_si512(converted1);
    packed         = _mm512_inserti64x4(packed, converted2, 1);

    if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 15) {
        return internal::mm1024_pack_epi32_avx512vbmi_9to15<BIT_WIDTH>(packed);
    } else {
        return mm1024_pack_aligned_epi32_avx512<BIT_WIDTH>(packed);
    }
}
}  // namespace pernix::bitpacking
#endif  // PERNIX_AVX512_VBMI_ENABLED

#endif  // PERNIX_BITPACKING_PACKING_AVX512_H
