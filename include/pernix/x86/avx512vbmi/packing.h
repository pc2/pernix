#ifndef PERNIX_AVX512VBMI_PACKING_H
#define PERNIX_AVX512VBMI_PACKING_H

#include <pernix/x86/avx512vbmi/tables.h>
#include <pernix/simd_compat.h>

namespace pernix::internal {
namespace m128 {
/**
 * @brief Pack 8 16-bit values for bit widths 9 through 16 using VBMI.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
[[gnu::always_inline]] inline __m128i mm_pack_epi16_avx512vbmi_9to16(const __m128i& input) {
    if constexpr (BIT_WIDTH == 16) {
        return input;
    } else {
        using tables         = pack_tables_avx512_16<BIT_WIDTH, __m128i>;
        const __m128i maskv  = _mm_set1_epi16(static_cast<int16_t>((1u << BIT_WIDTH) - 1u));
        const __m128i masked = _mm_and_si128(input, maskv);

        if constexpr (BIT_WIDTH == 12 || BIT_WIDTH == 14 || BIT_WIDTH == 15) {
            const __m128i permuted1 = _mm_permutexvar_epi16(tables::get_permute1(), masked);
            const __m128i permuted2 = _mm_permutexvar_epi16(tables::get_permute2(), masked);

            const __m128i shifted1 = _mm_sllv_epi16(permuted1, tables::get_shift1());
            const __m128i shifted2 = _mm_srlv_epi16(permuted2, tables::get_shift2());

            return _mm_or_si128(shifted1, shifted2);
        } else {
            const auto [mask1, mask2, mask3] = tables::get_permute_masks();

            const __m128i permuted1 = _mm_maskz_permutexvar_epi16(mask1, tables::get_permute1(), masked);
            const __m128i permuted2 = _mm_maskz_permutexvar_epi16(mask2, tables::get_permute2(), masked);
            const __m128i permuted3 = _mm_maskz_permutexvar_epi16(mask3, tables::get_permute3(), masked);

            const __m128i shifted1 = _mm_sllv_epi16(permuted1, tables::get_shift1());
            const __m128i shifted2 = _mm_sllv_epi16(permuted2, tables::get_shift2());
            const __m128i shifted3 = _mm_srlv_epi16(permuted3, tables::get_shift3());

            return _mm_or_si128(_mm_or_si128(shifted1, shifted2), shifted3);
        }
    }
}

/**
 * @brief Pack 16 8-bit values for bit widths 1 through 8 using VBMI.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8)
[[gnu::always_inline]] inline __m128i mm_pack_epi8_avx512vbmi_1to8(const __m128i& input) {
    if constexpr (BIT_WIDTH == 8) {
        return input;
    } else {
        const __m128i maskv  = _mm_set1_epi8(static_cast<int8_t>((1u << BIT_WIDTH) - 1u));
        const __m128i masked = _mm_and_si128(input, maskv);

        if constexpr (BIT_WIDTH == 1) {
            return _mm_set1_epi16(static_cast<int16_t>(_mm_cmpgt_epi8_mask(masked, _mm_setzero_si128())));
        } else if constexpr (BIT_WIDTH == 2) {
            const __m128i shifted  = _mm_srli_epi16(masked, 6);
            const __m128i combined = _mm_or_si128(masked, shifted);

            const __m128i shifted2  = _mm_srli_epi32(combined, 12);
            const __m128i combined2 = _mm_or_si128(shifted2, combined);

            return _mm_cvtepi32_epi8(combined2);
        } else if constexpr (BIT_WIDTH == 3) {
            const __m128i even = _mm_and_si128(masked, _mm_set1_epi16(0x00FF));
            const __m128i odd  = _mm_and_si128(masked, _mm_set1_epi16(0xFF00));

            const __m128i pair6    = _mm_or_si128(even, _mm_srli_epi16(odd, 5));
            const __m128i packed12 = _mm_or_si128(pair6, _mm_srli_epi32(pair6, 10));

            return m128::mm_pack_epi16_avx512vbmi_9to16<12>(_mm_cvtepi32_epi16(packed12));
        } else if constexpr (BIT_WIDTH == 4) {
            const __m128i shifted  = _mm_srli_epi16(masked, 4);
            const __m128i combined = _mm_or_si128(masked, shifted);

            return _mm_cvtepi16_epi8(combined);
        } else {
            const __m128i even = _mm_and_si128(masked, _mm_set1_epi16(0x00FF));
            const __m128i odd  = _mm_and_si128(masked, _mm_set1_epi16(0xFF00));

            const __m128i shifted = _mm_or_si128(even, _mm_srli_epi16(odd, 8 - BIT_WIDTH));
            return mm_pack_epi16_avx512vbmi_9to16<2 * BIT_WIDTH>(shifted);
        }
    }
}

/**
 * @brief Pack 4 32-bit values for bit widths 17 through 24 using VBMI.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24)
[[gnu::always_inline]] inline __m128i mm_pack_epi32_avx512vbmi_17to24(const __m128i& input) {
    using tables = pack_tables_avx512_24<BIT_WIDTH, __m128i>;

    const __m128i maskv  = _mm_set1_epi32(static_cast<int32_t>((1u << BIT_WIDTH) - 1u));
    const __m128i masked = _mm_and_si128(input, maskv);

    const __m128 permuted1 = _mm_permutevar_ps(_mm_castsi128_ps(masked), tables::get_permute1());
    const __m128 permuted2 = _mm_permutevar_ps(_mm_castsi128_ps(masked), tables::get_permute2());
    const __m128 permuted3 = _mm_permutevar_ps(_mm_castsi128_ps(masked), tables::get_permute3());

    const __m128i shifted1 = _mm_sllv_epi32(_mm_castps_si128(permuted1), tables::get_shift1());
    const __m128i shifted2 = _mm_sllv_epi32(_mm_castps_si128(permuted2), tables::get_shift2());
    const __m128i shifted3 = _mm_srlv_epi32(_mm_castps_si128(permuted3), tables::get_shift3());

    return _mm_or_si128(_mm_or_si128(shifted1, shifted2), shifted3);
}
} // namespace m128

namespace m256 {
/**
 * @brief Pack 16 16-bit values for bit widths 9 through 16 using VBMI.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
[[gnu::always_inline]] inline __m256i mm256_pack_epi16_avx512vbmi_9to16(const __m256i& input) {
    if constexpr (BIT_WIDTH == 16) {
        return input;
    } else {
        using tables         = pack_tables_avx512_16<BIT_WIDTH, __m256i>;
        const __m256i maskv  = _mm256_set1_epi16(static_cast<int16_t>((1u << BIT_WIDTH) - 1u));
        const __m256i masked = _mm256_and_si256(input, maskv);

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
}

/**
 * @brief Pack 32 8-bit values for bit widths 1 through 8 using VBMI.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8)
[[gnu::always_inline]] inline __m256i mm256_pack_epi8_avx512vbmi_1to8(const __m256i& input) {
    if constexpr (BIT_WIDTH == 8) {
        return input;
    } else {
        const __m256i maskv  = _mm256_set1_epi8(static_cast<int8_t>((1u << BIT_WIDTH) - 1u));
        const __m256i masked = _mm256_and_si256(input, maskv);

        if constexpr (BIT_WIDTH == 1) {
            return _mm256_set1_epi32(static_cast<int32_t>(_mm256_cmpgt_epi8_mask(masked, _mm256_setzero_si256())));
        } else if constexpr (BIT_WIDTH == 2) {
            const __m256i shifted  = _mm256_srli_epi16(masked, 6);
            const __m256i combined = _mm256_or_si256(masked, shifted);

            const __m256i shifted2  = _mm256_srli_epi32(combined, 12);
            const __m256i combined2 = _mm256_or_si256(shifted2, combined);

            return _mm256_castsi128_si256(_mm256_cvtepi32_epi8(combined2));
        } else if constexpr (BIT_WIDTH == 3) {
            const __m256i even = _mm256_and_si256(masked, _mm256_set1_epi16(0x00FF));
            const __m256i odd  = _mm256_and_si256(masked, _mm256_set1_epi16(0xFF00));

            const __m256i pair6    = _mm256_or_si256(even, _mm256_srli_epi16(odd, 5));
            const __m256i packed12 = _mm256_or_si256(pair6, _mm256_srli_epi32(pair6, 10));

            return m256::mm256_pack_epi16_avx512vbmi_9to16<12>(_mm256_castsi128_si256(_mm256_cvtepi32_epi16(packed12)));
        } else if constexpr (BIT_WIDTH == 4) {
            const __m256i shifted  = _mm256_srli_epi16(masked, 4);
            const __m256i combined = _mm256_or_si256(masked, shifted);

            return _mm256_castsi128_si256(_mm256_cvtepi16_epi8(combined));
        } else {
            const __m256i even = _mm256_and_si256(masked, _mm256_set1_epi16(0x00FF));
            const __m256i odd  = _mm256_and_si256(masked, _mm256_set1_epi16(0xFF00));

            const __m256i shifted = _mm256_or_si256(even, _mm256_srli_epi16(odd, 8 - BIT_WIDTH));
            return mm256_pack_epi16_avx512vbmi_9to16<2 * BIT_WIDTH>(shifted);
        }
    }
}

/**
 * @brief Pack 8 32-bit values for bit widths 17 through 24 using VBMI.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24)
[[gnu::always_inline]] inline __m256i mm256_pack_epi32_avx512vbmi_17to24(const __m256i& input) {
    using tables = pack_tables_avx512_24<BIT_WIDTH, __m256i>;

    const __m256i maskv  = _mm256_set1_epi32(static_cast<int32_t>((1u << BIT_WIDTH) - 1u));
    const __m256i masked = _mm256_and_si256(input, maskv);

    const __m256i permuted1 = _mm256_permutexvar_epi32(tables::get_permute1(), masked);
    const __m256i permuted2 = _mm256_permutexvar_epi32(tables::get_permute2(), masked);
    const __m256i permuted3 = _mm256_permutexvar_epi32(tables::get_permute3(), masked);

    const __m256i shifted1 = _mm256_sllv_epi32(permuted1, tables::get_shift1());
    const __m256i shifted2 = _mm256_sllv_epi32(permuted2, tables::get_shift2());
    const __m256i shifted3 = _mm256_srlv_epi32(permuted3, tables::get_shift3());

    return _mm256_or_si256(_mm256_or_si256(shifted1, shifted2), shifted3);
}
} // namespace m256

namespace m512 {
/**
 * @brief Pack 32 16-bit values for bit widths 9 through 16 using VBMI.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
[[gnu::always_inline]] inline __m512i mm512_pack_epi16_avx512vbmi_9to16(const __m512i& input) {
    if constexpr (BIT_WIDTH == 16) {
        return input;
    } else {
        using tables         = pack_tables_avx512_16<BIT_WIDTH, __m512i>;
        const __m512i maskv  = _mm512_set1_epi16(static_cast<int16_t>((1u << BIT_WIDTH) - 1u));
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
}

/**
 * @brief Pack 64 8-bit values for bit widths 1 through 8 using VBMI.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8)
[[gnu::always_inline]] inline __m512i mm512_pack_epi8_avx512vbmi_1to8(const __m512i& input) {
    if constexpr (BIT_WIDTH == 8) {
        return input;
    } else {
        const __m512i maskv  = _mm512_set1_epi8(static_cast<int8_t>((1u << BIT_WIDTH) - 1u));
        const __m512i masked = _mm512_and_si512(input, maskv);

        if constexpr (BIT_WIDTH == 1) {
            return _mm512_set1_epi64(static_cast<int64_t>(_mm512_cmpgt_epi8_mask(masked, _mm512_setzero_si512())));
        } else if constexpr (BIT_WIDTH == 2) {
            const __m512i shifted  = _mm512_srli_epi16(masked, 6);
            const __m512i combined = _mm512_or_si512(masked, shifted);

            const __m512i shifted2  = _mm512_srli_epi32(combined, 12);
            const __m512i combined2 = _mm512_or_si512(shifted2, combined);

            return _mm512_castsi128_si512(_mm512_cvtepi32_epi8(combined2));
        } else if constexpr (BIT_WIDTH == 3) {
            const __m512i even = _mm512_and_si512(masked, _mm512_set1_epi16(0x00FF));
            const __m512i odd  = _mm512_and_si512(masked, _mm512_set1_epi16(0xFF00));

            const __m512i pair6    = _mm512_or_si512(even, _mm512_srli_epi16(odd, 5));
            const __m512i packed12 = _mm512_or_si512(pair6, _mm512_srli_epi32(pair6, 10));

            return _mm512_castsi256_si512(m256::mm256_pack_epi16_avx512vbmi_9to16<12>(_mm512_cvtepi32_epi16(packed12)));
        } else if constexpr (BIT_WIDTH == 4) {
            const __m512i shifted  = _mm512_srli_epi16(masked, 4);
            const __m512i combined = _mm512_or_si512(masked, shifted);

            return _mm512_castsi256_si512(_mm512_cvtepi16_epi8(combined));
        } else {
            const __m512i even = _mm512_and_si512(masked, _mm512_set1_epi16(0x00FF));
            const __m512i odd  = _mm512_and_si512(masked, _mm512_set1_epi16(0xFF00));

            const __m512i shifted = _mm512_or_si512(even, _mm512_srli_epi16(odd, 8 - BIT_WIDTH));
            return mm512_pack_epi16_avx512vbmi_9to16<2 * BIT_WIDTH>(shifted);
        }
    }
}

/**
 * @brief Pack 16 32-bit values for bit widths 17 through 24 using VBMI.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24)
[[gnu::always_inline]] inline __m512i mm512_pack_epi32_avx512vbmi_17to24(const __m512i& input) {
    using tables = pack_tables_avx512_24<BIT_WIDTH, __m512i>;

    const __m512i maskv  = _mm512_set1_epi32(static_cast<int32_t>((1u << BIT_WIDTH) - 1u));
    const __m512i masked = _mm512_and_si512(input, maskv);

    const __m512i permuted1 = _mm512_permutexvar_epi32(tables::get_permute1(), masked);
    const __m512i permuted2 = _mm512_permutexvar_epi32(tables::get_permute2(), masked);
    const __m512i permuted3 = _mm512_permutexvar_epi32(tables::get_permute3(), masked);

    const __m512i shifted1 = _mm512_sllv_epi32(permuted1, tables::get_shift1());
    const __m512i shifted2 = _mm512_sllv_epi32(permuted2, tables::get_shift2());
    const __m512i shifted3 = _mm512_srlv_epi32(permuted3, tables::get_shift3());

    return _mm512_or_si512(_mm512_or_si512(shifted1, shifted2), shifted3);
}
} // namespace m512
} // namespace pernix::internal

#endif  // PERNIX_AVX512VBMI_PACKING_H
