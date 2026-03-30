#ifndef PERNIX_AVX512VBMI_PACKING_H
#define PERNIX_AVX512VBMI_PACKING_H

#include <pernix/avx512vbmi/tables.h>

namespace pernix::internal {

/**
 * @brief Pack eight 32-bit values for bit widths 9 through 15 using VBMI.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
[[gnu::always_inline]] inline __m128i mm256_pack_epi16_avx512vbmi_9to15(const __m128i& input) {
    using tables = pack_tables_avx512_16<BIT_WIDTH, __m128i>;

    constexpr uint16_t bit_mask = (1u << BIT_WIDTH) - 1u;

    const __m128i maskv  = _mm_set1_epi16(static_cast<int16_t>(bit_mask));
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

/**
 * @brief Pack four 16-bit values for bit widths 9 through 15 using VBMI.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
[[gnu::always_inline]] inline __m128i mm_pack_epi16_avx512vbmi_9to15(const __m128i& input) {
    return mm256_pack_epi16_avx512vbmi_9to15<BIT_WIDTH>(_mm256_castsi128_si256(input));
}

/**
 * @brief Pack sixteen 32-bit values for bit widths 9 through 15 using VBMI.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
[[gnu::always_inline]] inline __m256i mm512_pack_epi16_avx512vbmi_9to15(const __m256i& input) {
    using tables = pack_tables_avx512_16<BIT_WIDTH, __m256i>;

    constexpr uint16_t bit_mask = (1u << BIT_WIDTH) - 1u;

    const __m256i maskv  = _mm256_set1_epi16(static_cast<int16_t>(bit_mask));
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

/**
 * @brief Pack eight 32-bit values for bit widths 17 through 24 using AVX-512 permute instructions.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24)
[[gnu::always_inline]] inline __m256i mm256_pack_epi32_avx512_17to24(const __m256i& input) {
    using tables = pack_tables_avx512_24<BIT_WIDTH, __m256i>;

    constexpr uint32_t bitmask = (1u << BIT_WIDTH) - 1u;
    const __m256i masked       = _mm256_and_si256(input, _mm256_set1_epi32(static_cast<int32_t>(bitmask)));

    const __m256i permuted1 = _mm256_permutexvar_epi32(tables::get_permute1(), masked);
    const __m256i permuted2 = _mm256_permutexvar_epi32(tables::get_permute2(), masked);
    const __m256i permuted3 = _mm256_permutexvar_epi32(tables::get_permute3(), masked);

    const __m256i shifted1 = _mm256_sllv_epi32(permuted1, tables::get_shift1());
    const __m256i shifted2 = _mm256_sllv_epi32(permuted2, tables::get_shift2());
    const __m256i shifted3 = _mm256_srlv_epi32(permuted3, tables::get_shift3());
    return _mm256_or_si256(_mm256_or_si256(shifted1, shifted2), shifted3);
}

/**
 * @brief Pack thirty-two 16-bit values for bit widths 9 through 15 using VBMI.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
[[gnu::always_inline]] inline __m512i mm512_pack_epi16_avx512vbmi_9to15(const __m512i& input) {
    using tables                = pack_tables_avx512_16<BIT_WIDTH, __m512i>;
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

/**
 * @brief Pack sixteen 32-bit values for bit widths 17 through 24 using AVX-512 permute instructions.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24)
[[gnu::always_inline]] inline __m512i mm512_pack_epi32_avx512_17to24(const __m512i& input) {
    using tables = pack_tables_avx512_24<BIT_WIDTH, __m512i>;

    constexpr uint32_t bitmask = (1u << BIT_WIDTH) - 1u;
    const __m512i masked       = _mm512_and_si512(input, _mm512_set1_epi32(static_cast<int32_t>(bitmask)));

    const __m512i permuted1 = _mm512_permutexvar_epi32(tables::get_permute1(), masked);
    const __m512i permuted2 = _mm512_permutexvar_epi32(tables::get_permute2(), masked);
    const __m512i permuted3 = _mm512_permutexvar_epi32(tables::get_permute3(), masked);

    const __m512i shifted1 = _mm512_sllv_epi32(permuted1, tables::get_shift1());
    const __m512i shifted2 = _mm512_sllv_epi32(permuted2, tables::get_shift2());
    const __m512i shifted3 = _mm512_srlv_epi32(permuted3, tables::get_shift3());
    return _mm512_or_si512(_mm512_or_si512(shifted1, shifted2), shifted3);
}

/**
 * @brief Pack aligned 8-bit or 16-bit values from four 32-bit lanes using AVX-512 narrowing instructions.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH == 8 || BIT_WIDTH == 16)
[[gnu::always_inline]] inline __m128i mm_pack_aligned_epi32_avx512(const __m128i& input) {
    if constexpr (BIT_WIDTH == 8) {
        return _mm_cvtepi32_epi8(input);
    } else {
        return _mm_cvtepi32_epi16(input);
    }
}

/**
 * @brief Pack aligned 8-bit or 16-bit values from eight 32-bit lanes.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH == 8 || BIT_WIDTH == 16)
[[gnu::always_inline]] inline __m128i mm256_pack_aligned_epi32_avx512(const __m256i& input) {
    if constexpr (BIT_WIDTH == 8) {
        return _mm256_cvtepi32_epi8(input);
    } else {
        return _mm256_cvtepi32_epi16(input);
    }
}

/**
 * @brief Pack aligned 8-bit or 16-bit values from sixteen 32-bit lanes.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH == 8 || BIT_WIDTH == 16)
[[gnu::always_inline]] inline __m256i mm512_pack_aligned_epi32_avx512(const __m512i& input) {
    if constexpr (BIT_WIDTH == 8) {
        return _mm256_castsi128_si256(_mm512_cvtepi32_epi8(input));
    } else {
        return _mm512_cvtepi32_epi16(input);
    }
}

/**
 * @brief Pack aligned 8-bit or 16-bit values from thirty-two 16-bit lanes.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH == 8 || BIT_WIDTH == 16)
[[gnu::always_inline]] inline __m512i mm512_pack_aligned_epi16_avx512(const __m512i& input) {
    if constexpr (BIT_WIDTH == 8) {
        return _mm512_castsi256_si512(_mm512_cvtepi16_epi8(input));
    } else {
        return input;
    }
}

/**
 * @brief Dispatch to the appropriate 128-bit VBMI packer.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
[[gnu::always_inline]] inline __m128i mm_pack_epi32_avx512vbmi(const __m128i& input) {
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 7) {
        // TODO: Implement complete bitpacking path

        return _mm_setzero_si128();
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 15) {
        const __m128i converted = _mm_cvtepi32_epi16(input);

        return mm_pack_epi16_avx512vbmi_9to15<BIT_WIDTH>(converted);
    } else if constexpr (BIT_WIDTH >= 17 && BIT_WIDTH <= 24) {
        const __m256i widened = _mm256_castsi128_si256(input);
        return _mm256_castsi256_si128(mm256_pack_epi32_avx512_17to24<BIT_WIDTH>(widened));
    } else {
        return mm_pack_aligned_epi32_avx512<BIT_WIDTH>(input);
    }
}

/**
 * @brief Dispatch to the appropriate 256-bit VBMI packer.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
[[gnu::always_inline]] inline __m256i mm256_pack_epi32_avx512vbmi(const __m256i& input) {
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 7) {
        if constexpr (BIT_WIDTH <= 3) {
            // TODO: Implement complete bitpacking path

            return _mm256_setzero_si256();
        } else if constexpr (BIT_WIDTH == 4) {
            // TODO: Implement complete bitpacking path

            return _mm256_setzero_si256();
        } else {
            // TODO: Implement complete bitpacking path

            return _mm256_setzero_si256();
        }
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 15) {
        const __m128i converted = _mm256_cvtepi32_epi16(input);

        return _mm256_castsi128_si256(mm256_pack_epi16_avx512vbmi_9to15<BIT_WIDTH>(converted));
    } else if constexpr (BIT_WIDTH >= 17 && BIT_WIDTH <= 24) {
        return mm256_pack_epi32_avx512_17to24<BIT_WIDTH>(input);
    } else {
        return _mm256_castsi128_si256(mm256_pack_aligned_epi32_avx512<BIT_WIDTH>(input));
    }
}

/**
 * @brief Dispatch to the appropriate 512-bit VBMI packer.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
[[gnu::always_inline]] inline __m512i mm512_pack_epi32_avx512vbmi(const __m512i& input) {
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 7) {
        // TODO: Implement complete bitpacking path

        return _mm512_setzero_si512();
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 15) {
        const __m256i converted = _mm512_cvtepi32_epi16(input);

        return _mm512_castsi256_si512(mm512_pack_epi16_avx512vbmi_9to15<BIT_WIDTH>(converted));
    } else if constexpr (BIT_WIDTH >= 17 && BIT_WIDTH <= 24) {
        return mm512_pack_epi32_avx512_17to24<BIT_WIDTH>(input);
    } else {
        return _mm512_castsi256_si512(mm512_pack_aligned_epi32_avx512<BIT_WIDTH>(input));
    }
}

/**
 * @brief Dispatch to the appropriate dual-register VBMI packer.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 16)
[[gnu::always_inline]] inline __m512i mm1024_pack_epi32_avx512vbmi(const __m512i& input1, const __m512i& input2) {
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 7) {
        const __m128i converted1 = _mm512_cvtepi32_epi8(input1);
        const __m128i converted2 = _mm512_cvtepi32_epi8(input2);
        __m256i packed           = _mm256_castsi128_si256(converted1);
        packed                   = _mm256_inserti128_si256(packed, converted2, 1);

        return mm256_pack_epi8_avx512vbmi<BIT_WIDTH>(packed);
    } else {
        const __m256i converted1 = _mm512_cvtepi32_epi16(input1);
        const __m256i converted2 = _mm512_cvtepi32_epi16(input2);
        __m512i packed           = _mm512_castsi256_si512(converted1);
        packed                   = _mm512_inserti64x4(packed, converted2, 1);

        if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 15) {
            return mm512_pack_epi16_avx512vbmi_9to15<BIT_WIDTH>(packed);
        } else {
            return mm512_pack_aligned_epi16_avx512<BIT_WIDTH>(packed);
        }
    }
}

}  // namespace pernix::internal

#endif  // PERNIX_AVX512VBMI_PACKING_H
