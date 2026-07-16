#ifndef PERNIX_AVX512VBMI_PACKING_H
#define PERNIX_AVX512VBMI_PACKING_H

#include <pernix/detail/bits.h>
#include <pernix/detail/pack_tables.h>
#include <pernix/simd_compat.h>
#include <pernix/x86/avx512vbmi/tables.h>

#include <array>
#include <utility>

namespace pernix::internal {
template <typename Tables, typename Vec, typename PackContributor, typename Combine, usize... Indices>
__always_inline Vec combine_primary_contributors(const PackContributor& pack_contributor, const Combine& combine,
                                                 std::index_sequence<Indices...>) {
    static_assert(Tables::contributor_count > 0);
    static_assert(sizeof...(Indices) + 1 == Tables::contributor_count);

    Vec result = pack_contributor(Tables::primary_permutes[0], Tables::primary_shifts[0], Tables::primary_masks[0]);
    ((result = combine(result, pack_contributor(Tables::primary_permutes[Indices + 1], Tables::primary_shifts[Indices + 1],
                                                Tables::primary_masks[Indices + 1]))),
     ...);
    return result;
}

namespace m128 {
constexpr __mmask16 kPackAlternateByteMask16 = 0xAAAAULL;

__always_inline __m128i mm_pack_sllv_epi8(const __m128i input, const __m128i count) {
    const __m128i high_mask = _mm_set1_epi16(static_cast<i16>(0xff00u));
    const __m128i low       = _mm_sllv_epi16(input, _mm_andnot_si128(high_mask, count));
    const __m128i high      = _mm_sllv_epi16(_mm_and_si128(high_mask, input), _mm_srli_epi16(count, 8));
    return _mm_mask_blend_epi8(kPackAlternateByteMask16, low, high);
}

__always_inline __m128i mm_pack_srlv_epi8(const __m128i input, const __m128i count) {
    const __m128i low_mask = _mm_set1_epi16(0x00ff);
    const __m128i low      = _mm_srlv_epi16(_mm_and_si128(low_mask, input), _mm_and_si128(low_mask, count));
    const __m128i high     = _mm_srlv_epi16(input, _mm_srli_epi16(count, 8));
    return _mm_mask_blend_epi8(kPackAlternateByteMask16, low, high);
}

template <typename LaneType, u8 BIT_WIDTH>
__always_inline __m128i mm_pack_table_avx512vbmi(const __m128i values) {
    using tables                = detail::pack_table<LaneType, BIT_WIDTH, sizeof(__m128i)>;
    using contributor_table     = std::array<LaneType, tables::element_count>;
    const auto pack_contributor = [values](const contributor_table& permute_indices, const contributor_table& shift_counts,
                                           const u64 active_mask) {
        if constexpr (sizeof(LaneType) == sizeof(i8)) {
            const __m128i value =
                _mm_maskz_permutexvar_epi8(static_cast<__mmask16>(active_mask), load_table<__m128i>(permute_indices), values);
            return mm_pack_sllv_epi8(value, load_table<__m128i>(shift_counts));
        } else if constexpr (sizeof(LaneType) == sizeof(i16)) {
            const __m128i value =
                _mm_maskz_permutexvar_epi16(static_cast<__mmask8>(active_mask), load_table<__m128i>(permute_indices), values);
            return _mm_sllv_epi16(value, load_table<__m128i>(shift_counts));
        } else {
            const __m128i value =
                _mm_maskz_permutexvar_epi32(static_cast<__mmask8>(active_mask), load_table<__m128i>(permute_indices), values);
            return _mm_sllv_epi32(value, load_table<__m128i>(shift_counts));
        }
    };

    const auto combine = [](__m128i left, __m128i right) { return _mm_or_si128(left, right); };
    __m128i result =
        combine_primary_contributors<tables, __m128i>(pack_contributor, combine, std::make_index_sequence<tables::contributor_count - 1>{});

    if constexpr (tables::spill_mask != 0) {
        if constexpr (sizeof(LaneType) == sizeof(i8)) {
            const __m128i permuted =
                _mm_maskz_permutexvar_epi8(static_cast<__mmask16>(tables::spill_mask), load_table<__m128i>(tables::spill_permute), values);
            result = _mm_or_si128(result, mm_pack_srlv_epi8(permuted, load_table<__m128i>(tables::spill_shift)));
        } else if constexpr (sizeof(LaneType) == sizeof(i16)) {
            const __m128i permuted =
                _mm_maskz_permutexvar_epi16(static_cast<__mmask8>(tables::spill_mask), load_table<__m128i>(tables::spill_permute), values);
            result = _mm_or_si128(result, _mm_srlv_epi16(permuted, load_table<__m128i>(tables::spill_shift)));
        } else {
            const __m128i permuted =
                _mm_maskz_permutexvar_epi32(static_cast<__mmask8>(tables::spill_mask), load_table<__m128i>(tables::spill_permute), values);
            result = _mm_or_si128(result, _mm_srlv_epi32(permuted, load_table<__m128i>(tables::spill_shift)));
        }
    }
    return result;
}

/**
 * @brief Pack 8 16-bit values for bit widths 9 through 16 using VBMI.
 */
template <u8 BIT_WIDTH>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
__always_inline __m128i mm_pack_epi16_avx512vbmi_9to16(const __m128i input) {
    if constexpr (BIT_WIDTH == 16) {
        return input;
    }

    const __m128i maskv         = _mm_set1_epi16(static_cast<i16>(detail::low_bit_mask<BIT_WIDTH>()));
    const __m128i masked_values = _mm_and_si128(input, maskv);
    return mm_pack_table_avx512vbmi<i16, BIT_WIDTH>(masked_values);
}

/**
 * @brief Pack 16 8-bit values for bit widths 1 through 8 using VBMI.
 */
template <u8 BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8)
__always_inline __m128i mm_pack_epi8_avx512vbmi_1to8(const __m128i input) {
    if constexpr (BIT_WIDTH == 8) {
        return input;
    }

    const __m128i maskv         = _mm_set1_epi8(static_cast<i8>(detail::low_bit_mask<BIT_WIDTH>()));
    const __m128i masked_values = _mm_and_si128(input, maskv);
    return mm_pack_table_avx512vbmi<i8, BIT_WIDTH>(masked_values);
}

/**
 * @brief Pack 4 32-bit values for bit widths 17 through 24 using VBMI.
 */
template <u8 BIT_WIDTH>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24)
__always_inline __m128i mm_pack_epi32_avx512vbmi_17to24(const __m128i input) {
    const __m128i maskv         = _mm_set1_epi32(static_cast<i32>(detail::low_bit_mask<BIT_WIDTH>()));
    const __m128i masked_values = _mm_and_si128(input, maskv);
    return mm_pack_table_avx512vbmi<i32, BIT_WIDTH>(masked_values);
}
}  // namespace m128

namespace m256 {
constexpr __mmask32 kPackAlternateByteMask32 = 0xAAAAAAAAULL;

__always_inline __m256i mm256_pack_sllv_epi8(const __m256i input, const __m256i count) {
    const __m256i high_mask = _mm256_set1_epi16(static_cast<i16>(0xff00u));
    const __m256i low       = _mm256_sllv_epi16(input, _mm256_andnot_si256(high_mask, count));
    const __m256i high      = _mm256_sllv_epi16(_mm256_and_si256(high_mask, input), _mm256_srli_epi16(count, 8));
    return _mm256_mask_blend_epi8(kPackAlternateByteMask32, low, high);
}

__always_inline __m256i mm256_pack_srlv_epi8(const __m256i input, const __m256i count) {
    const __m256i low_mask = _mm256_set1_epi16(0x00ff);
    const __m256i low      = _mm256_srlv_epi16(_mm256_and_si256(low_mask, input), _mm256_and_si256(low_mask, count));
    const __m256i high     = _mm256_srlv_epi16(input, _mm256_srli_epi16(count, 8));
    return _mm256_mask_blend_epi8(kPackAlternateByteMask32, low, high);
}

template <typename LaneType, u8 BIT_WIDTH>
__always_inline __m256i mm256_pack_table_avx512vbmi(const __m256i values) {
    using tables                = detail::pack_table<LaneType, BIT_WIDTH, sizeof(__m256i)>;
    using contributor_table     = std::array<LaneType, tables::element_count>;
    const auto pack_contributor = [values](const contributor_table& permute_indices, const contributor_table& shift_counts,
                                           const u64 active_mask) {
        if constexpr (sizeof(LaneType) == sizeof(i8)) {
            const __m256i value =
                _mm256_maskz_permutexvar_epi8(static_cast<__mmask32>(active_mask), load_table<__m256i>(permute_indices), values);
            return mm256_pack_sllv_epi8(value, load_table<__m256i>(shift_counts));
        } else if constexpr (sizeof(LaneType) == sizeof(i16)) {
            const __m256i value =
                _mm256_maskz_permutexvar_epi16(static_cast<__mmask16>(active_mask), load_table<__m256i>(permute_indices), values);
            return _mm256_sllv_epi16(value, load_table<__m256i>(shift_counts));
        } else {
            const __m256i value =
                _mm256_maskz_permutexvar_epi32(static_cast<__mmask8>(active_mask), load_table<__m256i>(permute_indices), values);
            return _mm256_sllv_epi32(value, load_table<__m256i>(shift_counts));
        }
    };

    const auto combine = [](__m256i left, __m256i right) { return _mm256_or_si256(left, right); };
    __m256i result =
        combine_primary_contributors<tables, __m256i>(pack_contributor, combine, std::make_index_sequence<tables::contributor_count - 1>{});

    if constexpr (tables::spill_mask != 0) {
        if constexpr (sizeof(LaneType) == sizeof(i8)) {
            const __m256i permuted = _mm256_maskz_permutexvar_epi8(static_cast<__mmask32>(tables::spill_mask),
                                                                   load_table<__m256i>(tables::spill_permute), values);
            result                 = _mm256_or_si256(result, mm256_pack_srlv_epi8(permuted, load_table<__m256i>(tables::spill_shift)));
        } else if constexpr (sizeof(LaneType) == sizeof(i16)) {
            const __m256i permuted = _mm256_maskz_permutexvar_epi16(static_cast<__mmask16>(tables::spill_mask),
                                                                    load_table<__m256i>(tables::spill_permute), values);
            result                 = _mm256_or_si256(result, _mm256_srlv_epi16(permuted, load_table<__m256i>(tables::spill_shift)));
        } else {
            const __m256i permuted = _mm256_maskz_permutexvar_epi32(static_cast<__mmask8>(tables::spill_mask),
                                                                    load_table<__m256i>(tables::spill_permute), values);
            result                 = _mm256_or_si256(result, _mm256_srlv_epi32(permuted, load_table<__m256i>(tables::spill_shift)));
        }
    }
    return result;
}

/**
 * @brief Pack 16 16-bit values for bit widths 9 through 16 using VBMI.
 */
template <u8 BIT_WIDTH>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
__always_inline __m256i mm256_pack_epi16_avx512vbmi_9to16(const __m256i input) {
    if constexpr (BIT_WIDTH == 16) {
        return input;
    }

    const __m256i maskv         = _mm256_set1_epi16(static_cast<i16>(detail::low_bit_mask<BIT_WIDTH>()));
    const __m256i masked_values = _mm256_and_si256(input, maskv);
    return mm256_pack_table_avx512vbmi<i16, BIT_WIDTH>(masked_values);
}

/**
 * @brief Pack 32 8-bit values for bit widths 1 through 8 using VBMI.
 */
template <u8 BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8)
__always_inline __m256i mm256_pack_epi8_avx512vbmi_1to8(const __m256i input) {
    if constexpr (BIT_WIDTH == 8) {
        return input;
    }

    const __m256i maskv         = _mm256_set1_epi8(static_cast<i8>(detail::low_bit_mask<BIT_WIDTH>()));
    const __m256i masked_values = _mm256_and_si256(input, maskv);
    return mm256_pack_table_avx512vbmi<i8, BIT_WIDTH>(masked_values);
}

/**
 * @brief Pack 8 32-bit values for bit widths 17 through 24 using VBMI.
 */
template <u8 BIT_WIDTH>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24)
__always_inline __m256i mm256_pack_epi32_avx512vbmi_17to24(const __m256i input) {
    const __m256i maskv         = _mm256_set1_epi32(static_cast<i32>(detail::low_bit_mask<BIT_WIDTH>()));
    const __m256i masked_values = _mm256_and_si256(input, maskv);
    return mm256_pack_table_avx512vbmi<i32, BIT_WIDTH>(masked_values);
}
}  // namespace m256

namespace m512 {
constexpr __mmask64 kPackAlternateByteMask64 = 0xAAAAAAAAAAAAAAAAULL;

__always_inline __m512i mm512_pack_sllv_epi8(const __m512i input, const __m512i count) {
    const __m512i high_mask = _mm512_set1_epi16(static_cast<i16>(0xff00u));
    const __m512i low       = _mm512_sllv_epi16(input, _mm512_andnot_si512(high_mask, count));
    const __m512i high      = _mm512_sllv_epi16(_mm512_and_si512(high_mask, input), _mm512_srli_epi16(count, 8));
    return _mm512_mask_blend_epi8(kPackAlternateByteMask64, low, high);
}

__always_inline __m512i mm512_pack_srlv_epi8(const __m512i input, const __m512i count) {
    const __m512i low_mask = _mm512_set1_epi16(0x00ff);
    const __m512i low      = _mm512_srlv_epi16(_mm512_and_si512(low_mask, input), _mm512_and_si512(low_mask, count));
    const __m512i high     = _mm512_srlv_epi16(input, _mm512_srli_epi16(count, 8));
    return _mm512_mask_blend_epi8(kPackAlternateByteMask64, low, high);
}

template <typename LaneType, u8 BIT_WIDTH>
__always_inline __m512i mm512_pack_table_avx512vbmi(const __m512i values) {
    using tables                = detail::pack_table<LaneType, BIT_WIDTH, sizeof(__m512i)>;
    using contributor_table     = std::array<LaneType, tables::element_count>;
    const auto pack_contributor = [values](const contributor_table& permute_indices, const contributor_table& shift_counts,
                                           const u64 active_mask) {
        if constexpr (sizeof(LaneType) == sizeof(i8)) {
            const __m512i value =
                _mm512_maskz_permutexvar_epi8(static_cast<__mmask64>(active_mask), load_table<__m512i>(permute_indices), values);
            return mm512_pack_sllv_epi8(value, load_table<__m512i>(shift_counts));
        } else if constexpr (sizeof(LaneType) == sizeof(i16)) {
            const __m512i value =
                _mm512_maskz_permutexvar_epi16(static_cast<__mmask32>(active_mask), load_table<__m512i>(permute_indices), values);
            return _mm512_sllv_epi16(value, load_table<__m512i>(shift_counts));
        } else {
            const __m512i value =
                _mm512_maskz_permutexvar_epi32(static_cast<__mmask16>(active_mask), load_table<__m512i>(permute_indices), values);
            return _mm512_sllv_epi32(value, load_table<__m512i>(shift_counts));
        }
    };

    const auto combine = [](__m512i left, __m512i right) { return _mm512_or_si512(left, right); };
    __m512i result =
        combine_primary_contributors<tables, __m512i>(pack_contributor, combine, std::make_index_sequence<tables::contributor_count - 1>{});

    if constexpr (tables::spill_mask != 0) {
        if constexpr (sizeof(LaneType) == sizeof(i8)) {
            const __m512i permuted = _mm512_maskz_permutexvar_epi8(static_cast<__mmask64>(tables::spill_mask),
                                                                   load_table<__m512i>(tables::spill_permute), values);
            result                 = _mm512_or_si512(result, mm512_pack_srlv_epi8(permuted, load_table<__m512i>(tables::spill_shift)));
        } else if constexpr (sizeof(LaneType) == sizeof(i16)) {
            const __m512i permuted = _mm512_maskz_permutexvar_epi16(static_cast<__mmask32>(tables::spill_mask),
                                                                    load_table<__m512i>(tables::spill_permute), values);
            result                 = _mm512_or_si512(result, _mm512_srlv_epi16(permuted, load_table<__m512i>(tables::spill_shift)));
        } else {
            const __m512i permuted = _mm512_maskz_permutexvar_epi32(static_cast<__mmask16>(tables::spill_mask),
                                                                    load_table<__m512i>(tables::spill_permute), values);
            result                 = _mm512_or_si512(result, _mm512_srlv_epi32(permuted, load_table<__m512i>(tables::spill_shift)));
        }
    }
    return result;
}

/**
 * @brief Pack 32 16-bit values for bit widths 9 through 16 using VBMI.
 */
template <u8 BIT_WIDTH>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
__always_inline __m512i mm512_pack_epi16_avx512vbmi_9to16(const __m512i input) {
    if constexpr (BIT_WIDTH == 16) {
        return input;
    }

    const __m512i maskv         = _mm512_set1_epi16(static_cast<i16>(detail::low_bit_mask<BIT_WIDTH>()));
    const __m512i masked_values = _mm512_and_si512(input, maskv);
    return mm512_pack_table_avx512vbmi<i16, BIT_WIDTH>(masked_values);
}

/**
 * @brief Pack 64 8-bit values for bit widths 1 through 8 using VBMI.
 */
template <u8 BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8)
__always_inline __m512i mm512_pack_epi8_avx512vbmi_1to8(const __m512i input) {
    if constexpr (BIT_WIDTH == 8) {
        return input;
    }

    const __m512i maskv         = _mm512_set1_epi8(static_cast<i8>(detail::low_bit_mask<BIT_WIDTH>()));
    const __m512i masked_values = _mm512_and_si512(input, maskv);
    return mm512_pack_table_avx512vbmi<i8, BIT_WIDTH>(masked_values);
}

/**
 * @brief Pack 16 32-bit values for bit widths 17 through 24 using VBMI.
 */
template <u8 BIT_WIDTH>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24)
__always_inline __m512i mm512_pack_epi32_avx512vbmi_17to24(const __m512i input) {
    const __m512i maskv         = _mm512_set1_epi32(static_cast<i32>(detail::low_bit_mask<BIT_WIDTH>()));
    const __m512i masked_values = _mm512_and_si512(input, maskv);
    return mm512_pack_table_avx512vbmi<i32, BIT_WIDTH>(masked_values);
}
}  // namespace m512
}  // namespace pernix::internal

#endif  // PERNIX_AVX512VBMI_PACKING_H
