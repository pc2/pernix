#ifndef PERNIX_AVX512VBMI_UNPACKING_H
#define PERNIX_AVX512VBMI_UNPACKING_H

#include <pernix/detail/unpack_tables.h>
#include <pernix/simd_compat.h>
#include <pernix/x86/avx512vbmi/tables.h>

namespace pernix::internal {
namespace unpacking_detail {
template <usize SIZE, usize GROUP_SIZE>
alignas(SIZE) inline constexpr std::array<u8, SIZE> kInterleaveBytes = [] {
    std::array<u8, SIZE> indices{};
    constexpr usize group_count = SIZE / GROUP_SIZE;
    for (usize i = 0; i < SIZE; ++i) {
        indices[i] = static_cast<u8>((i % GROUP_SIZE) * group_count + i / GROUP_SIZE);
    }
    return indices;
}();

template <usize SIZE, usize GROUP_SIZE>
alignas(SIZE) inline constexpr std::array<u8, SIZE> kRepeatBytes = [] {
    std::array<u8, SIZE> indices{};
    for (usize i = 0; i < SIZE; ++i) {
        indices[i] = static_cast<u8>(i / GROUP_SIZE);
    }
    return indices;
}();

template <usize SIZE, usize GROUP_SIZE, u8 BIT_WIDTH>
alignas(SIZE) inline constexpr std::array<u8, SIZE> kGroupShifts = [] {
    std::array<u8, SIZE> shifts{};
    for (usize i = 0; i < SIZE; ++i) {
        shifts[i] = static_cast<u8>((i % GROUP_SIZE) * BIT_WIDTH);
    }
    return shifts;
}();
}  // namespace unpacking_detail

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

template <u8 BIT_WIDTH, bool SIGN_VALUES>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
__always_inline __m128i mm_unpack_epi16_avx512vbmi_9to16(__m128i input);

template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8)
__always_inline __m128i mm_unpack_epi8_avx512vbmi_1to8(__m128i input) {
    if constexpr (BIT_WIDTH == 8) {
        return input;
    } else {
        __m128i unpacked;

        if constexpr (BIT_WIDTH == 1) {
            const __m128i repeated = _mm_permutexvar_epi8(load_table<__m128i>(unpacking_detail::kRepeatBytes<16, 8>), input);
            unpacked =
                _mm_and_si128(_mm_srlv_epi8(repeated, load_table<__m128i>(unpacking_detail::kGroupShifts<16, 8, 1>)), _mm_set1_epi8(0x01));
        } else if constexpr (BIT_WIDTH == 2) {
            const __m128i repeated = _mm_permutexvar_epi8(load_table<__m128i>(unpacking_detail::kRepeatBytes<16, 4>), input);
            unpacked =
                _mm_and_si128(_mm_srlv_epi8(repeated, load_table<__m128i>(unpacking_detail::kGroupShifts<16, 4, 2>)), _mm_set1_epi8(0x03));
        } else if constexpr (BIT_WIDTH == 3) {
            const __m128i groups = mm_unpack_epi16_avx512vbmi_9to16<4 * BIT_WIDTH, false>(input);
            const __m128i mask   = _mm_set1_epi16(0x07);

            const __m128i values0 = _mm_cvtepi16_epi8(_mm_and_si128(groups, mask));
            const __m128i values1 = _mm_cvtepi16_epi8(_mm_and_si128(_mm_srli_epi16(groups, 3), mask));
            const __m128i values2 = _mm_cvtepi16_epi8(_mm_and_si128(_mm_srli_epi16(groups, 6), mask));
            const __m128i values3 = _mm_cvtepi16_epi8(_mm_srli_epi16(groups, 9));

            __m128i combined = values0;
            combined         = _mm_insert_epi32(combined, _mm_cvtsi128_si32(values1), 1);
            combined         = _mm_insert_epi32(combined, _mm_cvtsi128_si32(values2), 2);
            combined         = _mm_insert_epi32(combined, _mm_cvtsi128_si32(values3), 3);

            unpacked = _mm_permutexvar_epi8(load_table<__m128i>(unpacking_detail::kInterleaveBytes<16, 4>), combined);
        } else if constexpr (BIT_WIDTH == 4) {
            const __m128i mask     = _mm_set1_epi8(0x0f);
            const __m128i even     = _mm_and_si128(input, mask);
            const __m128i odd      = _mm_and_si128(_mm_srli_epi16(input, BIT_WIDTH), mask);
            const __m128i combined = _mm_unpacklo_epi64(even, odd);

            unpacked = _mm_permutexvar_epi8(load_table<__m128i>(unpacking_detail::kInterleaveBytes<16, 2>), combined);
        } else {
            constexpr u8 pair_bit_width = 2 * BIT_WIDTH;
            constexpr u16 value_mask    = (1U << BIT_WIDTH) - 1U;

            const __m128i pairs    = mm_unpack_epi16_avx512vbmi_9to16<pair_bit_width, false>(input);
            const __m128i even     = _mm_cvtepi16_epi8(_mm_and_si128(pairs, _mm_set1_epi16(static_cast<i16>(value_mask))));
            const __m128i odd      = _mm_cvtepi16_epi8(_mm_srli_epi16(pairs, BIT_WIDTH));
            const __m128i combined = _mm_unpacklo_epi64(even, odd);

            unpacked = _mm_permutexvar_epi8(load_table<__m128i>(unpacking_detail::kInterleaveBytes<16, 2>), combined);
        }

        if constexpr (SIGN_VALUES && BIT_WIDTH > 1) {
            const __m128i sign = _mm_set1_epi8(1U << (BIT_WIDTH - 1U));
            unpacked           = _mm_sub_epi8(_mm_xor_si128(unpacked, sign), sign);
        }

        return unpacked;
    }
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
__always_inline __m128i mm_unpack_epi16_avx512vbmi_9to16(__m128i input) {
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

        if constexpr (SIGN_VALUES) {
            constexpr u32 shift = 16 - BIT_WIDTH;
            shifted             = _mm_slli_epi16(shifted, shift);
            shifted             = _mm_srai_epi16(shifted, shift);
        } else {
            constexpr u16 value_mask = (1U << BIT_WIDTH) - 1U;
            shifted                  = _mm_and_si128(shifted, _mm_set1_epi16(static_cast<i16>(value_mask)));
        }

        return shifted;
    }
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24)
__always_inline __m128i mm_unpack_epi32_avx512vbmi_17to24(__m128i input) {
    using tables = detail::unpack_table<i32, BIT_WIDTH, sizeof(__m128i)>;

    const __m128i permuted    = _mm_permutexvar_epi8(load_table<__m128i>(tables::primary_permute), input);
    const __m128i right_shift = load_table<__m128i>(tables::right_shift_magnitude);

    constexpr u32 shift = 32 - BIT_WIDTH;
    __m128i shifted     = _mm_srlv_epi32(permuted, right_shift);
    shifted             = _mm_slli_epi32(shifted, shift);
    if constexpr (SIGN_VALUES) {
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

template <u8 BIT_WIDTH, bool SIGN_VALUES>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
__always_inline __m256i mm256_unpack_epi16_avx512vbmi_9to16(__m256i input);

template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8)
__always_inline __m256i mm256_unpack_epi8_avx512vbmi_1to8(__m256i input) {
    if constexpr (BIT_WIDTH == 8) {
        return input;
    } else {
        __m256i unpacked;

        if constexpr (BIT_WIDTH == 1) {
            const __m256i repeated = _mm256_permutexvar_epi8(load_table<__m256i>(unpacking_detail::kRepeatBytes<32, 8>), input);
            unpacked = _mm256_and_si256(_mm256_srlv_epi8(repeated, load_table<__m256i>(unpacking_detail::kGroupShifts<32, 8, 1>)),
                                        _mm256_set1_epi8(0x01));
        } else if constexpr (BIT_WIDTH == 2) {
            const __m256i repeated = _mm256_permutexvar_epi8(load_table<__m256i>(unpacking_detail::kRepeatBytes<32, 4>), input);
            unpacked = _mm256_and_si256(_mm256_srlv_epi8(repeated, load_table<__m256i>(unpacking_detail::kGroupShifts<32, 4, 2>)),
                                        _mm256_set1_epi8(0x03));
        } else if constexpr (BIT_WIDTH == 3) {
            const __m256i groups = mm256_unpack_epi16_avx512vbmi_9to16<4 * BIT_WIDTH, false>(input);
            const __m256i mask   = _mm256_set1_epi16(0x07);

            const __m128i values0 = _mm256_cvtepi16_epi8(_mm256_and_si256(groups, mask));
            const __m128i values1 = _mm256_cvtepi16_epi8(_mm256_and_si256(_mm256_srli_epi16(groups, 3), mask));
            const __m128i values2 = _mm256_cvtepi16_epi8(_mm256_and_si256(_mm256_srli_epi16(groups, 6), mask));
            const __m128i values3 = _mm256_cvtepi16_epi8(_mm256_srli_epi16(groups, 9));

            const __m128i combined01 = _mm_unpacklo_epi64(values0, values1);
            const __m128i combined23 = _mm_unpacklo_epi64(values2, values3);
            __m256i combined         = _mm256_castsi128_si256(combined01);
            combined                 = _mm256_inserti128_si256(combined, combined23, 1);

            unpacked = _mm256_permutexvar_epi8(load_table<__m256i>(unpacking_detail::kInterleaveBytes<32, 4>), combined);
        } else if constexpr (BIT_WIDTH == 4) {
            const __m256i mask = _mm256_set1_epi8(0x0f);
            const __m256i even = _mm256_and_si256(input, mask);
            const __m256i odd  = _mm256_and_si256(_mm256_srli_epi16(input, BIT_WIDTH), mask);

            __m256i combined = _mm256_castsi128_si256(_mm256_castsi256_si128(even));
            combined         = _mm256_inserti128_si256(combined, _mm256_castsi256_si128(odd), 1);

            unpacked = _mm256_permutexvar_epi8(load_table<__m256i>(unpacking_detail::kInterleaveBytes<32, 2>), combined);
        } else {
            constexpr u8 pair_bit_width = 2 * BIT_WIDTH;
            constexpr u16 value_mask    = (1U << BIT_WIDTH) - 1U;

            const __m256i pairs = mm256_unpack_epi16_avx512vbmi_9to16<pair_bit_width, false>(input);
            const __m128i even  = _mm256_cvtepi16_epi8(_mm256_and_si256(pairs, _mm256_set1_epi16(static_cast<i16>(value_mask))));
            const __m128i odd   = _mm256_cvtepi16_epi8(_mm256_srli_epi16(pairs, BIT_WIDTH));

            __m256i combined = _mm256_castsi128_si256(even);
            combined         = _mm256_inserti128_si256(combined, odd, 1);

            unpacked = _mm256_permutexvar_epi8(load_table<__m256i>(unpacking_detail::kInterleaveBytes<32, 2>), combined);
        }

        if constexpr (SIGN_VALUES && BIT_WIDTH > 1) {
            const __m256i sign = _mm256_set1_epi8(1U << (BIT_WIDTH - 1U));
            unpacked           = _mm256_sub_epi8(_mm256_xor_si256(unpacked, sign), sign);
        }

        return unpacked;
    }
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
__always_inline __m256i mm256_unpack_epi16_avx512vbmi_9to16(__m256i input) {
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

        if constexpr (SIGN_VALUES) {
            constexpr u32 shift = 16 - BIT_WIDTH;
            shifted             = _mm256_slli_epi16(shifted, shift);
            shifted             = _mm256_srai_epi16(shifted, shift);
        } else {
            constexpr u16 value_mask = (1U << BIT_WIDTH) - 1U;
            shifted                  = _mm256_and_si256(shifted, _mm256_set1_epi16(static_cast<i16>(value_mask)));
        }

        return shifted;
    }
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24)
__always_inline __m256i mm256_unpack_epi32_avx512vbmi_17to24(__m256i input) {
    using tables = detail::unpack_table<i32, BIT_WIDTH, sizeof(__m256i)>;

    const __m256i permuted    = _mm256_permutexvar_epi8(load_table<__m256i>(tables::primary_permute), input);
    const __m256i right_shift = load_table<__m256i>(tables::right_shift_magnitude);

    constexpr u32 shift = 32 - BIT_WIDTH;
    __m256i shifted     = _mm256_srlv_epi32(permuted, right_shift);
    shifted             = _mm256_slli_epi32(shifted, shift);
    if constexpr (SIGN_VALUES) {
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
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
__always_inline __m512i mm512_unpack_epi16_avx512vbmi_9to16(__m512i input) {
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

        if constexpr (SIGN_VALUES) {
            constexpr u32 shift = 16 - BIT_WIDTH;
            shifted             = _mm512_slli_epi16(shifted, shift);
            shifted             = _mm512_srai_epi16(shifted, shift);
        } else {
            constexpr u16 value_mask = (1U << BIT_WIDTH) - 1U;
            shifted                  = _mm512_and_si512(shifted, _mm512_set1_epi16(static_cast<i16>(value_mask)));
        }

        return shifted;
    }
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24)
__always_inline __m512i mm512_unpack_epi32_avx512vbmi_17to24(__m512i input) {
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

template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 7)
__always_inline __m512i mm512_unpack_epi8_avx512vbmi_1to7(__m512i input) {
    __m512i unpacked;

    if constexpr (BIT_WIDTH == 1) {
        const __m512i repeated = _mm512_permutexvar_epi8(load_table<__m512i>(unpacking_detail::kRepeatBytes<64, 8>), input);
        unpacked               = _mm512_and_si512(_mm512_srlv_epi8(repeated, load_table<__m512i>(unpacking_detail::kGroupShifts<64, 8, 1>)),
                                                  _mm512_set1_epi8(0x01));
    } else if constexpr (BIT_WIDTH == 2) {
        const __m512i repeated = _mm512_permutexvar_epi8(load_table<__m512i>(unpacking_detail::kRepeatBytes<64, 4>), input);
        unpacked               = _mm512_and_si512(_mm512_srlv_epi8(repeated, load_table<__m512i>(unpacking_detail::kGroupShifts<64, 4, 2>)),
                                                  _mm512_set1_epi8(0x03));
    } else if constexpr (BIT_WIDTH == 3) {
        const __m512i groups = mm512_unpack_epi16_avx512vbmi_9to16<4 * BIT_WIDTH, false>(input);
        const __m512i mask   = _mm512_set1_epi16(0x07);

        const __m256i values0 = _mm512_cvtepi16_epi8(_mm512_and_si512(groups, mask));
        const __m256i values1 = _mm512_cvtepi16_epi8(_mm512_and_si512(_mm512_srli_epi16(groups, 3), mask));
        const __m256i values2 = _mm512_cvtepi16_epi8(_mm512_and_si512(_mm512_srli_epi16(groups, 6), mask));
        const __m256i values3 = _mm512_cvtepi16_epi8(_mm512_srli_epi16(groups, 9));

        __m512i combined = _mm512_castsi128_si512(_mm256_castsi256_si128(values0));
        combined         = _mm512_inserti32x4(combined, _mm256_castsi256_si128(values1), 1);
        combined         = _mm512_inserti32x4(combined, _mm256_castsi256_si128(values2), 2);
        combined         = _mm512_inserti32x4(combined, _mm256_castsi256_si128(values3), 3);

        unpacked = _mm512_permutexvar_epi8(load_table<__m512i>(unpacking_detail::kInterleaveBytes<64, 4>), combined);
    } else if constexpr (BIT_WIDTH == 4) {
        const __m512i mask = _mm512_set1_epi8(0x0f);
        const __m512i even = _mm512_and_si512(input, mask);
        const __m512i odd  = _mm512_and_si512(_mm512_srli_epi16(input, BIT_WIDTH), mask);

        __m512i combined = _mm512_castsi256_si512(_mm512_castsi512_si256(even));
        combined         = _mm512_inserti64x4(combined, _mm512_castsi512_si256(odd), 1);

        unpacked = _mm512_permutexvar_epi8(load_table<__m512i>(unpacking_detail::kInterleaveBytes<64, 2>), combined);
    } else {
        constexpr u8 pair_bit_width = 2 * BIT_WIDTH;
        constexpr u16 value_mask    = (1U << BIT_WIDTH) - 1U;

        const __m512i pairs = mm512_unpack_epi16_avx512vbmi_9to16<pair_bit_width, false>(input);
        const __m512i even  = _mm512_and_si512(pairs, _mm512_set1_epi16(static_cast<i16>(value_mask)));
        const __m512i odd   = _mm512_srli_epi16(pairs, BIT_WIDTH);

        const __m256i even_bytes = _mm512_cvtepi16_epi8(even);
        const __m256i odd_bytes  = _mm512_cvtepi16_epi8(odd);

        __m512i combined = _mm512_castsi256_si512(even_bytes);
        combined         = _mm512_inserti64x4(combined, odd_bytes, 1);

        unpacked = _mm512_permutexvar_epi8(load_table<__m512i>(unpacking_detail::kInterleaveBytes<64, 2>), combined);
    }

    if constexpr (SIGN_VALUES && BIT_WIDTH > 1) {
        const __m512i sign = _mm512_set1_epi8(1U << (BIT_WIDTH - 1U));
        unpacked           = _mm512_sub_epi8(_mm512_xor_si512(unpacked, sign), sign);
    }

    return unpacked;
}

}  // namespace m512
}  // namespace pernix::internal

#endif  // PERNIX_AVX512VBMI_UNPACKING_H
