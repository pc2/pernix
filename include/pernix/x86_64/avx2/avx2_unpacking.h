#ifndef PERNIX_AVX2_UNPACKING_H
#define PERNIX_AVX2_UNPACKING_H

#include <pernix/simd_compat.h>

#include <array>
#include <tuple>

namespace pernix::internal::avx2 {

namespace tables {

template <u8 BIT_WIDTH>
struct unpack_constants;

template <u8 BIT_WIDTH>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24)
struct unpack_constants<BIT_WIDTH> {
    alignas(32) static constexpr std::array<u8, 32> SHUFFLE_MASK = [] {
        std::array<u8, 32> mask{};
        mask.fill(0x80);

        constexpr usize second_bit_offset = (4 * BIT_WIDTH) % 8;

        for (usize value = 0; value < 8; ++value) {
            const usize half        = value / 4;
            const usize local_value = value % 4;
            const usize initial_bit = half == 0 ? 0 : second_bit_offset;
            const usize bit_start   = initial_bit + local_value * BIT_WIDTH;
            const usize byte_start  = bit_start / 8;
            const usize bit_offset  = bit_start % 8;
            const usize byte_count  = (bit_offset + BIT_WIDTH + 7) / 8;

            const usize output_byte = value * sizeof(i32);
            for (usize byte = 0; byte < byte_count; ++byte) {
                mask[output_byte + byte] = static_cast<u8>(byte_start + byte);
            }
        }

        return mask;
    }();

    alignas(32) static constexpr std::array<i32, 8> LEFT_SHIFT_MASK = [] {
        std::array<i32, 8> mask{};

        constexpr usize second_bit_offset = (4 * BIT_WIDTH) % 8;

        for (usize value = 0; value < mask.size(); ++value) {
            const usize half        = value / 4;
            const usize local_value = value % 4;
            const usize initial_bit = half == 0 ? 0 : second_bit_offset;
            const usize bit_offset  = (initial_bit + local_value * BIT_WIDTH) % 8;
            mask[value]             = static_cast<i32>(32 - BIT_WIDTH - bit_offset);
        }

        return mask;
    }();
};

template <u8 BIT_WIDTH>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 15)
struct unpack_constants<BIT_WIDTH> {
    alignas(64) static constexpr std::array<u8, 32> SHUFFLE_MASK = [] {
        std::array<u8, 32> mask{};
        mask.fill(0x80);

        for (usize value = 0; value < 8; ++value) {
            const usize bit_start  = value * BIT_WIDTH;
            const usize byte_start = bit_start / 8;
            const usize bit_offset = bit_start % 8;
            const usize byte_count = (bit_offset + BIT_WIDTH + 7) / 8;

            const usize output_byte = value * sizeof(i32);
            for (usize byte = 0; byte < byte_count; ++byte) {
                mask[output_byte + byte] = static_cast<u8>(byte_start + byte);
            }
        }

        return mask;
    }();

    alignas(32) static constexpr std::array<i32, 8> LEFT_SHIFT_MASK = [] {
        std::array<i32, 8> mask{};

        for (usize value = 0; value < mask.size(); ++value) {
            const usize bit_offset = (value * BIT_WIDTH) % 8;
            mask[value]            = static_cast<i32>(32 - BIT_WIDTH - bit_offset);
        }

        return mask;
    }();
};

}  // namespace tables

namespace m256 {
namespace detail {}

struct unpacked_epi32_pair {
    __m256i low;
    __m256i high;
};

template <u8 BitWidth, bool SignValues = true>
    requires(BitWidth >= 17 && BitWidth <= 24)
__always_inline __m256i mm256_unpack_epi32_17to24(__m128i low, __m128i high) {
    using constants = tables::unpack_constants<BitWidth>;

    __m256i windows = _mm256_castsi128_si256(low);
    windows         = _mm256_inserti128_si256(windows, high, 1);

    const __m256i shuffled =
        _mm256_shuffle_epi8(windows, _mm256_load_si256(reinterpret_cast<const __m256i*>(constants::SHUFFLE_MASK.data())));
    __m256i shifted = _mm256_sllv_epi32(shuffled, _mm256_load_si256(reinterpret_cast<const __m256i*>(constants::LEFT_SHIFT_MASK.data())));

    if constexpr (SignValues) {
        shifted = _mm256_srai_epi32(shifted, 32 - BitWidth);
    } else {
        shifted = _mm256_srli_epi32(shifted, 32 - BitWidth);
    }

    return shifted;
}

template <u8 BitWidth, bool SignValues = true>
    requires(BitWidth >= 9 && BitWidth <= 15)
__always_inline __m256i mm256_unpack_epi32_9to15(__m128i input) {
    using constants = tables::unpack_constants<BitWidth>;

    const __m256i repeated = _mm256_broadcastsi128_si256(input);
    const __m256i shuffled =
        _mm256_shuffle_epi8(repeated, _mm256_load_si256(reinterpret_cast<const __m256i*>(constants::SHUFFLE_MASK.data())));
    __m256i shifted = _mm256_sllv_epi32(shuffled, _mm256_load_si256(reinterpret_cast<const __m256i*>(constants::LEFT_SHIFT_MASK.data())));

    if constexpr (SignValues) {
        shifted = _mm256_srai_epi32(shifted, 32 - BitWidth);
    } else {
        shifted = _mm256_srli_epi32(shifted, 32 - BitWidth);
    }

    return shifted;
}

template <u8 BitWidth, bool SignValues = true>
    requires(BitWidth >= 4 && BitWidth <= 7)
__always_inline std::array<__m256i, 2> mm256_unpack_epi32_4to7(__m128i input) {
    constexpr u8 pair_bit_width = 2 * BitWidth;
    constexpr u16 value_mask    = (1U << BitWidth) - 1U;

    __m256i pairs;
    if constexpr (BitWidth == 4) {
        pairs = _mm256_cvtepu8_epi32(input);
    } else {
        pairs = mm256_unpack_epi32_9to15<pair_bit_width, false>(input);
    }

    const __m256i mask = _mm256_set1_epi32(value_mask);
    __m256i even       = _mm256_and_si256(pairs, mask);
    __m256i odd        = _mm256_srli_epi32(pairs, BitWidth);

    if constexpr (SignValues) {
        constexpr u8 sign_shift = 32 - BitWidth;

        even = _mm256_srai_epi32(_mm256_slli_epi32(even, sign_shift), sign_shift);
        odd  = _mm256_srai_epi32(_mm256_slli_epi32(odd, sign_shift), sign_shift);
    }

    const __m256i interleaved_low  = _mm256_unpacklo_epi32(even, odd);
    const __m256i interleaved_high = _mm256_unpackhi_epi32(even, odd);

    return {
        _mm256_permute2x128_si256(interleaved_low, interleaved_high, 0x20),
        _mm256_permute2x128_si256(interleaved_low, interleaved_high, 0x31),
    };
}

template <u8 BitWidth, bool SignValues = true>
    requires(BitWidth >= 1 && BitWidth <= 3)
__always_inline std::array<__m256i, 4> mm256_unpack_epi32_1to3(__m128i input) {
    __m256i groups;
    if constexpr (BitWidth == 1) {
        const __m256i bytes = _mm256_cvtepu8_epi32(input);
        const __m256i low   = _mm256_and_si256(bytes, _mm256_set1_epi32(0x0f));
        const __m256i high  = _mm256_srli_epi32(bytes, 4);

        const __m256i interleaved_low  = _mm256_unpacklo_epi32(low, high);
        const __m256i interleaved_high = _mm256_unpackhi_epi32(low, high);
        groups                         = _mm256_permute2x128_si256(interleaved_low, interleaved_high, 0x20);
    } else if constexpr (BitWidth == 2) {
        groups = _mm256_cvtepu8_epi32(input);
    } else {
        groups = mm256_unpack_epi32_9to15<12, false>(input);
    }

    constexpr u32 value_mask = (1U << BitWidth) - 1U;
    const __m256i mask       = _mm256_set1_epi32(value_mask);
    __m256i values0          = _mm256_and_si256(groups, mask);
    __m256i values1          = _mm256_and_si256(_mm256_srli_epi32(groups, BitWidth), mask);
    __m256i values2          = _mm256_and_si256(_mm256_srli_epi32(groups, 2 * BitWidth), mask);
    __m256i values3          = _mm256_srli_epi32(groups, 3 * BitWidth);

    if constexpr (SignValues && BitWidth > 1) {
        constexpr u8 sign_shift = 32 - BitWidth;
        values0                 = _mm256_srai_epi32(_mm256_slli_epi32(values0, sign_shift), sign_shift);
        values1                 = _mm256_srai_epi32(_mm256_slli_epi32(values1, sign_shift), sign_shift);
        values2                 = _mm256_srai_epi32(_mm256_slli_epi32(values2, sign_shift), sign_shift);
        values3                 = _mm256_srai_epi32(_mm256_slli_epi32(values3, sign_shift), sign_shift);
    }

    const __m256i values01_low  = _mm256_unpacklo_epi32(values0, values1);
    const __m256i values01_high = _mm256_unpackhi_epi32(values0, values1);
    const __m256i values23_low  = _mm256_unpacklo_epi32(values2, values3);
    const __m256i values23_high = _mm256_unpackhi_epi32(values2, values3);

    const __m256i groups04 = _mm256_unpacklo_epi64(values01_low, values23_low);
    const __m256i groups15 = _mm256_unpackhi_epi64(values01_low, values23_low);
    const __m256i groups26 = _mm256_unpacklo_epi64(values01_high, values23_high);
    const __m256i groups37 = _mm256_unpackhi_epi64(values01_high, values23_high);

    return {
        _mm256_permute2x128_si256(groups04, groups15, 0x20),
        _mm256_permute2x128_si256(groups26, groups37, 0x20),
        _mm256_permute2x128_si256(groups04, groups15, 0x31),
        _mm256_permute2x128_si256(groups26, groups37, 0x31),
    };
}

}  // namespace m256

}  // namespace pernix::internal::avx2

#endif  // PERNIX_AVX2_UNPACKING_H
