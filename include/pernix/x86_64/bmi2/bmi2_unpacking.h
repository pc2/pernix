#ifndef PERNIX_BMI2_UNPACKING_H
#define PERNIX_BMI2_UNPACKING_H

#include <pernix/simd_compat.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <span>

namespace pernix::internal::bmi2 {

#define VALUE_MASK_32(bit_width) (static_cast<u32>((u64{1} << (bit_width)) - 1U))
#define VALUE_MASK_64(bit_width) (static_cast<u64>((u64{1} << (bit_width)) - 1U))

#define PDEP_MASK_8(mask) (0x0101010101010101ULL * mask)
#define PDEP_MASK_16(mask) (0x0001000100010001ULL * mask)
#define PDEP_MASK_32(mask) (0x0000000100000001ULL * mask)

#define PACKED_BYTES(elements, bit_width) (((elements * bit_width) + 7U) / 8U)

template <u8 BitWidth>
    requires(BitWidth <= 32)
__always_inline __m128i mm_sign_extend_epi32(__m128i value) noexcept {
    constexpr usize shift_mask = 32 - BitWidth;
    return _mm_srai_epi32(_mm_slli_epi32(value, shift_mask), shift_mask);
}

template <u8 BitWidth>
    requires(BitWidth <= 32)
__always_inline __m256i mm256_sign_extend_epi32(__m256i value) noexcept {
    constexpr usize shift_mask = 32 - BitWidth;
    return _mm256_srai_epi32(_mm256_slli_epi32(value, shift_mask), shift_mask);
}

template <u8 BitWidth, bool SignValues, usize Elements = 4>
    requires(BitWidth >= 1 && BitWidth <= 7 && Elements > 0 && Elements <= 4)
__always_inline __m128i mm_unpack_epi32_1to7(const std::span<const u8, PACKED_BYTES(Elements, BitWidth)> input) noexcept {
    constexpr u32 mask           = VALUE_MASK_32(BitWidth);
    constexpr u32 pdep_mask      = static_cast<u32>(PDEP_MASK_8(mask));
    constexpr usize packed_bytes = PACKED_BYTES(Elements, BitWidth);

    u32 value = 0;
    std::memcpy(&value, input.data(), packed_bytes);

    value = _pdep_u32(value, pdep_mask);

    __m128i result = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(static_cast<i32>(value)));
    if constexpr (SignValues && BitWidth > 1) {
        result = mm_sign_extend_epi32<BitWidth>(result);
    }

    return result;
}

template <bool SignValues, usize Elements = 4>
    requires(Elements > 0 && Elements <= 4)
__always_inline __m128i mm_unpack_epi32_8(const std::span<const u8, PACKED_BYTES(Elements, 8)> input) noexcept {
    u32 value = 0;
    std::memcpy(&value, input.data(), input.size());
    const __m128i source = _mm_cvtsi32_si128(static_cast<i32>(value));
    if constexpr (SignValues) {
        return _mm_cvtepi8_epi32(source);
    } else {
        return _mm_cvtepu8_epi32(source);
    }
}

template <u8 BitWidth, bool SignValues, usize Elements = 4>
    requires(BitWidth >= 9 && BitWidth <= 15 && Elements > 0 && Elements <= 4)
__always_inline __m128i mm_unpack_epi32_9to15(const std::span<const u8, PACKED_BYTES(Elements, BitWidth)> input) noexcept {
    constexpr u32 mask           = VALUE_MASK_32(BitWidth);
    constexpr u64 pdep_mask      = PDEP_MASK_16(mask);
    constexpr usize packed_bytes = PACKED_BYTES(Elements, BitWidth);

    u64 value = 0;
    std::memcpy(&value, input.data(), packed_bytes);

    value = _pdep_u64(value, pdep_mask);

    __m128i result = _mm_cvtepu16_epi32(_mm_set_epi64x(0, static_cast<i64>(value)));
    if constexpr (SignValues) {
        result = mm_sign_extend_epi32<BitWidth>(result);
    }

    return result;
}

template <bool SignValues, usize Elements = 4>
    requires(Elements > 0 && Elements <= 4)
__always_inline __m128i mm_unpack_epi32_16(const std::span<const u8, PACKED_BYTES(Elements, 16)> input) noexcept {
    u64 value = 0;
    std::memcpy(&value, input.data(), input.size());
    const __m128i source = _mm_set_epi64x(0, static_cast<i64>(value));
    if constexpr (SignValues) {
        return _mm_cvtepi16_epi32(source);
    } else {
        return _mm_cvtepu16_epi32(source);
    }
}

template <u8 BitWidth, bool SignValues, usize Elements = 4>
    requires(BitWidth >= 17 && BitWidth <= 24 && Elements > 0 && Elements <= 4)
__always_inline __m128i mm_unpack_epi32_17to24(const std::span<const u8, PACKED_BYTES(Elements, BitWidth)> input) noexcept {
    constexpr u32 mask             = VALUE_MASK_32(BitWidth);
    constexpr u64 pdep_mask        = PDEP_MASK_32(mask);
    constexpr usize packed_bytes   = PACKED_BYTES(Elements, BitWidth);
    constexpr u32 value_pair_shift = BitWidth * 2U;
    constexpr u32 carry_shift      = 64U - value_pair_shift;

    std::array<u64, 2> temp_values{};
    std::memcpy(temp_values.data(), input.data(), packed_bytes);

    const u64 value0 = _pdep_u64(temp_values.at(0), pdep_mask);
    const u64 value1 = _pdep_u64((temp_values.at(0) >> value_pair_shift) | (temp_values.at(1) << carry_shift), pdep_mask);

    __m128i result = _mm_set_epi64x(static_cast<i64>(value1), static_cast<i64>(value0));
    if constexpr (SignValues) {
        result = mm_sign_extend_epi32<BitWidth>(result);
    }

    return result;
}

template <u8 BitWidth, bool SignValues = true, usize Elements = 4>
    requires(BitWidth >= 1 && BitWidth <= 24 && Elements > 0 && Elements <= 4)
__always_inline __m128i mm_unpack_epi32(const std::span<const u8, PACKED_BYTES(Elements, BitWidth)> input) noexcept {
    if constexpr (BitWidth <= 7) {
        return mm_unpack_epi32_1to7<BitWidth, SignValues, Elements>(input);
    } else if constexpr (BitWidth == 8) {
        return mm_unpack_epi32_8<SignValues, Elements>(input);
    } else if constexpr (BitWidth <= 15) {
        return mm_unpack_epi32_9to15<BitWidth, SignValues, Elements>(input);
    } else if constexpr (BitWidth == 16) {
        return mm_unpack_epi32_16<SignValues, Elements>(input);
    } else {
        return mm_unpack_epi32_17to24<BitWidth, SignValues, Elements>(input);
    }
}

template <u8 BitWidth, bool SignValues, usize Elements = 8>
    requires(BitWidth >= 1 && BitWidth <= 7 && Elements > 0 && Elements <= 8)
__always_inline __m256i mm256_unpack_epi32_1to7(const std::span<const u8, PACKED_BYTES(Elements, BitWidth)> input) noexcept {
    if constexpr (Elements <= 4) {
        const __m128i unpacked = mm_unpack_epi32_1to7<BitWidth, SignValues, Elements>(input);
        return _mm256_inserti128_si256(_mm256_setzero_si256(), unpacked, 0);
    } else {
        constexpr u32 mask           = VALUE_MASK_32(BitWidth);
        constexpr u64 pdep_mask      = PDEP_MASK_8(mask);
        constexpr usize packed_bytes = PACKED_BYTES(Elements, BitWidth);

        u64 value = 0;
        std::memcpy(&value, input.data(), packed_bytes);

        value = _pdep_u64(value, pdep_mask);

        __m256i result = _mm256_cvtepu8_epi32(_mm_set_epi64x(0, static_cast<i64>(value)));
        if constexpr (SignValues && BitWidth > 1) {
            result = mm256_sign_extend_epi32<BitWidth>(result);
        }

        return result;
    }
}

template <bool SignValues, usize Elements = 8>
    requires(Elements > 0 && Elements <= 8)
__always_inline __m256i mm256_unpack_epi32_8(const std::span<const u8, PACKED_BYTES(Elements, 8)> input) noexcept {
    if constexpr (Elements <= 4) {
        const __m128i unpacked = mm_unpack_epi32_8<SignValues, Elements>(input);
        return _mm256_inserti128_si256(_mm256_setzero_si256(), unpacked, 0);
    } else {
        u64 value = 0;
        std::memcpy(&value, input.data(), input.size());
        const __m128i source = _mm_set_epi64x(0, static_cast<i64>(value));
        if constexpr (SignValues) {
            return _mm256_cvtepi8_epi32(source);
        } else {
            return _mm256_cvtepu8_epi32(source);
        }
    }
}

template <u8 BitWidth, bool SignValues, usize Elements = 8>
    requires(BitWidth >= 9 && BitWidth <= 15 && Elements > 0 && Elements <= 8)
__always_inline __m256i mm256_unpack_epi32_9to15(const std::span<const u8, PACKED_BYTES(Elements, BitWidth)> input) noexcept {
    if constexpr (Elements <= 4) {
        const __m128i unpacked = mm_unpack_epi32_9to15<BitWidth, SignValues, Elements>(input);
        return _mm256_inserti128_si256(_mm256_setzero_si256(), unpacked, 0);
    } else {
        constexpr u32 mask              = VALUE_MASK_32(BitWidth);
        constexpr u64 pdep_mask         = PDEP_MASK_16(mask);
        constexpr usize packed_bytes    = PACKED_BYTES(Elements, BitWidth);
        constexpr u32 value_group_shift = BitWidth * 4U;
        constexpr u32 carry_shift       = 64U - value_group_shift;

        std::array<u64, 2> temp_values{};
        std::memcpy(temp_values.data(), input.data(), packed_bytes);

        const u64 value0 = _pdep_u64(temp_values.at(0), pdep_mask);
        const u64 value1 = _pdep_u64((temp_values.at(0) >> value_group_shift) | (temp_values.at(1) << carry_shift), pdep_mask);

        __m256i result = _mm256_cvtepu16_epi32(_mm_set_epi64x(static_cast<i64>(value1), static_cast<i64>(value0)));
        if constexpr (SignValues) {
            result = mm256_sign_extend_epi32<BitWidth>(result);
        }

        return result;
    }
}

template <bool SignValues, usize Elements = 8>
    requires(Elements > 0 && Elements <= 8)
__always_inline __m256i mm256_unpack_epi32_16(const std::span<const u8, PACKED_BYTES(Elements, 16)> input) noexcept {
    if constexpr (Elements <= 4) {
        const __m128i unpacked = mm_unpack_epi32_16<SignValues, Elements>(input);
        return _mm256_inserti128_si256(_mm256_setzero_si256(), unpacked, 0);
    } else {
        std::array<u64, 2> value{};
        std::memcpy(value.data(), input.data(), input.size());
        const __m128i source = _mm_loadu_si128(reinterpret_cast<__m128i const*>(value.data()));
        if constexpr (SignValues) {
            return _mm256_cvtepi16_epi32(source);
        } else {
            return _mm256_cvtepu16_epi32(source);
        }
    }
}

template <u8 BitWidth, bool SignValues, usize Elements = 8>
    requires(BitWidth >= 17U && BitWidth <= 24U && Elements > 0 && Elements <= 8)
__always_inline __m256i mm256_unpack_epi32_17to24(const std::span<const u8, PACKED_BYTES(Elements, BitWidth)> input) noexcept {
    if constexpr (Elements <= 4) {
        const __m128i unpacked = mm_unpack_epi32_17to24<BitWidth, SignValues, Elements>(input);
        return _mm256_inserti128_si256(_mm256_setzero_si256(), unpacked, 0);
    } else {
        constexpr u32 mask      = VALUE_MASK_32(BitWidth);
        constexpr u64 pdep_mask = PDEP_MASK_32(mask);

        constexpr usize packed_bytes = PACKED_BYTES(Elements, BitWidth);
        constexpr usize group_bytes  = 2U * sizeof(u64);

        constexpr u32 value_pair_shift = BitWidth * 2U;
        constexpr u32 carry_shift      = 64U - value_pair_shift;

        constexpr usize second_group_byte_offset = BitWidth / 2U;
        constexpr usize second_group_bytes       = packed_bytes - second_group_byte_offset;

        static_assert(second_group_bytes <= group_bytes);

        std::array<u64, 2> first_group{};
        std::array<u64, 2> second_group{};

        std::memcpy(first_group.data(), input.data(), std::min(packed_bytes, group_bytes));
        std::memcpy(second_group.data(), input.data() + second_group_byte_offset, second_group_bytes);

        if constexpr ((BitWidth % 2U) != 0U) {
            second_group = {
                (second_group[0] >> 4U) | (second_group[1] << 60U),
                second_group[1] >> 4U,
            };
        }

        const u64 value0 = _pdep_u64(first_group[0], pdep_mask);

        const u64 value1 = _pdep_u64((first_group[0] >> value_pair_shift) | (first_group[1] << carry_shift), pdep_mask);

        const u64 value2 = _pdep_u64(second_group[0], pdep_mask);

        const u64 value3 = _pdep_u64((second_group[0] >> value_pair_shift) | (second_group[1] << carry_shift), pdep_mask);

        __m256i result =
            _mm256_set_epi64x(static_cast<i64>(value3), static_cast<i64>(value2), static_cast<i64>(value1), static_cast<i64>(value0));

        if constexpr (SignValues) {
            result = mm256_sign_extend_epi32<BitWidth>(result);
        }

        return result;
    }
}

template <u8 BitWidth, bool SignValues = true, usize Elements = 8>
    requires(BitWidth >= 1 && BitWidth <= 24 && Elements > 0 && Elements <= 8)
__always_inline __m256i mm256_unpack_epi32(const std::span<const u8, PACKED_BYTES(Elements, BitWidth)> input) noexcept {
    if constexpr (BitWidth <= 7) {
        return mm256_unpack_epi32_1to7<BitWidth, SignValues, Elements>(input);
    } else if constexpr (BitWidth == 8) {
        return mm256_unpack_epi32_8<SignValues, Elements>(input);
    } else if constexpr (BitWidth <= 15) {
        return mm256_unpack_epi32_9to15<BitWidth, SignValues, Elements>(input);
    } else if constexpr (BitWidth == 16) {
        return mm256_unpack_epi32_16<SignValues, Elements>(input);
    } else {
        return mm256_unpack_epi32_17to24<BitWidth, SignValues, Elements>(input);
    }
}

#undef VALUE_MASK_32
#undef VALUE_MASK_64
#undef PDEP_MASK_8
#undef PDEP_MASK_16
#undef PDEP_MASK_32
#undef PACKED_BYTES

}  // namespace pernix::internal::bmi2

#endif  // PERNIX_BMI2_UNPACKING_H
