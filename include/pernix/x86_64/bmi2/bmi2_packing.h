#ifndef PERNIX_BMI2_PACKING_H
#define PERNIX_BMI2_PACKING_H

#include <pernix/simd_compat.h>

#include <array>
#include <cstring>
#include <span>

namespace pernix::internal::bmi2 {

#define VALUE_MASK_32(bit_width) (static_cast<u32>((u64{1} << (bit_width)) - 1U))
#define VALUE_MASK_64(bit_width) (static_cast<u64>((u64{1} << (bit_width)) - 1U))

#define PEXT_MASK_8(mask) (0x0101010101010101ULL * mask)
#define PEXT_MASK_16(mask) (0x0001000100010001ULL * mask)
#define PEXT_MASK_32(mask) (0x0000000100000001ULL * mask)

#define PACKED_BYTES(elements, bit_width) (((elements * bit_width) + 7U) / 8U)
#define MAX_PACKED_BYTES(bit_width) PACKED_BYTES(8, bit_width)

template <u8 BitWidth, usize Elements = 4>
    requires(BitWidth >= 1 && BitWidth <= 7 && Elements > 0 && Elements <= 4)
__always_inline void mm_pack_epi32_1to7(__m128i input_value, const std::span<u8, PACKED_BYTES(Elements, BitWidth)> output) noexcept {
    constexpr u32 pext_mask      = static_cast<u32>(PEXT_MASK_8(VALUE_MASK_32(BitWidth)));
    constexpr usize packed_bytes = PACKED_BYTES(Elements, BitWidth);

    const __m128i packed16 = _mm_packs_epi32(input_value, _mm_setzero_si128());
    const __m128i packed8  = _mm_packs_epi16(packed16, _mm_setzero_si128());
    const u32 value        = _pext_u32(static_cast<u32>(_mm_extract_epi32(packed8, 0)), pext_mask);

    std::memcpy(output.data(), &value, packed_bytes);
}

template <usize Elements = 4>
    requires(Elements > 0 && Elements <= 4)
__always_inline void mm_pack_epi32_8(__m128i input_value, const std::span<u8, PACKED_BYTES(Elements, 8)> output) noexcept {
    const __m128i packed16 = _mm_packs_epi32(input_value, _mm_setzero_si128());
    const __m128i packed8  = _mm_packs_epi16(packed16, _mm_setzero_si128());
    std::memcpy(output.data(), &packed8, output.size());
}

template <u8 BitWidth, usize Elements = 4>
    requires(BitWidth >= 9 && BitWidth <= 15 && Elements > 0 && Elements <= 4)
__always_inline void mm_pack_epi32_9to15(__m128i input_value, const std::span<u8, PACKED_BYTES(Elements, BitWidth)> output) noexcept {
    constexpr u64 pext_mask      = PEXT_MASK_16(VALUE_MASK_32(BitWidth));
    constexpr usize packed_bytes = PACKED_BYTES(Elements, BitWidth);

    const __m128i packed16 = _mm_packs_epi32(input_value, _mm_setzero_si128());
    const u64 value        = _pext_u64(static_cast<u64>(_mm_extract_epi64(packed16, 0)), pext_mask);

    std::memcpy(output.data(), &value, packed_bytes);
}

template <usize Elements = 4>
    requires(Elements > 0 && Elements <= 4)
__always_inline void mm_pack_epi32_16(__m128i input_value, const std::span<u8, PACKED_BYTES(Elements, 16)> output) noexcept {
    const __m128i packed16 = _mm_packs_epi32(input_value, _mm_setzero_si128());
    std::memcpy(output.data(), &packed16, output.size());
}

template <u8 BitWidth, usize Elements = 4>
    requires(BitWidth >= 17 && BitWidth <= 24 && Elements > 0 && Elements <= 4)
__always_inline void mm_pack_epi32_17to24(__m128i input_value, const std::span<u8, PACKED_BYTES(Elements, BitWidth)> output) noexcept {
    constexpr u64 pext_mask      = PEXT_MASK_32(VALUE_MASK_32(BitWidth));
    constexpr usize packed_bytes = PACKED_BYTES(Elements, BitWidth);

    constexpr u32 value_pair_bits = BitWidth * 2U;
    constexpr u32 carry_shift     = 64U - value_pair_bits;

    const u64 value0 = _pext_u64(static_cast<u64>(_mm_extract_epi64(input_value, 0)), pext_mask);
    const u64 value1 = _pext_u64(static_cast<u64>(_mm_extract_epi64(input_value, 1)), pext_mask);

    const std::array<u64, 2> values{
        value0 | (value1 << value_pair_bits),
        value1 >> carry_shift,
    };

    std::memcpy(output.data(), values.data(), packed_bytes);
}

template <u8 BitWidth, usize Elements = 4>
    requires(BitWidth >= 1 && BitWidth <= 24)
__always_inline void mm_pack_epi32(__m128i input_value, const std::span<u8, PACKED_BYTES(Elements, BitWidth)> output) noexcept {
    if constexpr (BitWidth <= 7) {
        return mm_pack_epi32_1to7<BitWidth, Elements>(input_value, output);
    } else if constexpr (BitWidth == 8) {
        return mm_pack_epi32_8<Elements>(input_value, output);
    } else if constexpr (BitWidth <= 15) {
        return mm_pack_epi32_9to15<BitWidth, Elements>(input_value, output);
    } else if constexpr (BitWidth == 16) {
        return mm_pack_epi32_16<Elements>(input_value, output);
    } else {
        return mm_pack_epi32_17to24<BitWidth, Elements>(input_value, output);
    }
}

template <u8 BitWidth, usize Elements = 8>
    requires(BitWidth >= 1 && BitWidth <= 7 && Elements > 0 && Elements <= 8)
__always_inline void mm256_pack_epi32_1to7(__m256i input_value, const std::span<u8, PACKED_BYTES(Elements, BitWidth)> output) noexcept {
    if constexpr (Elements <= 4) {
        return mm_pack_epi32_1to7<BitWidth, Elements>(_mm256_castsi256_si128(input_value), output);
    } else {
        constexpr u64 pext_mask      = PEXT_MASK_8(VALUE_MASK_32(BitWidth));
        constexpr usize packed_bytes = PACKED_BYTES(Elements, BitWidth);

        const __m256i packed16 = _mm256_packs_epi32(input_value, _mm256_setzero_si256());
        const __m256i permuted = _mm256_permute4x64_epi64(packed16, _MM_SHUFFLE(3, 1, 2, 0));
        const __m256i packed8  = _mm256_packs_epi16(permuted, _mm256_setzero_si256());
        const u64 value        = _pext_u64(static_cast<u64>(_mm256_extract_epi64(packed8, 0)), pext_mask);

        std::memcpy(output.data(), &value, packed_bytes);
    }
}

template <usize Elements = 8>
    requires(Elements > 0 && Elements <= 8)
__always_inline void mm256_pack_epi32_8(__m256i input_value, const std::span<u8, PACKED_BYTES(Elements, 8)> output) noexcept {
    if constexpr (Elements <= 4) {
        return mm_pack_epi32_8<Elements>(_mm256_castsi256_si128(input_value), output);
    } else {
        const __m256i packed16 = _mm256_packs_epi32(input_value, _mm256_setzero_si256());
        const __m256i permuted = _mm256_permute4x64_epi64(packed16, _MM_SHUFFLE(3, 1, 2, 0));
        const __m256i packed8  = _mm256_packs_epi16(permuted, _mm256_setzero_si256());
        std::memcpy(output.data(), &packed8, output.size());
    }
}

template <u8 BitWidth, usize Elements = 8>
    requires(BitWidth >= 9 && BitWidth <= 15 && Elements > 0 && Elements <= 8)
__always_inline void mm256_pack_epi32_9to15(__m256i input_value, const std::span<u8, PACKED_BYTES(Elements, BitWidth)> output) noexcept {
    if constexpr (Elements <= 4) {
        return mm_pack_epi32_9to15<BitWidth, Elements>(_mm256_castsi256_si128(input_value), output);
    } else {
        constexpr u64 pext_mask      = PEXT_MASK_16(VALUE_MASK_32(BitWidth));
        constexpr usize packed_bytes = PACKED_BYTES(Elements, BitWidth);

        constexpr u32 lower_group_bits = BitWidth * 4U;
        constexpr u32 upper_shift      = 64U - lower_group_bits;

        const __m256i packed16 = _mm256_packs_epi32(input_value, _mm256_setzero_si256());

        const u64 value0 = _pext_u64(static_cast<u64>(_mm256_extract_epi64(packed16, 0)), pext_mask);
        const u64 value1 = _pext_u64(static_cast<u64>(_mm256_extract_epi64(packed16, 2)), pext_mask);

        const std::array<u64, 2> values{
            value0 | (value1 << lower_group_bits),
            value1 >> upper_shift,
        };

        std::memcpy(output.data(), values.data(), packed_bytes);
    }
}

template <usize Elements = 8>
    requires(Elements > 0 && Elements <= 8)
__always_inline void mm256_pack_epi32_16(__m256i input_value, const std::span<u8, PACKED_BYTES(Elements, 16)> output) noexcept {
    if constexpr (Elements <= 4) {
        return mm_pack_epi32_16<Elements>(_mm256_castsi256_si128(input_value), output);
    } else {
        const __m256i packed16 = _mm256_packs_epi32(input_value, _mm256_setzero_si256());
        const __m256i permuted = _mm256_permute4x64_epi64(packed16, _MM_SHUFFLE(3, 1, 2, 0));
        std::memcpy(output.data(), &permuted, output.size());
    }
}

template <u8 BitWidth, usize Elements = 8>
    requires(BitWidth >= 17 && BitWidth <= 24 && Elements > 0 && Elements <= 8)
__always_inline void mm256_pack_epi32_17to24(__m256i input_value, const std::span<u8, PACKED_BYTES(Elements, BitWidth)> output) noexcept {
    if constexpr (Elements <= 4) {
        return mm_pack_epi32_17to24<BitWidth, Elements>(_mm256_castsi256_si128(input_value), output);
    } else {
        constexpr u64 pext_mask      = PEXT_MASK_32(VALUE_MASK_32(BitWidth));
        constexpr usize packed_bytes = PACKED_BYTES(Elements, BitWidth);

        constexpr u32 lane_bits   = BitWidth * 2U;
        constexpr u32 upper_shift = 64U - lane_bits;

        constexpr usize group_bytes              = PACKED_BYTES(4, BitWidth);
        constexpr usize second_group_byte_offset = BitWidth / 2U;

        const u64 value0 = _pext_u64(static_cast<u64>(_mm256_extract_epi64(input_value, 0)), pext_mask);
        const u64 value1 = _pext_u64(static_cast<u64>(_mm256_extract_epi64(input_value, 1)), pext_mask);
        const u64 value2 = _pext_u64(static_cast<u64>(_mm256_extract_epi64(input_value, 2)), pext_mask);
        const u64 value3 = _pext_u64(static_cast<u64>(_mm256_extract_epi64(input_value, 3)), pext_mask);

        const std::array<u64, 2> first_values{
            value0 | (value1 << lane_bits),
            value1 >> upper_shift,
        };

        const std::array<u64, 2> second_values{
            value2 | (value3 << lane_bits),
            value3 >> upper_shift,
        };

        static_assert(group_bytes <= sizeof(first_values));

        std::array<u64, 4> packed{};

        std::memcpy(packed.data(), first_values.data(), group_bytes);

        if constexpr ((BitWidth % 2U) == 0U) {
            std::memcpy(reinterpret_cast<u8*>(packed.data()) + second_group_byte_offset, second_values.data(), group_bytes);
        } else {
            auto* const packed_bytes_ptr = reinterpret_cast<u8*>(packed.data());

            const u8 preserved_nibble = packed_bytes_ptr[second_group_byte_offset] & 0x0FU;

            const std::array<u64, 2> shifted_second_group{
                (second_values[0] << 4U) | preserved_nibble,
                (second_values[1] << 4U) | (second_values[0] >> 60U),
            };

            std::memcpy(packed_bytes_ptr + second_group_byte_offset, shifted_second_group.data(), packed_bytes - second_group_byte_offset);
        }

        std::memcpy(output.data(), packed.data(), packed_bytes);
    }
}

template <u8 BitWidth, usize Elements = 8>
    requires(BitWidth >= 1 && BitWidth <= 24 && Elements > 0 && Elements <= 8)
__always_inline void mm256_pack_epi32(__m256i input_value, const std::span<u8, PACKED_BYTES(Elements, BitWidth)> output) noexcept {
    if constexpr (BitWidth <= 7) {
        return mm256_pack_epi32_1to7<BitWidth, Elements>(input_value, output);
    } else if constexpr (BitWidth == 8) {
        return mm256_pack_epi32_8<Elements>(input_value, output);
    } else if constexpr (BitWidth <= 15) {
        return mm256_pack_epi32_9to15<BitWidth, Elements>(input_value, output);
    } else if constexpr (BitWidth == 16) {
        return mm256_pack_epi32_16<Elements>(input_value, output);
    } else {
        return mm256_pack_epi32_17to24<BitWidth, Elements>(input_value, output);
    }
}

#undef VALUE_MASK_32
#undef VALUE_MASK_64
#undef PEXT_MASK_8
#undef PEXT_MASK_16
#undef PEXT_MASK_32
#undef PACKED_BYTES

}  // namespace pernix::internal::bmi2

#endif  // PERNIX_BMI2_PACKING_H
