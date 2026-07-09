#ifndef PERNIX_ARM64_NEON_UNPACKING_H
#define PERNIX_ARM64_NEON_UNPACKING_H

#include <pernix/arm64/neon/tables.h>
#include <pernix/simd_compat.h>

using namespace pernix::arm64::neon::internal;

namespace pernix::arm64::neon::internal::b128 {
template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8)
__always_inline int8x16_t neon_unpack_epi8_1to8(const uint8x16_t& input) {
    if constexpr (BIT_WIDTH == 8) {
        return vreinterpretq_s8_u8(input);
    } else if constexpr (BIT_WIDTH == 1) {
        using tables = table_unpacking<BIT_WIDTH, 128>;

        const uint8x16_t permuted_u8 = vqtbl1q_u8(input, vld1q_u8(tables::permute1.data()));
        const uint8x16_t shifted     = vshlq_u8(permuted_u8, vld1q_s8(tables::shift1.data()));

        return vreinterpretq_s8_u8(vandq_u8(shifted, vdupq_n_u8(1)));
    } else {
        using tables = table_unpacking<BIT_WIDTH, 128>;

        const uint8x16_t permuted_u8 = vqtbl1q_u8(input, vld1q_u8(tables::permute1.data()));

        uint8x16_t shifted = vshlq_u8(permuted_u8, vld1q_s8(tables::shift1.data()));

        if constexpr (BIT_WIDTH == 3 || BIT_WIDTH == 5 || BIT_WIDTH == 6 || BIT_WIDTH == 7) {
            const uint8x16_t permuted2_u8 = vqtbl1q_u8(input, vld1q_u8(tables::permute2.data()));

            shifted = vorrq_u8(shifted, vshlq_u8(permuted2_u8, vld1q_s8(tables::shift2.data())));
        }

        constexpr int shift = 8 - BIT_WIDTH;
        shifted             = vshlq_n_u8(shifted, shift);

        if constexpr (SIGN_VALUES) {
            return vshlq_s8(vreinterpretq_s8_u8(shifted), vdupq_n_s8(-shift));
        } else {
            return vreinterpretq_s8_u8(vshlq_u8(shifted, vdupq_n_s8(-shift)));
        }
    }
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
__always_inline int16x8_t neon_unpack_epi16_9to16(const uint16x8_t& input) {
    if constexpr (BIT_WIDTH == 16) {
        return vreinterpretq_s16_u16(input);
    } else {
        using tables = table_unpacking<BIT_WIDTH, 128>;

        const uint8x16_t input_u8 = vreinterpretq_u8_u16(input);

        const uint8x16_t permuted1_u8 = vqtbl1q_u8(input_u8, vld1q_u8(tables::permute1.data()));

        uint16x8_t shifted = vshlq_u16(vreinterpretq_u16_u8(permuted1_u8), vld1q_s16(tables::shift1.data()));

        if constexpr (BIT_WIDTH == 11 || BIT_WIDTH == 13 || BIT_WIDTH == 14 || BIT_WIDTH == 15) {
            const uint8x16_t permuted2_u8 = vqtbl1q_u8(input_u8, vld1q_u8(tables::permute2.data()));

            const uint16x8_t shifted2 = vshlq_u16(vreinterpretq_u16_u8(permuted2_u8), vld1q_s16(tables::shift2.data()));

            shifted = vorrq_u16(shifted, shifted2);
        }

        constexpr int shift = 16 - BIT_WIDTH;
        shifted             = vshlq_n_u16(shifted, shift);

        if constexpr (SIGN_VALUES) {
            return vshlq_s16(vreinterpretq_s16_u16(shifted), vdupq_n_s16(-shift));
        } else {
            return vreinterpretq_s16_u16(vshlq_u16(shifted, vdupq_n_s16(-shift)));
        }
    }
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u8 START_BIT_OFFSET = 0>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24)
__always_inline int32x4_t neon_unpack_epi32_17to24(const uint32x4_t& input) {
    using tables = table_unpacking<BIT_WIDTH, 128, START_BIT_OFFSET>;

    const uint8x16_t input_8 = vreinterpretq_u8_u32(input);

    const uint8x16_t permuted_u8 = vqtbl1q_u8(input_8, vld1q_u8(tables::permute.data()));

    const uint32x4_t value = vshlq_u32(vreinterpretq_u32_u8(permuted_u8), vld1q_s32(tables::shift.data()));

    if constexpr (SIGN_VALUES) {
        constexpr int sign_shift = 32 - BIT_WIDTH;
        return vshrq_n_s32(vreinterpretq_s32_u32(vshlq_n_u32(value, sign_shift)), sign_shift);
    } else {
        constexpr u32 mask = (u32{1} << BIT_WIDTH) - 1u;
        return vreinterpretq_s32_u32(vandq_u32(value, vdupq_n_u32(mask)));
    }
}
}  // namespace pernix::arm64::neon::internal::b128

#endif  // PERNIX_ARM64_NEON_UNPACKING_H
