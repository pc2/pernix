#ifndef PERNIX_ARM64_SVE2_UNPACKING_H
#define PERNIX_ARM64_SVE2_UNPACKING_H

#include <pernix/simd_compat.h>

#include "tables.h"

namespace pernix::arm64::sve2::internal {
template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8)
__always_inline svint8_t sve2_unpack_epi8_1to8(const svuint8_t input, const svuint8_t permute, const svuint8_t shift,
                                               const svuint8_t spill_permute, const svuint8_t spill_shift) {
    if constexpr (BIT_WIDTH == 8) {
        return svreinterpret_s8(input);
    } else {
        const svbool_t pg = svptrue_b8();

        const svuint8_t permuted = svtbl_u8(input, permute);
        svuint8_t unpacked       = svlsr_u8_x(pg, permuted, shift);

        if constexpr (BIT_WIDTH == 3 || BIT_WIDTH == 5 || BIT_WIDTH == 6 || BIT_WIDTH == 7) {
            const svuint8_t spill_permuted_values = svtbl_u8(input, spill_permute);
            const svuint8_t spill_shifted         = svlsl_u8_x(pg, spill_permuted_values, spill_shift);
            unpacked                              = svorr_u8_x(pg, unpacked, spill_shifted);
        }

        if constexpr (BIT_WIDTH == 1) {
            unpacked = svand_n_u8_x(pg, unpacked, 1);
            return svreinterpret_s8(unpacked);
        } else {
            constexpr int sign_shift = 8 - BIT_WIDTH;

            unpacked = svlsl_n_u8_x(pg, unpacked, sign_shift);

            if constexpr (SIGN_VALUES) {
                return svasr_n_s8_x(pg, svreinterpret_s8_u8(unpacked), sign_shift);
            } else {
                return svreinterpret_s8_u8(svlsr_n_u8_x(pg, unpacked, sign_shift));
            }
        }
    }
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
__always_inline svint16_t sve2_unpack_epi16_9to16(const svuint16_t input, const svuint8_t permute, const svuint16_t shift,
                                                  const svuint8_t spill_permute, const svuint16_t spill_shift) {
    if constexpr (BIT_WIDTH == 16) {
        return svreinterpret_s16(input);
    } else {
        const svbool_t pg = svptrue_b16();

        const svuint8_t permuted = svtbl_u8(svreinterpret_u8_u16(input), permute);
        svuint16_t shifted       = svlsr_u16_x(pg, svreinterpret_u16_u8(permuted), shift);

        if constexpr (BIT_WIDTH == 11 || BIT_WIDTH == 13 || BIT_WIDTH == 14 || BIT_WIDTH == 15) {
            const svuint8_t spill_permuted_values = svtbl_u8(svreinterpret_u8_u16(input), spill_permute);
            const svuint16_t spill_shifted        = svlsl_u16_x(pg, svreinterpret_u16_u8(spill_permuted_values), spill_shift);
            shifted                               = svorr_u16_x(pg, shifted, spill_shifted);
        }

        constexpr int sign_shift = 16 - BIT_WIDTH;
        shifted                  = svlsl_n_u16_x(pg, shifted, sign_shift);

        if constexpr (SIGN_VALUES) {
            return svasr_n_s16_x(pg, svreinterpret_s16_u16(shifted), sign_shift);
        } else {
            return svreinterpret_s16_u16(svlsr_n_u16_x(pg, shifted, sign_shift));
        }
    }
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u8 START_BIT_OFFSET = 0>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24)
__always_inline svint32_t sve2_unpack_epi32_17to24(const svuint8_t input) {
    using table = table_unpacking<BIT_WIDTH, START_BIT_OFFSET>;

    const svbool_t pg         = svptrue_b32();
    const svuint8_t permuted  = svtbl_u8(input, table::permute());
    const svuint32_t unpacked = svlsr_u32_x(pg, svreinterpret_u32_u8(permuted), table::shift());

    if constexpr (SIGN_VALUES) {
        constexpr int sign_shift = 32 - BIT_WIDTH;
        return svasr_n_s32_x(pg, svreinterpret_s32_u32(svlsl_n_u32_x(pg, unpacked, sign_shift)), sign_shift);
    } else {
        constexpr u32 mask = (u32{1} << BIT_WIDTH) - 1u;
        return svreinterpret_s32_u32(svand_n_u32_x(pg, unpacked, mask));
    }
}
}  // namespace pernix::arm64::sve2::internal

#endif  // PERNIX_ARM64_SVE2_UNPACKING_H
