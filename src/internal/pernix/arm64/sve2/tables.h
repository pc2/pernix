#ifndef PERNIX_ARM64_SVE2_TABLES_H
#define PERNIX_ARM64_SVE2_TABLES_H

#include <pernix/simd_compat.h>

namespace pernix::arm64::sve2::internal {
template <u8 BIT_WIDTH, u8 START_BIT_OFFSET = 0>
struct table_unpacking {
    static constexpr u8 bit_width = BIT_WIDTH;

    static svbool_t pg_b8() { return svptrue_b8(); }

    static svbool_t pg_b16() { return svptrue_b16(); }

    static svbool_t pg_b32() { return svptrue_b32(); }
};

template <u8 BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8)
struct table_unpacking<BIT_WIDTH, 0> {
    static constexpr u8 bit_width = BIT_WIDTH;

    static svuint8_t permute() {
        const svbool_t pg = svptrue_b8();
        return svlsr_n_u8_x(pg, svindex_u8(0, BIT_WIDTH), 3);
    }

    static svuint8_t spill_permute() {
        const svbool_t pg = svptrue_b8();
        return svadd_n_u8_x(pg, permute(), 1);
    }

    static svuint8_t shift() {
        const svbool_t pg = svptrue_b8();
        return svand_n_u8_x(pg, svindex_u8(0, BIT_WIDTH), 7);
    }

    static svuint8_t spill_shift() {
        const svbool_t pg = svptrue_b8();
        return svsub_u8_x(pg, svdup_n_u8(8), shift());
    }
};

template <u8 BIT_WIDTH>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
struct table_unpacking<BIT_WIDTH, 0> {
    static constexpr u8 bit_width = BIT_WIDTH;

    static svuint8_t permute() {
        const svbool_t pg    = svptrue_b8();
        const svuint8_t lane = svindex_u8(0, 1);
        const svuint8_t elem = svlsr_n_u8_x(pg, lane, 1);
        const svuint8_t byte = svand_n_u8_x(pg, lane, 1);

        svuint8_t first;
        if constexpr (BIT_WIDTH == 16) {
            first = svlsl_n_u8_x(pg, elem, 1);
        } else {
            constexpr u8 extra_bits = BIT_WIDTH - 8u;
            const svuint8_t high    = svmul_n_u8_x(pg, svlsr_n_u8_x(pg, elem, 3), extra_bits);
            const svuint8_t low     = svlsr_n_u8_x(pg, svmul_n_u8_x(pg, svand_n_u8_x(pg, elem, 7), extra_bits), 3);
            first                   = svadd_u8_x(pg, elem, svadd_u8_x(pg, high, low));
        }

        return svadd_u8_x(pg, first, byte);
    }

    static svuint8_t spill_permute() {
        const svbool_t pg = svptrue_b8();
        return svadd_n_u8_x(pg, permute(), 2);
    }

    static svuint16_t shift() {
        const svbool_t pg = svptrue_b16();
        return svand_n_u16_x(pg, svmul_n_u16_x(pg, svindex_u16(0, 1), BIT_WIDTH), 7);
    }

    static svuint16_t spill_shift() {
        const svbool_t pg          = svptrue_b16();
        const svuint16_t bit_shift = shift();
        const svuint16_t spill     = svsub_u16_x(pg, svdup_n_u16(16), bit_shift);
        return svsel_u16(svcmpgt_n_u16(pg, bit_shift, 16u - BIT_WIDTH), spill, svdup_n_u16(16));
    }
};

template <u8 BIT_WIDTH, u8 START_BIT_OFFSET>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24 && START_BIT_OFFSET < 8)
struct table_unpacking<BIT_WIDTH, START_BIT_OFFSET> {
    static constexpr u8 bit_width = BIT_WIDTH;

    static svuint8_t permute() {
        const svbool_t pg    = svptrue_b8();
        const svuint8_t lane = svindex_u8(0, 1);
        const svuint8_t elem = svlsr_n_u8_x(pg, lane, 2);
        const svuint8_t byte = svand_n_u8_x(pg, lane, 3);

        svuint8_t first = svmul_n_u8_x(pg, elem, BIT_WIDTH / 8u);
        if constexpr (BIT_WIDTH % 8u != 0) {
            constexpr u8 extra_bits  = BIT_WIDTH % 8u;
            const svuint8_t high     = svmul_n_u8_x(pg, svlsr_n_u8_x(pg, elem, 3), extra_bits);
            const svuint8_t low_bits =
                svadd_n_u8_x(pg, svmul_n_u8_x(pg, svand_n_u8_x(pg, elem, 7), extra_bits), START_BIT_OFFSET);
            first = svadd_u8_x(pg, first, svadd_u8_x(pg, high, svlsr_n_u8_x(pg, low_bits, 3)));
        }

        return svadd_u8_x(pg, first, byte);
    }

    static svuint32_t shift() {
        const svbool_t pg = svptrue_b32();
        return svand_n_u32_x(pg, svadd_n_u32_x(pg, svmul_n_u32_x(pg, svindex_u32(0, 1), BIT_WIDTH),
                                               START_BIT_OFFSET), 7);
    }
};
} // namespace pernix::arm64::sve2::internal

#endif  // PERNIX_ARM64_SVE2_TABLES_H
