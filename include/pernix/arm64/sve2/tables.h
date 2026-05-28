#ifndef PERNIX_ARM64_SVE2_TABLES_H
#define PERNIX_ARM64_SVE2_TABLES_H

#include <pernix/simd_compat.h>

#include <cstdint>
#include <vector>

namespace pernix::arm64::sve2::internal {
template <std::uint8_t BIT_WIDTH, uint8_t START_BIT_OFFSET = 0>
struct table_unpacking {
    static constexpr uint8_t bit_width = BIT_WIDTH;

    static svbool_t pg_b8() { return svptrue_b8(); }

    static svbool_t pg_b16() { return svptrue_b16(); }

    static svbool_t pg_b32() { return svptrue_b32(); }
};

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8)
struct table_unpacking<BIT_WIDTH, 0> {
    static constexpr uint8_t bit_width = BIT_WIDTH;

    static svuint8_t permute() {
        std::vector<uint8_t> table(svcntb());
        for (uint32_t lane = 0; lane < table.size(); ++lane) {
            table[lane] = static_cast<uint8_t>((lane * BIT_WIDTH) / 8u);
        }

        return svld1_u8(svptrue_b8(), table.data());
    }

    static svuint8_t spill_permute() {
        std::vector<uint8_t> table(svcntb());
        for (uint32_t lane = 0; lane < table.size(); ++lane) {
            table[lane] = static_cast<uint8_t>((lane * BIT_WIDTH) / 8u + 1u);
        }

        return svld1_u8(svptrue_b8(), table.data());
    }

    static svuint8_t shift() {
        std::vector<uint8_t> table(svcntb());
        for (uint32_t lane = 0; lane < table.size(); ++lane) {
            table[lane] = static_cast<uint8_t>((lane * BIT_WIDTH) % 8u);
        }

        return svld1_u8(svptrue_b8(), table.data());
    }

    static svuint8_t spill_shift() {
        std::vector<uint8_t> table(svcntb());
        for (uint32_t lane = 0; lane < table.size(); ++lane) {
            table[lane] = static_cast<uint8_t>(8u - ((lane * BIT_WIDTH) % 8u));
        }

        return svld1_u8(svptrue_b8(), table.data());
    }
};

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
struct table_unpacking<BIT_WIDTH, 0> {
    static constexpr uint8_t bit_width = BIT_WIDTH;

    static svuint8_t permute() {
        std::vector<uint8_t> table(svcntb());
        for (uint32_t lane = 0; lane < table.size(); ++lane) {
            const uint32_t element = lane / 2u;
            const uint32_t byte    = lane % 2u;
            const uint32_t first   = (element * BIT_WIDTH) / 8u;

            table[lane] = static_cast<uint8_t>(first + byte);
        }

        return svld1_u8(svptrue_b8(), table.data());
    }

    static svuint8_t spill_permute() {
        std::vector<uint8_t> table(svcntb());
        for (uint32_t lane = 0; lane < table.size(); ++lane) {
            const uint32_t element = lane / 2u;
            const uint32_t byte    = lane % 2u;
            const uint32_t first   = (element * BIT_WIDTH) / 8u;

            table[lane] = static_cast<uint8_t>(first + 2u + byte);
        }

        return svld1_u8(svptrue_b8(), table.data());
    }

    static svuint16_t shift() {
        std::vector<uint16_t> table(svcnth());
        for (uint32_t lane = 0; lane < table.size(); ++lane) {
            table[lane] = static_cast<uint16_t>((lane * BIT_WIDTH) % 8u);
        }

        return svld1_u16(svptrue_b16(), table.data());
    }

    static svuint16_t spill_shift() {
        std::vector<uint16_t> table(svcnth());
        for (uint32_t lane = 0; lane < table.size(); ++lane) {
            const uint32_t bit_offset = (lane * BIT_WIDTH) % 8u;
            table[lane]               = bit_offset + BIT_WIDTH > 16u ? static_cast<uint16_t>(16u - bit_offset) : uint16_t{16};
        }

        return svld1_u16(svptrue_b16(), table.data());
    }
};

template <uint8_t BIT_WIDTH, uint8_t START_BIT_OFFSET>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24 && START_BIT_OFFSET < 8)
struct table_unpacking<BIT_WIDTH, START_BIT_OFFSET> {
    static constexpr uint8_t bit_width = BIT_WIDTH;

    static svuint8_t permute() {
        std::vector<uint8_t> table(svcntb());
        for (uint32_t lane = 0; lane < table.size(); ++lane) {
            const uint32_t element = lane / 4u;
            const uint32_t byte    = lane % 4u;
            const uint32_t first   = (START_BIT_OFFSET + element * BIT_WIDTH) / 8u;

            table[lane] = static_cast<uint8_t>(first + byte);
        }

        return svld1_u8(svptrue_b8(), table.data());
    }

    static svuint32_t shift() {
        std::vector<uint32_t> table(svcntw());
        for (uint32_t lane = 0; lane < table.size(); ++lane) {
            table[lane] = (START_BIT_OFFSET + lane * BIT_WIDTH) % 8u;
        }

        return svld1_u32(svptrue_b32(), table.data());
    }
};
}  // namespace pernix::arm64::sve2::internal

#endif  // PERNIX_ARM64_SVE2_TABLES_H
