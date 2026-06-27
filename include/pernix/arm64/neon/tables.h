#ifndef PERNIX_ARM64_NEON_TABLES_H
#define PERNIX_ARM64_NEON_TABLES_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace pernix::arm64::neon::internal {
namespace detail {
inline constexpr std::size_t neon_vector_width = 128;
inline constexpr u8 inactive_lane              = 0xff;

template <std::size_t Elements>
constexpr bool table_indices_are_valid(const std::array<u8, Elements>& table) {
    return std::ranges::all_of(table, [](const u8 index) {
        return index == inactive_lane || index < Elements;
    });
}

template <u8 BIT_WIDTH, std::size_t LANE_BITS, std::size_t ELEMENTS>
constexpr std::array<u8, ELEMENTS> make_primary_permute() {
    static_assert(LANE_BITS % 8 == 0);

    constexpr std::size_t lane_bytes = LANE_BITS / 8;
    static_assert(ELEMENTS % lane_bytes == 0);

    std::array<u8, ELEMENTS> table{};
    table.fill(inactive_lane);

    for (std::size_t entry = 0; entry < ELEMENTS / lane_bytes; ++entry) {
        const std::size_t bit_start  = entry * BIT_WIDTH;
        const std::size_t first_byte = bit_start / 8;
        const std::size_t base       = entry * lane_bytes;

        for (std::size_t lane_byte = 0; lane_byte < lane_bytes; ++lane_byte) {
            table[base + lane_byte] = static_cast<u8>(first_byte + lane_byte);
        }
    }

    return table;
}

template <u8 BIT_WIDTH, std::size_t LANE_BITS, std::size_t ELEMENTS>
constexpr std::array<u8, ELEMENTS> make_spill_permute() {
    static_assert(LANE_BITS % 8 == 0);

    constexpr std::size_t lane_bytes = LANE_BITS / 8;
    static_assert(ELEMENTS % lane_bytes == 0);

    std::array<u8, ELEMENTS> table{};
    table.fill(inactive_lane);

    for (std::size_t entry = 0; entry < ELEMENTS / lane_bytes; ++entry) {
        const std::size_t bit_start  = entry * BIT_WIDTH;
        const std::size_t first_byte = bit_start / 8;
        const std::size_t bit_offset = bit_start % 8;
        const std::size_t base       = entry * lane_bytes;

        if (bit_offset + BIT_WIDTH > LANE_BITS) {
            table[base] = static_cast<u8>(first_byte + lane_bytes);
        }
    }

    return table;
}

template <typename ShiftType, u8 BIT_WIDTH, std::size_t ELEMENTS>
constexpr std::array<ShiftType, ELEMENTS> make_shift_right() {
    std::array<ShiftType, ELEMENTS> table{};
    table.fill(0);

    for (std::size_t entry = 0; entry < ELEMENTS; ++entry) {
        const std::size_t bit_start  = entry * BIT_WIDTH;
        const std::size_t bit_offset = bit_start % 8u;

        table[entry] = -static_cast<ShiftType>(bit_offset);
    }

    return table;
}

template <typename ShiftType, u8 BIT_WIDTH, std::size_t LANE_BITS, std::size_t ELEMENTS>
constexpr std::array<ShiftType, ELEMENTS> make_shift_left_for_spill() {
    std::array<ShiftType, ELEMENTS> table{};
    table.fill(0);

    for (std::size_t entry = 0; entry < ELEMENTS; ++entry) {
        const std::size_t bit_start  = entry * BIT_WIDTH;
        const std::size_t bit_offset = bit_start % 8u;
        const bool spills            = bit_offset + BIT_WIDTH > LANE_BITS;

        table[entry] = spills ? static_cast<ShiftType>(LANE_BITS - bit_offset) : 0;
    }

    return table;
}

template <u8 BIT_WIDTH, u8 START_BIT_OFFSET, std::size_t ELEMENTS>
constexpr std::array<u8, ELEMENTS> make_contiguous_permute_32() {
    static_assert(ELEMENTS % 4 == 0);

    std::array<u8, ELEMENTS> table{};
    table.fill(inactive_lane);

    for (std::size_t entry = 0; entry < ELEMENTS / 4; ++entry) {
        const std::size_t bit_start  = START_BIT_OFFSET + entry * BIT_WIDTH;
        const std::size_t bit_end    = bit_start + BIT_WIDTH - 1;
        const std::size_t first_byte = bit_start / 8;
        const std::size_t last_byte  = bit_end / 8;
        const std::size_t base       = entry * 4;

        for (std::size_t byte = first_byte; byte <= last_byte; ++byte) {
            table[base + (byte - first_byte)] = static_cast<u8>(byte);
        }
    }

    return table;
}

template <u8 BIT_WIDTH, u8 START_BIT_OFFSET, std::size_t ELEMENTS>
constexpr std::array<i32, ELEMENTS> make_shift_right_32() {
    std::array<i32, ELEMENTS> table{};
    table.fill(0);

    for (std::size_t entry = 0; entry < ELEMENTS; ++entry) {
        const std::size_t bit_start = START_BIT_OFFSET + entry * BIT_WIDTH;

        table[entry] = -static_cast<i32>(bit_start % 8u);
    }

    return table;
}
} // namespace detail

template <u8 BIT_WIDTH, u8 VECTOR_WIDTH, u8 START_BIT_OFFSET = 0>
struct table_unpacking;

template <u8 BIT_WIDTH, u8 VECTOR_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8 && VECTOR_WIDTH == detail::neon_vector_width)
struct table_unpacking<BIT_WIDTH, VECTOR_WIDTH, 0> {
private:
    static constexpr std::size_t PERMUTE_ELEMENTS = VECTOR_WIDTH / 8;
    static constexpr std::size_t SHIFT_ELEMENTS   = VECTOR_WIDTH / 8;

public:
    static constexpr u8 bit_width = BIT_WIDTH;

    alignas(64) static constexpr std::array<u8, PERMUTE_ELEMENTS> permute1 =
        detail::make_primary_permute<BIT_WIDTH, 8, PERMUTE_ELEMENTS>();
    alignas(64) static constexpr std::array<u8, PERMUTE_ELEMENTS> permute2 =
        detail::make_spill_permute<BIT_WIDTH, 8, PERMUTE_ELEMENTS>();
    alignas(64) static constexpr std::array<i8, SHIFT_ELEMENTS> shift1 = detail::make_shift_right<i8,
        BIT_WIDTH, SHIFT_ELEMENTS>();
    alignas(64) static constexpr std::array<i8, SHIFT_ELEMENTS> shift2 =
        detail::make_shift_left_for_spill<i8, BIT_WIDTH, 8, SHIFT_ELEMENTS>();

    static_assert(PERMUTE_ELEMENTS == 16);
    static_assert(SHIFT_ELEMENTS == 16);
    static_assert(detail::table_indices_are_valid(permute1));
    static_assert(detail::table_indices_are_valid(permute2));
};

template <u8 BIT_WIDTH, u8 VECTOR_WIDTH>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16 && VECTOR_WIDTH == detail::neon_vector_width)
struct table_unpacking<BIT_WIDTH, VECTOR_WIDTH, 0> {
private:
    static constexpr std::size_t PERMUTE_ELEMENTS = VECTOR_WIDTH / 8;
    static constexpr std::size_t SHIFT_ELEMENTS   = VECTOR_WIDTH / 16;

public:
    static constexpr u8 bit_width = BIT_WIDTH;

    alignas(64) static constexpr std::array<u8, PERMUTE_ELEMENTS> permute1 =
        detail::make_primary_permute<BIT_WIDTH, 16, PERMUTE_ELEMENTS>();
    alignas(64) static constexpr std::array<u8, PERMUTE_ELEMENTS> permute2 =
        detail::make_spill_permute<BIT_WIDTH, 16, PERMUTE_ELEMENTS>();
    alignas(64) static constexpr std::array<i16, SHIFT_ELEMENTS> shift1 =
        detail::make_shift_right<i16, BIT_WIDTH, SHIFT_ELEMENTS>();
    alignas(64) static constexpr std::array<i16, SHIFT_ELEMENTS> shift2 =
        detail::make_shift_left_for_spill<i16, BIT_WIDTH, 16, SHIFT_ELEMENTS>();

    static_assert(PERMUTE_ELEMENTS == 16);
    static_assert(SHIFT_ELEMENTS == 8);
    static_assert(detail::table_indices_are_valid(permute1));
    static_assert(detail::table_indices_are_valid(permute2));
};

template <u8 BIT_WIDTH, u8 VECTOR_WIDTH, u8 START_BIT_OFFSET>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24 && VECTOR_WIDTH == detail::neon_vector_width && START_BIT_OFFSET <
             8)
struct table_unpacking<BIT_WIDTH, VECTOR_WIDTH, START_BIT_OFFSET> {
private:
    static constexpr std::size_t PERMUTE_ELEMENTS = VECTOR_WIDTH / 8;
    static constexpr std::size_t SHIFT_ELEMENTS   = VECTOR_WIDTH / 32;

public:
    static constexpr u8 bit_width = BIT_WIDTH;

    alignas(64) static constexpr std::array<u8, PERMUTE_ELEMENTS> permute =
        detail::make_contiguous_permute_32<BIT_WIDTH, START_BIT_OFFSET, PERMUTE_ELEMENTS>();
    alignas(64) static constexpr std::array<i32, SHIFT_ELEMENTS> shift =
        detail::make_shift_right_32<BIT_WIDTH, START_BIT_OFFSET, SHIFT_ELEMENTS>();

    static_assert(PERMUTE_ELEMENTS == 16);
    static_assert(SHIFT_ELEMENTS == 4);
    static_assert(detail::table_indices_are_valid(permute));
};

template <u8 BIT_WIDTH, u8 VECTOR_WIDTH, u8 START_BIT_OFFSET = 0>
struct table_packing;
} // namespace pernix::arm64::internal

#endif  // PERNIX_ARM64_NEON_TABLES_H
