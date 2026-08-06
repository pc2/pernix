#ifndef PERNIX_UNPACK_TABLES_H
#define PERNIX_UNPACK_TABLES_H

#include <pernix/compat.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <ranges>

namespace pernix::detail {
inline constexpr u8 inactive_lane = 0xff;

namespace internal {
template <usize Elements>
constexpr bool table_indices_are_valid(const std::array<u8, Elements>& table) {
    return std::ranges::all_of(table, [](const u8 index) { return index == inactive_lane || index < Elements; });
}

template <u8 BitWidth, usize IntWidth, usize Elements>
constexpr std::array<u8, Elements> make_primary_permute() {
    static_assert(Elements % IntWidth == 0);

    std::array<u8, Elements> table{};
    table.fill(inactive_lane);

    for (usize entry = 0; entry < Elements / IntWidth; ++entry) {
        const usize bit_start  = entry * BitWidth;
        const usize first_byte = bit_start / 8;
        const usize base       = entry * IntWidth;

        for (usize lane_byte = 0; lane_byte < IntWidth; ++lane_byte) {
            table[base + lane_byte] = static_cast<u8>(first_byte + lane_byte);
        }
    }

    return table;
}

template <u8 BitWidth, usize IntWidth, usize Elements>
constexpr std::array<u8, Elements> make_spill_permute() {
    static_assert(Elements % IntWidth == 0);

    std::array<u8, Elements> table{};
    table.fill(inactive_lane);

    for (usize entry = 0; entry < Elements / IntWidth; ++entry) {
        const usize bit_start  = entry * BitWidth;
        const usize first_byte = bit_start / 8;
        const usize bit_offset = bit_start % 8;
        const usize base       = entry * IntWidth;

        if (bit_offset + BitWidth > IntWidth * 8) {
            table[base] = static_cast<u8>(first_byte + IntWidth);
        }
    }

    return table;
}

template <typename ShiftType, u8 BitWidth, usize Elements>
constexpr std::array<ShiftType, Elements> make_shift_right_magnitude() {
    std::array<ShiftType, Elements> table{};

    for (usize entry = 0; entry < Elements; ++entry) {
        const usize bit_offset = (entry * BitWidth) % 8;
        table[entry]           = static_cast<ShiftType>(bit_offset);
    }

    return table;
}

template <typename ShiftType, u8 BitWidth, usize Elements>
constexpr std::array<ShiftType, Elements> make_shift_right() {
    auto table = make_shift_right_magnitude<ShiftType, BitWidth, Elements>();
    for (ShiftType& shift : table) {
        shift = -shift;
    }
    return table;
}

template <typename ShiftType, u8 BitWidth, usize IntWidth, usize Elements>
constexpr std::array<ShiftType, Elements> make_shift_left_for_spill() {
    std::array<ShiftType, Elements> table{};

    for (usize entry = 0; entry < Elements; ++entry) {
        const usize bit_offset = (entry * BitWidth) % 8;
        const bool spills      = bit_offset + BitWidth > IntWidth * 8;
        table[entry]           = spills ? static_cast<ShiftType>(IntWidth * 8 - bit_offset) : 0;
    }

    return table;
}
}  // namespace internal

template <typename ShiftType, u8 BitWidth, usize VectorBytes>
struct unpack_table;

template <typename ShiftType, u8 BitWidth, usize VectorBytes>
    requires std::signed_integral<ShiftType> && (sizeof(ShiftType) == sizeof(u8)) && (BitWidth >= 1 && BitWidth <= 8) && (VectorBytes > 0)
struct unpack_table<ShiftType, BitWidth, VectorBytes> {
    static constexpr usize shift_elements = VectorBytes / sizeof(ShiftType);
    static constexpr u8 bit_width         = BitWidth;

    alignas(64) static constexpr std::array<u8, VectorBytes> primary_permute =
        internal::make_primary_permute<BitWidth, sizeof(ShiftType), VectorBytes>();
    alignas(64) static constexpr std::array<u8, VectorBytes> spill_permute =
        internal::make_spill_permute<BitWidth, sizeof(ShiftType), VectorBytes>();
    alignas(64) static constexpr std::array<ShiftType, shift_elements> right_shift =
        internal::make_shift_right<ShiftType, BitWidth, shift_elements>();
    alignas(64) static constexpr std::array<ShiftType, shift_elements> right_shift_magnitude =
        internal::make_shift_right_magnitude<ShiftType, BitWidth, shift_elements>();
    alignas(64) static constexpr std::array<ShiftType, shift_elements> left_shift_for_spill =
        internal::make_shift_left_for_spill<ShiftType, BitWidth, sizeof(ShiftType), shift_elements>();
    static constexpr bool has_spill = std::ranges::any_of(left_shift_for_spill, [](const ShiftType shift) { return shift != 0; });

    static_assert(internal::table_indices_are_valid(primary_permute));
    static_assert(internal::table_indices_are_valid(spill_permute));
};

template <typename ShiftType, u8 BitWidth, usize VectorBytes>
    requires std::signed_integral<ShiftType> && (sizeof(ShiftType) == sizeof(u16)) && (BitWidth >= 9 && BitWidth <= 16) && (VectorBytes > 0)
struct unpack_table<ShiftType, BitWidth, VectorBytes> {
    static constexpr usize shift_elements = VectorBytes / sizeof(ShiftType);
    static constexpr u8 bit_width         = BitWidth;

    alignas(64) static constexpr std::array<u8, VectorBytes> primary_permute =
        internal::make_primary_permute<BitWidth, sizeof(ShiftType), VectorBytes>();
    alignas(64) static constexpr std::array<u8, VectorBytes> spill_permute =
        internal::make_spill_permute<BitWidth, sizeof(ShiftType), VectorBytes>();
    alignas(64) static constexpr std::array<ShiftType, shift_elements> right_shift =
        internal::make_shift_right<ShiftType, BitWidth, shift_elements>();
    alignas(64) static constexpr std::array<ShiftType, shift_elements> right_shift_magnitude =
        internal::make_shift_right_magnitude<ShiftType, BitWidth, shift_elements>();
    alignas(64) static constexpr std::array<ShiftType, shift_elements> left_shift_for_spill =
        internal::make_shift_left_for_spill<ShiftType, BitWidth, sizeof(ShiftType), shift_elements>();
    static constexpr bool has_spill = std::ranges::any_of(left_shift_for_spill, [](const ShiftType shift) { return shift != 0; });

    static_assert(internal::table_indices_are_valid(primary_permute));
    static_assert(internal::table_indices_are_valid(spill_permute));
};

template <typename ShiftType, u8 BitWidth, usize VectorBytes>
    requires std::signed_integral<ShiftType> && (sizeof(ShiftType) == sizeof(u32)) && (BitWidth >= 17 && BitWidth <= 24) &&
             (VectorBytes > 0)
struct unpack_table<ShiftType, BitWidth, VectorBytes> {
    static constexpr usize shift_elements = VectorBytes / sizeof(ShiftType);
    static constexpr u8 bit_width         = BitWidth;

    alignas(64) static constexpr std::array<u8, VectorBytes> primary_permute =
        internal::make_primary_permute<BitWidth, sizeof(ShiftType), VectorBytes>();
    alignas(64) static constexpr std::array<ShiftType, shift_elements> right_shift =
        internal::make_shift_right<ShiftType, BitWidth, shift_elements>();
    alignas(64) static constexpr std::array<ShiftType, shift_elements> right_shift_magnitude =
        internal::make_shift_right_magnitude<ShiftType, BitWidth, shift_elements>();

    static_assert(internal::table_indices_are_valid(primary_permute));
};
}  // namespace pernix::detail

#endif  // PERNIX_UNPACK_TABLES_H
