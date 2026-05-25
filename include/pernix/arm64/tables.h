#ifndef PERNIX_ARM64_TABLES_H
#define PERNIX_ARM64_TABLES_H

#include <array>
#include <cstddef>
#include <cstdint>

namespace pernix::arm64::internal {
namespace detail {
inline constexpr std::size_t neon_vector_width = 128;
inline constexpr uint8_t inactive_lane         = 0xff;

template <std::size_t Elements>
constexpr bool table_indices_are_valid(const std::array<uint8_t, Elements>& table) {
    for (const uint8_t index : table) {
        if (index != inactive_lane && index >= Elements) {
            return false;
        }
    }

    return true;
}

template <uint8_t BIT_WIDTH, std::size_t LANE_BITS, std::size_t ELEMENTS>
constexpr std::array<uint8_t, ELEMENTS> make_primary_permute() {
    static_assert(LANE_BITS % 8 == 0);

    constexpr std::size_t lane_bytes = LANE_BITS / 8;
    static_assert(ELEMENTS % lane_bytes == 0);

    std::array<uint8_t, ELEMENTS> table{};
    table.fill(inactive_lane);

    for (std::size_t entry = 0; entry < ELEMENTS / lane_bytes; ++entry) {
        const std::size_t bit_start  = entry * BIT_WIDTH;
        const std::size_t first_byte = bit_start / 8;
        const std::size_t base       = entry * lane_bytes;

        for (std::size_t lane_byte = 0; lane_byte < lane_bytes; ++lane_byte) {
            table[base + lane_byte] = static_cast<uint8_t>(first_byte + lane_byte);
        }
    }

    return table;
}

template <uint8_t BIT_WIDTH, std::size_t LANE_BITS, std::size_t ELEMENTS>
constexpr std::array<uint8_t, ELEMENTS> make_spill_permute() {
    static_assert(LANE_BITS % 8 == 0);

    constexpr std::size_t lane_bytes = LANE_BITS / 8;
    static_assert(ELEMENTS % lane_bytes == 0);

    std::array<uint8_t, ELEMENTS> table{};
    table.fill(inactive_lane);

    for (std::size_t entry = 0; entry < ELEMENTS / lane_bytes; ++entry) {
        const std::size_t bit_start  = entry * BIT_WIDTH;
        const std::size_t first_byte = bit_start / 8;
        const std::size_t bit_offset = bit_start % 8;
        const std::size_t base       = entry * lane_bytes;

        if (bit_offset + BIT_WIDTH > LANE_BITS) {
            table[base] = static_cast<uint8_t>(first_byte + lane_bytes);
        }
    }

    return table;
}

template <typename ShiftType, uint8_t BIT_WIDTH, std::size_t ELEMENTS>
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

template <typename ShiftType, uint8_t BIT_WIDTH, std::size_t LANE_BITS, std::size_t ELEMENTS>
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

template <uint8_t BIT_WIDTH, uint8_t START_BIT_OFFSET, std::size_t ELEMENTS>
constexpr std::array<uint8_t, ELEMENTS> make_contiguous_permute_32() {
    static_assert(ELEMENTS % 4 == 0);

    std::array<uint8_t, ELEMENTS> table{};
    table.fill(inactive_lane);

    for (std::size_t entry = 0; entry < ELEMENTS / 4; ++entry) {
        const std::size_t bit_start  = START_BIT_OFFSET + entry * BIT_WIDTH;
        const std::size_t bit_end    = bit_start + BIT_WIDTH - 1;
        const std::size_t first_byte = bit_start / 8;
        const std::size_t last_byte  = bit_end / 8;
        const std::size_t base       = entry * 4;

        for (std::size_t byte = first_byte; byte <= last_byte; ++byte) {
            table[base + (byte - first_byte)] = static_cast<uint8_t>(byte);
        }
    }

    return table;
}

template <uint8_t BIT_WIDTH, uint8_t START_BIT_OFFSET, std::size_t ELEMENTS>
constexpr std::array<int32_t, ELEMENTS> make_shift_right_32() {
    std::array<int32_t, ELEMENTS> table{};
    table.fill(0);

    for (std::size_t entry = 0; entry < ELEMENTS; ++entry) {
        const std::size_t bit_start = START_BIT_OFFSET + entry * BIT_WIDTH;

        table[entry] = -static_cast<int32_t>(bit_start % 8u);
    }

    return table;
}
}  // namespace detail

template <uint8_t BIT_WIDTH, uint8_t VECTOR_WIDTH, uint8_t START_BIT_OFFSET = 0>
struct table_unpacking;

template <uint8_t BIT_WIDTH, uint8_t VECTOR_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8 && VECTOR_WIDTH == detail::neon_vector_width)
struct table_unpacking<BIT_WIDTH, VECTOR_WIDTH, 0> {
private:
    static constexpr std::size_t PERMUTE_ELEMENTS = VECTOR_WIDTH / 8;
    static constexpr std::size_t SHIFT_ELEMENTS   = VECTOR_WIDTH / 8;

public:
    static constexpr uint8_t bit_width = BIT_WIDTH;

    alignas(64) static constexpr std::array<uint8_t, PERMUTE_ELEMENTS> permute1 =
        detail::make_primary_permute<BIT_WIDTH, 8, PERMUTE_ELEMENTS>();
    alignas(64) static constexpr std::array<uint8_t, PERMUTE_ELEMENTS> permute2 =
        detail::make_spill_permute<BIT_WIDTH, 8, PERMUTE_ELEMENTS>();
    alignas(64) static constexpr std::array<int8_t, SHIFT_ELEMENTS> shift1 = detail::make_shift_right<int8_t, BIT_WIDTH, SHIFT_ELEMENTS>();
    alignas(64) static constexpr std::array<int8_t, SHIFT_ELEMENTS> shift2 =
        detail::make_shift_left_for_spill<int8_t, BIT_WIDTH, 8, SHIFT_ELEMENTS>();

    static_assert(PERMUTE_ELEMENTS == 16);
    static_assert(SHIFT_ELEMENTS == 16);
    static_assert(detail::table_indices_are_valid(permute1));
    static_assert(detail::table_indices_are_valid(permute2));
};

template <uint8_t BIT_WIDTH, uint8_t VECTOR_WIDTH>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16 && VECTOR_WIDTH == detail::neon_vector_width)
struct table_unpacking<BIT_WIDTH, VECTOR_WIDTH, 0> {
private:
    static constexpr std::size_t PERMUTE_ELEMENTS = VECTOR_WIDTH / 8;
    static constexpr std::size_t SHIFT_ELEMENTS   = VECTOR_WIDTH / 16;

public:
    static constexpr uint8_t bit_width = BIT_WIDTH;

    alignas(64) static constexpr std::array<uint8_t, PERMUTE_ELEMENTS> permute1 =
        detail::make_primary_permute<BIT_WIDTH, 16, PERMUTE_ELEMENTS>();
    alignas(64) static constexpr std::array<uint8_t, PERMUTE_ELEMENTS> permute2 =
        detail::make_spill_permute<BIT_WIDTH, 16, PERMUTE_ELEMENTS>();
    alignas(64) static constexpr std::array<int16_t, SHIFT_ELEMENTS> shift1 =
        detail::make_shift_right<int16_t, BIT_WIDTH, SHIFT_ELEMENTS>();
    alignas(64) static constexpr std::array<int16_t, SHIFT_ELEMENTS> shift2 =
        detail::make_shift_left_for_spill<int16_t, BIT_WIDTH, 16, SHIFT_ELEMENTS>();

    static_assert(PERMUTE_ELEMENTS == 16);
    static_assert(SHIFT_ELEMENTS == 8);
    static_assert(detail::table_indices_are_valid(permute1));
    static_assert(detail::table_indices_are_valid(permute2));
};

template <uint8_t BIT_WIDTH, uint8_t VECTOR_WIDTH, uint8_t START_BIT_OFFSET>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24 && VECTOR_WIDTH == detail::neon_vector_width && START_BIT_OFFSET < 8)
struct table_unpacking<BIT_WIDTH, VECTOR_WIDTH, START_BIT_OFFSET> {
private:
    static constexpr std::size_t PERMUTE_ELEMENTS = VECTOR_WIDTH / 8;
    static constexpr std::size_t SHIFT_ELEMENTS   = VECTOR_WIDTH / 32;

public:
    static constexpr uint8_t bit_width = BIT_WIDTH;

    alignas(64) static constexpr std::array<uint8_t, PERMUTE_ELEMENTS> permute =
        detail::make_contiguous_permute_32<BIT_WIDTH, START_BIT_OFFSET, PERMUTE_ELEMENTS>();
    alignas(64) static constexpr std::array<int32_t, SHIFT_ELEMENTS> shift =
        detail::make_shift_right_32<BIT_WIDTH, START_BIT_OFFSET, SHIFT_ELEMENTS>();

    static_assert(PERMUTE_ELEMENTS == 16);
    static_assert(SHIFT_ELEMENTS == 4);
    static_assert(detail::table_indices_are_valid(permute));
};
}  // namespace pernix::arm64::internal

#endif  // PERNIX_ARM64_TABLES_H
