#include <pernix/detail/unpack_tables.h>

#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

namespace {
template <u8 BitWidth>
using shift_type_for = std::conditional_t<(BitWidth <= 8), i8, std::conditional_t<(BitWidth <= 16), i16, i32>>;

template <typename ShiftType, u8 BitWidth, usize VectorBytes>
constexpr bool validate_unpack_table() {
    using table = pernix::detail::unpack_table<ShiftType, BitWidth, VectorBytes>;

    constexpr usize lane_bytes = sizeof(ShiftType);
    constexpr usize lanes      = VectorBytes / lane_bytes;
    bool expected_has_spill    = false;

    if (table::primary_permute.size() != VectorBytes || table::right_shift.size() != lanes ||
        table::right_shift_magnitude.size() != lanes) {
        return false;
    }

    for (usize entry = 0; entry < lanes; ++entry) {
        const usize bit_start  = entry * BitWidth;
        const usize first_byte = bit_start / 8;
        const usize bit_offset = bit_start % 8;
        const usize base       = entry * lane_bytes;

        for (usize byte = 0; byte < lane_bytes; ++byte) {
            if (table::primary_permute[base + byte] != first_byte + byte) {
                return false;
            }
        }
        if (table::right_shift[entry] != -static_cast<ShiftType>(bit_offset)) {
            return false;
        }
        if (table::right_shift_magnitude[entry] != static_cast<ShiftType>(bit_offset)) {
            return false;
        }

        if constexpr (BitWidth <= 16) {
            const bool spills = bit_offset + BitWidth > lane_bytes * 8;
            expected_has_spill |= spills;
            for (usize byte = 0; byte < lane_bytes; ++byte) {
                const u8 expected = spills && byte == 0 ? static_cast<u8>(first_byte + lane_bytes) : pernix::detail::inactive_lane;
                if (table::spill_permute[base + byte] != expected) {
                    return false;
                }
            }

            const ShiftType expected_shift = spills ? static_cast<ShiftType>((lane_bytes * 8) - bit_offset) : 0;
            if (table::left_shift_for_spill[entry] != expected_shift) {
                return false;
            }
        }
    }

    if constexpr (BitWidth <= 16) {
        if (table::has_spill != expected_has_spill) {
            return false;
        }
    }

    return true;
}

template <usize VectorBytes, usize... Indices>
constexpr bool validate_all_bit_widths(std::index_sequence<Indices...>) {
    return (validate_unpack_table<shift_type_for<static_cast<u8>(Indices + 1)>, static_cast<u8>(Indices + 1), VectorBytes>() && ...);
}

static_assert(validate_all_bit_widths<16>(std::make_index_sequence<24>{}));
static_assert(validate_all_bit_widths<32>(std::make_index_sequence<24>{}));
static_assert(validate_all_bit_widths<64>(std::make_index_sequence<24>{}));
}  // namespace

TEST(UnpackTablesTest, PreservesSignedShiftCountsForNeonCompatibleConsumers) {
    using byte_table = pernix::detail::unpack_table<i8, 5, 16>;
    EXPECT_EQ(byte_table::right_shift[1], -5);
    EXPECT_EQ(byte_table::right_shift_magnitude[1], 5);
    EXPECT_EQ(byte_table::left_shift_for_spill[1], 3);

    using halfword_table = pernix::detail::unpack_table<i16, 13, 16>;
    EXPECT_EQ(halfword_table::right_shift[1], -5);
    EXPECT_EQ(halfword_table::right_shift_magnitude[1], 5);
    EXPECT_EQ(halfword_table::left_shift_for_spill[1], 11);

    using word_table = pernix::detail::unpack_table<i32, 19, 16>;
    EXPECT_EQ(word_table::right_shift[1], -3);
    EXPECT_EQ(word_table::right_shift_magnitude[1], 3);
}
