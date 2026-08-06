#include <pernix/detail/bits.h>
#include <pernix/detail/pack_tables.h>

#include <array>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

namespace {
static_assert(pernix::detail::low_bit_mask<1>() == 0x00000001U);
static_assert(pernix::detail::low_bit_mask<7>() == 0x0000007fU);
static_assert(pernix::detail::low_bit_mask<8>() == 0x000000ffU);
static_assert(pernix::detail::low_bit_mask<9>() == 0x000001ffU);
static_assert(pernix::detail::low_bit_mask<15>() == 0x00007fffU);
static_assert(pernix::detail::low_bit_mask<16>() == 0x0000ffffU);
static_assert(pernix::detail::low_bit_mask<17>() == 0x0001ffffU);
static_assert(pernix::detail::low_bit_mask<24>() == 0x00ffffffU);

template <typename LaneType, u8 BitWidth, usize VectorBytes>
constexpr bool validate_pack_table() {
    using table         = pernix::detail::pack_table<LaneType, BitWidth, VectorBytes>;
    using unsigned_type = std::make_unsigned_t<LaneType>;

    constexpr usize elements           = VectorBytes / sizeof(LaneType);
    constexpr usize lane_bits          = sizeof(LaneType) * 8;
    constexpr unsigned_type value_mask = (unsigned_type{1} << BitWidth) - 1;
    constexpr unsigned_type lane_mask  = static_cast<unsigned_type>(~unsigned_type{0});

    std::array<std::array<LaneType, elements>, table::contributor_count> reference_primary_permutes{};
    std::array<std::array<LaneType, elements>, table::contributor_count> reference_primary_shifts{};
    std::array<u64, table::contributor_count> reference_primary_masks{};
    std::array<LaneType, elements> reference_spill_permute{};
    std::array<LaneType, elements> reference_spill_shift{};
    reference_spill_permute.fill(static_cast<LaneType>(-1));
    reference_spill_shift.fill(static_cast<LaneType>(lane_bits));
    for (usize contributor = 0; contributor < table::contributor_count; ++contributor) {
        reference_primary_permutes[contributor].fill(static_cast<LaneType>(-1));
        reference_primary_shifts[contributor].fill(static_cast<LaneType>(lane_bits));
    }

    u64 reference_spill_mask = 0;
    for (usize output_lane = 0; output_lane < elements; ++output_lane) {
        const usize output_start = output_lane * lane_bits;
        usize contributor        = 0;
        usize previous_input     = elements;

        // Derive the plan independently from each packed output bit rather than interval arithmetic.
        for (usize output_bit = 0; output_bit < lane_bits; ++output_bit) {
            const usize packed_bit = output_start + output_bit;
            const usize input_lane = packed_bit / BitWidth;
            if (input_lane >= elements || input_lane == previous_input) {
                continue;
            }
            previous_input   = input_lane;
            const auto shift = static_cast<std::ptrdiff_t>(input_lane * BitWidth) - static_cast<std::ptrdiff_t>(output_start);
            if (shift < 0) {
                reference_spill_permute[output_lane] = static_cast<LaneType>(input_lane);
                reference_spill_shift[output_lane]   = static_cast<LaneType>(-shift);
                reference_spill_mask |= u64{1} << output_lane;
            } else {
                if (contributor >= table::contributor_count) {
                    return false;
                }
                reference_primary_permutes[contributor][output_lane] = static_cast<LaneType>(input_lane);
                reference_primary_shifts[contributor][output_lane]   = static_cast<LaneType>(shift);
                reference_primary_masks[contributor] |= u64{1} << output_lane;
                ++contributor;
            }
        }
    }

    if (table::primary_permutes != reference_primary_permutes || table::primary_shifts != reference_primary_shifts ||
        table::primary_masks != reference_primary_masks || table::spill_permute != reference_spill_permute ||
        table::spill_shift != reference_spill_shift || table::spill_mask != reference_spill_mask) {
        return false;
    }

    std::array<unsigned_type, elements> input{};
    for (usize lane = 0; lane < elements; ++lane) {
        input[lane] = static_cast<unsigned_type>(((lane * 0x9e37U) + 0x5a5aU) & value_mask);
    }

    for (usize output_lane = 0; output_lane < elements; ++output_lane) {
        unsigned_type generated = 0;
        const auto add_left     = [&](const LaneType index, const LaneType shift) {
            if (index >= 0) {
                generated |= static_cast<unsigned_type>((input[static_cast<usize>(index)] << shift) & lane_mask);
            }
        };
        for (usize slot = 0; slot < table::contributor_count; ++slot) {
            add_left(table::primary_permutes[slot][output_lane], table::primary_shifts[slot][output_lane]);
        }
        if (table::spill_permute[output_lane] >= 0) {
            generated |=
                static_cast<unsigned_type>(input[static_cast<usize>(table::spill_permute[output_lane])] >> table::spill_shift[output_lane]);
        }

        unsigned_type expected = 0;
        for (usize bit = 0; bit < lane_bits; ++bit) {
            const usize packed_bit = (output_lane * lane_bits) + bit;
            const usize input_lane = packed_bit / BitWidth;
            const usize input_bit  = packed_bit % BitWidth;
            if (input_lane < elements) {
                expected |= static_cast<unsigned_type>(((input[input_lane] >> input_bit) & 1U) << bit);
            }
        }
        if (generated != expected) {
            return false;
        }

        const auto mask_bit = u64{1} << output_lane;
        for (usize slot = 0; slot < table::contributor_count; ++slot) {
            if (((table::primary_masks[slot] & mask_bit) != 0) != (table::primary_permutes[slot][output_lane] >= 0)) {
                return false;
            }
        }
        if (((table::spill_mask & mask_bit) != 0) != (table::spill_permute[output_lane] >= 0)) {
            return false;
        }
    }

    return true;
}

template <usize VectorBytes, usize... Indices>
constexpr bool validate_8_bit_tables(std::index_sequence<Indices...>) {
    return (validate_pack_table<i8, static_cast<u8>(Indices + 1), VectorBytes>() && ...);
}

template <usize VectorBytes, usize... Indices>
constexpr bool validate_16_bit_tables(std::index_sequence<Indices...>) {
    return (validate_pack_table<i16, static_cast<u8>(Indices + 9), VectorBytes>() && ...);
}

template <usize VectorBytes, usize... Indices>
constexpr bool validate_32_bit_tables(std::index_sequence<Indices...>) {
    return (validate_pack_table<i32, static_cast<u8>(Indices + 17), VectorBytes>() && ...);
}

static_assert(validate_8_bit_tables<16>(std::make_index_sequence<8>{}));
static_assert(validate_8_bit_tables<32>(std::make_index_sequence<8>{}));
static_assert(validate_8_bit_tables<64>(std::make_index_sequence<8>{}));
static_assert(validate_16_bit_tables<16>(std::make_index_sequence<8>{}));
static_assert(validate_16_bit_tables<32>(std::make_index_sequence<8>{}));
static_assert(validate_16_bit_tables<64>(std::make_index_sequence<8>{}));
static_assert(validate_32_bit_tables<16>(std::make_index_sequence<8>{}));
static_assert(validate_32_bit_tables<32>(std::make_index_sequence<8>{}));
static_assert(validate_32_bit_tables<64>(std::make_index_sequence<8>{}));
}  // namespace

TEST(PackTablesTest, GeneratesAllAvx512LanePlansAtCompileTime) {
    SUCCEED();
}
