#ifndef PERNIX_PACK_TABLES_H
#define PERNIX_PACK_TABLES_H

#include <pernix/compat.h>

#include <array>
#include <concepts>
#include <cstddef>
#include <limits>
#include <utility>

namespace pernix::detail {
namespace internal {
template <typename LaneType, u8 BitWidth, usize Elements, usize Contributor>
constexpr std::array<LaneType, Elements> make_primary_permute() {
    constexpr usize lane_bits = sizeof(LaneType) * 8;
    std::array<LaneType, Elements> table{};
    table.fill(static_cast<LaneType>(-1));

    for (usize output_lane = 0; output_lane < Elements; ++output_lane) {
        const usize output_start = output_lane * lane_bits;
        const usize output_end   = output_start + lane_bits;
        usize contributor        = 0;

        for (usize input_lane = 0; input_lane < Elements; ++input_lane) {
            const usize input_start = input_lane * BitWidth;
            const usize input_end   = input_start + BitWidth;

            // Half-open intervals overlap when each starts before the other ends.
            if (!(input_start < output_end && input_end > output_start)) {
                continue;
            }

            const auto shift = static_cast<std::ptrdiff_t>(input_start) - static_cast<std::ptrdiff_t>(output_start);
            // Nonnegative shifts place input bits into this output lane and are primary contributors.
            // Negative shifts cross the output boundary and are emitted by the spill logic instead.
            if (shift >= 0 && contributor++ == Contributor) {
                table[output_lane] = static_cast<LaneType>(input_lane);
                break;
            }
        }
    }
    return table;
}

template <typename LaneType, u8 BitWidth, usize Elements, usize Contributor>
constexpr std::array<LaneType, Elements> make_primary_shift() {
    constexpr usize lane_bits = sizeof(LaneType) * 8;
    const auto permute        = make_primary_permute<LaneType, BitWidth, Elements, Contributor>();
    std::array<LaneType, Elements> table{};
    table.fill(static_cast<LaneType>(lane_bits));

    for (usize output_lane = 0; output_lane < Elements; ++output_lane) {
        if (permute[output_lane] >= 0) {
            table[output_lane] = static_cast<LaneType>(static_cast<usize>(permute[output_lane]) * BitWidth - output_lane * lane_bits);
        }
    }
    return table;
}

template <typename LaneType, u8 BitWidth, usize Elements>
constexpr std::array<LaneType, Elements> make_spill_permute() {
    constexpr usize lane_bits = sizeof(LaneType) * 8;
    std::array<LaneType, Elements> table{};
    table.fill(static_cast<LaneType>(-1));

    for (usize output_lane = 0; output_lane < Elements; ++output_lane) {
        const usize output_start = output_lane * lane_bits;
        for (usize input_lane = 0; input_lane < Elements; ++input_lane) {
            const usize input_start = input_lane * BitWidth;
            const usize input_end   = input_start + BitWidth;
            if (input_start < output_start && input_end > output_start) {
                table[output_lane] = static_cast<LaneType>(input_lane);
                break;
            }
        }
    }
    return table;
}

template <typename LaneType, u8 BitWidth, usize Elements>
constexpr std::array<LaneType, Elements> make_spill_shift() {
    constexpr usize lane_bits = sizeof(LaneType) * 8;
    const auto permute        = make_spill_permute<LaneType, BitWidth, Elements>();
    std::array<LaneType, Elements> table{};
    table.fill(static_cast<LaneType>(lane_bits));

    for (usize output_lane = 0; output_lane < Elements; ++output_lane) {
        if (permute[output_lane] >= 0) {
            table[output_lane] = static_cast<LaneType>(output_lane * lane_bits - static_cast<usize>(permute[output_lane]) * BitWidth);
        }
    }
    return table;
}

template <typename LaneType, usize Elements>
constexpr u64 make_active_mask(const std::array<LaneType, Elements>& permute) {
    static_assert(Elements <= 64);
    u64 mask = 0;
    for (usize lane = 0; lane < Elements; ++lane) {
        if (permute[lane] >= 0) {
            mask |= u64{1} << lane;
        }
    }
    return mask;
}

template <typename LaneType, u8 BitWidth, usize Elements, usize... Contributors>
constexpr std::array<std::array<LaneType, Elements>, sizeof...(Contributors)> make_primary_permutes(std::index_sequence<Contributors...>) {
    return {make_primary_permute<LaneType, BitWidth, Elements, Contributors>()...};
}

template <typename LaneType, u8 BitWidth, usize Elements, usize... Contributors>
constexpr std::array<std::array<LaneType, Elements>, sizeof...(Contributors)> make_primary_shifts(std::index_sequence<Contributors...>) {
    return {make_primary_shift<LaneType, BitWidth, Elements, Contributors>()...};
}

template <typename LaneType, usize Contributors, usize Elements>
constexpr std::array<u64, Contributors> make_primary_masks(const std::array<std::array<LaneType, Elements>, Contributors>& permutes) {
    std::array<u64, Contributors> masks{};
    for (usize contributor = 0; contributor < Contributors; ++contributor) {
        masks[contributor] = make_active_mask(permutes[contributor]);
    }
    return masks;
}
}  // namespace internal

// Unsupported lane types, bit widths, and vector sizes intentionally fail the focused assertions below.
template <typename LaneType, u8 BitWidth, usize VectorBytes>
struct pack_table {
    static_assert(std::signed_integral<LaneType>, "pack_table LaneType must be a signed integral type");
    static_assert(sizeof(LaneType) == sizeof(u8) || sizeof(LaneType) == sizeof(u16) || sizeof(LaneType) == sizeof(u32),
                  "pack_table LaneType must use 8-, 16-, or 32-bit lanes");
    static_assert(VectorBytes > 0, "pack_table VectorBytes must be greater than zero");
    static_assert(VectorBytes % sizeof(LaneType) == 0, "pack_table VectorBytes must contain a whole number of lanes");

    static constexpr usize lane_bits      = sizeof(LaneType) * 8;
    static constexpr bool valid_bit_width = (sizeof(LaneType) == sizeof(u8) && BitWidth >= 1 && BitWidth <= 8) ||
                                            (sizeof(LaneType) == sizeof(u16) && BitWidth >= 9 && BitWidth <= 16) ||
                                            (sizeof(LaneType) == sizeof(u32) && BitWidth >= 17 && BitWidth <= 24);
    static_assert(valid_bit_width, "pack_table BitWidth is outside the permitted range for LaneType");

    static constexpr usize element_count = VectorBytes / sizeof(LaneType);
    static_assert(element_count <= 64, "pack_table supports at most 64 lanes because active masks use u64");
    static_assert(element_count == 0 || element_count - 1 <= static_cast<usize>(std::numeric_limits<LaneType>::max()),
                  "pack_table permute indices must be representable by LaneType");
    static_assert(std::numeric_limits<LaneType>::lowest() <= -1 && static_cast<LaneType>(-1) < 0,
                  "pack_table LaneType must represent -1 as the inactive-entry sentinel");

    static constexpr usize contributor_count = BitWidth > 0 ? (lane_bits + BitWidth - 1) / BitWidth : 0;
    static_assert(contributor_count > 0, "pack_table must have at least one primary contributor");
    static constexpr u8 bit_width = BitWidth;

    alignas(64) static constexpr auto primary_permutes =
        internal::make_primary_permutes<LaneType, BitWidth, element_count>(std::make_index_sequence<contributor_count>{});
    alignas(64) static constexpr auto primary_shifts =
        internal::make_primary_shifts<LaneType, BitWidth, element_count>(std::make_index_sequence<contributor_count>{});
    alignas(64) static constexpr auto spill_permute = internal::make_spill_permute<LaneType, BitWidth, element_count>();
    alignas(64) static constexpr auto spill_shift   = internal::make_spill_shift<LaneType, BitWidth, element_count>();
    static constexpr auto primary_masks             = internal::make_primary_masks(primary_permutes);
    static constexpr u64 spill_mask                 = internal::make_active_mask(spill_permute);
};
}  // namespace pernix::detail

#endif  // PERNIX_PACK_TABLES_H
