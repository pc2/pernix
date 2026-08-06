#ifndef PERNIX_FALLBACK_STDPAR_COMPRESSION_H
#define PERNIX_FALLBACK_STDPAR_COMPRESSION_H

#include <pernix/fallback/common.h>
#include <pernix/fallback/scalar_compression.h>
#include <pernix/fallback/stdpar_common.h>

#include <algorithm>
#include <limits>
#include <span>
#include <type_traits>

namespace pernix {
template <u8 BIT_WIDTH, u32 BLOCK_SIZE, typename FloatT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<FloatT> && BLOCK_SIZE > 0 && BLOCK_SIZE % 32 == 0)
int compress_block_fallback_stdpar(const std::span<const FloatT> input, const FloatT scale, std::span<u8> destination) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8U) / BIT_WIDTH;
    constexpr u32 full_groups        = elements_per_block / internal::stdpar_group_size_v;
    constexpr u32 remainder          = elements_per_block % internal::stdpar_group_size_v;

    std::ranges::fill(destination, 0);

    internal::for_each_index_stdpar(full_groups, [&](const u32 group) {
        constexpr u32 group_size = internal::stdpar_group_size_v;

        const auto input_group = input.subspan(static_cast<usize>(group) * group_size, group_size);
        auto destination_group = destination.subspan(static_cast<usize>(group) * BIT_WIDTH, BIT_WIDTH);

        internal::quantize_and_pack_fallback<BIT_WIDTH>(input_group, scale, group_size, destination_group);
    });

    if constexpr (remainder != 0) {
        constexpr usize tail_input_offset  = full_groups * internal::stdpar_group_size_v;
        constexpr usize tail_output_offset = full_groups * BIT_WIDTH;
        constexpr usize tail_bytes         = internal::packed_group_bytes_v<static_cast<u8>(remainder), BIT_WIDTH>;

        internal::quantize_and_pack_fallback<BIT_WIDTH>(input.subspan(tail_input_offset, remainder), scale, remainder,
                                                        destination.subspan(tail_output_offset, tail_bytes));
    }

    return 0;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE, typename FloatT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<FloatT> && BLOCK_SIZE > 0 && BLOCK_SIZE % 32 == 0)
int compress_block_fallback_stdpar(const void* __restrict__ input_ptr, const FloatT scale, void* __restrict__ output_ptr) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8U) / BIT_WIDTH;

    const auto input_span = std::span<const FloatT, elements_per_block>(static_cast<const FloatT*>(input_ptr), elements_per_block);
    auto output_span      = std::span<u8, BLOCK_SIZE>(static_cast<u8*>(output_ptr), BLOCK_SIZE);

    return compress_block_fallback_stdpar<BIT_WIDTH, BLOCK_SIZE, FloatT>(input_span, scale, output_span);
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE, typename FloatT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<FloatT> && BLOCK_SIZE > 0 && BLOCK_SIZE % 32 == 0)
int compress_blocks_fallback_stdpar(const std::span<const FloatT> input, const FloatT scale, std::span<u8> destination, const u32 blocks) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8U) / BIT_WIDTH;

    internal::for_each_index_stdpar(blocks, [&](const u32 block) {
        const auto block_input = input.subspan(static_cast<usize>(block) * elements_per_block, elements_per_block);
        auto block_output      = destination.subspan(static_cast<usize>(block) * BLOCK_SIZE, BLOCK_SIZE);
        static_cast<void>(compress_block_fallback_stdpar<BIT_WIDTH, BLOCK_SIZE, FloatT>(block_input, scale, block_output));
    });

    return 0;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE, typename FloatT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<FloatT> && BLOCK_SIZE > 0 && BLOCK_SIZE % 32 == 0)
int compress_blocks_fallback_stdpar(const void* __restrict__ input_ptr, const FloatT scale, void* __restrict__ output_ptr,
                                    const u32 blocks) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8U) / BIT_WIDTH;

    const auto input_span = std::span<const FloatT>(static_cast<const FloatT*>(input_ptr), elements_per_block * blocks);
    auto output_span      = std::span<u8>(static_cast<u8*>(output_ptr), static_cast<usize>(blocks) * BLOCK_SIZE);

    return compress_blocks_fallback_stdpar<BIT_WIDTH, BLOCK_SIZE, FloatT>(input_span, scale, output_span, blocks);
}
}  // namespace pernix

#endif  // PERNIX_FALLBACK_STDPAR_COMPRESSION_H
