#ifndef PERNIX_FALLBACK_STDPAR_DECOMPRESSION_H
#define PERNIX_FALLBACK_STDPAR_DECOMPRESSION_H

#include <pernix/fallback/common.h>
#include <pernix/fallback/scalar_decompression.h>
#include <pernix/fallback/stdpar_common.h>

#include <limits>
#include <span>
#include <type_traits>

namespace pernix {
template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE, typename FloatT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<FloatT> && BLOCK_SIZE > 0 && BLOCK_SIZE % 32 == 0)
int decompress_block_fallback_stdpar(const std::span<const u8> input, const FloatT scale, std::span<FloatT> output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8U) / BIT_WIDTH;
    constexpr u32 full_groups        = elements_per_block / internal::stdpar_group_size_v;
    constexpr u32 remainder          = elements_per_block % internal::stdpar_group_size_v;

    internal::for_each_index_stdpar(full_groups, [&](const u32 group) {
        constexpr u32 group_size = internal::stdpar_group_size_v;

        const auto input_group = input.subspan(static_cast<usize>(group) * BIT_WIDTH, BIT_WIDTH);
        auto output_group      = output.subspan(static_cast<usize>(group) * group_size, group_size);

        internal::unpack_and_dequantize_fallback<BIT_WIDTH, SIGN_VALUES>(input_group, group_size, scale, output_group);
    });

    if constexpr (remainder != 0) {
        constexpr usize tail_input_offset  = full_groups * BIT_WIDTH;
        constexpr usize tail_output_offset = full_groups * internal::stdpar_group_size_v;
        constexpr usize tail_bytes         = internal::packed_group_bytes_v<static_cast<u8>(remainder), BIT_WIDTH>;

        internal::unpack_and_dequantize_fallback<BIT_WIDTH, SIGN_VALUES>(input.subspan(tail_input_offset, tail_bytes), remainder, scale,
                                                                         output.subspan(tail_output_offset, remainder));
    }

    return 0;
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE, typename FloatT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<FloatT> && BLOCK_SIZE > 0 && BLOCK_SIZE % 32 == 0)
int decompress_block_fallback_stdpar(const void* __restrict__ input_ptr, const FloatT scale, void* __restrict__ output_ptr) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8U) / BIT_WIDTH;

    const auto input_span = std::span<const u8, BLOCK_SIZE>(static_cast<const u8*>(input_ptr), BLOCK_SIZE);
    auto output_span      = std::span<FloatT, elements_per_block>(static_cast<FloatT*>(output_ptr), elements_per_block);

    return decompress_block_fallback_stdpar<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE, FloatT>(input_span, scale, output_span);
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE, typename FloatT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<FloatT> && BLOCK_SIZE > 0 && BLOCK_SIZE % 32 == 0)
int decompress_blocks_fallback_stdpar(const std::span<const u8> input, const FloatT scale, std::span<FloatT> output, const u32 blocks) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8U) / BIT_WIDTH;

    internal::for_each_index_stdpar(blocks, [&](const u32 block) {
        const auto block_input = input.subspan(static_cast<usize>(block) * BLOCK_SIZE, BLOCK_SIZE);
        auto block_output      = output.subspan(static_cast<usize>(block) * elements_per_block, elements_per_block);
        static_cast<void>(decompress_block_fallback_stdpar<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE, FloatT>(block_input, scale, block_output));
    });

    return 0;
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE, typename FloatT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<FloatT> && BLOCK_SIZE > 0 && BLOCK_SIZE % 32 == 0)
int decompress_blocks_fallback_stdpar(const void* __restrict__ input_ptr, const FloatT scale, void* __restrict__ output_ptr,
                                      const u32 blocks) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8U) / BIT_WIDTH;

    const auto input_span = std::span<const u8>(static_cast<const u8*>(input_ptr), static_cast<usize>(blocks) * BLOCK_SIZE);
    auto output_span      = std::span<FloatT>(static_cast<FloatT*>(output_ptr), elements_per_block * blocks);

    return decompress_blocks_fallback_stdpar<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE, FloatT>(input_span, scale, output_span, blocks);
}
}  // namespace pernix

#endif  // PERNIX_FALLBACK_STDPAR_DECOMPRESSION_H
