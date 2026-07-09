#ifndef PERNIX_FALLBACK_STDPAR_COMPRESSION_H
#define PERNIX_FALLBACK_STDPAR_COMPRESSION_H

#include <pernix/fallback/stdpar_common.h>
#include <pernix/pernix.h>

#include <cstring>
#include <execution>
#include <span>
#include <type_traits>

namespace pernix {
namespace internal {
template <u8 BIT_WIDTH, u8 GROUP_SIZE, typename ScaleType>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && GROUP_SIZE >= 1 && GROUP_SIZE <= 8 && std::is_floating_point_v<ScaleType>)
__always_inline stdpar_buffer<u32, GROUP_SIZE> quantize_epi32_group_fallback_stdpar(const std::span<const ScaleType> input,
                                                                                    const ScaleType scale) {
    stdpar_buffer<u32, GROUP_SIZE> output;

#pragma GCC unroll GROUP_SIZE
    for (usize i = 0; i < GROUP_SIZE; ++i) {
        output.data[i] = quantize_stdpar_value<BIT_WIDTH>(input[i], scale);
        ++output.size;
    }

    return output;
}

template <u8 BIT_WIDTH, u8 GROUP_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && GROUP_SIZE >= 1 && GROUP_SIZE <= 8)
__always_inline void pack_epi32_group_fallback_stdpar(const stdpar_buffer<u32, GROUP_SIZE>& input, const std::span<u8> destination) {
    constexpr u32 bitmask = (u32{1} << BIT_WIDTH) - 1U;

    usize idx            = 0;
    usize bits_in_buffer = 0;
    u64 buffer           = 0;

#pragma GCC unroll GROUP_SIZE
    for (usize i = 0; i < GROUP_SIZE; ++i) {
        buffer |= (static_cast<u64>(input.data[i] & bitmask) << bits_in_buffer);
        bits_in_buffer += BIT_WIDTH;

        while (bits_in_buffer >= 8) {
            destination[idx++] = static_cast<u8>(buffer & 0xFFU);
            buffer >>= 8;
            bits_in_buffer -= 8;
        }
    }

    if (bits_in_buffer > 0) {
        destination[idx] = static_cast<u8>(buffer & 0xFFU);
    }
}
}  // namespace internal

template <u8 BIT_WIDTH, u32 BLOCK_SIZE, typename ScaleType>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<ScaleType>)
int compress_block_fallback_stdpar(const void* input_ptr, ScaleType scale, void* output_ptr) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8U) / BIT_WIDTH;
    constexpr u32 full_groups        = elements_per_block / 8U;
    constexpr u32 remainder          = elements_per_block % 8U;

    const auto* input = static_cast<const ScaleType*>(input_ptr);
    auto* output      = static_cast<u8*>(output_ptr);

    std::memset(output, 0, BLOCK_SIZE);

    auto first = internal::counting_iterator<u32>{0};
    std::for_each_n(std::execution::par_unseq, first, full_groups, [&](const u32 group) {
        constexpr u8 group_size            = 8;
        constexpr usize packed_group_bytes = internal::packed_group_bytes_v<group_size, BIT_WIDTH>;

        const auto quantized = internal::quantize_epi32_group_fallback_stdpar<BIT_WIDTH, group_size>(
            std::span<const ScaleType>(input + static_cast<usize>(group) * group_size, group_size), scale);
        internal::pack_epi32_group_fallback_stdpar<BIT_WIDTH, group_size>(
            quantized, std::span<u8>(output + static_cast<usize>(group) * BIT_WIDTH, packed_group_bytes));
    });

    if constexpr (remainder != 0) {
        constexpr u8 tail_group_size       = static_cast<u8>(remainder);
        constexpr usize tail_input_offset  = full_groups * 8U;
        constexpr usize tail_output_offset = full_groups * BIT_WIDTH;
        constexpr usize tail_bytes         = internal::packed_group_bytes_v<tail_group_size, BIT_WIDTH>;

        const auto quantized = internal::quantize_epi32_group_fallback_stdpar<BIT_WIDTH, tail_group_size>(
            std::span<const ScaleType>(input + tail_input_offset, tail_group_size), scale);
        internal::pack_epi32_group_fallback_stdpar<BIT_WIDTH, tail_group_size>(quantized,
                                                                               std::span<u8>(output + tail_output_offset, tail_bytes));
    }

    return PERNIX_STATUS_OK;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE, typename ScaleType>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<ScaleType>)
int compress_blocks_fallback_stdpar(const void* input_ptr, ScaleType scale, void* output_ptr, u32 blocks) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8U) / BIT_WIDTH;

    const auto* input = static_cast<const ScaleType*>(input_ptr);
    auto* output      = static_cast<u8*>(output_ptr);

    auto first = internal::counting_iterator<u32>{0};
    std::for_each_n(std::execution::par_unseq, first, blocks, [&](const u32 block) {
        const auto* block_input = input + (static_cast<usize>(block) * elements_per_block);
        auto* block_output      = output + (static_cast<usize>(block) * BLOCK_SIZE);
        static_cast<void>(compress_block_fallback_stdpar<BIT_WIDTH, BLOCK_SIZE, ScaleType>(block_input, scale, block_output));
    });

    return PERNIX_STATUS_OK;
}
}  // namespace pernix

#endif  // PERNIX_FALLBACK_STDPAR_COMPRESSION_H
