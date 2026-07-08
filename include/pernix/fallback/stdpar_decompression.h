#ifndef PERNIX_FALLBACK_STDPAR_DECOMPRESSION_H
#define PERNIX_FALLBACK_STDPAR_DECOMPRESSION_H

#include <pernix/fallback/stdpar_common.h>

#include <execution>
#include <span>
#include <type_traits>

#include "scalar_decompression.h"

namespace pernix {
namespace internal {
// Eight packed values consume exactly BIT_WIDTH bytes.
template <u8 BIT_WIDTH, bool SIGN_VALUES, u8 GROUP_SIZE = 8>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && GROUP_SIZE >= 1 && GROUP_SIZE <= 8)
__always_inline stdpar_buffer<i32, GROUP_SIZE> unpack_epi32_group_fallback_stdpar(const std::span<const u8> input) {
    constexpr auto bit_offset = 0; // TODO: for now
    constexpr u32 bitmask     = (u32{1} << BIT_WIDTH) - 1U;

    stdpar_buffer<i32, GROUP_SIZE> output;

    // Unpack one fixed-size byte group using the scalar bit-buffer logic.

    usize idx         = 0;
    u8 bits_in_buffer = 8 - bit_offset;
    u64 buffer        = static_cast<u64>(input[idx++]) >> bit_offset;

#pragma GCC unroll GROUP_SIZE
    for (usize i = 0; i < GROUP_SIZE; i++) {
        while (BIT_WIDTH > bits_in_buffer) {
            const auto next_value = static_cast<u64>(input[idx++]) << bits_in_buffer;
            buffer                |= next_value;
            bits_in_buffer        += 8;
        }

        const u32 raw_value = static_cast<u32>(buffer & bitmask);
        if constexpr (SIGN_VALUES) {
            output.data[i] = sign_extend_stdpar<BIT_WIDTH>(raw_value);
        } else {
            output.data[i] = static_cast<i32>(raw_value);
        }
        ++output.size;

        buffer         >>= BIT_WIDTH;
        bits_in_buffer -= BIT_WIDTH;
    }
    return output;
}
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE, typename ScaleType>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24 && std::is_floating_point_v<ScaleType>)
int decompress_block_fallback_stdpar(const void* __restrict__ input_ptr, const ScaleType scale,
                                     void* __restrict__ output_ptr) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;
    constexpr u32 full_groups        = elements_per_block / 8;
    constexpr u32 remainder          = elements_per_block % 8;

    const auto input_span = std::span<const u8, BLOCK_SIZE>(static_cast<const u8*>(input_ptr), BLOCK_SIZE);
    auto output_span      = std::span<ScaleType, elements_per_block>(static_cast<ScaleType*>(output_ptr), elements_per_block);

    auto first = internal::counting_iterator<u32>{0};
    std::for_each_n(std::execution::par_unseq, first, full_groups, [&](const u32 group) {
        const auto input_subspan = input_span.subspan(static_cast<usize>(group) * BIT_WIDTH, BIT_WIDTH);
        const auto unpacked      = internal::unpack_epi32_group_fallback_stdpar<BIT_WIDTH, SIGN_VALUES>(input_subspan);

        for (u32 i = 0; i < unpacked.size; ++i) {
            output_span[static_cast<usize>(group) * 8U + i] = internal::dequantize_stdpar_value(unpacked.data[i], scale);
        }
    });

    if constexpr (remainder != 0) {
        constexpr usize tail_input_offset  = full_groups * BIT_WIDTH;
        constexpr usize tail_output_offset = full_groups * 8U;
        constexpr usize tail_bytes = internal::packed_group_bytes_v<static_cast<u8>(remainder), BIT_WIDTH>;

        const auto unpacked = internal::unpack_epi32_group_fallback_stdpar<
            BIT_WIDTH,
            SIGN_VALUES,
            static_cast<u8>(remainder)>(
            input_span.subspan(tail_input_offset, tail_bytes));

        for (u32 i = 0; i < remainder; ++i) {
            output_span[tail_output_offset + i] =
                internal::dequantize_stdpar_value(unpacked.data[i], scale);
        }
    }

    return 0;
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE, typename ScaleType>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24 && std::is_floating_point_v<ScaleType>)
int decompress_blocks_fallback_stdpar(const void* __restrict__ input_ptr, const ScaleType scale,
                                      void* __restrict__ output_ptr, const u32 blocks) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    const auto* input = static_cast<const u8*>(input_ptr);
    auto* output      = static_cast<ScaleType*>(output_ptr);

    auto first = internal::counting_iterator<u32>{0};
    std::for_each_n(std::execution::par_unseq, first, blocks, [&](const u32 block) {
        const auto* block_input = input + (static_cast<usize>(block) * BLOCK_SIZE);
        auto* block_output      = output + (static_cast<usize>(block) * elements_per_block);
        static_cast<void>(decompress_block_fallback_stdpar<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE, ScaleType>(
            block_input, scale, block_output));
    });

    return 0;
}
}

#endif // PERNIX_FALLBACK_STDPAR_DECOMPRESSION_H
