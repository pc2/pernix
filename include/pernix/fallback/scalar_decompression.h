#ifndef PERNIX_FALLBACK_SCALAR_DECOMPRESSION_H
#define PERNIX_FALLBACK_SCALAR_DECOMPRESSION_H

#include <pernix/compat.h>
#include <pernix/fallback/common.h>

#include <limits>
#include <span>
#include <type_traits>

namespace pernix {
namespace internal {
template <typename T, u8 BIT_WIDTH, bool SIGN_VALUES = true, typename FloatT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_integral_v<T> && std::is_unsigned_v<T> && std::is_floating_point_v<FloatT>)
__always_inline void unpack_and_dequantize_fallback_inner(const std::span<const u8> input, const u8 bit_offset, const u32 elements,
                                                          const FloatT scale, std::span<FloatT> output) {
    constexpr u32 bits_in_type = sizeof(T) * 8;
    constexpr T bitmask        = BIT_WIDTH == bits_in_type ? std::numeric_limits<T>::max() : static_cast<T>((1U << BIT_WIDTH) - 1U);

    usize idx         = 0;
    u8 bits_in_buffer = 8 - bit_offset;
    u64 buffer        = static_cast<u64>(input[idx++]) >> bit_offset;

#pragma GCC unroll 512
    for (usize i = 0; i < elements; i++) {
        while (BIT_WIDTH > bits_in_buffer) {
            const auto next_value = static_cast<u64>(input[idx++]) << bits_in_buffer;
            buffer |= next_value;
            bits_in_buffer += 8;
        }

        const u32 raw_value = static_cast<u32>(buffer & bitmask);
        i32 unpacked        = 0;
        if constexpr (SIGN_VALUES) {
            unpacked = sign_extend<BIT_WIDTH>(raw_value);
        } else {
            unpacked = static_cast<i32>(raw_value);
        }
        output[i] = static_cast<FloatT>(unpacked) * scale;

        buffer >>= BIT_WIDTH;
        bits_in_buffer -= BIT_WIDTH;
    }
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true, typename FloatT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<FloatT>)
__always_inline void unpack_and_dequantize_fallback(const std::span<const u8> input, const u32 elements, const FloatT scale,
                                                    std::span<FloatT> output) {
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
        unpack_and_dequantize_fallback_inner<u8, BIT_WIDTH, SIGN_VALUES>(input, 0, elements, scale, output);
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
        unpack_and_dequantize_fallback_inner<u16, BIT_WIDTH, SIGN_VALUES>(input, 0, elements, scale, output);
    } else {
        unpack_and_dequantize_fallback_inner<u32, BIT_WIDTH, SIGN_VALUES>(input, 0, elements, scale, output);
    }
}
}  // namespace internal

template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE, typename FloatT>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24 && std::is_floating_point_v<FloatT> && BLOCK_SIZE > 0 && BLOCK_SIZE % 32 == 0)
int decompress_block_fallback(const std::span<const u8> input, const FloatT scale, std::span<FloatT> output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    internal::unpack_and_dequantize_fallback<BIT_WIDTH, SIGN_VALUES>(input, elements_per_block, scale, output);
    return 0;
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE, typename FloatT>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24 && std::is_floating_point_v<FloatT> && BLOCK_SIZE > 0 && BLOCK_SIZE % 32 == 0)
int decompress_block_fallback(const void* __restrict__ input_ptr, const FloatT scale, void* __restrict__ output_ptr) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    const auto input = std::span<const u8, BLOCK_SIZE>(static_cast<const u8*>(input_ptr), BLOCK_SIZE);
    auto output      = std::span<FloatT, elements_per_block>(static_cast<FloatT*>(output_ptr), elements_per_block);

    return decompress_block_fallback<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE, FloatT>(input, scale, output);
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE, typename FloatT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && BLOCK_SIZE > 0 && BLOCK_SIZE % 32 == 0 && std::is_floating_point_v<FloatT>)
int decompress_blocks_fallback(const std::span<const u8> input, const FloatT scale, std::span<FloatT> output, const u32 blocks) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    for (u32 block = 0; block < blocks; block++) {
        const auto block_input = input.subspan(block * BLOCK_SIZE, BLOCK_SIZE);
        auto block_output      = output.subspan(block * elements_per_block, elements_per_block);

        decompress_block_fallback<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE, FloatT>(block_input, scale, block_output);
    }

    return 0;
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE, typename FloatT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && BLOCK_SIZE > 0 && BLOCK_SIZE % 32 == 0 && std::is_floating_point_v<FloatT>)
int decompress_blocks_fallback(const void* __restrict__ input_ptr, const FloatT scale, void* __restrict__ output_ptr, const u32 blocks) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    const auto input_span = std::span(static_cast<const u8*>(input_ptr), blocks * BLOCK_SIZE);
    auto output_span      = std::span<FloatT>(static_cast<FloatT*>(output_ptr), elements_per_block * blocks);

    return decompress_blocks_fallback<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE, FloatT>(input_span, scale, output_span, blocks);
}

}  // namespace pernix

#endif  // PERNIX_FALLBACK_SCALAR_DECOMPRESSION_H
