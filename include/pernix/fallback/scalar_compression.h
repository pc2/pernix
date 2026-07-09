#ifndef PERNIX_FALLBACK_SCALAR_COMPRESSION_H
#define PERNIX_FALLBACK_SCALAR_COMPRESSION_H

#include <pernix/compat.h>
#include <pernix/fallback/common.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <type_traits>

namespace pernix {
namespace internal {
template <typename T, u8 BIT_WIDTH, typename FloatT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_integral_v<T> && std::is_unsigned_v<T> && std::is_floating_point_v<FloatT>)
__always_inline void quantize_and_pack_fallback_inner(const std::span<const FloatT> input, const FloatT scale, const u32 elements,
                                                      const u8 bit_offset, std::span<u8> destination) {
    constexpr u32 bits_in_type = sizeof(T) * 8;
    constexpr T bitmask        = BIT_WIDTH == bits_in_type ? std::numeric_limits<T>::max() : static_cast<T>((1U << BIT_WIDTH) - 1U);

    usize idx            = 0;
    usize bits_in_buffer = bit_offset;
    u64 buffer           = bit_offset ? static_cast<u64>(destination[0] & ((1U << bit_offset) - 1U)) : 0;

#pragma GCC unroll 512
    for (usize i = 0; i < elements; i++) {
        const i32 quantized  = quantize_clamped<BIT_WIDTH, FloatT, i32>(input[i], scale);
        const u32 next_value = static_cast<u32>(quantized) & static_cast<u32>(bitmask);

        buffer |= static_cast<u64>(next_value) << bits_in_buffer;
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

template <u8 BIT_WIDTH, typename FloatT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<FloatT>)
__always_inline void quantize_and_pack_fallback(const std::span<const FloatT> input, const FloatT scale, const u32 elements,
                                                std::span<u8> destination) {
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
        quantize_and_pack_fallback_inner<u8, BIT_WIDTH>(input, scale, elements, 0, destination);
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
        quantize_and_pack_fallback_inner<u16, BIT_WIDTH>(input, scale, elements, 0, destination);
    } else {
        quantize_and_pack_fallback_inner<u32, BIT_WIDTH>(input, scale, elements, 0, destination);
    }
}
}  // namespace internal

template <u8 BIT_WIDTH, u32 BLOCK_SIZE, typename FloatT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<FloatT> && BLOCK_SIZE > 0 && BLOCK_SIZE % 32 == 0)
int compress_block_fallback(const std::span<const FloatT> input, const FloatT scale, std::span<u8> destination) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    std::ranges::fill(destination, 0);

    internal::quantize_and_pack_fallback<BIT_WIDTH>(input, scale, elements_per_block, destination);
    return 0;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE, typename FloatT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<FloatT> && BLOCK_SIZE > 0 && BLOCK_SIZE % 32 == 0)
int compress_block_fallback(const void* __restrict__ input_ptr, const FloatT scale, void* __restrict__ output_ptr) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    const auto input_span = std::span<const FloatT, elements_per_block>(static_cast<const FloatT*>(input_ptr), elements_per_block);
    auto output_span      = std::span<u8, BLOCK_SIZE>(static_cast<u8*>(output_ptr), BLOCK_SIZE);

    return compress_block_fallback<BIT_WIDTH, BLOCK_SIZE, FloatT>(input_span, scale, output_span);
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE, typename FloatT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<FloatT> && BLOCK_SIZE > 0 && BLOCK_SIZE % 32 == 0)
int compress_blocks_fallback(const std::span<const FloatT> input, const FloatT scale, std::span<u8> destination, const u32 blocks) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    for (usize i = 0; i < blocks; i++) {
        const auto block_input = input.subspan(i * elements_per_block, elements_per_block);
        auto block_output      = destination.subspan(i * BLOCK_SIZE, BLOCK_SIZE);
        compress_block_fallback<BIT_WIDTH, BLOCK_SIZE>(block_input, scale, block_output);
    }

    return 0;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE, typename FloatT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<FloatT> && BLOCK_SIZE > 0 && BLOCK_SIZE % 32 == 0)
int compress_blocks_fallback(const void* __restrict__ input_ptr, const FloatT scale, void* __restrict__ output_ptr, const u32 blocks) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    const auto input_span = std::span<const FloatT>(static_cast<const FloatT*>(input_ptr), elements_per_block * blocks);
    auto output_span      = std::span(static_cast<u8*>(output_ptr), blocks * BLOCK_SIZE);

    return compress_blocks_fallback<BIT_WIDTH, BLOCK_SIZE, FloatT>(input_span, scale, output_span, blocks);
}
}  // namespace pernix

#endif  // PERNIX_FALLBACK_SCALAR_COMPRESSION_H
