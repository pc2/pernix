#ifndef PERNIX_FALLBACK_SCALAR_DECOMPRESSION_H
#define PERNIX_FALLBACK_SCALAR_DECOMPRESSION_H

#include <limits>
#include <type_traits>
#include <vector>

namespace pernix {
namespace internal {
__always_inline float dequantize_epi32(const i32 input, const float scale) {
    return static_cast<float>(input) * scale;
}

__always_inline f64 dequantize_epi64(const i64 input, const f64 scale) {
    return static_cast<f64>(input) * scale;
}

template <u8 BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
__always_inline auto sign_extend(const u32 value) -> i32 {
    if constexpr (BIT_WIDTH == 1) {
        return static_cast<i32>(value & 1U);
    }

    constexpr u32 sign_bit = u32{1} << (BIT_WIDTH - 1);
    constexpr u32 mask     = (u32{1} << BIT_WIDTH) - 1;
    const u32 masked       = value & mask;
    return static_cast<i32>((static_cast<i64>(masked ^ sign_bit)) - static_cast<i64>(sign_bit));
}

template <typename T, u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_integral_v<T> && std::is_unsigned_v<T>)
__always_inline auto unpack_epi32_fallback_inner(const u8* __restrict__ input, const u8 bit_offset,
                                                 const std::size_t elements)
    -> std::vector<i32> {
    constexpr u32 bits_in_type = sizeof(T) * 8;
    constexpr u32 bitmask      = BIT_WIDTH == bits_in_type
                                ? std::numeric_limits<T>::max()
                                : (1U << BIT_WIDTH) - 1U;

    std::vector<i32> output(elements);

    std::size_t idx   = 0;
    u8 bits_in_buffer = 8 - bit_offset;
    u64 buffer        = static_cast<u64>(input[idx++]) >> bit_offset;

#pragma GCC unroll 64
    for (u32 i = 0; i < elements; i++) {
        while (BIT_WIDTH > bits_in_buffer) {
            const auto next_value = static_cast<u64>(input[idx++]) << bits_in_buffer;
            buffer                |= next_value;
            bits_in_buffer        += 8;
        }

        const u32 raw_value = static_cast<u32>(buffer & bitmask);
        if constexpr (SIGN_VALUES) {
            output[i] = sign_extend<BIT_WIDTH>(raw_value);
        } else {
            output[i] = static_cast<i32>(raw_value);
        }

        buffer         >>= BIT_WIDTH;
        bits_in_buffer -= BIT_WIDTH;
    }

    return output;
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
__always_inline auto unpack_epi32_fallback(const u8* __restrict__ input,
                                           const std::size_t elements) -> std::vector<i32> {
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
        return unpack_epi32_fallback_inner<u8, BIT_WIDTH, SIGN_VALUES>(input, 0, elements);
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
        return unpack_epi32_fallback_inner<u16, BIT_WIDTH, SIGN_VALUES>(input, 0, elements);
    } else {
        return unpack_epi32_fallback_inner<u32, BIT_WIDTH, SIGN_VALUES>(input, 0, elements);
    }
}
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
int decompress_block_fallback(const void* __restrict__ input_ptr, const f32 scale,
                              void* __restrict__ output_ptr) {
    const auto* input = static_cast<const u8*>(input_ptr);
    auto* output      = static_cast<f32*>(output_ptr);

    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    const std::vector<i32> block_values = internal::unpack_epi32_fallback<BIT_WIDTH, SIGN_VALUES>(
        input, elements_per_block);

#pragma GCC unroll 512
    for (u32 i = 0; i < elements_per_block; i++) {
        output[i] = internal::dequantize_epi32(block_values[i], scale);
    }

    return 0;
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
int decompress_block_fallback(const void* __restrict__ input_ptr, const f64 scale,
                              void* __restrict__ output_ptr) {
    const auto* input = static_cast<const u8*>(input_ptr);
    auto* output      = static_cast<f64*>(output_ptr);

    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    const std::vector<i32> block_values = internal::unpack_epi32_fallback<BIT_WIDTH, SIGN_VALUES>(
        input, elements_per_block);

#pragma GCC unroll 512
    for (u32 i = 0; i < elements_per_block; i++) {
        output[i] = internal::dequantize_epi64(block_values[i], scale);
    }

    return 0;
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
int decompress_blocks_fallback(const void* __restrict__ input_ptr, const f32 scale,
                               void* __restrict__ output_ptr, const u32 blocks) {
    const auto* input = static_cast<const u8*>(input_ptr);
    auto* output      = static_cast<f32*>(output_ptr);

    const u8* block_input = input;
    f32* block_output     = output;

    for (u32 block = 0; block < blocks; block++) {
        decompress_block_fallback<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(block_input, scale, block_output);
        block_input  += BLOCK_SIZE;
        block_output += (BLOCK_SIZE * 8) / BIT_WIDTH;
    }

    return 0;
}

template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
int decompress_blocks_fallback(const void* __restrict__ input_ptr, const f64 scale,
                               void* __restrict__ output_ptr, const u32 blocks) {
    const auto* input = static_cast<const u8*>(input_ptr);
    auto* output      = static_cast<f64*>(output_ptr);

    const u8* block_input = input;
    f64* block_output     = output;

    for (u32 block = 0; block < blocks; block++) {
        decompress_block_fallback<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(block_input, scale, block_output);
        block_input  += BLOCK_SIZE;
        block_output += (BLOCK_SIZE * 8) / BIT_WIDTH;
    }

    return 0;
}
}

#endif // PERNIX_FALLBACK_SCALAR_DECOMPRESSION_H
