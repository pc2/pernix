#ifndef PERNIX_FALLBACK_SCALAR_COMPRESSION_H
#define PERNIX_FALLBACK_SCALAR_COMPRESSION_H

#include <pernix/compat.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace pernix {
namespace internal {
__always_inline i32 quantize_ps_epi32(const float input, const float scale) {
    return static_cast<i32>(std::lroundf(input * scale));
}

__always_inline i64 quantize_pd_epi64(const f64 input, const f64 scale) {
    return std::llround(input * scale);
}

template <u8 BIT_WIDTH, typename T>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<T>)
__always_inline i32 quantize_clamped(const T input, const T scale) {
    constexpr i64 min_value = BIT_WIDTH == 1 ? 0 : -(i64{1} << (BIT_WIDTH - 1));
    constexpr i64 max_value = BIT_WIDTH == 1 ? 1 : ((i64{1} << (BIT_WIDTH - 1)) - 1);

    const long double scaled = static_cast<long double>(input) * static_cast<long double>(scale);
    if (std::isnan(scaled)) {
        return 0;
    }
    if (scaled <= static_cast<long double>(min_value)) {
        return static_cast<i32>(min_value);
    }
    if (scaled >= static_cast<long double>(max_value)) {
        return static_cast<i32>(max_value);
    }

    return static_cast<i32>(std::llround(scaled));
}

template <u8 BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
__always_inline i32 clamp_signed_quantized(const i64 value) {
    if constexpr (BIT_WIDTH == 1) {
        return static_cast<i32>(std::clamp<i64>(value, 0, 1));
    }

    constexpr i32 min_value = -(1 << (BIT_WIDTH - 1));
    constexpr i32 max_value = (1 << (BIT_WIDTH - 1)) - 1;
    return static_cast<i32>(std::clamp<i64>(value, min_value, max_value));
}

template <typename T, u8 BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_integral_v<T> && std::is_unsigned_v<T>)
void pack_epi32_fallback_inner(const std::span<const u32> input, const u8 bit_offset,
                               u8* __restrict__ destination) {
    constexpr u32 bits_in_type = sizeof(T) * 8;
    constexpr u32 bitmask      = BIT_WIDTH == bits_in_type
                                ? std::numeric_limits<T>::max()
                                : (1U << BIT_WIDTH) - 1U;

    std::size_t idx            = 0;
    std::size_t bits_in_buffer = bit_offset;
    u64 buffer                 = bit_offset ? static_cast<u64>(destination[0] & ((1U << bit_offset) - 1U)) : 0;

#pragma GCC unroll 64
    for (u32 raw_value : input) {
        const u32 next_value = raw_value & bitmask;

        buffer         |= static_cast<u64>(next_value) << bits_in_buffer;
        bits_in_buffer += BIT_WIDTH;

        while (bits_in_buffer >= 8) {
            destination[idx++] = static_cast<u8>(buffer & 0xFFU);
            buffer             >>= 8;
            bits_in_buffer     -= 8;
        }
    }

    if (bits_in_buffer > 0) {
        destination[idx] = static_cast<u8>(buffer & 0xFFU);
    }
}

template <u8 BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
void pack_epi32_fallback(const std::span<const u32> input, u8* __restrict__ destination) {
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
        return internal::pack_epi32_fallback_inner<u8, BIT_WIDTH>(input, 0, destination);
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
        return internal::pack_epi32_fallback_inner<u16, BIT_WIDTH>(input, 0, destination);
    } else {
        return internal::pack_epi32_fallback_inner<u32, BIT_WIDTH>(input, 0, destination);
    }
}

template <u8 BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
void pack_epi32_fallback(const std::vector<u32>& input, u8* __restrict__ destination) {
    return pack_epi32_fallback<BIT_WIDTH>(std::span<const u32>(input.data(), input.size()), destination);
}

template <typename T, u8 BIT_WIDTH, typename FloatT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_integral_v<T> && std::is_unsigned_v<T> &&
             std::is_floating_point_v<FloatT>)
void quantize_and_pack_fallback_inner(const FloatT* __restrict__ input, const FloatT scale,
                                      const u32 elements, const u8 bit_offset,
                                      u8* __restrict__ destination) {
    constexpr u32 bits_in_type = sizeof(T) * 8;
    constexpr u32 bitmask      = BIT_WIDTH == bits_in_type
                                ? std::numeric_limits<T>::max()
                                : (1U << BIT_WIDTH) - 1U;

    std::size_t idx            = 0;
    std::size_t bits_in_buffer = bit_offset;
    u64 buffer                 = bit_offset ? static_cast<u64>(destination[0] & ((1U << bit_offset) - 1U)) : 0;

#pragma GCC unroll 64
    for (u32 i = 0; i < elements; i++) {
        const i32 quantized  = quantize_clamped<BIT_WIDTH>(input[i], scale);
        const u32 next_value = static_cast<u32>(quantized) & bitmask;

        buffer         |= static_cast<u64>(next_value) << bits_in_buffer;
        bits_in_buffer += BIT_WIDTH;

        while (bits_in_buffer >= 8) {
            destination[idx++] = static_cast<u8>(buffer & 0xFFU);
            buffer             >>= 8;
            bits_in_buffer     -= 8;
        }
    }

    if (bits_in_buffer > 0) {
        destination[idx] = static_cast<u8>(buffer & 0xFFU);
    }
}

template <u8 BIT_WIDTH, typename FloatT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<FloatT>)
void quantize_and_pack_fallback(const FloatT* __restrict__ input, const FloatT scale,
                                const u32 elements, u8* __restrict__ destination) {
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
        return quantize_and_pack_fallback_inner<u8, BIT_WIDTH>(input, scale, elements, 0, destination);
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
        return quantize_and_pack_fallback_inner<u16, BIT_WIDTH>(input, scale, elements, 0, destination);
    } else {
        return quantize_and_pack_fallback_inner<u32, BIT_WIDTH>(input, scale, elements, 0, destination);
    }
}
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
int compress_block_fallback(const void* __restrict__ input_ptr, const f32 scale,
                            void* __restrict__ output_ptr) {
    const auto* input = static_cast<const float*>(input_ptr);
    auto* output      = static_cast<u8*>(output_ptr);

    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    std::memset(output, 0, BLOCK_SIZE);

    internal::quantize_and_pack_fallback<BIT_WIDTH>(input, scale, elements_per_block, output);
    return 0;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
int compress_block_fallback(const void* __restrict__ input_ptr, const f64 scale,
                            void* __restrict__ output_ptr) {
    const auto* input = static_cast<const double*>(input_ptr);
    auto* output      = static_cast<u8*>(output_ptr);

    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    std::memset(output, 0, BLOCK_SIZE);

    internal::quantize_and_pack_fallback<BIT_WIDTH>(input, scale, elements_per_block, output);
    return 0;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
int compress_blocks_fallback(const void* __restrict__ input_ptr, float scale, void* __restrict__ output_ptr,
                             u32 blocks) {
    const auto* input = static_cast<const float*>(input_ptr);
    auto* output      = static_cast<u8*>(output_ptr);

    const f32* block_input = input;
    u8* block_output       = output;

    for (u32 block = 0; block < blocks; block++) {
        compress_block_fallback<BIT_WIDTH, BLOCK_SIZE>(block_input, scale, block_output);
        block_input  += (BLOCK_SIZE * 8) / BIT_WIDTH;
        block_output += BLOCK_SIZE;
    }

    return 0;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
int compress_blocks_fallback(const void* __restrict__ input_ptr, const f64 scale,
                             void* __restrict__ output_ptr,
                             const unsigned int blocks) {
    const auto* input = static_cast<const double*>(input_ptr);
    auto* output      = static_cast<u8*>(output_ptr);

    const f64* block_input = input;
    u8* block_output       = output;

    for (u32 block = 0; block < blocks; block++) {
        compress_block_fallback<BIT_WIDTH, BLOCK_SIZE>(block_input, scale, block_output);
        block_input  += (BLOCK_SIZE * 8) / BIT_WIDTH;
        block_output += BLOCK_SIZE;
    }
    return 0;
}
}

#endif // PERNIX_FALLBACK_SCALAR_COMPRESSION_H
