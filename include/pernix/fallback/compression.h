#ifndef PERNIX_FALLBACK_COMPRESSION_H
#define PERNIX_FALLBACK_COMPRESSION_H

#include <pernix/simd_compat.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace pernix {
namespace internal {
/**
 * @brief Quantize a single float value to int32_t using the provided scale.
 *
 * @param input input float value to be quantized.
 * @param scale scaling factor used during quantization.
 * @return int32_t quantized integer value.
 */
__always_inline int32_t quantize_ps_epi32(const float input, const float scale) {
    return static_cast<int32_t>(std::lroundf(input * scale));
}

/**
 * @brief Quantize a single double value to int64_t using the provided scale.
 *
 * @param input input double value to be quantized.
 * @param scale scaling factor used during quantization.
 * @return int64_t quantized integer value.
 */
__always_inline int64_t quantize_pd_epi64(const double_t input, const double_t scale) {
    return std::llround(input * scale);
}

/**
 * @brief Clamp a signed quantized value to the representable range of BIT_WIDTH bits.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
__always_inline int32_t clamp_signed_quantized(const int64_t value) {
    if constexpr (BIT_WIDTH == 1) {
        // 1-bit fallback is treated as binary quantization (0/1).
        return static_cast<int32_t>(std::clamp<int64_t>(value, 0, 1));
    }

    constexpr int32_t min_value = -(1 << (BIT_WIDTH - 1));
    constexpr int32_t max_value = (1 << (BIT_WIDTH - 1)) - 1;
    return static_cast<int32_t>(std::clamp<int64_t>(value, min_value, max_value));
}

/**
 * @brief Append packed scalar values into an output buffer using the selected
 * storage width.
 *
 * @tparam T unsigned integer type used as the packing word.
 * @tparam BIT_WIDTH bit width per value in the packed representation.
 * @param input vector of quantized values to pack.
 * @param bit_offset starting bit offset in the destination buffer.
 * @param destination pointer to the output buffer.
 */
template <typename T, uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_integral_v<T> && std::is_unsigned_v<T>)
void pack_epi32_fallback_inner(const std::vector<uint32_t>& input, const uint8_t bit_offset, uint8_t* __restrict__ destination) {
    constexpr uint32_t bits_in_type = sizeof(T) * 8;
    constexpr uint32_t bitmask      = BIT_WIDTH == bits_in_type ? std::numeric_limits<T>::max() : (1U << BIT_WIDTH) - 1U;

    std::size_t idx            = 0;
    std::size_t bits_in_buffer = bit_offset;
    uint64_t buffer            = bit_offset ? static_cast<uint64_t>(destination[0] & ((1U << bit_offset) - 1U)) : 0;

#pragma GCC unroll 64
    for (uint32_t raw_value : input) {
        const uint32_t next_value = raw_value & bitmask;

        buffer         |= static_cast<uint64_t>(next_value) << bits_in_buffer;
        bits_in_buffer += BIT_WIDTH;

        while (bits_in_buffer >= 8) {
            destination[idx++] = static_cast<uint8_t>(buffer & 0xFFU);
            buffer             >>= 8;
            bits_in_buffer     -= 8;
        }
    }

    if (bits_in_buffer > 0) {
        destination[idx] = static_cast<uint8_t>(buffer & 0xFFU);
    }
}

/**
 * @brief Pack a vector of uint32_t values into a compact byte representation using fallback scalar implementation.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam BLOCK_SIZE size of each block in bytes (default 64 for 512 bits).
 *
 * @param input vector of uint32_t values to be packed.
 * @param destination pointer to the output buffer where packed bytes will be stored.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
void pack_epi32_fallback(const std::vector<uint32_t>& input, uint8_t* __restrict__ destination) {
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
        return internal::pack_epi32_fallback_inner<uint8_t, BIT_WIDTH>(input, 0, destination);
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
        return internal::pack_epi32_fallback_inner<uint16_t, BIT_WIDTH>(input, 0, destination);
    } else {
        return internal::pack_epi32_fallback_inner<uint32_t, BIT_WIDTH>(input, 0, destination);
    }
}
} // namespace internal

/**
 * @brief Compress a single 512-bit block using fallback scalar implementation.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam BLOCK_SIZE size of each block in bytes (default 64 for 512 bits).
 *
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @return int status code (0 for success).
 */
template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
int compress_block_fallback(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    std::vector<uint32_t> block_values(elements_per_block);
#pragma GCC unroll 64
    for (uint32_t i = 0; i < elements_per_block; i++) {
        const int32_t quantized = internal::clamp_signed_quantized<BIT_WIDTH>(internal::quantize_ps_epi32(input[i], scale));
        block_values[i]         = static_cast<uint32_t>(quantized);
    }

    internal::pack_epi32_fallback<BIT_WIDTH>(block_values, output);
    return 0;
}

/**
 * @brief Compress a single block of double values using the fallback scalar implementation.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam BLOCK_SIZE size of each block in bytes (default 64 for 512 bits).
 * @param input pointer to the start of the input double values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @return int status code (0 for success).
 */
template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
int compress_block_fallback(const double_t* __restrict__ input, const double_t scale, uint8_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;
    std::vector<uint32_t> block_values(elements_per_block);
#pragma GCC unroll 32
    for (uint32_t i = 0; i < elements_per_block; i++) {
        const int32_t quantized = internal::clamp_signed_quantized<BIT_WIDTH>(internal::quantize_pd_epi64(input[i], scale));
        block_values[i]         = static_cast<uint32_t>(quantized);
    }

    internal::pack_epi32_fallback<BIT_WIDTH>(block_values, output);
    return 0;
}

/**
 * @brief Compress multiple 512-bit blocks using fallback scalar implementation.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam BLOCK_SIZE size of each block in bytes (default 64 for 512 bits).
 *
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @param blocks number of 512-bit blocks to compress.
 * @return int status code (0 for success).
 */
template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
int compress_blocks_fallback(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output, const uint32_t blocks) {
    const float_t* block_input = input;
    uint8_t* block_output      = output;

    for (uint32_t block = 0; block < blocks; block++) {
        compress_block_fallback<BIT_WIDTH, BLOCK_SIZE>(block_input, scale, block_output);
        block_input  += (BLOCK_SIZE * 8) / BIT_WIDTH;
        block_output += BLOCK_SIZE;
    }

    return 0;
}

/**
 * @brief Compress multiple blocks of double values using the fallback scalar implementation.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam BLOCK_SIZE size of each block in bytes (default 64 for 512 bits).
 * @param input pointer to the start of the input double values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @param blocks number of blocks to compress.
 * @return int status code (0 for success).
 */
template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
int compress_blocks_fallback(const double_t* __restrict__ input, const double_t scale, uint8_t* __restrict__ output,
                             const uint32_t blocks) {
    const double_t* block_input = input;
    uint8_t* block_output       = output;

    for (uint32_t block = 0; block < blocks; block++) {
        compress_block_fallback<BIT_WIDTH, BLOCK_SIZE>(block_input, scale, block_output);
        block_input  += (BLOCK_SIZE * 8) / BIT_WIDTH;
        block_output += BLOCK_SIZE;
    }
    return 0;
}
} // namespace pernix

#ifdef __cplusplus
namespace pernix {
extern "C" {
#endif

/**
 * @brief Compress a single 512-bit block using fallback scalar implementation.
 *
 * @param bit_width bit width per value in the packed representation (1 to 24).
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @return int status code (0 for success).
 */
int compress_block_fallback(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output);

/**
 * @brief Compress a single 512-bit block using fallback scalar implementation.
 *
 * @param bit_width bit width per value in the packed representation (1 to 24).
 * @param input pointer to the start of the input double values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @return int status code (0 for success).
 */
int compress_block_fallback_f64(uint8_t bit_width, const double_t* __restrict__ input, double_t scale, uint8_t* __restrict__ output);

/**
 * @brief Compress multiple 512-bit blocks using fallback scalar implementation.
 *
 * @param bit_width bit width per value in the packed representation (1 to 24).
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @param blocks number of 512-bit blocks to compress.
 * @return int status code (0 for success).
 */
int compress_blocks_fallback(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output,
                             uint32_t blocks);

/**
 * @brief Compress multiple 512-bit blocks using fallback scalar implementation.
 *
 * @param bit_width bit width per value in the packed representation (1 to 24).
 * @param input pointer to the start of the input double values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @param blocks number of 512-bit blocks to compress.
 * @return int status code (0 for success).
 */
int compress_blocks_fallback_f64(uint8_t bit_width, const double_t* __restrict__ input, double_t scale, uint8_t* __restrict__ output,
                                 uint32_t blocks);

#ifdef __cplusplus
}
} // namespace pernix
#endif

#endif  // PERNIX_FALLBACK_COMPRESSION_H
