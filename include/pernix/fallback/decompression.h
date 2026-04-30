#ifndef PERNIX_FALLBACK_DECOMPRESSION_H
#define PERNIX_FALLBACK_DECOMPRESSION_H

#include <pernix/simd_compat.h>

#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace pernix {
namespace internal {
/**
 * @brief Dequantize a single int32_t value to float using the provided scale.
 *
 * @param input input int32_t value to be dequantized.
 * @param scale scaling factor used during quantization.
 * @return float dequantized float value.
 */
__always_inline float dequantize_epi32(const int32_t input, const float scale) {
    return static_cast<float>(input) * scale;
}

/**
 * @brief Dequantize a single int64_t value to double using the provided scale.
 *
 * @param input input int64_t value to be dequantized.
 * @param scale scaling factor used during quantization.
 * @return double_t dequantized double value.
 */
__always_inline double_t dequantize_epi64(const int64_t input, const double_t scale) {
    return static_cast<double_t>(input) * scale;
}

/**
 * @brief Sign-extend a packed integer value stored in the low bits of a 32-bit word.
 *
 * @tparam BIT_WIDTH number of significant bits in the encoded value.
 * @param value unsigned packed value.
 * @return int32_t sign-extended value.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
__always_inline auto sign_extend(const uint32_t value) -> int32_t {
    if constexpr (BIT_WIDTH == 1) {
        return static_cast<int32_t>(value & 1U);
    }

    constexpr uint32_t shift = 32 - BIT_WIDTH;
    return (static_cast<int32_t>(value) << shift) >> shift;
}

/**
 * @brief Unpack bit-packed values from a typed input span into signed 32-bit integers.
 *
 * @tparam T unsigned integer type used to read the source buffer.
 * @tparam BIT_WIDTH bit width per packed value.
 * @tparam SIGN_VALUES whether to sign-extend unpacked values.
 * @param input pointer to the typed packed input buffer.
 * @param bit_offset starting bit offset in the first input word.
 * @param elements number of values to unpack.
 * @return std::vector<int32_t> unpacked values.
 */
template <typename T, uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_integral_v<T> && std::is_unsigned_v<T>)
__always_inline auto unpack_epi32_fallback_inner(const uint8_t* __restrict__ input, const uint8_t bit_offset, const std::size_t elements)
    -> std::vector<int32_t> {
    constexpr uint32_t bits_in_type = sizeof(T) * 8;
    constexpr uint32_t bitmask      = BIT_WIDTH == bits_in_type ? std::numeric_limits<T>::max() : (1U << BIT_WIDTH) - 1U;

    std::vector<int32_t> output(elements);

    std::size_t idx        = 0;
    uint8_t bits_in_buffer = 8 - bit_offset;
    uint64_t buffer        = static_cast<uint64_t>(input[idx++]) >> bit_offset;

#pragma GCC unroll 64
    for (uint32_t i = 0; i < elements; i++) {
        while (BIT_WIDTH > bits_in_buffer) {
            const auto next_value = static_cast<uint64_t>(input[idx++]) << bits_in_buffer;
            buffer                |= next_value;
            bits_in_buffer        += 8;
        }

        const uint32_t raw_value = static_cast<uint32_t>(buffer & bitmask);
        if constexpr (SIGN_VALUES) {
            output[i] = sign_extend<BIT_WIDTH>(raw_value);
        } else {
            output[i] = static_cast<int32_t>(raw_value);
        }

        buffer         >>= BIT_WIDTH;
        bits_in_buffer -= BIT_WIDTH;
    }

    return output;
}

/**
 * @brief Unpack packed int32_t values from the input buffer using fallback scalar implementation.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 * @param input pointer to the start of the packed data.
 * @param elements number of elements to unpack.
 * @return std::vector<int32_t> unpacked int32_t values.
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
__always_inline auto unpack_epi32_fallback(const uint8_t* __restrict__ input, const std::size_t elements) -> std::vector<int32_t> {
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
        return unpack_epi32_fallback_inner<uint8_t, BIT_WIDTH, SIGN_VALUES>(input, 0, elements);
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
        return unpack_epi32_fallback_inner<uint16_t, BIT_WIDTH, SIGN_VALUES>(input, 0, elements);
    } else {
        return unpack_epi32_fallback_inner<uint32_t, BIT_WIDTH, SIGN_VALUES>(input, 0, elements);
    }
}
} // namespace internal

/**
 * @brief Decompress a single 512\-bit block using fallback scalar implementation.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 * @param input pointer to the start of the compressed block.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @return int status code (0 for success).
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
int decompress_block_fallback(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    const std::vector<int32_t> block_values = internal::unpack_epi32_fallback<BIT_WIDTH, SIGN_VALUES>(input, elements_per_block);

#pragma GCC unroll 512
    for (uint32_t i = 0; i < elements_per_block; i++) {
        output[i] = internal::dequantize_epi32(block_values[i], scale);
    }

    return 0;
}

/**
 * @brief Decompress a single block to double values using the fallback scalar implementation.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 * @param input pointer to the start of the compressed block.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed double values will be stored.
 * @return int status code (0 for success).
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
int decompress_block_fallback(const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    const std::vector<int32_t> block_values = internal::unpack_epi32_fallback<BIT_WIDTH, SIGN_VALUES>(input, elements_per_block);

#pragma GCC unroll 512
    for (uint32_t i = 0; i < elements_per_block; i++) {
        output[i] = internal::dequantize_epi64(block_values[i], scale);
    }

    return 0;
}

/**
 * @brief Decompress multiple 512\-bit blocks using fallback scalar implementation.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 * @param input pointer to the start of the compressed data.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @param blocks number of 512-bit blocks to decompress.
 * @return int status code (0 for success).
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
int decompress_blocks_fallback(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output,
                               const uint32_t blocks) {
    const uint8_t* block_input = input;
    float_t* block_output      = output;

    for (uint32_t block = 0; block < blocks; block++) {
        decompress_block_fallback<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(block_input, scale, block_output);
        block_input  += BLOCK_SIZE;
        block_output += (BLOCK_SIZE * 8) / BIT_WIDTH;
    }

    return 0;
}

/**
 * @brief Decompress multiple blocks to double values using the fallback scalar implementation.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 * @param input pointer to the start of the compressed data.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed double values will be stored.
 * @param blocks number of blocks to decompress.
 * @return int status code (0 for success).
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
int decompress_blocks_fallback(const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output,
                               const uint32_t blocks) {
    const uint8_t* block_input = input;
    double_t* block_output     = output;

    for (uint32_t block = 0; block < blocks; block++) {
        decompress_block_fallback<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(block_input, scale, block_output);
        block_input  += BLOCK_SIZE;
        block_output += (BLOCK_SIZE * 8) / BIT_WIDTH;
    }

    return 0;
}
} // namespace pernix

#ifdef __cplusplus
namespace pernix {
extern "C" {
#endif

/**
 * @brief Decompress a single 512-bit block using fallback scalar implementation.
 *
 * @param bit_width bit width per value in the packed representation (1 to 24).
 * @param input pointer to the start of the compressed block.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @return int status code (0 for success).
 */
int decompress_block_fallback(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output);

/**
 * @brief Decompress a single 512-bit block using fallback scalar implementation.
 *
 * @param bit_width bit width per value in the packed representation (1 to 24).
 * @param input pointer to the start of the compressed block.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @return int status code (0 for success).
 */
int decompress_block_fallback_f64(uint8_t bit_width, const uint8_t* __restrict__ input, double_t scale, double_t* __restrict__ output);

/**
 * @brief Decompress multiple 512-bit blocks using fallback scalar implementation.
 *
 * @param bit_width bit width per value in the packed representation (1 to 24).
 * @param input pointer to the start of the compressed data.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @param blocks number of 512-bit blocks to decompress.
 * @return int status code (0 for success).
 */
int decompress_blocks_fallback(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output,
                               uint32_t blocks);

/**
 * @brief Decompress multiple 512-bit blocks using fallback scalar implementation.
 *
 * @param bit_width bit width per value in the packed representation (1 to 24).
 * @param input pointer to the start of the compressed data.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @param blocks number of 512-bit blocks to decompress.
 * @return int status code (0 for success).
 */
int decompress_blocks_fallback_f64(uint8_t bit_width, const uint8_t* __restrict__ input, double_t scale, double_t* __restrict__ output,
                                   uint32_t blocks);

#ifdef __cplusplus
}
} // namespace pernix
#endif

#endif  // PERNIX_FALLBACK_DECOMPRESSION_H
