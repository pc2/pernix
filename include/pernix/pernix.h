#ifndef PERNIX_H
#define PERNIX_H

#include <pernix/detection.h>

// Include architecture-specific headers based on detected capabilities
// AVX2
#ifdef PERNIX_AVX2_ENABLED
#include <pernix/avx2/compression.h>
#include <pernix/avx2/decompression.h>

// BMI2: Needs AVX2 as well
#ifdef PERNIX_BMI2_ENABLED
#include <pernix/bmi2/compression.h>
#include <pernix/bmi2/decompression.h>
#endif  // PERNIX_BMI2_ENABLED

// AVX512 VBMI: Needs AVX2 as well
#ifdef PERNIX_AVX512_VBMI_ENABLED
#include <pernix/avx512vbmi/compression.h>
#include <pernix/avx512vbmi/decompression.h>
#endif  // PERNIX_AVX512_VBMI_ENABLED

#endif  // PERNIX_AVX2_ENABLED

// Fallback (non-SIMD) implementations
#include <pernix/fallback/compression.h>
#include <pernix/fallback/decompression.h>

namespace pernix {

/**
 * @brief Compress a single block of floating-point data into a bit-packed format using the specified bit width and scale.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam BLOCK_SIZE size of the block in bytes (must be a multiple of 32).
 *
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 *
 * @return int status code (0 for success).
 *
 * @note This function will dispatch to the best available implementation based on detected CPU features at compile time.
 */
template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int compress_block(const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output);

/**
 * @brief Compress a single block of double-precision values into a bit-packed representation.
 *
 * @tparam BIT_WIDTH bit width per quantized value (1 to 24).
 * @tparam BLOCK_SIZE size of the block in bytes (must be a multiple of 32).
 *
 * @param input pointer to the input block.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the destination compressed bytes.
 *
 * @return int status code (0 for success).
 */
template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int compress_block(const double_t* __restrict__ input, double_t scale, uint8_t* __restrict__ output);

/**
 * @brief Compress multiple blocks of single-precision values.
 *
 * @tparam BIT_WIDTH bit width per quantized value (1 to 24).
 * @tparam BLOCK_SIZE size of each block in bytes (must be a multiple of 32).
 *
 * @param input pointer to the first input value.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the destination compressed bytes.
 * @param blocks number of consecutive blocks to process.
 *
 * @return int status code (0 for success).
 */
template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int compress_blocks(const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output, uint32_t blocks);

/**
 * @brief Compress multiple blocks of double-precision values.
 *
 * @tparam BIT_WIDTH bit width per quantized value (1 to 24).
 * @tparam BLOCK_SIZE size of each block in bytes (must be a multiple of 32).
 *
 * @param input pointer to the first input value.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the destination compressed bytes.
 * @param blocks number of consecutive blocks to process.
 *
 * @return int status code (0 for success).
 */
template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int compress_blocks(const double_t* __restrict__ input, double_t scale, uint8_t* __restrict__ output, uint32_t blocks);

/**
 * @brief Decompress a single block of packed values into single-precision values.
 *
 * @tparam BIT_WIDTH bit width per packed value (1 to 24).
 * @tparam SIGN_VALUES true for signed values, false for unsigned values.
 * @tparam BLOCK_SIZE size of the block in bytes (must be a multiple of 32).
 *
 * @param input pointer to the packed input block.
 * @param scale scaling factor used to reconstruct floating-point values.
 * @param output pointer to the destination decompressed values.
 *
 * @return int status code (0 for success).
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int decompress_block(const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output);

/**
 * @brief Decompress a single block of packed values into double-precision values.
 *
 * @tparam BIT_WIDTH bit width per packed value (1 to 24).
 * @tparam SIGN_VALUES true for signed values, false for unsigned values.
 * @tparam BLOCK_SIZE size of the block in bytes (must be a multiple of 32).
 *
 * @param input pointer to the packed input block.
 * @param scale scaling factor used to reconstruct floating-point values.
 * @param output pointer to the destination decompressed values.
 *
 * @return int status code (0 for success).
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int decompress_block(const uint8_t* __restrict__ input, double_t scale, double_t* __restrict__ output);

/**
 * @brief Decompress multiple blocks of packed values into single-precision values.
 *
 * @tparam BIT_WIDTH bit width per packed value (1 to 24).
 * @tparam SIGN_VALUES true for signed values, false for unsigned values.
 * @tparam BLOCK_SIZE size of each block in bytes (must be a multiple of 32).
 *
 * @param input pointer to the first packed block.
 * @param scale scaling factor used to reconstruct floating-point values.
 * @param output pointer to the destination decompressed values.
 * @param blocks number of consecutive blocks to process.
 *
 * @return int status code (0 for success).
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int decompress_blocks(const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output, uint32_t blocks);

/**
 * @brief Decompress multiple blocks of packed values into double-precision values.
 *
 * @tparam BIT_WIDTH bit width per packed value (1 to 24).
 * @tparam SIGN_VALUES true for signed values, false for unsigned values.
 * @tparam BLOCK_SIZE size of each block in bytes (must be a multiple of 32).
 *
 * @param input pointer to the first packed block.
 * @param scale scaling factor used to reconstruct floating-point values.
 * @param output pointer to the destination decompressed values.
 * @param blocks number of consecutive blocks to process.
 *
 * @return int status code (0 for success).
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int decompress_blocks(const uint8_t* __restrict__ input, double_t scale, double_t* __restrict__ output, uint32_t blocks);

// Use the best available implementation based on detected CPU features at compile time.
#ifdef PERNIX_AVX2_ENABLED
#ifdef PERNIX_AVX512_VBMI_ENABLED
template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int compress_block(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output) {
    return mm512_compress_block_avx512vbmi<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
}

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int compress_block(const double_t* __restrict__ input, const double_t scale, uint8_t* __restrict__ output) {
    return mm512_compress_block_avx512vbmi<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
}

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int compress_blocks(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output, const uint32_t blocks) {
    return mm512_compress_blocks_avx512vbmi<BIT_WIDTH, BLOCK_SIZE>(input, scale, output, blocks);
}

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int compress_blocks(const double_t* __restrict__ input, const double_t scale, uint8_t* __restrict__ output, const uint32_t blocks) {
    return mm512_compress_blocks_avx512vbmi<BIT_WIDTH, BLOCK_SIZE>(input, scale, output, blocks);
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int decompress_block(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output) {
    return mm512_decompress_block_avx512vbmi<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int decompress_block(const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output) {
    return mm512_decompress_block_avx512vbmi<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int decompress_blocks(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output, const uint32_t blocks) {
    return mm512_decompress_blocks_avx512vbmi<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output, blocks);
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int decompress_blocks(const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output, const uint32_t blocks) {
    return mm512_decompress_blocks_avx512vbmi<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output, blocks);
}
#else
template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int compress_block(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output) {
    return mm256_compress_block_avx2<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
}

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int compress_block(const double_t* __restrict__ input, const double_t scale, uint8_t* __restrict__ output) {
    return mm256_compress_block_avx2<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
}

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int compress_blocks(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output, const uint32_t blocks) {
    return mm256_compress_blocks_avx2<BIT_WIDTH, BLOCK_SIZE>(input, scale, output, blocks);
}

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int compress_blocks(const double_t* __restrict__ input, const double_t scale, uint8_t* __restrict__ output, const uint32_t blocks) {
    return mm256_compress_blocks_avx2<BIT_WIDTH, BLOCK_SIZE>(input, scale, output, blocks);
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int decompress_block(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output) {
    return mm256_decompress_block_avx2<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int decompress_block(const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output) {
    return mm256_decompress_block_avx2<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int decompress_blocks(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output, const uint32_t blocks) {
    return mm256_decompress_blocks_avx2<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output, blocks);
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int decompress_blocks(const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output, const uint32_t blocks) {
    return mm256_decompress_blocks_avx2<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output, blocks);
}
#endif
#else
template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int compress_block(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output) {
    return compress_block_fallback<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
}

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int compress_block(const double_t* __restrict__ input, const double_t scale, uint8_t* __restrict__ output) {
    return compress_block_fallback<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
}

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int compress_blocks(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output, const uint32_t blocks) {
    return compress_blocks_fallback<BIT_WIDTH, BLOCK_SIZE>(input, scale, output, blocks);
}

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int compress_blocks(const double_t* __restrict__ input, const double_t scale, uint8_t* __restrict__ output, const uint32_t blocks) {
    return compress_blocks_fallback<BIT_WIDTH, BLOCK_SIZE>(input, scale, output, blocks);
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int decompress_block(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output) {
    return decompress_block_fallback<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int decompress_block(const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output) {
    return decompress_block_fallback<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int decompress_blocks(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output, const uint32_t blocks) {
    return decompress_blocks_fallback<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output, blocks);
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int decompress_blocks(const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output, const uint32_t blocks) {
    return decompress_blocks_fallback<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output, blocks);
}
#endif
}  // namespace pernix

#ifdef __cplusplus
namespace pernix {
extern "C" {
#endif

/**
 * @brief C ABI wrapper for compressing one single-precision block.
 *
 * @param bit_width bit width per quantized value (8 to 16).
 * @param input pointer to the input block.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the destination compressed bytes.
 * @return int status code (0 for success, non-zero for invalid arguments or unsupported bit width).
 */
int compress_block(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output);

/**
 * @brief C ABI wrapper for compressing one double-precision block.
 *
 * @param bit_width bit width per quantized value (8 to 16).
 * @param input pointer to the input block.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the destination compressed bytes.
 * @return int status code (0 for success, non-zero for invalid arguments or unsupported bit width).
 */
int compress_block_f64(uint8_t bit_width, const double_t* __restrict__ input, double_t scale, uint8_t* __restrict__ output);

/**
 * @brief C ABI wrapper for compressing multiple single-precision blocks.
 *
 * @param bit_width bit width per quantized value (8 to 16).
 * @param input pointer to the first input value.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the destination compressed bytes.
 * @param blocks number of consecutive blocks to process.
 * @return int status code (0 for success, non-zero for invalid arguments or unsupported bit width).
 */
int compress_blocks(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output, uint32_t blocks);

/**
 * @brief C ABI wrapper for compressing multiple double-precision blocks.
 *
 * @param bit_width bit width per quantized value (8 to 16).
 * @param input pointer to the first input value.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the destination compressed bytes.
 * @param blocks number of consecutive blocks to process.
 * @return int status code (0 for success, non-zero for invalid arguments or unsupported bit width).
 */
int compress_blocks_f64(uint8_t bit_width, const double_t* __restrict__ input, double_t scale, uint8_t* __restrict__ output,
                        uint32_t blocks);

/**
 * @brief C ABI wrapper for decompressing one single-precision block.
 *
 * @param bit_width bit width per packed value (1 to 24).
 * @param input pointer to the packed input block.
 * @param scale scaling factor used to reconstruct floating-point values.
 * @param output pointer to the destination decompressed values.
 * @return int status code (0 for success, non-zero for invalid arguments or unsupported bit width).
 */
int decompress_block(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output);

/**
 * @brief C ABI wrapper for decompressing one double-precision block.
 *
 * @param bit_width bit width per packed value (1 to 24).
 * @param input pointer to the packed input block.
 * @param scale scaling factor used to reconstruct floating-point values.
 * @param output pointer to the destination decompressed values.
 * @return int status code (0 for success, non-zero for invalid arguments or unsupported bit width).
 */
int decompress_block_f64(uint8_t bit_width, const uint8_t* __restrict__ input, double_t scale, double_t* __restrict__ output);

/**
 * @brief C ABI wrapper for decompressing multiple single-precision blocks.
 *
 * @param bit_width bit width per packed value (1 to 24).
 * @param input pointer to the first packed block.
 * @param scale scaling factor used to reconstruct floating-point values.
 * @param output pointer to the destination decompressed values.
 * @param blocks number of consecutive blocks to process.
 * @return int status code (0 for success, non-zero for invalid arguments or unsupported bit width).
 */
int decompress_blocks(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output, uint32_t blocks);

/**
 * @brief C ABI wrapper for decompressing multiple double-precision blocks.
 *
 * @param bit_width bit width per packed value (1 to 24).
 * @param input pointer to the first packed block.
 * @param scale scaling factor used to reconstruct floating-point values.
 * @param output pointer to the destination decompressed values.
 * @param blocks number of consecutive blocks to process.
 * @return int status code (0 for success, non-zero for invalid arguments or unsupported bit width).
 */
int decompress_blocks_f64(uint8_t bit_width, const uint8_t* __restrict__ input, double_t scale, double_t* __restrict__ output,
                          uint32_t blocks);

#ifdef __cplusplus
}
}  // namespace pernix
#endif

#endif  // PERNIX_H