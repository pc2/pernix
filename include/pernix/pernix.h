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
 * Dispatching front-end for block compression.
 *
 * The implementation is selected at compile time from the best enabled backend
 * in this order: AVX-512 VBMI, AVX2, then the scalar fallback path.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
int compress_block(const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output);

/**
 * Compress multiple fixed-size blocks using the same backend selection rules as
 * `compress_block`.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
int compress_blocks(const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output, uint32_t blocks);

/**
 * Dispatching front-end for block decompression.
 *
 * The selected backend matches `compress_block` so compression and
 * decompression use compatible packing logic.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
int decompress_block(const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output);

/**
 * Decompress multiple fixed-size blocks using the best enabled backend.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
int decompress_blocks(const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output, uint32_t blocks);

// Use the best available implementation based on detected CPU features at compile time.
#ifdef PERNIX_AVX2_ENABLED
#ifdef PERNIX_AVX512_VBMI_ENABLED
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
int compress_block(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output) {
    return mm512_compress_block_avx512vbmi<BIT_WIDTH>(input, scale, output);
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
int compress_blocks(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output, const uint32_t blocks) {
    return mm512_compress_blocks_avx512vbmi<BIT_WIDTH>(input, scale, output, blocks);
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
int decompress_block(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output) {
    return mm512_decompress_block_avx512vbmi<BIT_WIDTH>(input, scale, output);
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
int decompress_blocks(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output, const uint32_t blocks) {
    return mm512_decompress_blocks_avx512vbmi<BIT_WIDTH>(input, scale, output, blocks);
}
#else
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
int compress_block(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output) {
    return mm256_compress_block_avx2<BIT_WIDTH>(input, scale, output);
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
int compress_blocks(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output, const uint32_t blocks) {
    return mm256_compress_blocks_avx2<BIT_WIDTH>(input, scale, output, blocks);
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
int decompress_block(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output) {
    return mm256_decompress_block_avx2<BIT_WIDTH>(input, scale, output);
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
int decompress_blocks(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output, const uint32_t blocks) {
    return mm256_decompress_blocks_avx2<BIT_WIDTH>(input, scale, output, blocks);
}
#endif
#else
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
int compress_block(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output) {
    return fallback_compress_block<BIT_WIDTH>(input, scale, output);
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
int compress_blocks(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output, const uint32_t blocks) {
    return fallback_compress_blocks<BIT_WIDTH>(input, scale, output, blocks);
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
int decompress_block(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output) {
    return fallback_decompress_block<BIT_WIDTH>(input, scale, output);
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
int decompress_blocks(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output, const uint32_t blocks) {
    return fallback_decompress_blocks<BIT_WIDTH>(input, scale, output, blocks);
}
#endif
}  // namespace pernix

#ifdef __cplusplus
namespace pernix {
extern "C" {
#endif

/**
 * Runtime-dispatched C ABI for single-block compression.
 *
 * `bit_width` is validated at runtime so C callers do not need template
 * instantiations.
 */
int compress_block(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output);
/**
 * Runtime-dispatched C ABI for multi-block compression.
 */
int compress_blocks(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output, uint32_t blocks);

/**
 * Runtime-dispatched C ABI for single-block decompression.
 */
int decompress_block(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output);
/**
 * Runtime-dispatched C ABI for multi-block decompression.
 */
int decompress_blocks(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output, uint32_t blocks);

#ifdef __cplusplus
}
}  // namespace pernix
#endif

#endif  // PERNIX_H
