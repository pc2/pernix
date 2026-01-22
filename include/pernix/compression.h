#ifndef PERNIX_COMPRESSION_H
#define PERNIX_COMPRESSION_H

#include <pernix/bitpacking/packing.h>
#include <pernix/helper.h>
#include <pernix/quantization/quantization.h>

#include <cmath>
#include <cstdint>

using namespace pernix::bitpacking;
using namespace pernix::quantization;

namespace pernix {
#ifdef PERNIX_AVX2_ENABLED

/**
 * @brief Compress a single 512-bit block using AVX2 and BMI2 instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 16).
 *
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 and BMI2 support.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 16)
int mm256_compress_block_bmi2(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = 512 / BIT_WIDTH;
    constexpr uint32_t iterations_8       = elements_per_block / 8;
    constexpr uint8_t remaining           = elements_per_block - iterations_8 * 8;

    const __m256 scale_v = _mm256_set1_ps(scale);
#pragma GCC unroll 4
    for (uint32_t iter = 0; iter < iterations_8; iter++) {
        const __m256 source     = _mm256_loadu_ps(input);
        const __m256i quantized = mm256_quantize_ps_epi32(source, scale_v);
        const __m256i packed    = mm256_pack_epi32_bmi2<BIT_WIDTH>(quantized);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output), _mm256_castsi256_si128(packed));
        input += 8;
        output += BIT_WIDTH;
    }

    if constexpr (remaining) {
        std::vector<uint32_t> block_values(remaining);
#pragma GCC unroll 8
        for (uint32_t i = 0; i < remaining; i++) {
            block_values[i] = static_cast<uint32_t>(quantize_ps_epi32(input[i], scale));
        }

        pack_epi32_fallback<BIT_WIDTH>(block_values, output);
    }

    return 0;
}

/**
 * @brief Compress multiple 512-bit blocks using AVX2 and BMI2 instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 16).
 *
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @param blocks number of 512-bit blocks to compress.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 and BMI2 support.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 16)
int mm256_compress_blocks_bmi2(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output,
                               const uint32_t blocks) {
    const float_t* block_input = input;
    uint8_t* block_output      = output;

    for (uint32_t block = 0; block < blocks; block++) {
        mm256_compress_block_bmi2<BIT_WIDTH>(block_input, scale, block_output);
        block_input += 512 / BIT_WIDTH;
        block_output += (BIT_WIDTH * 512) / 8;
    }

    return 0;
}

/**
 * @brief Compress a single 512-bit block using AVX2 and BMI2 instructions.
 *
 * @param bit_width bit width per value in the packed representation (1 to 16).
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 and BMI2 support.
 */
int mm256_compress_block_bmi2(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output);

/**
 * @brief Compress multiple 512-bit blocks using AVX2 and BMI2 instructions.
 *
 * @param bit_width bit width per value in the packed representation (1 to 16).
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @param blocks number of 512-bit blocks to compress.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 and BMI2 support.
 */
int mm256_compress_blocks_bmi2(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output,
                               uint32_t blocks);

/**
 * @brief Compress a single 512-bit block using AVX2 instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 16).
 *
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 support.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 16)
int mm256_compress_block_avx2(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = 512 / BIT_WIDTH;
    constexpr uint32_t iterations_8       = elements_per_block / 8;
    constexpr uint8_t remaining           = elements_per_block - iterations_8 * 8;

    const __m256 scale_v = _mm256_set1_ps(scale);
#pragma GCC unroll 8
    for (uint32_t iter = 0; iter < iterations_8; iter++) {
        const __m256 source     = _mm256_loadu_ps(input);
        const __m256i quantized = mm256_quantize_ps_epi32(source, scale_v);
        const __m256i packed    = mm256_pack_epi32_avx2<BIT_WIDTH>(quantized);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output), _mm256_castsi256_si128(packed));
        input += 8;
        output += BIT_WIDTH;
    }

    if constexpr (remaining) {
        std::vector<uint32_t> block_values(remaining);
#pragma GCC unroll 8
        for (uint32_t i = 0; i < remaining; i++) {
            block_values[i] = static_cast<uint32_t>(quantize_ps_epi32(input[i], scale));
        }

        pack_epi32_fallback<BIT_WIDTH>(block_values, output);
    }

    return 0;
}

/**
 * @brief Compress multiple 512-bit blocks using AVX2 instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 16).
 *
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @param blocks number of 512-bit blocks to compress.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 support.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 16)
int mm256_compress_blocks_avx2(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output,
                               const uint32_t blocks) {
    const float_t* block_input = input;
    uint8_t* block_output      = output;

    for (uint32_t block = 0; block < blocks; block++) {
        mm256_compress_block_avx2<BIT_WIDTH>(block_input, scale, block_output);
        block_input += 512 / BIT_WIDTH;
        block_output += (BIT_WIDTH * 512) / 8;
    }

    return 0;
}

/**
 * @brief Compress a single 512-bit block using AVX2 instructions.
 *
 * @param bit_width bit width per value in the packed representation (1 to 16).
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 support.
 */
int mm256_compress_block_avx2(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output);

/**
 * @brief Compress multiple 512-bit blocks using AVX2 instructions.
 *
 * @param bit_width bit width per value in the packed representation (1 to 16).
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @param blocks number of 512-bit blocks to compress.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 and BMI2 support.
 */
int mm256_compress_blocks_avx2(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output,
                               uint32_t blocks);

#endif  // PERNIX_AVX2_ENABLED

#ifdef PERNIX_AVX512_VBMI_ENABLED

/**
 * @brief Compress a single 512-bit block using AVX-512 and AVX-512-VBMI instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (8 to 16).
 *
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX-512 and AVX-512-VBMI support.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
int mm512_compress_block_avx512vbmi(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = 512 / BIT_WIDTH;
    constexpr uint32_t iterations_32      = elements_per_block / 32;
    constexpr uint8_t remaining           = elements_per_block - iterations_32 * 32;

    const __m512 scale_v    = _mm512_set1_ps(scale);
    const __m256 scale_v256 = _mm256_set1_ps(scale);
#pragma GCC unroll 2
    for (uint32_t iter = 0; iter < iterations_32; iter++) {
        const __m512 source1     = _mm512_loadu_ps(input);
        const __m512 source2     = _mm512_loadu_ps(input + 16);
        const __m512i quantized1 = mm512_quantize_ps_epi32(source1, scale_v);
        const __m512i quantized2 = mm512_quantize_ps_epi32(source2, scale_v);
        const __m512i packed     = mm1024_pack_epi32_avx512vbmi<BIT_WIDTH>(quantized1, quantized2);
        if constexpr (BIT_WIDTH == 8) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(output), _mm512_castsi512_si256(packed));
        } else {
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(output), packed);
        }

        input += 16;
        output += 2 * BIT_WIDTH;
    }

    constexpr __mmask16 store_mask = (1U << ((remaining * BIT_WIDTH) / 8)) - 1;
    if constexpr (remaining > 16) {
        const __m512 source1     = _mm512_loadu_ps(input);
        const __m512 source2     = _mm512_loadu_ps(input + 16);
        const __m512i quantized1 = mm512_quantize_ps_epi32(source1, scale_v);
        const __m512i quantized2 = mm512_quantize_ps_epi32(source2, scale_v);
        const __m512i packed     = mm1024_pack_epi32_avx512vbmi<BIT_WIDTH>(quantized1, quantized2);
        _mm512_mask_storeu_epi8(reinterpret_cast<__m512i*>(output), store_mask, packed);
    } else if constexpr (remaining > 8) {
        const __m512 source     = _mm512_loadu_ps(input);
        const __m512i quantized = mm512_quantize_ps_epi32(source, scale_v);
        const __m256i packed    = mm512_pack_epi32_avx512vbmi<BIT_WIDTH>(quantized);
        _mm256_mask_storeu_epi8(reinterpret_cast<__m256i*>(output), store_mask, packed);
    } else if constexpr (remaining > 0) {
        const __m256 source     = _mm256_loadu_ps(input);
        const __m256i quantized = mm256_quantize_ps_epi32(source, scale_v256);
        const __m128i packed    = mm256_pack_epi32_avx512vbmi<BIT_WIDTH>(quantized);
        _mm_mask_storeu_epi8(reinterpret_cast<__m128i*>(output), store_mask, packed);
    }

    return 0;
}

/**
 * @brief Compress multiple 512-bit blocks using AVX-512 and AVX-512-VBMI instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 16).
 *
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @param blocks number of 512-bit blocks to compress.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX-512 and AVX-512-VBMI support.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 16)
int mm512_compress_blocks_avx512vbmi(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output,
                                     const uint32_t blocks) {
    const float_t* block_input = input;
    uint8_t* block_output      = output;

    for (uint32_t block = 0; block < blocks; block++) {
        mm512_compress_block_avx512vbmi<BIT_WIDTH>(block_input, scale, block_output);
        block_input += 512 / BIT_WIDTH;
        block_output += (BIT_WIDTH * 512) / 8;
    }

    return 0;
}

/**
 * @brief Compress a single 512-bit block using AVX-512 and AVX-512-VBMI instructions.
 *
 * @param bit_width bit width per value in the packed representation (1 to 16).
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX-512 and AVX-512-VBMI support.
 */
int mm512_compress_block_avx512vbmi(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output);

/**
 * @brief Compress multiple 512-bit blocks using AVX-512 and AVX-512-VBMI instructions.
 *
 * @param bit_width bit width per value in the packed representation (1 to 16).
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @param blocks number of 512-bit blocks to compress.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX-512 and AVX-512-VBMI support.
 */
int mm512_compress_blocks_avx512vbmi(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output,
                                     uint32_t blocks);

#endif  // PERNIX_AVX512_VBMI_ENABLED

/**
 * @brief Compress a single 512-bit block using fallback scalar implementation.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 16).
 *
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @return int status code (0 for success).
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 16)
int compress_block_fallback(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = 512 / BIT_WIDTH;

    std::vector<uint32_t> block_values(elements_per_block);
#pragma GCC unroll 512
    for (uint32_t i = 0; i < elements_per_block; i++) {
        block_values[i] = static_cast<uint32_t>(quantize_ps_epi32(input[i], scale));
    }

    pack_epi32_fallback<BIT_WIDTH>(block_values, output);
    return 0;
}

/**
 * @brief Compress multiple 512-bit blocks using fallback scalar implementation.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 16).
 *
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @param blocks number of 512-bit blocks to compress.
 * @return int status code (0 for success).
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 16)
int compress_blocks_fallback(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output, const uint32_t blocks) {
    const float_t* block_input = input;
    uint8_t* block_output      = output;

    for (uint32_t block = 0; block < blocks; block++) {
        compress_block_fallback<BIT_WIDTH>(block_input, scale, block_output);
        block_input += 512 / BIT_WIDTH;
        block_output += (BIT_WIDTH * 512) / 8;
    }

    return 0;
}

/**
 * @brief Compress a single 512-bit block using fallback scalar implementation.
 *
 * @param bit_width bit width per value in the packed representation (1 to 16).
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @return int status code (0 for success).
 */
int compress_block_fallback(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output);

/**
 * @brief Compress multiple 512-bit blocks using fallback scalar implementation.
 *
 * @param bit_width bit width per value in the packed representation (1 to 16).
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @param blocks number of 512-bit blocks to compress.
 * @return int status code (0 for success).
 */
int compress_blocks_fallback(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output,
                             uint32_t blocks);

/**
 * @brief Compress multiple 512-bit blocks using the specified implementation.
 *
 * @param implementation the available implementation to use for compression.
 * @param bit_width bit width per value in the packed representation (1 to 16).
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @param blocks number of 512-bit blocks to compress.
 * @return int status code (0 for success).
 */
int compress_blocks(AvailableImplementations implementation, uint8_t bit_width, const float_t* input, float_t scale, uint8_t* output,
                    uint32_t blocks);

/**
 * @brief Compress a single 512-bit block using the specified implementation.
 *
 * @param implementation the available implementation to use for compression.
 * @param bit_width bit width per value in the packed representation (1 to 16).
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @return int status code (0 for success).
 */
int compress_block(AvailableImplementations implementation, uint8_t bit_width, const float_t* input, float_t scale, uint8_t* output);
}  // namespace pernix

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
#ifdef PERNIX_AVX2_ENABLED
int mm256_compress_block_bmi2_c(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output);

int mm256_compress_blocks_bmi2_c(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output,
                                 uint32_t blocks);

int mm256_compress_block_avx2_c(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output);

int mm256_compress_blocks_avx2_c(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output,
                                 uint32_t blocks);
#endif  // PERNIX_AVX2_ENABLED

#ifdef PERNIX_AVX512_VBMI_ENABLED

int mm512_compress_block_avx512vbmi_c(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output);

int mm512_compress_blocks_avx512vbmi_c(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output,
                                       uint32_t blocks);

#endif  // PERNIX_AVX512_VBMI_ENABLED

int compress_block_fallback_c(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output);

int compress_blocks_fallback_c(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output,
                               uint32_t blocks);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // PERNIX_COMPRESSION_H
