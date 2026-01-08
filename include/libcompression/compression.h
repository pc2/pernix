#ifndef LIBCOMPRESSION_COMPRESSION_H
#define LIBCOMPRESSION_COMPRESSION_H

#include <libcompression/bitpacking/packing.h>
#include <libcompression/helper.h>
#include <libcompression/quantization/quantization.h>

#include <cmath>
#include <cstdint>

using namespace libcompression::bitpacking;
using namespace libcompression::quantization;

namespace libcompression {
#ifdef LIBCOMPRESSION_AVX2_ENABLED
    template<uint8_t BIT_WIDTH>
        requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
    int mm256_compress_block_bmi2(const float_t * __restrict__ input, const float_t scale,
                                  uint8_t * __restrict__ output) {
        constexpr uint32_t elements_per_block = 512 / BIT_WIDTH;
        constexpr uint32_t iterations_8 = elements_per_block / 8;
        constexpr uint8_t remaining = elements_per_block - iterations_8 * 8;

        const __m256 scale_v = _mm256_set1_ps(scale);
#pragma GCC unroll 4
        for (uint32_t iter = 0; iter < iterations_8; iter++) {
            const __m256 source = _mm256_loadu_ps(input);
            const __m256i quantized = mm256_quantize_ps_epi32(source, scale_v);
            const __m256i packed = mm256_pack_epi32_bmi2 < BIT_WIDTH > (quantized);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(output), packed);
            input += 8;
            output += BIT_WIDTH;
        }

        if constexpr (remaining) {
            // TODO: handle remaining values
        }

        return 0;
    }

    template<uint8_t BIT_WIDTH>
        requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
    int mm256_compress_blocks_bmi2(const float_t * __restrict__ input, const float_t scale,
                                   uint8_t * __restrict__ output,
                                   const uint32_t blocks) {
        const float_t *block_input = input;
        uint8_t *block_output = output;

        for (uint32_t block = 0; block < blocks; block++) {
            mm256_compress_block_bmi2<BIT_WIDTH>(block_input, scale, block_output);
            block_input += 512 / BIT_WIDTH;
            block_output += (BIT_WIDTH * 512) / 8;
        }

        return 0;
    }

    int mm256_compress_block_bmi2(uint8_t bit_width, const float_t * __restrict__ input, float_t scale,
                                  uint8_t * __restrict__ output);

    int mm256_compress_blocks_bmi2(uint8_t bit_width, const float_t * __restrict__ input, float_t scale,
                                   uint8_t * __restrict__ output,
                                   uint32_t blocks);

    template<uint8_t BIT_WIDTH>
        requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
    int mm256_compress_block_avx2(const float_t * __restrict__ input, const float_t scale,
                                  uint8_t * __restrict__ output) {
        constexpr uint32_t elements_per_block = 512 / BIT_WIDTH;
        constexpr uint32_t iterations_8 = elements_per_block / 8;
        constexpr uint8_t remaining = elements_per_block - iterations_8 * 8;

        const __m256 scale_v = _mm256_set1_ps(scale);
#pragma GCC unroll 8
        for (uint32_t iter = 0; iter < iterations_8; iter++) {
            const __m256 source = _mm256_loadu_ps(input);
            const __m256i quantized = mm256_quantize_ps_epi32(source, scale_v);
            const __m256i packed = mm256_pack_epi32_avx2 < BIT_WIDTH > (quantized);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(output), _mm256_castsi256_si128(packed));
            input += 8;
            output += BIT_WIDTH;
        }

        if constexpr (remaining) {
            // TODO: handle remaining values
        }

        return 0;
    }

    template<uint8_t BIT_WIDTH>
        requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
    int mm256_compress_blocks_avx2(const float_t * __restrict__ input, const float_t scale,
                                   uint8_t * __restrict__ output,
                                   const uint32_t blocks) {
        return 0;
    }

    int mm256_compress_block_avx2(uint8_t bit_width, const float_t * __restrict__ input, float_t scale,
                                  uint8_t * __restrict__ output);

    int mm256_compress_blocks_avx2(uint8_t bit_width, const float_t * __restrict__ input, float_t scale,
                                   uint8_t * __restrict__ output,
                                   uint32_t blocks);

#endif  // LIBCOMPRESSION_AVX2_ENABLED

#ifdef LIBCOMPRESSION_AVX512_VBMI_ENABLED

    template<uint8_t BIT_WIDTH>
        requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
    int mm512_compress_block_avx512vbmi(const float_t * __restrict__ input, const float_t scale,
                                        uint8_t * __restrict__ output) {
        constexpr uint32_t elements_per_block = 512 / BIT_WIDTH;
        constexpr uint32_t iterations_16 = elements_per_block / 16;
        constexpr uint8_t remaining = elements_per_block - iterations_16 * 16;

        const __m512 scale_v = _mm512_set1_ps(scale);
#pragma GCC unroll 4
        for (uint32_t iter = 0; iter < iterations_16; iter++) {
            const __m512 source = _mm512_loadu_ps(input);
            const __m512i quantized = mm512_quantize_ps_epi32(source, scale_v);
            const __m256i packed = mm512_pack_epi32_avx512vbmi < BIT_WIDTH > (quantized);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(output), packed);
            input += 16;
            output += 2 * BIT_WIDTH;
        }

        if constexpr (remaining) {
            const __m512 source = _mm512_loadu_ps(input);
            const __m512i quantized = mm512_quantize_ps_epi32(source, scale_v);
            const __m256i packed = mm512_pack_epi32_avx512vbmi < BIT_WIDTH > (quantized);
            // _mm256_storeu_si256(reinterpret_cast<__m256i *>(output), packed);
        }

        return 0;
    }

    template<uint8_t BIT_WIDTH>
        requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
    int mm512_compress_blocks_avx512vbmi(const float_t * __restrict__ input, const float_t scale,
                                         uint8_t * __restrict__ output,
                                         const uint32_t blocks) {
        return 0;
    }

    int mm512_compress_block_avx512vbmi(uint8_t bit_width, const float_t * __restrict__ input, float_t scale,
                                        uint8_t * __restrict__ output);

    int mm512_compress_blocks_avx512vbmi(uint8_t bit_width, const float_t * __restrict__ input, float_t scale,
                                         uint8_t * __restrict__ output,
                                         uint32_t blocks);

#endif  // LIBCOMPRESSION_AVX512_VBMI_ENABLED

    template<uint8_t BIT_WIDTH>
        requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
    int compress_block_fallback(const float_t * __restrict__ input, const float_t scale,
                                uint8_t * __restrict__ output) {
        constexpr uint32_t elements_per_block = 512 / BIT_WIDTH;

        // const auto packed = pack_epi32_fallback<BIT_WIDTH>();

        return 0;
    }

    template<uint8_t BIT_WIDTH>
        requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
    int compress_blocks_fallback(const float_t * __restrict__ input, const float_t scale, uint8_t * __restrict__ output,
                                 const uint32_t blocks) {
        return 0;
    }

    int compress_block_fallback(uint8_t bit_width, const float_t * __restrict__ input, float_t scale,
                                uint8_t * __restrict__ output);

    int compress_blocks_fallback(uint8_t bit_width, const float_t * __restrict__ input, float_t scale,
                                 uint8_t * __restrict__ output,
                                 uint32_t blocks);

    int compress_blocks(AvailableImplementations implementation, uint8_t bit_width, const uint8_t *input, float_t scale,
                        float_t *output,
                        uint32_t blocks);

    int compress_block(AvailableImplementations implementation, uint8_t bit_width, const uint8_t *input, float_t scale,
                       float_t *output);
} // namespace libcompression

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
#ifdef LIBCOMPRESSION_AVX2_ENABLED
int mm256_compress_block_bmi2_c(uint8_t bit_width, const float_t * __restrict__ input, float_t scale,
                                uint8_t * __restrict__ output);

int mm256_compress_blocks_bmi2_c(uint8_t bit_width, const float_t * __restrict__ input, float_t scale,
                                 uint8_t * __restrict__ output,
                                 uint32_t blocks);

int mm256_compress_block_avx2_c(uint8_t bit_width, const float_t * __restrict__ input, float_t scale,
                                uint8_t * __restrict__ output);

int mm256_compress_blocks_avx2_c(uint8_t bit_width, const float_t * __restrict__ input, float_t scale,
                                 uint8_t * __restrict__ output,
                                 uint32_t blocks);
#endif  // LIBCOMPRESSION_AVX2_ENABLED

#ifdef LIBCOMPRESSION_AVX512_VBMI_ENABLED

int mm512_compress_block_avx512vbmi_c(uint8_t bit_width, const float_t * __restrict__ input, float_t scale,
                                      uint8_t * __restrict__ output);

int mm512_compress_blocks_avx512vbmi_c(uint8_t bit_width, const float_t * __restrict__ input, float_t scale,
                                       uint8_t * __restrict__ output,
                                       uint32_t blocks);

#endif  // LIBCOMPRESSION_AVX512_VBMI_ENABLED

int compress_block_fallback_c(uint8_t bit_width, const float_t * __restrict__ input, float_t scale,
                              uint8_t * __restrict__ output);

int compress_blocks_fallback_c(uint8_t bit_width, const float_t * __restrict__ input, float_t scale,
                               uint8_t * __restrict__ output,
                               uint32_t blocks);

#ifdef __cplusplus
} // extern "C"
#endif  // __cplusplus

#endif  // LIBCOMPRESSION_COMPRESSION_H