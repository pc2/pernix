#ifndef LIBCOMPRESSION_DECOMPRESSION_H
#define LIBCOMPRESSION_DECOMPRESSION_H

#include <libcompression/helper.h>
#include <cstdint>
#include <cmath>
#include <vector>

#include <libcompression/bitpacking/unpacking.h>
#include <libcompression/quantization/dequantization.h>

using namespace libcompression::bitpacking;
using namespace libcompression::quantization;

namespace libcompression {
#ifdef LIBCOMPRESSION_AVX2_ENABLED
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
int mm256_decompress_block_bmi2(const uint8_t* __restrict__ input, const float_t scale,
                                float_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = 512 / BIT_WIDTH;
    constexpr uint32_t iterations_8       = elements_per_block / 8;
    constexpr uint8_t remaining           = elements_per_block - iterations_8 * 8;

    const __m256 scale_v = _mm256_set1_ps(scale);
#pragma GCC unroll 4
    for (uint32_t iter = 0; iter < iterations_8; iter++) {
        const __m256i unpacked   = mm256_unpack_epi32_bmi2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m256 dequantized = mm256_dequantize_epi32(unpacked, scale_v);
        _mm256_storeu_ps(output, dequantized);
        input += BIT_WIDTH;
        output += 8;
    }

    constexpr __mmask8 remaining_mask = (1 << remaining) - 1;
    if constexpr (remaining > 0) {
        const __m256i unpacked   = mm256_unpack_epi32_bmi2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m256 dequantized = mm256_dequantize_epi32(unpacked, scale_v);
        _mm256_maskstore_ps(output, libcompression::internal::mm256_convert_vmask_epi32(remaining_mask),
                            dequantized);
    }

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
int mm256_decompress_blocks_bmi2(const uint8_t* __restrict__ input, const float_t scale,
                                 float_t* __restrict__ output, const uint32_t blocks) {
    const uint8_t* block_input = input;
    float_t* block_output      = output;

    for (uint32_t block = 0; block < blocks; block++) {
        mm256_decompress_block_bmi2<BIT_WIDTH, SIGN_VALUES>(block_input, scale, block_output);
        block_input += (BIT_WIDTH * 512) / 8;
        block_output += 512 / BIT_WIDTH;
    }

    return 0;
}

int mm256_decompress_block_bmi2(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale,
                                float_t* __restrict__ output);

int mm256_decompress_blocks_bmi2(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale,
                                 float_t* __restrict__ output, uint32_t blocks);


template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
int mm256_decompress_block_avx2(const uint8_t* __restrict__ input, const float_t scale,
                                float_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = 512 / BIT_WIDTH;
    constexpr uint32_t iterations_8       = elements_per_block / 8;
    constexpr uint8_t remaining           = elements_per_block - iterations_8 * 8;

    const __m256 scale_v = _mm256_set1_ps(scale);
#pragma GCC unroll 4
    for (uint32_t iter = 0; iter < iterations_8; iter++) {
        const __m256i unpacked   = mm256_unpack_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m256 dequantized = mm256_dequantize_epi32(unpacked, scale_v);
        _mm256_storeu_ps(output, dequantized);
        input += BIT_WIDTH;
        output += 8;
    }

    constexpr __mmask8 remaining_mask = (1 << remaining) - 1;
    if constexpr (remaining > 0) {
        const __m256i unpacked   = mm256_unpack_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m256 dequantized = mm256_dequantize_epi32(unpacked, scale_v);
        _mm256_maskstore_ps(output, internal::mm256_convert_vmask_epi32(remaining_mask), dequantized);
    }

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
int mm256_decompress_blocks_avx2(const uint8_t* __restrict__ input, const float_t scale,
                                 float_t* __restrict__ output, const uint32_t blocks) {
    const uint8_t* block_input = input;
    float_t* block_output      = output;

    for (uint32_t block = 0; block < blocks; block++) {
        mm256_decompress_block_avx2<BIT_WIDTH, SIGN_VALUES>(block_input, scale, block_output);
        block_input += (BIT_WIDTH * 512) / 8;
        block_output += 512 / BIT_WIDTH;
    }

    return 0;
}

int mm256_decompress_block_avx2(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale,
                                float_t* __restrict__ output);

int mm256_decompress_blocks_avx2(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale,
                                 float_t* __restrict__ output, uint32_t blocks);
# endif // LIBCOMPRESSION_AVX2_ENABLED

#ifdef LIBCOMPRESSION_AVX512_VBMI_ENABLED

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
int mm512_decompress_block_avx512vbmi(const uint8_t* __restrict__ input, const float_t scale,
                                      float_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = 512 / BIT_WIDTH;
    constexpr uint32_t iterations_16      = elements_per_block / 16;
    constexpr uint8_t remaining           = elements_per_block - iterations_16 * 16;

    const __m512 scale_v = _mm512_set1_ps(scale);
#pragma GCC unroll 4
    for (uint32_t iter = 0; iter < iterations_16; iter++) {
        const __m512i unpacked   = mm512_unpack_epi32_avx512vbmi<BIT_WIDTH, SIGN_VALUES>(input);
        const __m512 dequantized = mm512_dequantize_epi32(unpacked, scale_v);
        _mm512_storeu_ps(output, dequantized);
        input += BIT_WIDTH;
        output += 8;
    }

    constexpr __mmask16 remaining_mask = (1 << remaining) - 1;
    if constexpr (remaining > 8) {
        const __m512i unpacked   = mm512_unpack_epi32_avx512vbmi<BIT_WIDTH, SIGN_VALUES>(input);
        const __m512 dequantized = mm512_dequantize_epi32(unpacked, scale_v);
        // _mm512_maskstore_ps(output, internal::mm256_convert_vmask_epi32(remaining_mask), dequantized);
        _mm512_mask_storeu_ps(output, remaining_mask, dequantized);
    } else if constexpr (remaining > 0) {
        const __m256 scale_v256  = _mm256_set1_ps(scale);
        const __m256i unpacked   = mm256_unpack_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m256 dequantized = mm256_dequantize_epi32(unpacked, scale_v256);
        _mm256_maskstore_ps(output, internal::mm256_convert_vmask_epi32(remaining_mask), dequantized);
    }

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
int mm512_decompress_blocks_avx512vbmi(const uint8_t* __restrict__ input, const float_t scale,
                                       float_t* __restrict__ output, const uint32_t blocks) {
    const uint8_t* block_input = input;
    float_t* block_output      = output;

    for (uint32_t block = 0; block < blocks; block++) {
        mm512_decompress_block_avx512vbmi<BIT_WIDTH, SIGN_VALUES>(block_input, scale, block_output);
        block_input += (BIT_WIDTH * 512) / 8;
        block_output += 512 / BIT_WIDTH;
    }

    return 0;
}

int mm512_decompress_block_avx512vbmi(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale,
                                      float_t* __restrict__ output);

int mm512_decompress_blocks_avx512vbmi(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale,
                                       float_t* __restrict__ output, uint32_t blocks);

#endif // LIBCOMPRESSION_AVX512_VBMI_ENABLED

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
int decompress_block_fallback(const uint8_t* __restrict__ input, const float_t scale,
                              float_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = 512 / BIT_WIDTH;

    const std::vector<int32_t> block_values = unpack_epi32_fallback<BIT_WIDTH, SIGN_VALUES>(
        input, elements_per_block);

#pragma GCC unroll 512
    for (uint32_t i = 0; i < elements_per_block; i++) {
        output[i] = dequantize_epi32(block_values[i], scale);
    }

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
int decompress_blocks_fallback(const uint8_t* __restrict__ input, const float_t scale,
                               float_t* __restrict__ output, const uint32_t blocks) {
    const uint8_t* block_input = input;
    float_t* block_output      = output;

    for (uint32_t block = 0; block < blocks; block++) {
        decompress_block_fallback<BIT_WIDTH, SIGN_VALUES>(block_input, scale, block_output);
        block_input += (BIT_WIDTH * 512) / 8;
        block_output += 512 / BIT_WIDTH;
    }

    return 0;
}


int decompress_block_fallback(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale,
                              float_t* __restrict__ output);

int decompress_blocks_fallback(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale,
                               float_t* __restrict__ output, uint32_t blocks);

int decompress_blocks(AvailableImplementations implementation, uint8_t bit_width, const uint8_t* input, float_t scale, float_t* output,
                      uint32_t blocks);

int decompress_block(AvailableImplementations implementation, uint8_t bit_width, const uint8_t* input, float_t scale, float_t* output);
} // namespace libcompression

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
#ifdef LIBCOMPRESSION_AVX2_ENABLED
int mm256_decompress_block_bmi2_c(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale,
                                  float_t* __restrict__ output);

int mm256_decompress_blocks_bmi2_c(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale,
                                   float_t* __restrict__ output, uint32_t blocks);

int mm256_decompress_block_avx2_c(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale,
                                  float_t* __restrict__ output);

int mm256_decompress_blocks_avx2_c(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale,
                                   float_t* __restrict__ output, uint32_t blocks);
#endif // LIBCOMPRESSION_AVX2_ENABLED

#ifdef LIBCOMPRESSION_AVX512_VBMI_ENABLED

int mm512_decompress_block_avx512vbmi_c(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale,
                                        float_t* __restrict__ output);

int mm512_decompress_blocks_avx512vbmi_c(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale,
                                         float_t* __restrict__ output, uint32_t blocks);

#endif // LIBCOMPRESSION_AVX512_VBMI_ENABLED

int decompress_block_fallback_c(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale,
                                float_t* __restrict__ output);

int decompress_blocks_fallback_c(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale,
                                 float_t* __restrict__ output, uint32_t blocks);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif  // LIBCOMPRESSION_DECOMPRESSION_H
