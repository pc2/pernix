#include "libcompression/decompression.h"

namespace libcompression {
#ifdef LIBCOMPRESSION_AVX2_ENABLED
int mm256_decompress_block_bmi2(const uint8_t bit_width, const uint8_t* input, float_t scale, float_t* output) {
    switch (bit_width) {
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 1, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 2, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 3, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 4, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 5, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 6, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 7, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 8, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 9, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 10, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 11, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 12, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 13, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 14, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 15, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 16, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 17, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 18, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 19, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 20, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 21, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 22, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 23, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 24, true)
        default:
            return -1;
    }
}

int mm256_decompress_blocks_bmi2(const uint8_t bit_width, const uint8_t* input, const float_t scale, float_t* output,
                                 const uint32_t blocks) {
    switch (bit_width) {
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 1, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 2, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 3, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 4, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 5, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 6, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 7, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 8, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 9, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 10, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 11, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 12, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 13, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 14, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 15, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 16, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 17, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 18, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 19, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 20, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 21, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 22, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 23, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 24, true)
        default:
            return -1;
    }
}

int mm256_decompress_block_avx2(const uint8_t bit_width, const uint8_t* input, float_t scale, float_t* output) {
    switch (bit_width) {
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 1, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 2, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 3, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 4, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 5, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 6, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 7, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 8, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 9, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 10, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 11, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 12, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 13, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 14, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 15, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 16, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 17, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 18, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 19, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 20, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 21, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 22, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 23, true)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 24, true)
        default:
            return -1;
    }
}

int mm256_decompress_blocks_avx2(const uint8_t bit_width, const uint8_t* input, const float_t scale, float_t* output,
                                 const uint32_t blocks) {
    switch (bit_width) {
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 1, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 2, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 3, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 4, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 5, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 6, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 7, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 8, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 9, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 10, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 11, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 12, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 13, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 14, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 15, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 16, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 17, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 18, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 19, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 20, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 21, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 22, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 23, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 24, true)
        default:
            return -1;
    }
}
#endif // LIBCOMPRESSION_AVX2_ENABLED

int decompress_block_fallback(const uint8_t bit_width, const uint8_t* input, const float_t scale, float_t* output) {
    switch (bit_width) {
        COMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 1, true)
        COMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 2, true)
        COMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 3, true)
        COMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 4, true)
        COMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 5, true)
        COMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 6, true)
        COMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 7, true)
        COMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 8, true)
        COMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 9, true)
        COMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 10, true)
        COMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 11, true)
        COMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 12, true)
        COMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 13, true)
        COMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 14, true)
        COMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 15, true)
        COMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 16, true)
        COMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 17, true)
        COMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 18, true)
        COMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 19, true)
        COMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 20, true)
        COMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 21, true)
        COMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 22, true)
        COMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 23, true)
        COMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 24, true)
        default:
            return -1;
    }
}

int decompress_blocks_fallback(const uint8_t bit_width, const uint8_t* input, const float_t scale, float_t* output, const uint32_t blocks) {
    switch (bit_width) {
        COMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 1, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 2, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 3, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 4, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 5, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 6, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 7, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 8, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 9, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 10, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 11, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 12, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 13, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 14, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 15, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 16, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 17, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 18, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 19, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 20, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 21, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 22, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 23, true)
        COMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 24, true)
        default:
            return -1;
    }
}

int decompress_blocks(const AvailableImplementations implementation, const uint8_t bit_width, const uint8_t* input, const float_t scale,
                      float_t* output, const uint32_t blocks) {
    switch (implementation) {
#ifdef LIBCOMPRESSION_AVX512_VBMI_ENABLED
        case AvailableImplementations::AVX512_VBMI:
            return mm512_decompress_blocks_avx512vbmi(bit_width, input, scale, output, blocks);
#endif // LIBCOMPRESSION_AVX512_VBMI_ENABLED
#ifdef LIBCOMPRESSION_AVX512_ENABLED
        case AvailableImplementations::AVX512:
            return decompress_blocks_fallback(bit_width, input, scale, output, blocks);
#endif // LIBCOMPRESSION_AVX512_ENABLED
#ifdef LIBCOMPRESSION_AVX2_ENABLED
        case AvailableImplementations::AVX2_BMI2:
            return mm256_decompress_blocks_bmi2(bit_width, input, scale, output, blocks);
        case AvailableImplementations::AVX2:
            return mm256_decompress_blocks_avx2(bit_width, input, scale, output, blocks);
#endif // LIBCOMPRESSION_AVX2_ENABLED
        case AvailableImplementations::FALLBACK:
        default:
            return decompress_blocks_fallback(bit_width, input, scale, output, blocks);
    }
}

int decompress_block(const AvailableImplementations implementation, const uint8_t bit_width, const uint8_t* input, const float_t scale,
                     float_t* output) {
    switch (implementation) {
#ifdef LIBCOMPRESSION_AVX512_VBMI_ENABLED
        case AvailableImplementations::AVX512_VBMI:
            return mm512_decompress_block_avx512vbmi(bit_width, input, scale, output);
#endif // LIBCOMPRESSION_AVX512_VBMI_ENABLED
#ifdef LIBCOMPRESSION_AVX512_ENABLED
        case AvailableImplementations::AVX512:
            return decompress_block_fallback(bit_width, input, scale, output);
#endif // LIBCOMPRESSION_AVX512_ENABLED
#ifdef LIBCOMPRESSION_AVX2_ENABLED
        case AvailableImplementations::AVX2_BMI2:
            return mm256_decompress_block_bmi2(bit_width, input, scale, output);
        case AvailableImplementations::AVX2:
            return mm256_decompress_block_avx2(bit_width, input, scale, output);
#endif // LIBCOMPRESSION_AVX2_ENABLED
        case AvailableImplementations::FALLBACK:
        default:
            return decompress_block_fallback(bit_width, input, scale, output);
    }
}
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
#ifdef LIBCOMPRESSION_AVX2_ENABLED
int mm256_decompress_block_bmi2_c(const uint8_t bit_width, const uint8_t* input, const float_t scale, float_t* output) {
    return libcompression::mm256_decompress_block_bmi2(bit_width, input, scale, output);
}

int mm256_decompress_blocks_bmi2_c(const uint8_t bit_width, const uint8_t* input, const float_t scale, float_t* output,
                                   const uint32_t blocks) {
    return libcompression::mm256_decompress_blocks_bmi2(bit_width, input, scale, output, blocks);
}

int mm256_decompress_block_avx2_c(const uint8_t bit_width, const uint8_t* __restrict__ input, const float_t scale,
                                  float_t* __restrict__ output) {
    return libcompression::mm256_decompress_block_avx2(bit_width, input, scale, output);
}

int mm256_decompress_blocks_avx2_c(const uint8_t bit_width, const uint8_t* input, const float_t scale, float_t* output,
                                   const uint32_t blocks) {
    return libcompression::mm256_decompress_blocks_avx2(bit_width, input, scale, output, blocks);
}
#endif // LIBCOMPRESSION_AVX2_ENABLED

int decompress_block_fallback_c(const uint8_t bit_width, const uint8_t* __restrict__ input, const float_t scale,
                                float_t* __restrict__ output) {
    return libcompression::decompress_block_fallback(bit_width, input, scale, output);
}

int decompress_blocks_fallback_c(const uint8_t bit_width, const uint8_t* input, const float_t scale, float_t* output,
                                 const uint32_t blocks) {
    return libcompression::decompress_blocks_fallback(bit_width, input, scale, output, blocks);
}

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
