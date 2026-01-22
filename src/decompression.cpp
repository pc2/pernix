#include "pernix/decompression.h"

namespace pernix {
#ifdef PERNIX_AVX2_ENABLED
int mm256_decompress_block_bmi2(const uint8_t bit_width, const uint8_t* input, const float_t scale, float_t* output) {
    switch (bit_width) {
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 8, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 9, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 10, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 11, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 12, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 13, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 14, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 15, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_bmi2, 16, true)
        default:
            return -1;
    }
}

int mm256_decompress_blocks_bmi2(const uint8_t bit_width, const uint8_t* input, const float_t scale, float_t* output,
                                 const uint32_t blocks) {
    switch (bit_width) {
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 8, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 9, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 10, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 11, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 12, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 13, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 14, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 15, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_bmi2, 16, true)
        default:
            return -1;
    }
}

int mm256_decompress_block_avx2(const uint8_t bit_width, const uint8_t* input, float_t scale, float_t* output) {
    switch (bit_width) {
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 8, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 9, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 10, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 11, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 12, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 13, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 14, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 15, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm256_decompress_block_avx2, 16, true)
        default:
            return -1;
    }
}

int mm256_decompress_blocks_avx2(const uint8_t bit_width, const uint8_t* input, const float_t scale, float_t* output,
                                 const uint32_t blocks) {
    switch (bit_width) {
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 8, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 9, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 10, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 11, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 12, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 13, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 14, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 15, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm256_decompress_blocks_avx2, 16, true)
        default:
            return -1;
    }
}
#endif  // PERNIX_AVX2_ENABLED

#ifdef PERNIX_AVX512_VBMI_ENABLED

int mm512_decompress_block_avx512vbmi(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output) {
    switch (bit_width) {
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm512_decompress_block_avx512vbmi, 8, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm512_decompress_block_avx512vbmi, 9, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm512_decompress_block_avx512vbmi, 10, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm512_decompress_block_avx512vbmi, 11, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm512_decompress_block_avx512vbmi, 12, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm512_decompress_block_avx512vbmi, 13, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm512_decompress_block_avx512vbmi, 14, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm512_decompress_block_avx512vbmi, 15, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(mm512_decompress_block_avx512vbmi, 16, true)
        default:
            return -1;
    }
}

int mm512_decompress_blocks_avx512vbmi(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output,
                                       uint32_t blocks) {
    switch (bit_width) {
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm512_decompress_blocks_avx512vbmi, 8, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm512_decompress_blocks_avx512vbmi, 9, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm512_decompress_blocks_avx512vbmi, 10, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm512_decompress_blocks_avx512vbmi, 11, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm512_decompress_blocks_avx512vbmi, 12, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm512_decompress_blocks_avx512vbmi, 13, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm512_decompress_blocks_avx512vbmi, 14, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm512_decompress_blocks_avx512vbmi, 15, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(mm512_decompress_blocks_avx512vbmi, 16, true)
        default:
            return -1;
    }
}

#endif

int decompress_block_fallback(const uint8_t bit_width, const uint8_t* input, const float_t scale, float_t* output) {
    switch (bit_width) {
        DECOMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 8, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 9, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 10, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 11, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 12, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 13, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 14, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 15, true)
        DECOMPRESSION_BLOCK_SWITCH_CASE(decompress_block_fallback, 16, true)
        default:
            return -1;
    }
}

int decompress_blocks_fallback(const uint8_t bit_width, const uint8_t* input, const float_t scale, float_t* output, const uint32_t blocks) {
    switch (bit_width) {
        DECOMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 8, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 9, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 10, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 11, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 12, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 13, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 14, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 15, true)
        DECOMPRESSION_BLOCKS_SWITCH_CASE(decompress_blocks_fallback, 16, true)
        default:
            return -1;
    }
}

int decompress_blocks(const AvailableImplementations implementation, const uint8_t bit_width, const uint8_t* input, const float_t scale,
                      float_t* output, const uint32_t blocks) {
    switch (implementation) {
#ifdef PERNIX_AVX512_VBMI_ENABLED
        case AvailableImplementations::AVX512_VBMI:
            return mm512_decompress_blocks_avx512vbmi(bit_width, input, scale, output, blocks);
#endif  // PERNIX_AVX512_VBMI_ENABLED
#ifdef PERNIX_AVX512_ENABLED
        case AvailableImplementations::AVX512:
            return decompress_blocks_fallback(bit_width, input, scale, output, blocks);
#endif  // PERNIX_AVX512_ENABLED
#ifdef PERNIX_AVX2_ENABLED
        case AvailableImplementations::AVX2_BMI2:
            return mm256_decompress_blocks_bmi2(bit_width, input, scale, output, blocks);
        case AvailableImplementations::AVX2:
            return mm256_decompress_blocks_avx2(bit_width, input, scale, output, blocks);
#endif  // PERNIX_AVX2_ENABLED
        case AvailableImplementations::FALLBACK:
        default:
            return decompress_blocks_fallback(bit_width, input, scale, output, blocks);
    }
}

int decompress_block(const AvailableImplementations implementation, const uint8_t bit_width, const uint8_t* input, const float_t scale,
                     float_t* output) {
    switch (implementation) {
#ifdef PERNIX_AVX512_VBMI_ENABLED
        case AvailableImplementations::AVX512_VBMI:
            return mm512_decompress_block_avx512vbmi(bit_width, input, scale, output);
#endif  // PERNIX_AVX512_VBMI_ENABLED
#ifdef PERNIX_AVX512_ENABLED
        case AvailableImplementations::AVX512:
            return decompress_block_fallback(bit_width, input, scale, output);
#endif  // PERNIX_AVX512_ENABLED
#ifdef PERNIX_AVX2_ENABLED
        case AvailableImplementations::AVX2_BMI2:
            return mm256_decompress_block_bmi2(bit_width, input, scale, output);
        case AvailableImplementations::AVX2:
            return mm256_decompress_block_avx2(bit_width, input, scale, output);
#endif  // PERNIX_AVX2_ENABLED
        case AvailableImplementations::FALLBACK:
        default:
            return decompress_block_fallback(bit_width, input, scale, output);
    }
}
}  // namespace pernix

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
#ifdef PERNIX_AVX2_ENABLED
int mm256_decompress_block_bmi2_c(const uint8_t bit_width, const uint8_t* input, const float_t scale, float_t* output) {
    return pernix::mm256_decompress_block_bmi2(bit_width, input, scale, output);
}

int mm256_decompress_blocks_bmi2_c(const uint8_t bit_width, const uint8_t* input, const float_t scale, float_t* output,
                                   const uint32_t blocks) {
    return pernix::mm256_decompress_blocks_bmi2(bit_width, input, scale, output, blocks);
}

int mm256_decompress_block_avx2_c(const uint8_t bit_width, const uint8_t* __restrict__ input, const float_t scale,
                                  float_t* __restrict__ output) {
    return pernix::mm256_decompress_block_avx2(bit_width, input, scale, output);
}

int mm256_decompress_blocks_avx2_c(const uint8_t bit_width, const uint8_t* input, const float_t scale, float_t* output,
                                   const uint32_t blocks) {
    return pernix::mm256_decompress_blocks_avx2(bit_width, input, scale, output, blocks);
}
#endif  // PERNIX_AVX2_ENABLED

#ifdef PERNIX_AVX512_VBMI_ENABLED

int mm512_decompress_block_avx512vbmi_c(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output) {
    return pernix::mm512_decompress_block_avx512vbmi(bit_width, input, scale, output);
}

int mm512_decompress_blocks_avx512vbmi_c(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output,
                                         uint32_t blocks) {
    return pernix::mm512_decompress_blocks_avx512vbmi(bit_width, input, scale, output, blocks);
}

#endif  // PERNIX_AVX512_VBMI_ENABLED

int decompress_block_fallback_c(const uint8_t bit_width, const uint8_t* __restrict__ input, const float_t scale,
                                float_t* __restrict__ output) {
    return pernix::decompress_block_fallback(bit_width, input, scale, output);
}

int decompress_blocks_fallback_c(const uint8_t bit_width, const uint8_t* input, const float_t scale, float_t* output,
                                 const uint32_t blocks) {
    return pernix::decompress_blocks_fallback(bit_width, input, scale, output, blocks);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
