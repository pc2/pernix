#include <pernix/compression.h>

namespace pernix {
#ifdef PERNIX_AVX2_ENABLED

int mm256_compress_block_bmi2(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale,
                              uint8_t* __restrict__ output) {
    switch (bit_width) {
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_compress_block_bmi2, 8)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_compress_block_bmi2, 9)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_compress_block_bmi2, 10)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_compress_block_bmi2, 11)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_compress_block_bmi2, 12)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_compress_block_bmi2, 13)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_compress_block_bmi2, 14)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_compress_block_bmi2, 15)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_compress_block_bmi2, 16)
        default:
            return -1;
    }
}

int mm256_compress_blocks_bmi2(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale,
                               uint8_t* __restrict__ output, uint32_t blocks) {
    switch (bit_width) {
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_compress_blocks_bmi2, 8)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_compress_blocks_bmi2, 9)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_compress_blocks_bmi2, 10)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_compress_blocks_bmi2, 11)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_compress_blocks_bmi2, 12)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_compress_blocks_bmi2, 13)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_compress_blocks_bmi2, 14)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_compress_blocks_bmi2, 15)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_compress_blocks_bmi2, 16)
        default:
            return -1;
    }
}

int mm256_compress_block_avx2(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale,
                              uint8_t* __restrict__ output) {
    switch (bit_width) {
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_compress_block_avx2, 8)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_compress_block_avx2, 9)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_compress_block_avx2, 10)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_compress_block_avx2, 11)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_compress_block_avx2, 12)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_compress_block_avx2, 13)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_compress_block_avx2, 14)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_compress_block_avx2, 15)
        COMPRESSION_BLOCK_SWITCH_CASE(mm256_compress_block_avx2, 16)
        default:
            return -1;
    }
}

int mm256_compress_blocks_avx2(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale,
                               uint8_t* __restrict__ output, uint32_t blocks) {
    switch (bit_width) {
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_compress_blocks_avx2, 8)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_compress_blocks_avx2, 9)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_compress_blocks_avx2, 10)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_compress_blocks_avx2, 11)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_compress_blocks_avx2, 12)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_compress_blocks_avx2, 13)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_compress_blocks_avx2, 14)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_compress_blocks_avx2, 15)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm256_compress_blocks_avx2, 16)
        default:
            return -1;
    }
}

#endif

#ifdef PERNIX_AVX512_VBMI_ENABLED

int mm512_compress_block_avx512vbmi(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale,
                                    uint8_t* __restrict__ output) {
    switch (bit_width) {
        COMPRESSION_BLOCK_SWITCH_CASE(mm512_compress_block_avx512vbmi, 8)
        COMPRESSION_BLOCK_SWITCH_CASE(mm512_compress_block_avx512vbmi, 9)
        COMPRESSION_BLOCK_SWITCH_CASE(mm512_compress_block_avx512vbmi, 10)
        COMPRESSION_BLOCK_SWITCH_CASE(mm512_compress_block_avx512vbmi, 11)
        COMPRESSION_BLOCK_SWITCH_CASE(mm512_compress_block_avx512vbmi, 12)
        COMPRESSION_BLOCK_SWITCH_CASE(mm512_compress_block_avx512vbmi, 13)
        COMPRESSION_BLOCK_SWITCH_CASE(mm512_compress_block_avx512vbmi, 14)
        COMPRESSION_BLOCK_SWITCH_CASE(mm512_compress_block_avx512vbmi, 15)
        COMPRESSION_BLOCK_SWITCH_CASE(mm512_compress_block_avx512vbmi, 16)
        default:
            return -1;
    }
}

int mm512_compress_blocks_avx512vbmi(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale,
                                     uint8_t* __restrict__ output, uint32_t blocks) {
    switch (bit_width) {
        COMPRESSION_BLOCKS_SWITCH_CASE(mm512_compress_blocks_avx512vbmi, 8)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm512_compress_blocks_avx512vbmi, 9)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm512_compress_blocks_avx512vbmi, 10)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm512_compress_blocks_avx512vbmi, 11)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm512_compress_blocks_avx512vbmi, 12)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm512_compress_blocks_avx512vbmi, 13)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm512_compress_blocks_avx512vbmi, 14)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm512_compress_blocks_avx512vbmi, 15)
        COMPRESSION_BLOCKS_SWITCH_CASE(mm512_compress_blocks_avx512vbmi, 16)
        default:
            return -1;
    }
}

#endif

int compress_block_fallback(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output) {
    switch (bit_width) {
        COMPRESSION_BLOCK_SWITCH_CASE(compress_block_fallback, 8)
        COMPRESSION_BLOCK_SWITCH_CASE(compress_block_fallback, 9)
        COMPRESSION_BLOCK_SWITCH_CASE(compress_block_fallback, 10)
        COMPRESSION_BLOCK_SWITCH_CASE(compress_block_fallback, 11)
        COMPRESSION_BLOCK_SWITCH_CASE(compress_block_fallback, 12)
        COMPRESSION_BLOCK_SWITCH_CASE(compress_block_fallback, 13)
        COMPRESSION_BLOCK_SWITCH_CASE(compress_block_fallback, 14)
        COMPRESSION_BLOCK_SWITCH_CASE(compress_block_fallback, 15)
        COMPRESSION_BLOCK_SWITCH_CASE(compress_block_fallback, 16)
        default:
            return -1;
    }
}

int compress_blocks_fallback(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output,
                             uint32_t blocks) {
    switch (bit_width) {
        COMPRESSION_BLOCKS_SWITCH_CASE(compress_blocks_fallback, 8)
        COMPRESSION_BLOCKS_SWITCH_CASE(compress_blocks_fallback, 9)
        COMPRESSION_BLOCKS_SWITCH_CASE(compress_blocks_fallback, 10)
        COMPRESSION_BLOCKS_SWITCH_CASE(compress_blocks_fallback, 11)
        COMPRESSION_BLOCKS_SWITCH_CASE(compress_blocks_fallback, 12)
        COMPRESSION_BLOCKS_SWITCH_CASE(compress_blocks_fallback, 13)
        COMPRESSION_BLOCKS_SWITCH_CASE(compress_blocks_fallback, 14)
        COMPRESSION_BLOCKS_SWITCH_CASE(compress_blocks_fallback, 15)
        COMPRESSION_BLOCKS_SWITCH_CASE(compress_blocks_fallback, 16)
        default:
            return -1;
    }
}

int compress_blocks(AvailableImplementations implementation, const uint8_t bit_width, const float_t* input, const float_t scale,
                    uint8_t* output, uint32_t blocks) {
    switch (implementation) {
#ifdef PERNIX_AVX512_VBMI_ENABLED
        case AvailableImplementations::AVX512_VBMI:
            return mm512_compress_blocks_avx512vbmi(bit_width, input, scale, output, blocks);
#endif  // PERNIX_AVX512_VBMI_ENABLED
#ifdef PERNIX_AVX512_ENABLED
        case AvailableImplementations::AVX512:
            return compress_blocks_fallback(bit_width, input, scale, output, blocks);
#endif  // PERNIX_AVX512_ENABLED
#ifdef PERNIX_AVX2_ENABLED
        case AvailableImplementations::AVX2_BMI2:
            return mm256_compress_blocks_bmi2(bit_width, input, scale, output, blocks);
        case AvailableImplementations::AVX2:
            return mm256_compress_blocks_avx2(bit_width, input, scale, output, blocks);
#endif  // PERNIX_AVX2_ENABLED
        case AvailableImplementations::FALLBACK:
        default:
            return compress_blocks_fallback(bit_width, input, scale, output, blocks);
    }
}

int compress_block(AvailableImplementations implementation, const uint8_t bit_width, const float_t* input, const float_t scale,
                   uint8_t* output) {
    switch (implementation) {
#ifdef PERNIX_AVX512_VBMI_ENABLED
        case AvailableImplementations::AVX512_VBMI:
            return mm512_compress_block_avx512vbmi(bit_width, input, scale, output);
#endif  // PERNIX_AVX512_VBMI_ENABLED
#ifdef PERNIX_AVX512_ENABLED
        case AvailableImplementations::AVX512:
            return compress_block_fallback(bit_width, input, scale, output);
#endif  // PERNIX_AVX512_ENABLED
#ifdef PERNIX_AVX2_ENABLED
        case AvailableImplementations::AVX2_BMI2:
            return mm256_compress_block_bmi2(bit_width, input, scale, output);
        case AvailableImplementations::AVX2:
            return mm256_compress_block_avx2(bit_width, input, scale, output);
#endif  // PERNIX_AVX2_ENABLED
        case AvailableImplementations::FALLBACK:
        default:
            return compress_block_fallback(bit_width, input, scale, output);
    }
}
}  // namespace pernix

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
#ifdef PERNIX_AVX2_ENABLED
int mm256_compress_block_bmi2_c(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale,
                                uint8_t* __restrict__ output) {
    return pernix::mm256_compress_block_bmi2(bit_width, input, scale, output);
}

int mm256_compress_blocks_bmi2_c(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale,
                                 uint8_t* __restrict__ output, uint32_t blocks) {
    return pernix::mm256_compress_blocks_bmi2(bit_width, input, scale, output, blocks);
}

int mm256_compress_block_avx2_c(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale,
                                uint8_t* __restrict__ output) {
    return pernix::mm256_compress_block_avx2(bit_width, input, scale, output);
}

int mm256_compress_blocks_avx2_c(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale,
                                 uint8_t* __restrict__ output, uint32_t blocks) {
    return pernix::mm256_compress_blocks_avx2(bit_width, input, scale, output, blocks);
}
#endif  // PERNIX_AVX2_ENABLED

#ifdef PERNIX_AVX512_VBMI_ENABLED

int mm512_compress_block_avx512vbmi_c(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale,
                                      uint8_t* __restrict__ output) {
    return pernix::mm512_compress_block_avx512vbmi(bit_width, input, scale, output);
}

int mm512_compress_blocks_avx512vbmi_c(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale,
                                       uint8_t* __restrict__ output, uint32_t blocks) {
    return pernix::mm512_compress_blocks_avx512vbmi(bit_width, input, scale, output, blocks);
}

#endif  // PERNIX_AVX512_VBMI_ENABLED

int compress_block_fallback_c(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale,
                              uint8_t* __restrict__ output) {
    return pernix::compress_block_fallback(bit_width, input, scale, output);
}

int compress_blocks_fallback_c(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale,
                               uint8_t* __restrict__ output, uint32_t blocks) {
    return pernix::compress_blocks_fallback(bit_width, input, scale, output, blocks);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus