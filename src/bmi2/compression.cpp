#include <pernix/bmi2/compression.h>
#include <pernix/detection.h>

#if defined(PERNIX_AVX2_ENABLED) && defined(PERNIX_BMI2_ENABLED)
#ifdef __cplusplus
namespace pernix {
extern "C" {
#endif

#define PERNIX_COMPRESS_BLOCK_CASE(N) \
    case N:                           \
        return mm256_compress_block_bmi2<N>(input, scale, output);

#define PERNIX_COMPRESS_BLOCKS_CASE(N) \
    case N:                            \
        return mm256_compress_blocks_bmi2<N>(input, scale, output, blocks);

int mm256_compress_block_bmi2(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale,
                              uint8_t* __restrict__ output) {
    switch (bit_width) {
        PERNIX_COMPRESS_BLOCK_CASE(1)
        PERNIX_COMPRESS_BLOCK_CASE(2)
        PERNIX_COMPRESS_BLOCK_CASE(3)
        PERNIX_COMPRESS_BLOCK_CASE(4)
        PERNIX_COMPRESS_BLOCK_CASE(5)
        PERNIX_COMPRESS_BLOCK_CASE(6)
        PERNIX_COMPRESS_BLOCK_CASE(7)
        PERNIX_COMPRESS_BLOCK_CASE(8)
        PERNIX_COMPRESS_BLOCK_CASE(9)
        PERNIX_COMPRESS_BLOCK_CASE(10)
        PERNIX_COMPRESS_BLOCK_CASE(11)
        PERNIX_COMPRESS_BLOCK_CASE(12)
        PERNIX_COMPRESS_BLOCK_CASE(13)
        PERNIX_COMPRESS_BLOCK_CASE(14)
        PERNIX_COMPRESS_BLOCK_CASE(15)
        PERNIX_COMPRESS_BLOCK_CASE(16)
        PERNIX_COMPRESS_BLOCK_CASE(17)
        PERNIX_COMPRESS_BLOCK_CASE(18)
        PERNIX_COMPRESS_BLOCK_CASE(19)
        PERNIX_COMPRESS_BLOCK_CASE(20)
        PERNIX_COMPRESS_BLOCK_CASE(21)
        PERNIX_COMPRESS_BLOCK_CASE(22)
        PERNIX_COMPRESS_BLOCK_CASE(23)
        PERNIX_COMPRESS_BLOCK_CASE(24)
        default:
            return -1;
    }
}

int mm256_compress_block_f64_bmi2(const uint8_t bit_width, const double_t* __restrict__ input, const double_t scale,
                                  uint8_t* __restrict__ output) {
    switch (bit_width) {
        PERNIX_COMPRESS_BLOCK_CASE(1)
        PERNIX_COMPRESS_BLOCK_CASE(2)
        PERNIX_COMPRESS_BLOCK_CASE(3)
        PERNIX_COMPRESS_BLOCK_CASE(4)
        PERNIX_COMPRESS_BLOCK_CASE(5)
        PERNIX_COMPRESS_BLOCK_CASE(6)
        PERNIX_COMPRESS_BLOCK_CASE(7)
        PERNIX_COMPRESS_BLOCK_CASE(8)
        PERNIX_COMPRESS_BLOCK_CASE(9)
        PERNIX_COMPRESS_BLOCK_CASE(10)
        PERNIX_COMPRESS_BLOCK_CASE(11)
        PERNIX_COMPRESS_BLOCK_CASE(12)
        PERNIX_COMPRESS_BLOCK_CASE(13)
        PERNIX_COMPRESS_BLOCK_CASE(14)
        PERNIX_COMPRESS_BLOCK_CASE(15)
        PERNIX_COMPRESS_BLOCK_CASE(16)
        PERNIX_COMPRESS_BLOCK_CASE(17)
        PERNIX_COMPRESS_BLOCK_CASE(18)
        PERNIX_COMPRESS_BLOCK_CASE(19)
        PERNIX_COMPRESS_BLOCK_CASE(20)
        PERNIX_COMPRESS_BLOCK_CASE(21)
        PERNIX_COMPRESS_BLOCK_CASE(22)
        PERNIX_COMPRESS_BLOCK_CASE(23)
        PERNIX_COMPRESS_BLOCK_CASE(24)
        default:
            return -1;
    }
}

int mm256_compress_blocks_bmi2(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale,
                               uint8_t* __restrict__ output, const uint32_t blocks) {
    switch (bit_width) {
        PERNIX_COMPRESS_BLOCKS_CASE(1)
        PERNIX_COMPRESS_BLOCKS_CASE(2)
        PERNIX_COMPRESS_BLOCKS_CASE(3)
        PERNIX_COMPRESS_BLOCKS_CASE(4)
        PERNIX_COMPRESS_BLOCKS_CASE(5)
        PERNIX_COMPRESS_BLOCKS_CASE(6)
        PERNIX_COMPRESS_BLOCKS_CASE(7)
        PERNIX_COMPRESS_BLOCKS_CASE(8)
        PERNIX_COMPRESS_BLOCKS_CASE(9)
        PERNIX_COMPRESS_BLOCKS_CASE(10)
        PERNIX_COMPRESS_BLOCKS_CASE(11)
        PERNIX_COMPRESS_BLOCKS_CASE(12)
        PERNIX_COMPRESS_BLOCKS_CASE(13)
        PERNIX_COMPRESS_BLOCKS_CASE(14)
        PERNIX_COMPRESS_BLOCKS_CASE(15)
        PERNIX_COMPRESS_BLOCKS_CASE(16)
        PERNIX_COMPRESS_BLOCKS_CASE(17)
        PERNIX_COMPRESS_BLOCKS_CASE(18)
        PERNIX_COMPRESS_BLOCKS_CASE(19)
        PERNIX_COMPRESS_BLOCKS_CASE(20)
        PERNIX_COMPRESS_BLOCKS_CASE(21)
        PERNIX_COMPRESS_BLOCKS_CASE(22)
        PERNIX_COMPRESS_BLOCKS_CASE(23)
        PERNIX_COMPRESS_BLOCKS_CASE(24)
        default:
            return -1;
    }
}

int mm256_compress_blocks_f64_bmi2(const uint8_t bit_width, const double_t* __restrict__ input, const double_t scale,
                                   uint8_t* __restrict__ output, const uint32_t blocks) {
    switch (bit_width) {
        PERNIX_COMPRESS_BLOCKS_CASE(1)
        PERNIX_COMPRESS_BLOCKS_CASE(2)
        PERNIX_COMPRESS_BLOCKS_CASE(3)
        PERNIX_COMPRESS_BLOCKS_CASE(4)
        PERNIX_COMPRESS_BLOCKS_CASE(5)
        PERNIX_COMPRESS_BLOCKS_CASE(6)
        PERNIX_COMPRESS_BLOCKS_CASE(7)
        PERNIX_COMPRESS_BLOCKS_CASE(8)
        PERNIX_COMPRESS_BLOCKS_CASE(9)
        PERNIX_COMPRESS_BLOCKS_CASE(10)
        PERNIX_COMPRESS_BLOCKS_CASE(11)
        PERNIX_COMPRESS_BLOCKS_CASE(12)
        PERNIX_COMPRESS_BLOCKS_CASE(13)
        PERNIX_COMPRESS_BLOCKS_CASE(14)
        PERNIX_COMPRESS_BLOCKS_CASE(15)
        PERNIX_COMPRESS_BLOCKS_CASE(16)
        PERNIX_COMPRESS_BLOCKS_CASE(17)
        PERNIX_COMPRESS_BLOCKS_CASE(18)
        PERNIX_COMPRESS_BLOCKS_CASE(19)
        PERNIX_COMPRESS_BLOCKS_CASE(20)
        PERNIX_COMPRESS_BLOCKS_CASE(21)
        PERNIX_COMPRESS_BLOCKS_CASE(22)
        PERNIX_COMPRESS_BLOCKS_CASE(23)
        PERNIX_COMPRESS_BLOCKS_CASE(24)
        default:
            return -1;
    }
}

#undef PERNIX_COMPRESS_BLOCK_CASE
#undef PERNIX_COMPRESS_BLOCKS_CASE

#ifdef __cplusplus
}
}  // namespace pernix
#endif  // __cplusplus
#endif  // PERNIX_AVX2_ENABLED && PERNIX_BMI2_ENABLED
