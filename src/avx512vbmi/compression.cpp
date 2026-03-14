#include <pernix/avx512vbmi/compression.h>
#include <pernix/detection.h>

#if defined(PERNIX_AVX2_ENABLED) && defined(PERNIX_AVX512_VBMI_ENABLED)
#ifdef __cplusplus
namespace pernix {
extern "C" {
#endif

#define PERNIX_COMPRESS_BLOCK_CASE(N) \
    case N:                           \
        return mm512_compress_block_avx512vbmi<N>(input, scale, output);

#define PERNIX_COMPRESS_BLOCKS_CASE(N) \
    case N:                            \
        return mm512_compress_blocks_avx512vbmi<N>(input, scale, output, blocks);

int mm512_compress_block_avx512vbmi(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale,
                                    uint8_t* __restrict__ output) {
    switch (bit_width) {
        PERNIX_COMPRESS_BLOCK_CASE(8)
        PERNIX_COMPRESS_BLOCK_CASE(9)
        PERNIX_COMPRESS_BLOCK_CASE(10)
        PERNIX_COMPRESS_BLOCK_CASE(11)
        PERNIX_COMPRESS_BLOCK_CASE(12)
        PERNIX_COMPRESS_BLOCK_CASE(13)
        PERNIX_COMPRESS_BLOCK_CASE(14)
        PERNIX_COMPRESS_BLOCK_CASE(15)
        PERNIX_COMPRESS_BLOCK_CASE(16)
        default:
            return -1;
    }
}

int mm512_compress_block_f64_avx512vbmi(const uint8_t bit_width, const double_t* __restrict__ input, const double_t scale,
                                        uint8_t* __restrict__ output) {
    switch (bit_width) {
        PERNIX_COMPRESS_BLOCK_CASE(8)
        PERNIX_COMPRESS_BLOCK_CASE(9)
        PERNIX_COMPRESS_BLOCK_CASE(10)
        PERNIX_COMPRESS_BLOCK_CASE(11)
        PERNIX_COMPRESS_BLOCK_CASE(12)
        PERNIX_COMPRESS_BLOCK_CASE(13)
        PERNIX_COMPRESS_BLOCK_CASE(14)
        PERNIX_COMPRESS_BLOCK_CASE(15)
        PERNIX_COMPRESS_BLOCK_CASE(16)
        default:
            return -1;
    }
}

int mm512_compress_blocks_avx512vbmi(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale,
                                     uint8_t* __restrict__ output, const uint32_t blocks) {
    switch (bit_width) {
        PERNIX_COMPRESS_BLOCKS_CASE(8)
        PERNIX_COMPRESS_BLOCKS_CASE(9)
        PERNIX_COMPRESS_BLOCKS_CASE(10)
        PERNIX_COMPRESS_BLOCKS_CASE(11)
        PERNIX_COMPRESS_BLOCKS_CASE(12)
        PERNIX_COMPRESS_BLOCKS_CASE(13)
        PERNIX_COMPRESS_BLOCKS_CASE(14)
        PERNIX_COMPRESS_BLOCKS_CASE(15)
        PERNIX_COMPRESS_BLOCKS_CASE(16)
        default:
            return -1;
    }
}

int mm512_compress_blocks_f64_avx512vbmi(const uint8_t bit_width, const double_t* __restrict__ input, const double_t scale,
                                         uint8_t* __restrict__ output, const uint32_t blocks) {
    switch (bit_width) {
        PERNIX_COMPRESS_BLOCKS_CASE(8)
        PERNIX_COMPRESS_BLOCKS_CASE(9)
        PERNIX_COMPRESS_BLOCKS_CASE(10)
        PERNIX_COMPRESS_BLOCKS_CASE(11)
        PERNIX_COMPRESS_BLOCKS_CASE(12)
        PERNIX_COMPRESS_BLOCKS_CASE(13)
        PERNIX_COMPRESS_BLOCKS_CASE(14)
        PERNIX_COMPRESS_BLOCKS_CASE(15)
        PERNIX_COMPRESS_BLOCKS_CASE(16)
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
#endif  // defined(PERNIX_AVX2_ENABLED) && defined(PERNIX_AVX512_VBMI_ENABLED)