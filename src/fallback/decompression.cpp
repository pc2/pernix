#include <pernix/fallback/decompression.h>

#ifdef __cplusplus
namespace pernix {
extern "C" {
#endif

#define PERNIX_DECOMPRESS_BLOCK_CASE(N) \
    case N:                             \
        return decompress_block_fallback<N>(input, scale, output);

#define PERNIX_DECOMPRESS_BLOCKS_CASE(N) \
    case N:                              \
        return decompress_blocks_fallback<N>(input, scale, output, blocks);

int decompress_block_fallback(const uint8_t bit_width, const uint8_t* __restrict__ input, const float_t scale,
                              float_t* __restrict__ output) {
    switch (bit_width) {
        PERNIX_DECOMPRESS_BLOCK_CASE(1)
        PERNIX_DECOMPRESS_BLOCK_CASE(2)
        PERNIX_DECOMPRESS_BLOCK_CASE(3)
        PERNIX_DECOMPRESS_BLOCK_CASE(4)
        PERNIX_DECOMPRESS_BLOCK_CASE(5)
        PERNIX_DECOMPRESS_BLOCK_CASE(6)
        PERNIX_DECOMPRESS_BLOCK_CASE(7)
        PERNIX_DECOMPRESS_BLOCK_CASE(8)
        PERNIX_DECOMPRESS_BLOCK_CASE(9)
        PERNIX_DECOMPRESS_BLOCK_CASE(10)
        PERNIX_DECOMPRESS_BLOCK_CASE(11)
        PERNIX_DECOMPRESS_BLOCK_CASE(12)
        PERNIX_DECOMPRESS_BLOCK_CASE(13)
        PERNIX_DECOMPRESS_BLOCK_CASE(14)
        PERNIX_DECOMPRESS_BLOCK_CASE(15)
        PERNIX_DECOMPRESS_BLOCK_CASE(16)
        PERNIX_DECOMPRESS_BLOCK_CASE(17)
        PERNIX_DECOMPRESS_BLOCK_CASE(18)
        PERNIX_DECOMPRESS_BLOCK_CASE(19)
        PERNIX_DECOMPRESS_BLOCK_CASE(20)
        PERNIX_DECOMPRESS_BLOCK_CASE(21)
        PERNIX_DECOMPRESS_BLOCK_CASE(22)
        PERNIX_DECOMPRESS_BLOCK_CASE(23)
        PERNIX_DECOMPRESS_BLOCK_CASE(24)
        default:
            return -1;
    }
}

int decompress_block_fallback_f64(const uint8_t bit_width, const uint8_t* __restrict__ input, const double_t scale,
                                  double_t* __restrict__ output) {
    switch (bit_width) {
        PERNIX_DECOMPRESS_BLOCK_CASE(1)
        PERNIX_DECOMPRESS_BLOCK_CASE(2)
        PERNIX_DECOMPRESS_BLOCK_CASE(3)
        PERNIX_DECOMPRESS_BLOCK_CASE(4)
        PERNIX_DECOMPRESS_BLOCK_CASE(5)
        PERNIX_DECOMPRESS_BLOCK_CASE(6)
        PERNIX_DECOMPRESS_BLOCK_CASE(7)
        PERNIX_DECOMPRESS_BLOCK_CASE(8)
        PERNIX_DECOMPRESS_BLOCK_CASE(9)
        PERNIX_DECOMPRESS_BLOCK_CASE(10)
        PERNIX_DECOMPRESS_BLOCK_CASE(11)
        PERNIX_DECOMPRESS_BLOCK_CASE(12)
        PERNIX_DECOMPRESS_BLOCK_CASE(13)
        PERNIX_DECOMPRESS_BLOCK_CASE(14)
        PERNIX_DECOMPRESS_BLOCK_CASE(15)
        PERNIX_DECOMPRESS_BLOCK_CASE(16)
        PERNIX_DECOMPRESS_BLOCK_CASE(17)
        PERNIX_DECOMPRESS_BLOCK_CASE(18)
        PERNIX_DECOMPRESS_BLOCK_CASE(19)
        PERNIX_DECOMPRESS_BLOCK_CASE(20)
        PERNIX_DECOMPRESS_BLOCK_CASE(21)
        PERNIX_DECOMPRESS_BLOCK_CASE(22)
        PERNIX_DECOMPRESS_BLOCK_CASE(23)
        PERNIX_DECOMPRESS_BLOCK_CASE(24)
        default:
            return -1;
    }
}

int decompress_blocks_fallback(const uint8_t bit_width, const uint8_t* __restrict__ input, const float_t scale,
                               float_t* __restrict__ output, const uint32_t blocks) {
    switch (bit_width) {
        PERNIX_DECOMPRESS_BLOCKS_CASE(1)
        PERNIX_DECOMPRESS_BLOCKS_CASE(2)
        PERNIX_DECOMPRESS_BLOCKS_CASE(3)
        PERNIX_DECOMPRESS_BLOCKS_CASE(4)
        PERNIX_DECOMPRESS_BLOCKS_CASE(5)
        PERNIX_DECOMPRESS_BLOCKS_CASE(6)
        PERNIX_DECOMPRESS_BLOCKS_CASE(7)
        PERNIX_DECOMPRESS_BLOCKS_CASE(8)
        PERNIX_DECOMPRESS_BLOCKS_CASE(9)
        PERNIX_DECOMPRESS_BLOCKS_CASE(10)
        PERNIX_DECOMPRESS_BLOCKS_CASE(11)
        PERNIX_DECOMPRESS_BLOCKS_CASE(12)
        PERNIX_DECOMPRESS_BLOCKS_CASE(13)
        PERNIX_DECOMPRESS_BLOCKS_CASE(14)
        PERNIX_DECOMPRESS_BLOCKS_CASE(15)
        PERNIX_DECOMPRESS_BLOCKS_CASE(16)
        PERNIX_DECOMPRESS_BLOCKS_CASE(17)
        PERNIX_DECOMPRESS_BLOCKS_CASE(18)
        PERNIX_DECOMPRESS_BLOCKS_CASE(19)
        PERNIX_DECOMPRESS_BLOCKS_CASE(20)
        PERNIX_DECOMPRESS_BLOCKS_CASE(21)
        PERNIX_DECOMPRESS_BLOCKS_CASE(22)
        PERNIX_DECOMPRESS_BLOCKS_CASE(23)
        PERNIX_DECOMPRESS_BLOCKS_CASE(24)
        default:
            return -1;
    }
}

int decompress_blocks_fallback_f64(const uint8_t bit_width, const uint8_t* __restrict__ input, const double_t scale,
                                   double_t* __restrict__ output, const uint32_t blocks) {
    switch (bit_width) {
        PERNIX_DECOMPRESS_BLOCKS_CASE(1)
        PERNIX_DECOMPRESS_BLOCKS_CASE(2)
        PERNIX_DECOMPRESS_BLOCKS_CASE(3)
        PERNIX_DECOMPRESS_BLOCKS_CASE(4)
        PERNIX_DECOMPRESS_BLOCKS_CASE(5)
        PERNIX_DECOMPRESS_BLOCKS_CASE(6)
        PERNIX_DECOMPRESS_BLOCKS_CASE(7)
        PERNIX_DECOMPRESS_BLOCKS_CASE(8)
        PERNIX_DECOMPRESS_BLOCKS_CASE(9)
        PERNIX_DECOMPRESS_BLOCKS_CASE(10)
        PERNIX_DECOMPRESS_BLOCKS_CASE(11)
        PERNIX_DECOMPRESS_BLOCKS_CASE(12)
        PERNIX_DECOMPRESS_BLOCKS_CASE(13)
        PERNIX_DECOMPRESS_BLOCKS_CASE(14)
        PERNIX_DECOMPRESS_BLOCKS_CASE(15)
        PERNIX_DECOMPRESS_BLOCKS_CASE(16)
        PERNIX_DECOMPRESS_BLOCKS_CASE(17)
        PERNIX_DECOMPRESS_BLOCKS_CASE(18)
        PERNIX_DECOMPRESS_BLOCKS_CASE(19)
        PERNIX_DECOMPRESS_BLOCKS_CASE(20)
        PERNIX_DECOMPRESS_BLOCKS_CASE(21)
        PERNIX_DECOMPRESS_BLOCKS_CASE(22)
        PERNIX_DECOMPRESS_BLOCKS_CASE(23)
        PERNIX_DECOMPRESS_BLOCKS_CASE(24)
        default:
            return -1;
    }
}

#undef PERNIX_DECOMPRESS_BLOCK_CASE
#undef PERNIX_DECOMPRESS_BLOCKS_CASE

#ifdef __cplusplus
}
}  // namespace pernix
#endif  // __cplusplus