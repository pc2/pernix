#include <pernix/arm64/sve2/decompression.h>

namespace pernix {
extern "C" {
#define PERNIX_SVE2_DECOMPRESS_BLOCK_CASE(N) \
    case N:                                  \
        return arm64::sve2::sve2_decompress_block<N>(input, scale, output);

#define PERNIX_SVE2_DECOMPRESS_BLOCKS_CASE(N) \
    case N:                                   \
        return arm64::sve2::sve2_decompress_blocks<N>(input, scale, output, blocks);

int sve2_decompress_block(const uint8_t bit_width, const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output) {
    switch (bit_width) {
        PERNIX_SVE2_DECOMPRESS_BLOCK_CASE(1)
        PERNIX_SVE2_DECOMPRESS_BLOCK_CASE(2)
        PERNIX_SVE2_DECOMPRESS_BLOCK_CASE(3)
        PERNIX_SVE2_DECOMPRESS_BLOCK_CASE(4)
        PERNIX_SVE2_DECOMPRESS_BLOCK_CASE(5)
        PERNIX_SVE2_DECOMPRESS_BLOCK_CASE(6)
        PERNIX_SVE2_DECOMPRESS_BLOCK_CASE(7)
        PERNIX_SVE2_DECOMPRESS_BLOCK_CASE(8)
        default:
            return -1;
    }
}

int sve2_decompress_block_f64(const uint8_t bit_width, const uint8_t* __restrict__ input, const double_t scale,
                              double_t* __restrict__ output) {
    switch (bit_width) {
        PERNIX_SVE2_DECOMPRESS_BLOCK_CASE(1)
        PERNIX_SVE2_DECOMPRESS_BLOCK_CASE(2)
        PERNIX_SVE2_DECOMPRESS_BLOCK_CASE(3)
        PERNIX_SVE2_DECOMPRESS_BLOCK_CASE(4)
        PERNIX_SVE2_DECOMPRESS_BLOCK_CASE(5)
        PERNIX_SVE2_DECOMPRESS_BLOCK_CASE(6)
        PERNIX_SVE2_DECOMPRESS_BLOCK_CASE(7)
        PERNIX_SVE2_DECOMPRESS_BLOCK_CASE(8)
        default:
            return -1;
    }
}

int sve2_decompress_blocks(const uint8_t bit_width, const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output,
                           const uint32_t blocks) {
    switch (bit_width) {
        PERNIX_SVE2_DECOMPRESS_BLOCKS_CASE(1)
        PERNIX_SVE2_DECOMPRESS_BLOCKS_CASE(2)
        PERNIX_SVE2_DECOMPRESS_BLOCKS_CASE(3)
        PERNIX_SVE2_DECOMPRESS_BLOCKS_CASE(4)
        PERNIX_SVE2_DECOMPRESS_BLOCKS_CASE(5)
        PERNIX_SVE2_DECOMPRESS_BLOCKS_CASE(6)
        PERNIX_SVE2_DECOMPRESS_BLOCKS_CASE(7)
        PERNIX_SVE2_DECOMPRESS_BLOCKS_CASE(8)
        default:
            return -1;
    }
}

int sve2_decompress_blocks_f64(const uint8_t bit_width, const uint8_t* __restrict__ input, const double_t scale,
                               double_t* __restrict__ output, const uint32_t blocks) {
    switch (bit_width) {
        PERNIX_SVE2_DECOMPRESS_BLOCKS_CASE(1)
        PERNIX_SVE2_DECOMPRESS_BLOCKS_CASE(2)
        PERNIX_SVE2_DECOMPRESS_BLOCKS_CASE(3)
        PERNIX_SVE2_DECOMPRESS_BLOCKS_CASE(4)
        PERNIX_SVE2_DECOMPRESS_BLOCKS_CASE(5)
        PERNIX_SVE2_DECOMPRESS_BLOCKS_CASE(6)
        PERNIX_SVE2_DECOMPRESS_BLOCKS_CASE(7)
        PERNIX_SVE2_DECOMPRESS_BLOCKS_CASE(8)
        default:
            return -1;
    }
}

#undef PERNIX_SVE2_DECOMPRESS_BLOCK_CASE
#undef PERNIX_SVE2_DECOMPRESS_BLOCKS_CASE
}
}  // namespace pernix
