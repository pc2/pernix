#include <pernix/arm64/sve/decompression.h>

namespace pernix {
extern "C" {
int sve_decompress_block(uint8_t, const uint8_t*, float_t, float_t*) {
    return -1;
}

int sve_decompress_block_f64(uint8_t, const uint8_t*, double_t, double_t*) {
    return -1;
}

int sve_decompress_blocks(uint8_t, const uint8_t*, float_t, float_t*, uint32_t) {
    return -1;
}

int sve_decompress_blocks_f64(uint8_t, const uint8_t*, double_t, double_t*, uint32_t) {
    return -1;
}
}
} // namespace pernix
