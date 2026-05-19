#include <pernix/arm64/sve/compression.h>

namespace pernix {
extern "C" {
int sve_compress_block(uint8_t, const float_t*, float_t, uint8_t*) {
    return -1;
}

int sve_compress_block_f64(uint8_t, const double_t*, double_t, uint8_t*) {
    return -1;
}

int sve_compress_blocks(uint8_t, const float_t*, float_t, uint8_t*, uint32_t) {
    return -1;
}

int sve_compress_blocks_f64(uint8_t, const double_t*, double_t, uint8_t*, uint32_t) {
    return -1;
}
}
} // namespace pernix
