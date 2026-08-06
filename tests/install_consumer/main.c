#include <pernix/pernix.h>

int main(void) {
    enum {
        bit_width  = 8,
        block_size = 64,
        elements   = (block_size * 8) / bit_width
    };

    float input[elements];
    float restored[elements];
    u8 compressed[block_size];

    for (int i = 0; i < elements; ++i) {
        input[i] = (float)((i % 17) - 8) * 0.125f;
    }

    if (pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, bit_width, block_size, input, 8.0f, compressed) != PERNIX_STATUS_OK) {
        return 1;
    }
    if (pernix_decompress_block_f32(PERNIX_BACKEND_FALLBACK, bit_width, block_size, compressed, 0.125f, restored, true) !=
        PERNIX_STATUS_OK) {
        return 2;
    }

    for (int i = 0; i < elements; ++i) {
        if (input[i] != restored[i]) {
            return 3;
        }
    }
    return 0;
}
