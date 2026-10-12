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
    pernix_dispatcher* dispatcher = 0;

    for (int i = 0; i < elements; ++i) {
        input[i] = (float)((i % 17) - 8) * 0.125f;
    }

    if (pernix_dispatcher_create(PERNIX_BACKEND_FALLBACK, &dispatcher) != PERNIX_STATUS_OK) {
        return 1;
    }
    if (pernix_compress_block_f32(dispatcher, bit_width, block_size, input, 8.0f, compressed) != PERNIX_STATUS_OK) {
        pernix_dispatcher_destroy(dispatcher);
        return 2;
    }
    if (pernix_decompress_block_f32(dispatcher, bit_width, block_size, compressed, 0.125f, restored, true) != PERNIX_STATUS_OK) {
        pernix_dispatcher_destroy(dispatcher);
        return 3;
    }
    pernix_dispatcher_destroy(dispatcher);

    for (int i = 0; i < elements; ++i) {
        if (input[i] != restored[i]) {
            return 4;
        }
    }
    return 0;
}
