#include <pernix/pernix.h>

int main(void) {
    enum { bit_width = 16, block_size = 64, elements = (block_size * 8) / bit_width };
    float input[elements];
    float restored[elements];
    u8 compressed[block_size];
    float bmax = 0.0f;

    if (!pernix_is_valid_bit_width(bit_width) || pernix_compressed_block_size() != block_size ||
        pernix_elements_per_block(bit_width) != elements) {
        return 1;
    }

    for (int i = 0; i < elements; ++i) {
        input[i] = ((float)i - 16.0f) * 0.125f;
        const float magnitude = input[i] < 0.0f ? -input[i] : input[i];
        bmax = bmax < magnitude ? magnitude : bmax;
    }

    float scale = 0.0f;
    float inverse_scale = 0.0f;
    if (pernix_decompression_scale_f32(bmax, bit_width, &scale) != PERNIX_STATUS_OK ||
        pernix_inverse_scale_f32(scale, &inverse_scale) != PERNIX_STATUS_OK) {
        return 2;
    }

    if (pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, bit_width, block_size, input, inverse_scale,
                                  compressed) != PERNIX_STATUS_OK) {
        return 3;
    }
    if (pernix_decompress_block_f32(PERNIX_BACKEND_FALLBACK, bit_width, block_size, compressed, scale, restored,
                                    true) != PERNIX_STATUS_OK) {
        return 4;
    }

    for (int i = 0; i < elements; ++i) {
        const float diff = restored[i] - input[i];
        const float error = diff < 0.0f ? -diff : diff;
        if (error > scale) {
            return 5;
        }
    }
    return 0;
}
