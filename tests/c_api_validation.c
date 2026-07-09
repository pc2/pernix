#include <math.h>
#include <pernix/pernix.h>
#include <string.h>

enum { bit_width = 8, block_size = 64, elements = (block_size * 8) / bit_width };

static int expect_status(pernix_status actual, pernix_status expected) {
    return actual == expected ? 0 : 1;
}

int main(void) {
    float input_f32[elements]   = {0};
    float output_f32[elements]  = {0};
    double input_f64[elements]  = {0};
    double output_f64[elements] = {0};
    u8 compressed[block_size]   = {0};
    float scale_f32             = 0.0f;
    double scale_f64            = 0.0;

    if (expect_status(pernix_scale_f32(1.0f, bit_width, &scale_f32), PERNIX_STATUS_OK) != 0) {
        return 1;
    }
    if (expect_status(pernix_scale_f64(1.0, bit_width, &scale_f64), PERNIX_STATUS_OK) != 0) {
        return 2;
    }
    if (expect_status(pernix_scale_f32(1.0f, bit_width, 0), PERNIX_STATUS_INVALID_ARGUMENT) != 0) {
        return 3;
    }
    if (expect_status(pernix_scale_f32(-1.0f, bit_width, &scale_f32), PERNIX_STATUS_INVALID_ARGUMENT) != 0) {
        return 4;
    }
    if (expect_status(pernix_scale_f64(INFINITY, bit_width, &scale_f64), PERNIX_STATUS_INVALID_ARGUMENT) != 0) {
        return 5;
    }

    if (!pernix_is_valid_bit_width(pernix_min_bit_width()) || !pernix_is_valid_bit_width(pernix_max_bit_width()) ||
        pernix_is_valid_bit_width(0) || pernix_is_valid_bit_width((u8)(pernix_max_bit_width() + 1U))) {
        return 6;
    }
    if (pernix_compressed_block_size() != block_size || pernix_elements_per_block(bit_width) != elements) {
        return 7;
    }
    if (!pernix_is_valid_block_size(block_size) || pernix_is_valid_block_size(96)) {
        return 8;
    }

    if (strcmp(pernix_status_string(PERNIX_STATUS_OK), "PERNIX_STATUS_OK") != 0 ||
        strcmp(pernix_status_string(PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION), "PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION") != 0 ||
        strcmp(pernix_status_string((pernix_status)123), "PERNIX_STATUS_UNKNOWN") != 0) {
        return 23;
    }

    float decompression_scale_f32 = 0.0f;
    float compression_scale_f32   = 0.0f;
    float inverse_scale_f32       = 0.0f;
    if (expect_status(pernix_decompression_scale_f32(127.0f, bit_width, &decompression_scale_f32), PERNIX_STATUS_OK) != 0 ||
        expect_status(pernix_compression_scale_f32(127.0f, bit_width, &compression_scale_f32), PERNIX_STATUS_OK) != 0 ||
        expect_status(pernix_inverse_scale_f32(decompression_scale_f32, &inverse_scale_f32), PERNIX_STATUS_OK) != 0) {
        return 24;
    }
    if (fabsf(decompression_scale_f32 - 1.0f) > 0.0f || fabsf(compression_scale_f32 - 1.0f) > 0.0f ||
        fabsf(inverse_scale_f32 - 1.0f) > 0.0f) {
        return 25;
    }
    if (expect_status(pernix_inverse_scale_f32(0.0f, &inverse_scale_f32), PERNIX_STATUS_INVALID_ARGUMENT) != 0 ||
        expect_status(pernix_compression_scale_f32(1.0f, 0, &compression_scale_f32), PERNIX_STATUS_INVALID_ARGUMENT) != 0 ||
        expect_status(pernix_decompression_scale_f32(1.0f, bit_width, 0), PERNIX_STATUS_INVALID_ARGUMENT) != 0) {
        return 26;
    }

    double decompression_scale_f64 = 0.0;
    double compression_scale_f64   = 0.0;
    double inverse_scale_f64       = 0.0;
    if (expect_status(pernix_decompression_scale_f64(127.0, bit_width, &decompression_scale_f64), PERNIX_STATUS_OK) != 0 ||
        expect_status(pernix_compression_scale_f64(127.0, bit_width, &compression_scale_f64), PERNIX_STATUS_OK) != 0 ||
        expect_status(pernix_inverse_scale_f64(decompression_scale_f64, &inverse_scale_f64), PERNIX_STATUS_OK) != 0) {
        return 27;
    }
    if (fabs(decompression_scale_f64 - 1.0) > 0.0 || fabs(compression_scale_f64 - 1.0) > 0.0 || fabs(inverse_scale_f64 - 1.0) > 0.0) {
        return 28;
    }

    if (expect_status(pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, 0, block_size, input_f32, 1.0f, compressed),
                      PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH) != 0) {
        return 9;
    }
    if (expect_status(pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, bit_width, 96, input_f32, 1.0f, compressed),
                      PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE) != 0) {
        return 10;
    }
    if (expect_status(pernix_compress_block_f32((pernix_backend)255, bit_width, block_size, input_f32, 1.0f, compressed),
                      PERNIX_STATUS_UNSUPPORTED_BACKEND) != 0) {
        return 11;
    }
    if (expect_status(pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, bit_width, block_size, 0, 1.0f, compressed),
                      PERNIX_STATUS_INVALID_ARGUMENT) != 0) {
        return 12;
    }
    if (expect_status(pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, bit_width, block_size, input_f32, 1.0f, 0),
                      PERNIX_STATUS_INVALID_ARGUMENT) != 0) {
        return 13;
    }
    if (expect_status(pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, bit_width, block_size, input_f32, 0.0f, compressed),
                      PERNIX_STATUS_INVALID_ARGUMENT) != 0) {
        return 14;
    }
    if (expect_status(pernix_decompress_block_f32(PERNIX_BACKEND_FALLBACK, bit_width, block_size, 0, 1.0f, output_f32, true),
                      PERNIX_STATUS_INVALID_ARGUMENT) != 0) {
        return 15;
    }
    if (expect_status(pernix_decompress_block_f32(PERNIX_BACKEND_FALLBACK, bit_width, block_size, compressed, 1.0f, 0, true),
                      PERNIX_STATUS_INVALID_ARGUMENT) != 0) {
        return 16;
    }
    if (expect_status(pernix_decompress_block_f32(PERNIX_BACKEND_FALLBACK, bit_width, block_size, compressed, -1.0f, output_f32, true),
                      PERNIX_STATUS_INVALID_ARGUMENT) != 0) {
        return 17;
    }
    if (expect_status(pernix_compress_blocks_f32(PERNIX_BACKEND_FALLBACK, bit_width, block_size, input_f32, 1.0f, compressed, 0),
                      PERNIX_STATUS_INVALID_ARGUMENT) != 0) {
        return 18;
    }

    if (expect_status(pernix_compress_block_f64(PERNIX_BACKEND_FALLBACK, bit_width, block_size, input_f64, 1.0, compressed),
                      PERNIX_STATUS_OK) != 0) {
        return 19;
    }
    if (expect_status(pernix_decompress_block_f64(PERNIX_BACKEND_FALLBACK, bit_width, block_size, compressed, 1.0, output_f64, true),
                      PERNIX_STATUS_OK) != 0) {
        return 20;
    }
    if (expect_status(pernix_compress_blocks_f64(PERNIX_BACKEND_FALLBACK, bit_width, block_size, 0, 1.0, compressed, 1),
                      PERNIX_STATUS_INVALID_ARGUMENT) != 0) {
        return 21;
    }
    if (expect_status(pernix_decompress_blocks_f64(PERNIX_BACKEND_FALLBACK, bit_width, block_size, compressed, 1.0, 0, 1, true),
                      PERNIX_STATUS_INVALID_ARGUMENT) != 0) {
        return 22;
    }

    return 0;
}
