#include <pernix/pernix.h>
#include <pernix/dispatch/select.h>

namespace {
    bool is_valid_block_size(u32 block_size) {
        return block_size == 64 || block_size == 128 || block_size == 256 || block_size == 512 || block_size == 1024;
    }
}

extern "C" {
pernix_status pernix_compress_block_f32(pernix_backend backend, u8 bit_width, u32 block_size, const void *input,
                                        float scale, void *output) {
    if (input == nullptr || output == nullptr) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }
    if (!is_valid_block_size(block_size)) {
        return PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE;
    }

    const auto kernel = pernix::internal::select_compress_block_f32(static_cast<pernix::Backend>(backend), bit_width,
                                                                    block_size);

    if (!kernel) {
        return PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH;
    }

    return static_cast<pernix_status>(kernel.func(input, scale, output));
}

pernix_status pernix_compress_blocks_f32(pernix_backend backend, u8 bit_width, u32 block_size, const void *input,
                                         float scale, void *output, u32 blocks) {
    if (input == nullptr || output == nullptr) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }

    if (!is_valid_block_size(block_size)) {
        return PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE;
    }

    const auto kernel = pernix::internal::select_compress_blocks_f32(static_cast<pernix::Backend>(backend), bit_width,
                                                                     block_size);

    if (!kernel) {
        return PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH;
    }

    return static_cast<pernix_status>(kernel.func(input, scale, output, blocks));
}

pernix_status pernix_decompress_block_f32(pernix_backend backend, u8 bit_width, u32 block_size, const void *input,
                                          float scale, void *output, bool sign_values) {
    if (input == nullptr || output == nullptr) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }

    if (!is_valid_block_size(block_size)) {
        return PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE;
    }

    const auto kernel = pernix::internal::select_decompress_block_f32(static_cast<pernix::Backend>(backend), bit_width,
                                                                      block_size,
                                                                      sign_values);

    if (!kernel) {
        return PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH;
    }

    return static_cast<pernix_status>(kernel.func(input, scale, output));
}

pernix_status pernix_decompress_blocks_f32(pernix_backend backend, u8 bit_width, u32 block_size, const void *input,
                                           float scale, void *output, u32 blocks, bool sign_values) {
    if (input == nullptr || output == nullptr) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }

    if (!is_valid_block_size(block_size)) {
        return PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE;
    }

    const auto kernel = pernix::internal::select_decompress_blocks_f32(static_cast<pernix::Backend>(backend), bit_width,
                                                                       block_size,
                                                                       sign_values);

    if (!kernel) {
        return PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH;
    }

    return static_cast<pernix_status>(kernel.func(input, scale, output, blocks));
}

pernix_status pernix_compress_block_f64(pernix_backend backend, u8 bit_width, u32 block_size, const void *input,
                                        double scale, void *output) {
    if (input == nullptr || output == nullptr) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }

    if (!is_valid_block_size(block_size)) {
        return PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE;
    }

    const auto kernel = pernix::internal::select_compress_block_f64(static_cast<pernix::Backend>(backend), bit_width,
                                                                    block_size);

    if (!kernel) {
        return PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH;
    }

    return static_cast<pernix_status>(kernel.func(input, scale, output));
}

pernix_status pernix_compress_blocks_f64(pernix_backend backend, u8 bit_width, u32 block_size, const void *input,
                                         double scale, void *output, u32 blocks) {
    if (input == nullptr || output == nullptr) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }

    if (!is_valid_block_size(block_size)) {
        return PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE;
    }

    const auto kernel = pernix::internal::select_compress_blocks_f64(static_cast<pernix::Backend>(backend), bit_width,
                                                                     block_size);

    if (!kernel) {
        return PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH;
    }

    return static_cast<pernix_status>(kernel.func(input, scale, output, blocks));
}

pernix_status pernix_decompress_block_f64(pernix_backend backend, u8 bit_width, u32 block_size, const void *input,
                                          double scale, void *output, bool sign_values) {
    if (input == nullptr || output == nullptr) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }

    if (!is_valid_block_size(block_size)) {
        return PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE;
    }

    const auto kernel = pernix::internal::select_decompress_block_f64(static_cast<pernix::Backend>(backend), bit_width,
                                                                      block_size,
                                                                      sign_values);

    if (!kernel) {
        return PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH;
    }

    return static_cast<pernix_status>(kernel.func(input, scale, output));
}

pernix_status pernix_decompress_blocks_f64(pernix_backend backend, u8 bit_width, u32 block_size, const void *input,
                                           double scale, void *output, u32 blocks, bool sign_values) {
    if (input == nullptr || output == nullptr) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }

    if (!is_valid_block_size(block_size)) {
        return PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE;
    }

    const auto kernel = pernix::internal::select_decompress_blocks_f64(static_cast<pernix::Backend>(backend), bit_width,
                                                                       block_size,
                                                                       sign_values);

    if (!kernel) {
        return PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH;
    }

    return static_cast<pernix_status>(kernel.func(input, scale, output, blocks));
}
}
