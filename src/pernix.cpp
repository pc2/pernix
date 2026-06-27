#include <pernix/detail/api.hpp>

extern "C" {
pernix_status pernix_compress_block_f32(pernix_backend backend, u8 bit_width, u32 block_size, const void *input,
                                        float scale, void *output) {
    return pernix::detail::compress_block(static_cast<pernix::Backend>(backend), bit_width, block_size, input, scale,
                                          output);
}

pernix_status pernix_compress_blocks_f32(pernix_backend backend, u8 bit_width, u32 block_size, const void *input,
                                         float scale, void *output, u32 blocks) {
    return pernix::detail::compress_blocks(static_cast<pernix::Backend>(backend), bit_width, block_size, input,
                                           scale, output, blocks);
}

pernix_status pernix_decompress_block_f32(pernix_backend backend, u8 bit_width, u32 block_size, const void *input,
                                          float scale, void *output, bool sign_values) {
    return pernix::detail::decompress_block(static_cast<pernix::Backend>(backend), bit_width, block_size, input,
                                            scale, output, sign_values);
}

pernix_status pernix_decompress_blocks_f32(pernix_backend backend, u8 bit_width, u32 block_size, const void *input,
                                           float scale, void *output, u32 blocks, bool sign_values) {
    return pernix::detail::decompress_blocks(static_cast<pernix::Backend>(backend), bit_width, block_size, input,
                                             scale, output, blocks, sign_values);
}

pernix_status pernix_compress_block_f64(pernix_backend backend, u8 bit_width, u32 block_size, const void *input,
                                        double scale, void *output) {
    return pernix::detail::compress_block(static_cast<pernix::Backend>(backend), bit_width, block_size, input, scale,
                                          output);
}

pernix_status pernix_compress_blocks_f64(pernix_backend backend, u8 bit_width, u32 block_size, const void *input,
                                         double scale, void *output, u32 blocks) {
    return pernix::detail::compress_blocks(static_cast<pernix::Backend>(backend), bit_width, block_size, input,
                                           scale, output, blocks);
}

pernix_status pernix_decompress_block_f64(pernix_backend backend, u8 bit_width, u32 block_size, const void *input,
                                          double scale, void *output, bool sign_values) {
    return pernix::detail::decompress_block(static_cast<pernix::Backend>(backend), bit_width, block_size, input,
                                            scale, output, sign_values);
}

pernix_status pernix_decompress_blocks_f64(pernix_backend backend, u8 bit_width, u32 block_size, const void *input,
                                           double scale, void *output, u32 blocks, bool sign_values) {
    return pernix::detail::decompress_blocks(static_cast<pernix::Backend>(backend), bit_width, block_size, input,
                                             scale, output, blocks, sign_values);
}
}
