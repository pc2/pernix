#include <pernix/dispatch/select.h>
#include <pernix/arm64/sve2/compression.h>

namespace pernix::internal {
Kernel<KernelBlockF32Func> select_sve2_compress_block_f32(const uint8_t bit_width, const uint32_t block_size) {
    (void)bit_width;
    (void)block_size;
    return {"sve2", nullptr};
}

Kernel<KernelBlocksF32Func> select_sve2_compress_blocks_f32(const uint8_t bit_width, const uint32_t block_size) {
    (void)bit_width;
    (void)block_size;
    return {"sve2", nullptr};
}

Kernel<KernelBlockF64Func> select_sve2_compress_block_f64(const uint8_t bit_width, const uint32_t block_size) {
    (void)bit_width;
    (void)block_size;
    return {"sve2", nullptr};
}

Kernel<KernelBlocksF64Func> select_sve2_compress_blocks_f64(const uint8_t bit_width, const uint32_t block_size) {
    (void)bit_width;
    (void)block_size;
    return {"sve2", nullptr};
}
}
