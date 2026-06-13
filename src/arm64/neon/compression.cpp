#include <pernix/dispatch/select.h>
#include <pernix/arm64/neon/compression.h>

namespace pernix::internal {
Kernel<KernelBlockF32Func> select_neon_compress_block_f32(const uint8_t bit_width, const uint32_t block_size) {
    (void)bit_width;
    (void)block_size;
    return {"neon", nullptr};
}

Kernel<KernelBlocksF32Func> select_neon_compress_blocks_f32(const uint8_t bit_width, const uint32_t block_size) {
    (void)bit_width;
    (void)block_size;
    return {"neon", nullptr};
}

Kernel<KernelBlockF64Func> select_neon_compress_block_f64(const uint8_t bit_width, const uint32_t block_size) {
    (void)bit_width;
    (void)block_size;
    return {"neon", nullptr};
}

Kernel<KernelBlocksF64Func> select_neon_compress_blocks_f64(const uint8_t bit_width, const uint32_t block_size) {
    (void)bit_width;
    (void)block_size;
    return {"neon", nullptr};
}
}
