#include <pernix/arm64/sve2/compression.h>
#include <pernix/dispatch/select.h>

namespace pernix::internal {
Kernel<KernelBlockF32Func> select_sve2_compress_block_f32(const u8 bit_width, const u32 block_size) {
    (void)bit_width;
    (void)block_size;
    return {"sve2", nullptr};
}

Kernel<KernelBlocksF32Func> select_sve2_compress_blocks_f32(const u8 bit_width, const u32 block_size) {
    (void)bit_width;
    (void)block_size;
    return {"sve2", nullptr};
}

Kernel<KernelBlockF64Func> select_sve2_compress_block_f64(const u8 bit_width, const u32 block_size) {
    (void)bit_width;
    (void)block_size;
    return {"sve2", nullptr};
}

Kernel<KernelBlocksF64Func> select_sve2_compress_blocks_f64(const u8 bit_width, const u32 block_size) {
    (void)bit_width;
    (void)block_size;
    return {"sve2", nullptr};
}
}  // namespace pernix::internal
