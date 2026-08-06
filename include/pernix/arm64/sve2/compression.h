#ifndef PERNIX_ARM64_SVE2_COMPRESSION_H
#define PERNIX_ARM64_SVE2_COMPRESSION_H

#include <pernix/simd_compat.h>

#include <cmath>
#include <cstdint>

namespace pernix {
namespace internal {
template <u8>
inline constexpr bool sve2_compression_unimplemented_v = false;
}  // namespace internal

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int sve2_compress_block(const f32*, f32, u8*) {
    static_assert(internal::sve2_compression_unimplemented_v<BIT_WIDTH>, "ARM64 SVE2 compression is not implemented yet");
    return -1;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int sve2_compress_block(const f64*, f64, u8*) {
    static_assert(internal::sve2_compression_unimplemented_v<BIT_WIDTH>, "ARM64 SVE2 compression is not implemented yet");
    return -1;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int sve2_compress_blocks(const f32*, f32, u8*, u32) {
    static_assert(internal::sve2_compression_unimplemented_v<BIT_WIDTH>, "ARM64 SVE2 compression is not implemented yet");
    return -1;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int sve2_compress_blocks(const f64*, f64, u8*, u32) {
    static_assert(internal::sve2_compression_unimplemented_v<BIT_WIDTH>, "ARM64 SVE2 compression is not implemented yet");
    return -1;
}
}  // namespace pernix

#endif  // PERNIX_ARM64_SVE2_COMPRESSION_H
