#ifndef PERNIX_ARM64_NEON_COMPRESSION_H
#define PERNIX_ARM64_NEON_COMPRESSION_H

#include <pernix/simd_compat.h>
#include <pernix/arm64/neon/packing.h>

#include <cmath>
#include <cstdint>

namespace pernix::arm64::neon {
namespace internal {
template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8) && (BLOCK_SIZE % 32 == 0)
__always_inline int neon_compress_block_1to8(const u8* __restrict__ input, const f32 scale,
                                             f32* __restrict__ output) {
    static_assert(true, "Not yet implemented");
    return -1;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16) && (BLOCK_SIZE % 32 == 0)
__always_inline int neon_compress_block_9to16(const u8* __restrict__ input, const f32 scale,
                                              f32* __restrict__ output) {
    static_assert(true, "Not yet implemented");
    return -1;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int neon_compress_block_17to24(const u8* __restrict__ input, const f32 scale,
                                               f32* __restrict__ output) {
    static_assert(true, "Not yet implemented");
    return -1;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8) && (BLOCK_SIZE % 32 == 0)
__always_inline int neon_compress_block_1to8(const u8* __restrict__ input, const f64 scale,
                                             f64* __restrict__ output) {
    static_assert(true, "Not yet implemented");
    return -1;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16) && (BLOCK_SIZE % 32 == 0)
__always_inline int neon_compress_block_9to16(const u8* __restrict__ input, const f64 scale,
                                              f64* __restrict__ output) {
    static_assert(true, "Not yet implemented");
    return -1;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int neon_compress_block_17to24(const u8* __restrict__ input, const f64 scale,
                                               f64* __restrict__ output) {
    static_assert(true, "Not yet implemented");
    return -1;
}
} // namespace internal

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int neon_compress_block(const u8* __restrict__ input, const f32 scale,
                                        f32* __restrict__ output) {
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
        return internal::neon_compress_block_1to8<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
        return internal::neon_compress_block_9to16<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH >= 17 && BIT_WIDTH <= 24) {
        return internal::neon_compress_block_17to24<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    }
    return 0;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int neon_compress_block(const u8* __restrict__ input, const f64 scale,
                                        f64* __restrict__ output) {
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
        return internal::neon_compress_block_1to8<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
        return internal::neon_compress_block_9to16<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH >= 17 && BIT_WIDTH <= 24) {
        return internal::neon_compress_block_17to24<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    }
    return 0;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int neon_compress_blocks(const u8* __restrict__ input, const f32 scale, f32* __restrict__ output,
                         const u32 blocks) {
    const u8* block_input = input;
    f32* block_output     = output;

    for (u32 block = 0; block < blocks; ++block) {
        neon_compress_block<BIT_WIDTH, BLOCK_SIZE>(block_input, scale, block_output);
        block_input  += BLOCK_SIZE;
        block_output += (BLOCK_SIZE * 8) / BIT_WIDTH;
    }

    return 0;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int neon_compress_blocks(const u8* __restrict__ input, const f64 scale, f64* __restrict__ output,
                         const u32 blocks) {
    const u8* block_input = input;
    f64* block_output     = output;

    for (u32 block = 0; block < blocks; ++block) {
        neon_compress_block<BIT_WIDTH, BLOCK_SIZE>(block_input, scale, block_output);
        block_input  += BLOCK_SIZE;
        block_output += (BLOCK_SIZE * 8) / BIT_WIDTH;
    }
    return 0;
}
} // namespace pernix::arm64::neon

#endif  // PERNIX_ARM64_NEON_COMPRESSION_H
