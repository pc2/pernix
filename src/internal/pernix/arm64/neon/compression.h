#ifndef PERNIX_ARM64_NEON_COMPRESSION_H
#define PERNIX_ARM64_NEON_COMPRESSION_H

#include <pernix/simd_compat.h>
#include <pernix/arm64/neon/packing.h>

#include <cmath>
#include <cstdint>

namespace pernix::arm64::neon {
namespace internal {
template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8) && (BLOCK_SIZE % 32 == 0)
__always_inline int neon_compress_block_1to8(const uint8_t* __restrict__ input, const float_t scale,
                                             float_t* __restrict__ output) {
    static_assert(true, "Not yet implemented");
    return -1;
}

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16) && (BLOCK_SIZE % 32 == 0)
__always_inline int neon_compress_block_9to16(const uint8_t* __restrict__ input, const float_t scale,
                                              float_t* __restrict__ output) {
    static_assert(true, "Not yet implemented");
    return -1;
}

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int neon_compress_block_17to24(const uint8_t* __restrict__ input, const float_t scale,
                                               float_t* __restrict__ output) {
    static_assert(true, "Not yet implemented");
    return -1;
}

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8) && (BLOCK_SIZE % 32 == 0)
__always_inline int neon_compress_block_1to8(const uint8_t* __restrict__ input, const double_t scale,
                                             double_t* __restrict__ output) {
    static_assert(true, "Not yet implemented");
    return -1;
}

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16) && (BLOCK_SIZE % 32 == 0)
__always_inline int neon_compress_block_9to16(const uint8_t* __restrict__ input, const double_t scale,
                                              double_t* __restrict__ output) {
    static_assert(true, "Not yet implemented");
    return -1;
}

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int neon_compress_block_17to24(const uint8_t* __restrict__ input, const double_t scale,
                                               double_t* __restrict__ output) {
    static_assert(true, "Not yet implemented");
    return -1;
}
} // namespace internal

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int neon_compress_block(const uint8_t* __restrict__ input, const float_t scale,
                                        float_t* __restrict__ output) {
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
        return internal::neon_compress_block_1to8<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
        return internal::neon_compress_block_9to16<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH >= 17 && BIT_WIDTH <= 24) {
        return internal::neon_compress_block_17to24<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    }
    return 0;
}

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int neon_compress_block(const uint8_t* __restrict__ input, const double_t scale,
                                        double_t* __restrict__ output) {
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
        return internal::neon_compress_block_1to8<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
        return internal::neon_compress_block_9to16<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH >= 17 && BIT_WIDTH <= 24) {
        return internal::neon_compress_block_17to24<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    }
    return 0;
}

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int neon_compress_blocks(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output,
                         const uint32_t blocks) {
    const uint8_t* block_input = input;
    float_t* block_output      = output;

    for (uint32_t block = 0; block < blocks; ++block) {
        neon_compress_block<BIT_WIDTH, BLOCK_SIZE>(block_input, scale, block_output);
        block_input  += BLOCK_SIZE;
        block_output += (BLOCK_SIZE * 8) / BIT_WIDTH;
    }

    return 0;
}

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int neon_compress_blocks(const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output,
                         const uint32_t blocks) {
    const uint8_t* block_input = input;
    double_t* block_output     = output;

    for (uint32_t block = 0; block < blocks; ++block) {
        neon_compress_block<BIT_WIDTH, BLOCK_SIZE>(block_input, scale, block_output);
        block_input  += BLOCK_SIZE;
        block_output += (BLOCK_SIZE * 8) / BIT_WIDTH;
    }
    return 0;
}
} // namespace pernix::arm64::neon

#endif  // PERNIX_ARM64_NEON_COMPRESSION_H
