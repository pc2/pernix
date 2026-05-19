#ifndef PERNIX_ARM64_SVE_DECOMPRESSION_H
#define PERNIX_ARM64_SVE_DECOMPRESSION_H

#include <pernix/simd_compat.h>
#include <pernix/arm64/sve/unpacking.h>

#include <cmath>
#include <cstdint>

namespace pernix::arm64::sve {
namespace internal {
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8) && (BLOCK_SIZE % 32 == 0)
__always_inline int sve_decompress_block_1to8(const uint8_t* __restrict__ input, const float_t scale,
                                              float_t* __restrict__ output) {
    static_assert(true, "Not yet implemented");
    return -1;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16) && (BLOCK_SIZE % 32 == 0)
__always_inline int sve_decompress_block_9to16(const uint8_t* __restrict__ input, const float_t scale,
                                               float_t* __restrict__ output) {
    static_assert(true, "Not yet implemented");
    return -1;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int sve_decompress_block_17to24(const uint8_t* __restrict__ input, const float_t scale,
                                                float_t* __restrict__ output) {
    static_assert(true, "Not yet implemented");
    return -1;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8) && (BLOCK_SIZE % 32 == 0)
__always_inline int sve_decompress_block_1to8(const uint8_t* __restrict__ input, const double_t scale,
                                              double_t* __restrict__ output) {
    static_assert(true, "Not yet implemented");
    return -1;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16) && (BLOCK_SIZE % 32 == 0)
__always_inline int sve_decompress_block_9to16(const uint8_t* __restrict__ input, const double_t scale,
                                               double_t* __restrict__ output) {
    static_assert(true, "Not yet implemented");
    return -1;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int sve_decompress_block_17to24(const uint8_t* __restrict__ input, const double_t scale,
                                                double_t* __restrict__ output) {
    static_assert(true, "Not yet implemented");
    return -1;
}
} // namespace internal

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int sve_decompress_block(const uint8_t* __restrict__ input, const float_t scale,
                                         float_t* __restrict__ output) {
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
        return internal::sve_decompress_block_1to8<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
        return internal::sve_decompress_block_9to16<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH >= 17 && BIT_WIDTH <= 24) {
        return internal::sve_decompress_block_17to24<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    }
    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int sve_decompress_block(const uint8_t* __restrict__ input, const double_t scale,
                                         double_t* __restrict__ output) {
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
        return internal::sve_decompress_block_1to8<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
        return internal::sve_decompress_block_9to16<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH >= 17 && BIT_WIDTH <= 24) {
        return internal::sve_decompress_block_17to24<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    }
    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int sve_decompress_blocks(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output,
                          const uint32_t blocks) {
    const uint8_t* block_input = input;
    float_t* block_output      = output;

    for (uint32_t block = 0; block < blocks; ++block) {
        sve_decompress_block<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(block_input, scale, block_output);
        block_input  += BLOCK_SIZE;
        block_output += (BLOCK_SIZE * 8) / BIT_WIDTH;
    }

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int sve_decompress_blocks(const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output,
                          const uint32_t blocks) {
    const uint8_t* block_input = input;
    double_t* block_output     = output;

    for (uint32_t block = 0; block < blocks; ++block) {
        sve_decompress_block<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(block_input, scale, block_output);
        block_input  += BLOCK_SIZE;
        block_output += (BLOCK_SIZE * 8) / BIT_WIDTH;
    }
    return 0;
}

#ifdef __cplusplus
extern "C" {
#endif

int sve_decompress_block(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output);


int sve_decompress_block_f64(uint8_t bit_width, const uint8_t* __restrict__ input, double_t scale,
                             double_t* __restrict__ output);

int sve_decompress_blocks(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output,
                          uint32_t blocks);

int sve_decompress_blocks_f64(uint8_t bit_width, const uint8_t* __restrict__ input, double_t scale,
                              double_t* __restrict__ output, uint32_t blocks);

#ifdef __cplusplus
}
#endif
} // namespace pernix::arm64::sve

#endif  // PERNIX_ARM64_SVE_DECOMPRESSION_H
