#ifndef PERNIX_ARM64_SVE2_DECOMPRESSION_H
#define PERNIX_ARM64_SVE2_DECOMPRESSION_H

#include <pernix/simd_compat.h>

#include <cmath>
#include <cstdint>

namespace pernix {
namespace internal {
template <uint8_t>
inline constexpr bool sve2_decompression_unimplemented_v = false;
} // namespace internal

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int sve2_decompress_block(const uint8_t*, float_t, float_t*) {
    static_assert(internal::sve2_decompression_unimplemented_v<BIT_WIDTH>, "ARM64 SVE2 decompression is not implemented yet");
    return -1;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int sve2_decompress_block(const uint8_t*, double_t, double_t*) {
    static_assert(internal::sve2_decompression_unimplemented_v<BIT_WIDTH>, "ARM64 SVE2 decompression is not implemented yet");
    return -1;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int sve2_decompress_blocks(const uint8_t*, float_t, float_t*, uint32_t) {
    static_assert(internal::sve2_decompression_unimplemented_v<BIT_WIDTH>, "ARM64 SVE2 decompression is not implemented yet");
    return -1;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int sve2_decompress_blocks(const uint8_t*, double_t, double_t*, uint32_t) {
    static_assert(internal::sve2_decompression_unimplemented_v<BIT_WIDTH>, "ARM64 SVE2 decompression is not implemented yet");
    return -1;
}

#ifdef __cplusplus
extern "C" {
#endif

int sve2_decompress_block(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output);
int sve2_decompress_block_f64(uint8_t bit_width, const uint8_t* __restrict__ input, double_t scale, double_t* __restrict__ output);
int sve2_decompress_blocks(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output,
                           uint32_t blocks);
int sve2_decompress_blocks_f64(uint8_t bit_width, const uint8_t* __restrict__ input, double_t scale, double_t* __restrict__ output,
                               uint32_t blocks);

#ifdef __cplusplus
}
#endif
} // namespace pernix

#endif  // PERNIX_ARM64_SVE2_DECOMPRESSION_H
