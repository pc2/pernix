#ifndef PERNIX_ARM64_SVE_COMPRESSION_H
#define PERNIX_ARM64_SVE_COMPRESSION_H

#include <pernix/simd_compat.h>

#include <cmath>
#include <cstdint>

namespace pernix {
namespace internal {
template <uint8_t>
inline constexpr bool sve_compression_unimplemented_v = false;
} // namespace internal

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int sve_compress_block(const float_t*, float_t, uint8_t*) {
    static_assert(internal::sve_compression_unimplemented_v<BIT_WIDTH>, "ARM64 SVE compression is not implemented yet");
    return -1;
}

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int sve_compress_block(const double_t*, double_t, uint8_t*) {
    static_assert(internal::sve_compression_unimplemented_v<BIT_WIDTH>, "ARM64 SVE compression is not implemented yet");
    return -1;
}

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int sve_compress_blocks(const float_t*, float_t, uint8_t*, uint32_t) {
    static_assert(internal::sve_compression_unimplemented_v<BIT_WIDTH>, "ARM64 SVE compression is not implemented yet");
    return -1;
}

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int sve_compress_blocks(const double_t*, double_t, uint8_t*, uint32_t) {
    static_assert(internal::sve_compression_unimplemented_v<BIT_WIDTH>, "ARM64 SVE compression is not implemented yet");
    return -1;
}

#ifdef __cplusplus
extern "C" {
#endif

int sve_compress_block(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output);
int sve_compress_block_f64(uint8_t bit_width, const double_t* __restrict__ input, double_t scale, uint8_t* __restrict__ output);
int sve_compress_blocks(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output,
                        uint32_t blocks);
int sve_compress_blocks_f64(uint8_t bit_width, const double_t* __restrict__ input, double_t scale, uint8_t* __restrict__ output,
                            uint32_t blocks);

#ifdef __cplusplus
}
#endif
} // namespace pernix

#endif  // PERNIX_ARM64_SVE_COMPRESSION_H
