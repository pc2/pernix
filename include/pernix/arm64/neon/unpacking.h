#ifndef PERNIX_ARM64_NEON_UNPACKING_H
#define PERNIX_ARM64_NEON_UNPACKING_H

#include <pernix/simd_compat.h>

namespace pernix::arm64::neon::internal::b128 {
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8)
__always_inline int8x16_t neon_unpack_epi8_1to8(const int8x16_t& input) {
    return input;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
__always_inline int16x8_t neon_unpack_epi8_9to16(const int16x8_t& input) {
    return input;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24)
__always_inline int32x4_t neon_unpack_epi8_17to24(const int32x4_t& input) {
    return input;
}
} // namespace pernix::arm64::neon::internal::b128

#endif  // PERNIX_ARM64_NEON_UNPACKING_H
