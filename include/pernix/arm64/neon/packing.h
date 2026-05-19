#ifndef PERNIX_ARM64_NEON_PACKING_H
#define PERNIX_ARM64_NEON_PACKING_H

#include <pernix/simd_compat.h>

namespace pernix::arm64::neon::internal {
template <uint8_t>
inline constexpr bool packing_unimplemented_v = false;
} // namespace pernix::arm64::neon::internal

#endif  // PERNIX_ARM64_NEON_PACKING_H
