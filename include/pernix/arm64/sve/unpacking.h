#ifndef PERNIX_ARM64_SVE_UNPACKING_H
#define PERNIX_ARM64_SVE_UNPACKING_H

#include <pernix/simd_compat.h>

namespace pernix::arm64::sve::internal {
template <uint8_t>
inline constexpr bool unpacking_unimplemented_v = false;
} // namespace pernix::arm64::sve::internal

#endif  // PERNIX_ARM64_SVE_UNPACKING_H
