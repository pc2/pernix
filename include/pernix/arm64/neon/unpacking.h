#ifndef PERNIX_ARM64_NEON_UNPACKING_H
#define PERNIX_ARM64_NEON_UNPACKING_H

#include <pernix/simd_compat.h>

namespace pernix::arm64::neon::internal {

template <uint8_t>
inline constexpr bool unpacking_unimplemented_v = false;

namespace b64 {

} // namespace b64


} // namespace pernix::arm64::neon::internal



#endif  // PERNIX_ARM64_NEON_UNPACKING_H
