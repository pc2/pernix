#ifndef PERNIX_X86_UTILS_H
#define PERNIX_X86_UTILS_H

#include <pernix/compat.h>

namespace pernix::x86::internal {
static constexpr u32 tail_bytes(const u8 bit_width, const u32 remaining_elements) {
    const u32 tail_bits  = remaining_elements * bit_width;
    const u32 tail_bytes = (tail_bits + 7u) / 8u;
    return tail_bytes;
}
}  // namespace pernix::x86::internal

#endif  // PERNIX_X86_UTILS_H
