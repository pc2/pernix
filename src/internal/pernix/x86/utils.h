#ifndef PERNIX_X86_UTILS_H
#define PERNIX_X86_UTILS_H

#include <cstdint>

namespace pernix::x86::internal {

static constexpr uint32_t tail_bytes(const uint8_t bit_width, const uint32_t remaining_elements) {
    const uint32_t tail_bits  = remaining_elements * bit_width;
    const uint32_t tail_bytes = (tail_bits + 7u) / 8u;
    return tail_bytes;
}

}  // namespace pernix::x86::internal

#endif  // PERNIX_X86_UTILS_H
