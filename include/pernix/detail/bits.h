#ifndef PERNIX_DETAIL_BITS_H
#define PERNIX_DETAIL_BITS_H

#include <pernix/compat.h>

namespace pernix::detail {
template <u8 BitWidth>
constexpr u32 low_bit_mask() {
    static_assert(BitWidth >= 1, "low_bit_mask requires at least one bit");
    static_assert(BitWidth < 32, "low_bit_mask requires fewer than 32 bits");
    return (u32{1} << BitWidth) - 1u;
}
}  // namespace pernix::detail

#endif  // PERNIX_DETAIL_BITS_H
