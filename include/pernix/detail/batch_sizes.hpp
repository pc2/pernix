#ifndef PERNIX_DETAIL_BATCH_SIZES_HPP
#define PERNIX_DETAIL_BATCH_SIZES_HPP

#include <pernix/compat.h>

#include <limits>

namespace pernix::detail {
inline bool checked_batch_sizes(const u32 block_size, const u32 elements_per_block, const u32 blocks, usize& values,
                                usize& bytes) noexcept {
    if (block_size == 0 || elements_per_block == 0) {
        return false;
    }

    const usize count = static_cast<usize>(blocks);
    constexpr usize max_size = std::numeric_limits<usize>::max();
    if (count > max_size / static_cast<usize>(block_size) || count > max_size / static_cast<usize>(elements_per_block)) {
        return false;
    }

    values = count * static_cast<usize>(elements_per_block);
    bytes  = count * static_cast<usize>(block_size);
    return true;
}
}  // namespace pernix::detail

#endif  // PERNIX_DETAIL_BATCH_SIZES_HPP
