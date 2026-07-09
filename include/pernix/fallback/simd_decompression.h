#ifndef PERNIX_FALLBACK_SIMD_DECOMPRESSION_H
#define PERNIX_FALLBACK_SIMD_DECOMPRESSION_H

#include <pernix/pernix.h>

#include <cstddef>
#include <type_traits>
#include <version>

#if defined(__has_include)
#if __has_include(<simd>)
#include <simd>
#endif
#endif

namespace pernix {
template <u8 BIT_WIDTH, bool SIGN_VALUES, u32 BLOCK_SIZE, typename ScaleType>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<ScaleType>)
int decompress_block_fallback_simd(const void* input_ptr, ScaleType scale, void* output_ptr) {
#if defined(__cpp_lib_simd)
    using simd_placeholder                            = std::simd<i32>;
    [[maybe_unused]] constexpr std::size_t simd_lanes = simd_placeholder::size();
#endif
    [[maybe_unused]] constexpr bool sign_values       = SIGN_VALUES;
    [[maybe_unused]] constexpr u32 elements_per_block = (BLOCK_SIZE * 8U) / BIT_WIDTH;
    static_cast<void>(input_ptr);
    static_cast<void>(scale);
    static_cast<void>(output_ptr);
    static_cast<void>(sign_values);
    static_cast<void>(elements_per_block);
    return PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION;
}

template <u8 BIT_WIDTH, bool SIGN_VALUES, u32 BLOCK_SIZE, typename ScaleType>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<ScaleType>)
int decompress_blocks_fallback_simd(const void* input_ptr, ScaleType scale, void* output_ptr, u32 blocks) {
#if defined(__cpp_lib_simd)
    using simd_placeholder                            = std::simd<i32>;
    [[maybe_unused]] constexpr std::size_t simd_lanes = simd_placeholder::size();
#endif
    [[maybe_unused]] constexpr bool sign_values = SIGN_VALUES;
    static_cast<void>(input_ptr);
    static_cast<void>(scale);
    static_cast<void>(output_ptr);
    static_cast<void>(blocks);
    static_cast<void>(sign_values);
    return PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION;
}
}  // namespace pernix

#endif  // PERNIX_FALLBACK_SIMD_DECOMPRESSION_H
