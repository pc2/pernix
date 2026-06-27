#ifndef PERNIX_FALLBACK_STDPAR_DECOMPRESSION_H
#define PERNIX_FALLBACK_STDPAR_DECOMPRESSION_H

#include <pernix/pernix.h>

#include <execution>
#include <type_traits>

namespace pernix {
template <u8 BIT_WIDTH, bool SIGN_VALUES, u32 BLOCK_SIZE, typename ScaleType>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<ScaleType>)
int decompress_block_fallback_stdpar(const void *input_ptr, ScaleType scale, void *output_ptr) {
    [[maybe_unused]] const auto policy = std::execution::par_unseq;
    [[maybe_unused]] constexpr bool sign_values = SIGN_VALUES;
    [[maybe_unused]] constexpr u32 elements_per_block = (BLOCK_SIZE * 8U) / BIT_WIDTH;
#if defined(PERNIX_FALLBACK_STDPAR_USE_NVCC)
    [[maybe_unused]] constexpr bool nvcc_execution_path = true;
#endif
    static_cast<void>(input_ptr);
    static_cast<void>(scale);
    static_cast<void>(output_ptr);
    static_cast<void>(sign_values);
    static_cast<void>(elements_per_block);
    return PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION;
}

template <u8 BIT_WIDTH, bool SIGN_VALUES, u32 BLOCK_SIZE, typename ScaleType>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<ScaleType>)
int decompress_blocks_fallback_stdpar(const void *input_ptr, ScaleType scale, void *output_ptr, u32 blocks) {
    [[maybe_unused]] const auto policy = std::execution::par_unseq;
    [[maybe_unused]] constexpr bool sign_values = SIGN_VALUES;
#if defined(PERNIX_FALLBACK_STDPAR_USE_NVCC)
    [[maybe_unused]] constexpr bool nvcc_execution_path = true;
#endif
    static_cast<void>(input_ptr);
    static_cast<void>(scale);
    static_cast<void>(output_ptr);
    static_cast<void>(blocks);
    static_cast<void>(sign_values);
    return PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION;
}
}

#endif // PERNIX_FALLBACK_STDPAR_DECOMPRESSION_H
