#ifndef PERNIX_DETAIL_API_HPP
#define PERNIX_DETAIL_API_HPP

#include <pernix/backend.hpp>
#include <pernix/dispatch/select.h>

#include <string_view>
#include <type_traits>

namespace pernix::detail {
constexpr bool is_valid_block_size(const u32 block_size) {
    return block_size == 64 || block_size == 128 || block_size == 256 || block_size == 512 || block_size == 1024;
}

inline pernix_status select_error_status(const std::string_view kernel_name) {
    if (kernel_name == "invalid_backend") {
        return PERNIX_STATUS_UNSUPPORTED_BACKEND;
    }
    if (kernel_name == "unsupported_implementation") {
        return PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION;
    }
    return PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH;
}

template <typename ScaleType>
inline pernix_status compress_block(const Backend backend, const u8 bit_width, const u32 block_size,
                                    const void *input, const ScaleType scale, void *output) {
    if (input == nullptr || output == nullptr) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }
    if (!is_valid_block_size(block_size)) {
        return PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE;
    }

    const auto kernel = [&] {
        if constexpr (std::is_same_v<ScaleType, f32>) {
            return internal::select_compress_block_f32(backend, bit_width, block_size);
        } else {
            return internal::select_compress_block_f64(backend, bit_width, block_size);
        }
    }();
    if (!kernel) {
        return select_error_status(kernel.name);
    }

    return static_cast<pernix_status>(kernel.func(input, scale, output));
}

template <typename ScaleType>
inline pernix_status compress_blocks(const Backend backend, const u8 bit_width, const u32 block_size,
                                     const void *input, const ScaleType scale, void *output, const u32 blocks) {
    if (input == nullptr || output == nullptr) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }
    if (!is_valid_block_size(block_size)) {
        return PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE;
    }

    const auto kernel = [&] {
        if constexpr (std::is_same_v<ScaleType, f32>) {
            return internal::select_compress_blocks_f32(backend, bit_width, block_size);
        } else {
            return internal::select_compress_blocks_f64(backend, bit_width, block_size);
        }
    }();
    if (!kernel) {
        return select_error_status(kernel.name);
    }

    return static_cast<pernix_status>(kernel.func(input, scale, output, blocks));
}

template <typename ScaleType>
inline pernix_status decompress_block(const Backend backend, const u8 bit_width, const u32 block_size,
                                      const void *input, const ScaleType scale, void *output,
                                      const bool sign_values) {
    if (input == nullptr || output == nullptr) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }
    if (!is_valid_block_size(block_size)) {
        return PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE;
    }

    const auto kernel = [&] {
        if constexpr (std::is_same_v<ScaleType, f32>) {
            return internal::select_decompress_block_f32(backend, bit_width, block_size, sign_values);
        } else {
            return internal::select_decompress_block_f64(backend, bit_width, block_size, sign_values);
        }
    }();
    if (!kernel) {
        return select_error_status(kernel.name);
    }

    return static_cast<pernix_status>(kernel.func(input, scale, output));
}

template <typename ScaleType>
inline pernix_status decompress_blocks(const Backend backend, const u8 bit_width, const u32 block_size,
                                       const void *input, const ScaleType scale, void *output, const u32 blocks,
                                       const bool sign_values) {
    if (input == nullptr || output == nullptr) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }
    if (!is_valid_block_size(block_size)) {
        return PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE;
    }

    const auto kernel = [&] {
        if constexpr (std::is_same_v<ScaleType, f32>) {
            return internal::select_decompress_blocks_f32(backend, bit_width, block_size, sign_values);
        } else {
            return internal::select_decompress_blocks_f64(backend, bit_width, block_size, sign_values);
        }
    }();
    if (!kernel) {
        return select_error_status(kernel.name);
    }

    return static_cast<pernix_status>(kernel.func(input, scale, output, blocks));
}
}

#endif //PERNIX_DETAIL_API_HPP
