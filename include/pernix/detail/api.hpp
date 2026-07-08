#ifndef PERNIX_DETAIL_API_HPP
#define PERNIX_DETAIL_API_HPP

#include <pernix/backend.hpp>
#include <pernix/dispatch/select.h>

#include <cmath>
#include <limits>
#include <string_view>
#include <type_traits>

namespace pernix::detail {
constexpr u8 min_bit_width = 1;
constexpr u8 max_bit_width = 24;
constexpr u32 fixed_block_size = 64;

constexpr bool is_valid_bit_width(const u8 bit_width) {
    return bit_width >= min_bit_width && bit_width <= max_bit_width;
}

constexpr bool is_valid_block_size(const u32 block_size) {
    return block_size == fixed_block_size || block_size == 128 || block_size == 256 || block_size == 512 ||
           block_size == 1024;
}

constexpr u32 elements_per_block(const u8 bit_width, const u32 block_size = fixed_block_size) {
    return is_valid_bit_width(bit_width) && is_valid_block_size(block_size) ? (block_size * 8U) / bit_width : 0;
}

template <typename ScaleType>
constexpr ScaleType quantization_levels(const u8 bit_width) {
    return bit_width == 1 ? ScaleType{1} : static_cast<ScaleType>((1U << (bit_width - 1U)) - 1U);
}

template <typename ScaleType>
inline pernix_status scale_from_bmax(const ScaleType bmax, const u8 bit_width, ScaleType *scale) {
    if (scale == nullptr || !is_valid_bit_width(bit_width) || !std::isfinite(bmax) || bmax < ScaleType{0}) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }

    if (bmax <= ScaleType{0}) {
        *scale = std::numeric_limits<ScaleType>::epsilon();
    } else {
        *scale = bmax / quantization_levels<ScaleType>(bit_width);
    }
    return PERNIX_STATUS_OK;
}

template <typename ScaleType>
inline bool is_valid_scale(const ScaleType scale) {
    return std::isfinite(scale) && scale > ScaleType{0};
}

template <typename ScaleType>
inline pernix_status inverse_scale(const ScaleType scale, ScaleType *inverse_scale_value) {
    if (inverse_scale_value == nullptr || !is_valid_scale(scale)) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }

    *inverse_scale_value = ScaleType{1} / scale;
    return PERNIX_STATUS_OK;
}

template <typename ScaleType>
inline pernix_status compression_scale_from_bmax(const ScaleType bmax, const u8 bit_width,
                                                 ScaleType *inverse_scale_value) {
    ScaleType scale = ScaleType{0};
    const auto status = scale_from_bmax(bmax, bit_width, &scale);
    if (status != PERNIX_STATUS_OK) {
        return status;
    }
    return inverse_scale(scale, inverse_scale_value);
}

inline const char *status_string(const pernix_status status) {
    switch (status) {
    case PERNIX_STATUS_OK:
        return "PERNIX_STATUS_OK";
    case PERNIX_STATUS_INVALID_ARGUMENT:
        return "PERNIX_STATUS_INVALID_ARGUMENT";
    case PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH:
        return "PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH";
    case PERNIX_STATUS_UNSUPPORTED_BACKEND:
        return "PERNIX_STATUS_UNSUPPORTED_BACKEND";
    case PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE:
        return "PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE";
    case PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION:
        return "PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION";
    default:
        return "PERNIX_STATUS_UNKNOWN";
    }
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
    if (!is_valid_scale(scale)) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
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
    if (blocks == 0) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }
    if (!is_valid_block_size(block_size)) {
        return PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE;
    }
    if (!is_valid_scale(scale)) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
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
    if (!is_valid_scale(scale)) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
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
    if (blocks == 0) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }
    if (!is_valid_block_size(block_size)) {
        return PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE;
    }
    if (!is_valid_scale(scale)) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
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
