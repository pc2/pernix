#ifndef PERNIX_HPP
#define PERNIX_HPP

#include <limits>
#include <pernix/backend.hpp>
#include <pernix/detail/api.hpp>
#include <span>

namespace pernix {
using Status = pernix_status;

constexpr u8 min_bit_width() {
    return detail::min_bit_width;
}

constexpr u8 max_bit_width() {
    return detail::max_bit_width;
}

constexpr bool is_valid_bit_width(const u8 bit_width) {
    return detail::is_valid_bit_width(bit_width);
}

constexpr bool is_valid_block_size(const u32 block_size) {
    return detail::is_valid_block_size(block_size);
}

constexpr u32 compressed_block_size() {
    return detail::fixed_block_size;
}

constexpr u32 elements_per_block(const u8 bit_width) {
    return detail::elements_per_block(bit_width);
}

template <typename FloatT>
__always_inline Status scale_from_bmax(const FloatT bmax, const u8 bit_width, FloatT& scale) {
    return detail::scale_from_bmax(bmax, bit_width, &scale);
}

template <typename FloatT>
__always_inline Status decompression_scale_from_bmax(const FloatT bmax, const u8 bit_width, FloatT& scale) {
    return scale_from_bmax(bmax, bit_width, scale);
}

template <typename FloatT>
__always_inline Status inverse_scale(const FloatT scale, FloatT& inverse_scale_value) {
    return detail::inverse_scale(scale, &inverse_scale_value);
}

template <typename FloatT>
__always_inline Status compression_scale_from_bmax(const FloatT bmax, const u8 bit_width, FloatT& inverse_scale_value) {
    return detail::compression_scale_from_bmax(bmax, bit_width, &inverse_scale_value);
}

__always_inline const char* status_string(const Status status) {
    return detail::status_string(status);
}

namespace detail {
template <typename FloatT>
__always_inline Status validate_compress_spans(const u8 bit_width, const u32 block_size, const std::span<const FloatT> input,
                                               const std::span<u8> output, const u32 blocks) {
    if (!is_valid_bit_width(bit_width)) {
        return PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH;
    }
    if (!is_valid_block_size(block_size)) {
        return PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE;
    }
    if (blocks == 0) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }

    const usize required_input  = static_cast<usize>(elements_per_block(bit_width, block_size)) * blocks;
    const usize required_output = static_cast<usize>(block_size) * blocks;
    return input.size() >= required_input && output.size() >= required_output ? PERNIX_STATUS_OK : PERNIX_STATUS_INVALID_ARGUMENT;
}

template <typename FloatT>
__always_inline Status validate_decompress_spans(const u8 bit_width, const u32 block_size, const std::span<const u8> input,
                                                 const std::span<FloatT> output, const u32 blocks) {
    if (!is_valid_bit_width(bit_width)) {
        return PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH;
    }
    if (!is_valid_block_size(block_size)) {
        return PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE;
    }
    if (blocks == 0) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }

    const usize required_input  = static_cast<usize>(block_size) * blocks;
    const usize required_output = static_cast<usize>(elements_per_block(bit_width, block_size)) * blocks;
    return input.size() >= required_input && output.size() >= required_output ? PERNIX_STATUS_OK : PERNIX_STATUS_INVALID_ARGUMENT;
}
}  // namespace detail

PERNIX_API static Status compress_block(Backend backend, const u8 bit_width, const u32 block_size, const std::span<const float> input,
                                        const float inverse_scale, std::span<u8> output) {
    if (const Status status = detail::validate_compress_spans(bit_width, block_size, input, output, 1); status != PERNIX_STATUS_OK) {
        return status;
    }
    return detail::compress_block(backend, bit_width, block_size, input.data(), inverse_scale, output.data());
}

PERNIX_API static Status compress_block(Backend backend, const u8 bit_width, const u32 block_size, const std::span<const double> input,
                                        const double inverse_scale, std::span<u8> output) {
    if (const Status status = detail::validate_compress_spans(bit_width, block_size, input, output, 1); status != PERNIX_STATUS_OK) {
        return status;
    }
    return detail::compress_block(backend, bit_width, block_size, input.data(), inverse_scale, output.data());
}

PERNIX_API static Status decompress_block(Backend backend, const u8 bit_width, const u32 block_size, const std::span<const u8> input,
                                          const float scale, std::span<float> output, const bool sign_values = true) {
    if (const Status status = detail::validate_decompress_spans(bit_width, block_size, input, output, 1); status != PERNIX_STATUS_OK) {
        return status;
    }
    return detail::decompress_block(backend, bit_width, block_size, input.data(), scale, output.data(), sign_values);
}

PERNIX_API static Status decompress_block(Backend backend, const u8 bit_width, const u32 block_size, const std::span<const u8> input,
                                          const double scale, std::span<double> output, const bool sign_values = true) {
    if (const Status status = detail::validate_decompress_spans(bit_width, block_size, input, output, 1); status != PERNIX_STATUS_OK) {
        return status;
    }
    return detail::decompress_block(backend, bit_width, block_size, input.data(), scale, output.data(), sign_values);
}

PERNIX_API static Status compress_blocks(Backend backend, const u8 bit_width, const u32 block_size, const std::span<const float> input,
                                         const float inverse_scale, std::span<u8> output, const u32 blocks) {
    if (const Status status = detail::validate_compress_spans(bit_width, block_size, input, output, blocks); status != PERNIX_STATUS_OK) {
        return status;
    }
    return detail::compress_blocks(backend, bit_width, block_size, input.data(), inverse_scale, output.data(), blocks);
}

PERNIX_API static Status compress_blocks(Backend backend, const u8 bit_width, const u32 block_size, const std::span<const double> input,
                                         const double inverse_scale, std::span<u8> output, const u32 blocks) {
    if (const Status status = detail::validate_compress_spans(bit_width, block_size, input, output, blocks); status != PERNIX_STATUS_OK) {
        return status;
    }
    return detail::compress_blocks(backend, bit_width, block_size, input.data(), inverse_scale, output.data(), blocks);
}

PERNIX_API static Status decompress_blocks(Backend backend, const u8 bit_width, const u32 block_size, const std::span<const u8> input,
                                           const float scale, std::span<float> output, const u32 blocks, const bool sign_values = true) {
    if (const Status status = detail::validate_decompress_spans(bit_width, block_size, input, output, blocks); status != PERNIX_STATUS_OK) {
        return status;
    }
    return detail::decompress_blocks(backend, bit_width, block_size, input.data(), scale, output.data(), blocks, sign_values);
}

PERNIX_API static Status decompress_blocks(Backend backend, const u8 bit_width, const u32 block_size, const std::span<const u8> input,
                                           const double scale, std::span<double> output, const u32 blocks, const bool sign_values = true) {
    if (const Status status = detail::validate_decompress_spans(bit_width, block_size, input, output, blocks); status != PERNIX_STATUS_OK) {
        return status;
    }
    return detail::decompress_blocks(backend, bit_width, block_size, input.data(), scale, output.data(), blocks, sign_values);
}

// convenience overloads without backend (defaults to Auto)
PERNIX_API static Status compress_block(const u8 bit_width, const u32 block_size, const std::span<const float> input,
                                        const float inverse_scale, const std::span<u8> output) {
    return compress_block(Backend::Auto, bit_width, block_size, input, inverse_scale, output);
}

PERNIX_API static Status compress_block(const u8 bit_width, const u32 block_size, const std::span<const double> input,
                                        const double inverse_scale, const std::span<u8> output) {
    return compress_block(Backend::Auto, bit_width, block_size, input, inverse_scale, output);
}

PERNIX_API static Status decompress_block(const u8 bit_width, const u32 block_size, const std::span<const u8> input, const float scale,
                                          const std::span<float> output, const bool sign_values = true) {
    return decompress_block(Backend::Auto, bit_width, block_size, input, scale, output, sign_values);
}

PERNIX_API static Status decompress_block(const u8 bit_width, const u32 block_size, const std::span<const u8> input, const double scale,
                                          const std::span<double> output, const bool sign_values = true) {
    return decompress_block(Backend::Auto, bit_width, block_size, input, scale, output, sign_values);
}

PERNIX_API static Status compress_blocks(const u8 bit_width, const u32 block_size, const std::span<const float> input,
                                         const float inverse_scale, const std::span<u8> output, const u32 blocks) {
    return compress_blocks(Backend::Auto, bit_width, block_size, input, inverse_scale, output, blocks);
}

PERNIX_API static Status compress_blocks(const u8 bit_width, const u32 block_size, const std::span<const double> input,
                                         const double inverse_scale, const std::span<u8> output, const u32 blocks) {
    return compress_blocks(Backend::Auto, bit_width, block_size, input, inverse_scale, output, blocks);
}

PERNIX_API static Status decompress_blocks(const u8 bit_width, const u32 block_size, const std::span<const u8> input, const float scale,
                                           const std::span<float> output, const u32 blocks, const bool sign_values = true) {
    return decompress_blocks(Backend::Auto, bit_width, block_size, input, scale, output, blocks, sign_values);
}

PERNIX_API static Status decompress_blocks(const u8 bit_width, const u32 block_size, const std::span<const u8> input, const double scale,
                                           const std::span<double> output, const u32 blocks, const bool sign_values = true) {
    return decompress_blocks(Backend::Auto, bit_width, block_size, input, scale, output, blocks, sign_values);
}

// convenience overloads without backend and without block_size (defaults to 64)
PERNIX_API static Status compress_block(const u8 bit_width, const std::span<const float> input, const float inverse_scale,
                                        const std::span<u8> output) {
    return compress_block(Backend::Auto, bit_width, 64, input, inverse_scale, output);
}

PERNIX_API static Status compress_block(const u8 bit_width, const std::span<const double> input, const double inverse_scale,
                                        const std::span<u8> output) {
    return compress_block(Backend::Auto, bit_width, 64, input, inverse_scale, output);
}

PERNIX_API static Status decompress_block(const u8 bit_width, const std::span<const u8> input, const float scale,
                                          const std::span<float> output, const bool sign_values = true) {
    return decompress_block(Backend::Auto, bit_width, 64, input, scale, output, sign_values);
}

PERNIX_API static Status decompress_block(const u8 bit_width, const std::span<const u8> input, const double scale,
                                          const std::span<double> output, const bool sign_values = true) {
    return decompress_block(Backend::Auto, bit_width, 64, input, scale, output, sign_values);
}

PERNIX_API static Status compress_blocks(const u8 bit_width, const std::span<const float> input, const float inverse_scale,
                                         const std::span<u8> output, const u32 blocks) {
    return compress_blocks(Backend::Auto, bit_width, 64, input, inverse_scale, output, blocks);
}

PERNIX_API static Status compress_blocks(const u8 bit_width, const std::span<const double> input, const double inverse_scale,
                                         const std::span<u8> output, const u32 blocks) {
    return compress_blocks(Backend::Auto, bit_width, 64, input, inverse_scale, output, blocks);
}

PERNIX_API static Status decompress_blocks(const u8 bit_width, const std::span<const u8> input, const float scale,
                                           const std::span<float> output, const u32 blocks, const bool sign_values = true) {
    return decompress_blocks(Backend::Auto, bit_width, 64, input, scale, output, blocks, sign_values);
}

PERNIX_API static Status decompress_blocks(const u8 bit_width, const std::span<const u8> input, const double scale,
                                           const std::span<double> output, const u32 blocks, const bool sign_values = true) {
    return decompress_blocks(Backend::Auto, bit_width, 64, input, scale, output, blocks, sign_values);
}
}  // namespace pernix

#endif  // PERNIX_HPP
