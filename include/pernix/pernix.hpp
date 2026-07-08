#ifndef PERNIX_HPP
#define PERNIX_HPP

#include <pernix/backend.hpp>
#include <pernix/detail/api.hpp>

#include <limits>
#include <span>

namespace pernix {
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
__always_inline int scale_from_bmax(const FloatT bmax, const u8 bit_width, FloatT &scale) {
    return detail::scale_from_bmax(bmax, bit_width, &scale);
}

namespace detail {
template <typename FloatT>
__always_inline int validate_compress_spans(const u8 bit_width, const u32 block_size,
                                            const std::span<const FloatT> input, const std::span<u8> output,
                                            const u32 blocks) {
    if (!is_valid_bit_width(bit_width)) {
        return PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH;
    }
    if (!is_valid_block_size(block_size)) {
        return PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE;
    }
    if (blocks == 0) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }

    const usize required_input = static_cast<usize>(elements_per_block(bit_width, block_size)) * blocks;
    const usize required_output = static_cast<usize>(block_size) * blocks;
    return input.size() >= required_input && output.size() >= required_output ? PERNIX_STATUS_OK
                                                                              : PERNIX_STATUS_INVALID_ARGUMENT;
}

template <typename FloatT>
__always_inline int validate_decompress_spans(const u8 bit_width, const u32 block_size, const std::span<const u8> input,
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

    const usize required_input = static_cast<usize>(block_size) * blocks;
    const usize required_output = static_cast<usize>(elements_per_block(bit_width, block_size)) * blocks;
    return input.size() >= required_input && output.size() >= required_output ? PERNIX_STATUS_OK
                                                                             : PERNIX_STATUS_INVALID_ARGUMENT;
}
} // namespace detail

__always_inline int compress_block(Backend backend, const u8 bit_width, const u32 block_size,
                                   const std::span<const float> input, const float scale, std::span<u8> output) {
    if (const int status = detail::validate_compress_spans(bit_width, block_size, input, output, 1);
        status != PERNIX_STATUS_OK) {
        return status;
    }
    return detail::compress_block(backend, bit_width, block_size, input.data(), scale, output.data());
}

__always_inline int compress_block(Backend backend, const u8 bit_width, const u32 block_size,
                                   const std::span<const double> input, const double scale, std::span<u8> output) {
    if (const int status = detail::validate_compress_spans(bit_width, block_size, input, output, 1);
        status != PERNIX_STATUS_OK) {
        return status;
    }
    return detail::compress_block(backend, bit_width, block_size, input.data(), scale, output.data());
}

__always_inline int decompress_block(Backend backend, const u8 bit_width, const u32 block_size,
                                     const std::span<const u8> input, const float scale, std::span<float> output,
                                     const bool sign_values = true) {
    if (const int status = detail::validate_decompress_spans(bit_width, block_size, input, output, 1);
        status != PERNIX_STATUS_OK) {
        return status;
    }
    return detail::decompress_block(backend, bit_width, block_size, input.data(), scale, output.data(), sign_values);
}

__always_inline int decompress_block(Backend backend, const u8 bit_width, const u32 block_size,
                                     const std::span<const u8> input, const double scale, std::span<double> output,
                                     const bool sign_values = true) {
    if (const int status = detail::validate_decompress_spans(bit_width, block_size, input, output, 1);
        status != PERNIX_STATUS_OK) {
        return status;
    }
    return detail::decompress_block(backend, bit_width, block_size, input.data(), scale, output.data(), sign_values);
}

__always_inline int compress_blocks(Backend backend, const u8 bit_width, const u32 block_size,
                                    const std::span<const float> input, const float scale, std::span<u8> output,
                                    const u32 blocks) {
    if (const int status = detail::validate_compress_spans(bit_width, block_size, input, output, blocks);
        status != PERNIX_STATUS_OK) {
        return status;
    }
    return detail::compress_blocks(backend, bit_width, block_size, input.data(), scale, output.data(), blocks);
}

__always_inline int compress_blocks(Backend backend, const u8 bit_width, const u32 block_size,
                                    const std::span<const double> input, const double scale, std::span<u8> output,
                                    const u32 blocks) {
    if (const int status = detail::validate_compress_spans(bit_width, block_size, input, output, blocks);
        status != PERNIX_STATUS_OK) {
        return status;
    }
    return detail::compress_blocks(backend, bit_width, block_size, input.data(), scale, output.data(), blocks);
}

__always_inline int decompress_blocks(Backend backend, const u8 bit_width, const u32 block_size,
                                      const std::span<const u8> input, const float scale, std::span<float> output,
                                      const u32 blocks, const bool sign_values = true) {
    if (const int status = detail::validate_decompress_spans(bit_width, block_size, input, output, blocks);
        status != PERNIX_STATUS_OK) {
        return status;
    }
    return detail::decompress_blocks(backend, bit_width, block_size, input.data(), scale, output.data(), blocks,
                                     sign_values);
}

__always_inline int decompress_blocks(Backend backend, const u8 bit_width, const u32 block_size,
                                      const std::span<const u8> input, const double scale, std::span<double> output,
                                      const u32 blocks, const bool sign_values = true) {
    if (const int status = detail::validate_decompress_spans(bit_width, block_size, input, output, blocks);
        status != PERNIX_STATUS_OK) {
        return status;
    }
    return detail::decompress_blocks(backend, bit_width, block_size, input.data(), scale, output.data(), blocks,
                                     sign_values);
}

// convenience overloads without backend (defaults to Auto)
__always_inline int compress_block(const u8 bit_width, const u32 block_size, const std::span<const float> input,
                                   const float scale, const std::span<u8> output) {
    return compress_block(Backend::Auto, bit_width, block_size, input, scale, output);
}

__always_inline int compress_block(const u8 bit_width, const u32 block_size, const std::span<const double> input,
                                   const double scale, const std::span<u8> output) {
    return compress_block(Backend::Auto, bit_width, block_size, input, scale, output);
}

__always_inline int decompress_block(const u8 bit_width, const u32 block_size, const std::span<const u8> input,
                                     const float scale, const std::span<float> output, const bool sign_values = true) {
    return decompress_block(Backend::Auto, bit_width, block_size, input, scale, output, sign_values);
}

__always_inline int decompress_block(const u8 bit_width, const u32 block_size, const std::span<const u8> input,
                                     const double scale, const std::span<double> output,
                                     const bool sign_values = true) {
    return decompress_block(Backend::Auto, bit_width, block_size, input, scale, output, sign_values);
}

__always_inline int compress_blocks(const u8 bit_width, const u32 block_size, const std::span<const float> input,
                                    const float scale, const std::span<u8> output, const u32 blocks) {
    return compress_blocks(Backend::Auto, bit_width, block_size, input, scale, output, blocks);
}

__always_inline int compress_blocks(const u8 bit_width, const u32 block_size, const std::span<const double> input,
                                    const double scale, const std::span<u8> output, const u32 blocks) {
    return compress_blocks(Backend::Auto, bit_width, block_size, input, scale, output, blocks);
}

__always_inline int decompress_blocks(const u8 bit_width, const u32 block_size, const std::span<const u8> input,
                                      const float scale, const std::span<float> output, const u32 blocks,
                                      const bool sign_values = true) {
    return decompress_blocks(Backend::Auto, bit_width, block_size, input, scale, output, blocks, sign_values);
}

__always_inline int decompress_blocks(const u8 bit_width, const u32 block_size, const std::span<const u8> input,
                                      const double scale, const std::span<double> output, const u32 blocks,
                                      const bool sign_values = true) {
    return decompress_blocks(Backend::Auto, bit_width, block_size, input, scale, output, blocks, sign_values);
}

// convenience overloads without backend and without block_size (defaults to 64)
__always_inline int compress_block(const u8 bit_width, const std::span<const float> input, const float scale,
                                   const std::span<u8> output) {
    return compress_block(Backend::Auto, bit_width, 64, input, scale, output);
}

__always_inline int compress_block(const u8 bit_width, const std::span<const double> input, const double scale,
                                   const std::span<u8> output) {
    return compress_block(Backend::Auto, bit_width, 64, input, scale, output);
}

__always_inline int decompress_block(const u8 bit_width, const std::span<const u8> input, const float scale,
                                     const std::span<float> output, const bool sign_values = true) {
    return decompress_block(Backend::Auto, bit_width, 64, input, scale, output, sign_values);
}

__always_inline int decompress_block(const u8 bit_width, const std::span<const u8> input, const double scale,
                                     const std::span<double> output, const bool sign_values = true) {
    return decompress_block(Backend::Auto, bit_width, 64, input, scale, output, sign_values);
}

__always_inline int compress_blocks(const u8 bit_width, const std::span<const float> input, const float scale,
                                    const std::span<u8> output, const u32 blocks) {
    return compress_blocks(Backend::Auto, bit_width, 64, input, scale, output, blocks);
}

__always_inline int compress_blocks(const u8 bit_width, const std::span<const double> input, const double scale,
                                    const std::span<u8> output, const u32 blocks) {
    return compress_blocks(Backend::Auto, bit_width, 64, input, scale, output, blocks);
}

__always_inline int decompress_blocks(const u8 bit_width, const std::span<const u8> input, const float scale,
                                      const std::span<float> output, const u32 blocks,
                                      const bool sign_values = true) {
    return decompress_blocks(Backend::Auto, bit_width, 64, input, scale, output, blocks, sign_values);
}

__always_inline int decompress_blocks(const u8 bit_width, const std::span<const u8> input, const double scale,
                                      const std::span<double> output, const u32 blocks,
                                      const bool sign_values = true) {
    return decompress_blocks(Backend::Auto, bit_width, 64, input, scale, output, blocks, sign_values);
}
} // namespace pernix

#endif //PERNIX_HPP
