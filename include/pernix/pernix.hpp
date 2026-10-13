#ifndef PERNIX_HPP
#define PERNIX_HPP

#include <pernix/backend.hpp>
#include <pernix/detail/api.hpp>

#include <span>
#include <type_traits>
#include <utility>

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
__always_inline Status decompression_scale_from_bmax(const FloatT bmax, const u8 bit_width, FloatT& scale) {
    return detail::scale_from_bmax(bmax, bit_width, &scale);
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

class Dispatcher {
public:
    Dispatcher() = default;

    Dispatcher(const Dispatcher&)            = delete;
    Dispatcher& operator=(const Dispatcher&) = delete;

    Dispatcher(Dispatcher&& other) noexcept
        : backend_(std::exchange(other.backend_, Backend::Auto)), configured_(std::exchange(other.configured_, false))
#if defined(PERNIX_USE_COMPILED_LIB)
          ,
          handle_(std::exchange(other.handle_, nullptr))
#endif
    {
    }

    Dispatcher& operator=(Dispatcher&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        release();
        backend_    = std::exchange(other.backend_, Backend::Auto);
        configured_ = std::exchange(other.configured_, false);
#if defined(PERNIX_USE_COMPILED_LIB)
        handle_ = std::exchange(other.handle_, nullptr);
#endif
        return *this;
    }

    ~Dispatcher() { release(); }

    static Status create(const Backend requested, Dispatcher& output) noexcept {
        Dispatcher configured;
#if defined(PERNIX_USE_COMPILED_LIB)
        Status status = pernix_dispatcher_create(static_cast<pernix_backend>(requested), &configured.handle_);
        if (status != PERNIX_STATUS_OK) {
            return status;
        }
        pernix_backend resolved = PERNIX_BACKEND_AUTO;
        status                  = pernix_dispatcher_backend(configured.handle_, &resolved);
        if (status != PERNIX_STATUS_OK) {
            return status;
        }
        configured.backend_ = static_cast<Backend>(resolved);
#else
        Backend resolved    = Backend::Fallback;
        const Status status = internal::resolve_backend(requested, &resolved);
        if (status != PERNIX_STATUS_OK) {
            return status;
        }
        configured.backend_ = resolved;
#endif
        configured.configured_ = true;
        output                 = std::move(configured);
        return PERNIX_STATUS_OK;
    }

    Backend backend() const noexcept { return configured_ ? backend_ : Backend::Auto; }

    bool configured() const noexcept { return configured_; }

    template <typename FloatT>
        requires(std::is_same_v<FloatT, f32> || std::is_same_v<FloatT, f64>)
    Status compress_block(const u8 bit_width, const u32 block_size, const std::span<const FloatT> input, const FloatT inverse_scale,
                          const std::span<u8> output) const {
        if (!configured_) {
            return PERNIX_STATUS_INVALID_ARGUMENT;
        }
        if (const Status status = detail::validate_compress_spans(bit_width, block_size, input, output, 1); status != PERNIX_STATUS_OK) {
            return status;
        }
        if constexpr (std::is_same_v<FloatT, f32>) {
#if defined(PERNIX_USE_COMPILED_LIB)
            return pernix_compress_block_f32(handle_, bit_width, block_size, input.data(), inverse_scale, output.data());
#else
            return detail::compress_block(backend_, bit_width, block_size, input.data(), inverse_scale, output.data());
#endif
        } else {
#if defined(PERNIX_USE_COMPILED_LIB)
            return pernix_compress_block_f64(handle_, bit_width, block_size, input.data(), inverse_scale, output.data());
#else
            return detail::compress_block(backend_, bit_width, block_size, input.data(), inverse_scale, output.data());
#endif
        }
    }

    template <typename FloatT>
        requires(std::is_same_v<FloatT, f32> || std::is_same_v<FloatT, f64>)
    Status decompress_block(const u8 bit_width, const u32 block_size, const std::span<const u8> input, const FloatT scale,
                            const std::span<FloatT> output, const bool sign_values = true) const {
        if (!configured_) {
            return PERNIX_STATUS_INVALID_ARGUMENT;
        }
        if (const Status status = detail::validate_decompress_spans(bit_width, block_size, input, output, 1); status != PERNIX_STATUS_OK) {
            return status;
        }
        if constexpr (std::is_same_v<FloatT, f32>) {
#if defined(PERNIX_USE_COMPILED_LIB)
            return pernix_decompress_block_f32(handle_, bit_width, block_size, input.data(), scale, output.data(), sign_values);
#else
            return detail::decompress_block(backend_, bit_width, block_size, input.data(), scale, output.data(), sign_values);
#endif
        } else {
#if defined(PERNIX_USE_COMPILED_LIB)
            return pernix_decompress_block_f64(handle_, bit_width, block_size, input.data(), scale, output.data(), sign_values);
#else
            return detail::decompress_block(backend_, bit_width, block_size, input.data(), scale, output.data(), sign_values);
#endif
        }
    }

    template <typename FloatT>
        requires(std::is_same_v<FloatT, f32> || std::is_same_v<FloatT, f64>)
    Status compress_blocks(const u8 bit_width, const u32 block_size, const std::span<const FloatT> input, const FloatT inverse_scale,
                           const std::span<u8> output, const u32 blocks) const {
        if (!configured_) {
            return PERNIX_STATUS_INVALID_ARGUMENT;
        }
        if (const Status status = detail::validate_compress_spans(bit_width, block_size, input, output, blocks);
            status != PERNIX_STATUS_OK) {
            return status;
        }
        if constexpr (std::is_same_v<FloatT, f32>) {
#if defined(PERNIX_USE_COMPILED_LIB)
            return pernix_compress_blocks_f32(handle_, bit_width, block_size, input.data(), inverse_scale, output.data(), blocks);
#else
            return detail::compress_blocks(backend_, bit_width, block_size, input.data(), inverse_scale, output.data(), blocks);
#endif
        } else {
#if defined(PERNIX_USE_COMPILED_LIB)
            return pernix_compress_blocks_f64(handle_, bit_width, block_size, input.data(), inverse_scale, output.data(), blocks);
#else
            return detail::compress_blocks(backend_, bit_width, block_size, input.data(), inverse_scale, output.data(), blocks);
#endif
        }
    }

    template <typename FloatT>
        requires(std::is_same_v<FloatT, f32> || std::is_same_v<FloatT, f64>)
    Status decompress_blocks(const u8 bit_width, const u32 block_size, const std::span<const u8> input, const FloatT scale,
                             const std::span<FloatT> output, const u32 blocks, const bool sign_values = true) const {
        if (!configured_) {
            return PERNIX_STATUS_INVALID_ARGUMENT;
        }
        if (const Status status = detail::validate_decompress_spans(bit_width, block_size, input, output, blocks);
            status != PERNIX_STATUS_OK) {
            return status;
        }
        if constexpr (std::is_same_v<FloatT, f32>) {
#if defined(PERNIX_USE_COMPILED_LIB)
            return pernix_decompress_blocks_f32(handle_, bit_width, block_size, input.data(), scale, output.data(), blocks, sign_values);
#else
            return detail::decompress_blocks(backend_, bit_width, block_size, input.data(), scale, output.data(), blocks, sign_values);
#endif
        } else {
#if defined(PERNIX_USE_COMPILED_LIB)
            return pernix_decompress_blocks_f64(handle_, bit_width, block_size, input.data(), scale, output.data(), blocks, sign_values);
#else
            return detail::decompress_blocks(backend_, bit_width, block_size, input.data(), scale, output.data(), blocks, sign_values);
#endif
        }
    }

    template <typename FloatT>
        requires(std::is_same_v<FloatT, f32> || std::is_same_v<FloatT, f64>)
    Status compress_block(const u8 bit_width, const std::span<const FloatT> input, const FloatT inverse_scale,
                          const std::span<u8> output) const {
        return compress_block(bit_width, detail::fixed_block_size, input, inverse_scale, output);
    }

    template <typename FloatT>
        requires(std::is_same_v<FloatT, f32> || std::is_same_v<FloatT, f64>)
    Status decompress_block(const u8 bit_width, const std::span<const u8> input, const FloatT scale, const std::span<FloatT> output,
                            const bool sign_values = true) const {
        return decompress_block(bit_width, detail::fixed_block_size, input, scale, output, sign_values);
    }

    template <typename FloatT>
        requires(std::is_same_v<FloatT, f32> || std::is_same_v<FloatT, f64>)
    Status compress_blocks(const u8 bit_width, const std::span<const FloatT> input, const FloatT inverse_scale, const std::span<u8> output,
                           const u32 blocks) const {
        return compress_blocks(bit_width, detail::fixed_block_size, input, inverse_scale, output, blocks);
    }

    template <typename FloatT>
        requires(std::is_same_v<FloatT, f32> || std::is_same_v<FloatT, f64>)
    Status decompress_blocks(const u8 bit_width, const std::span<const u8> input, const FloatT scale, const std::span<FloatT> output,
                             const u32 blocks, const bool sign_values = true) const {
        return decompress_blocks(bit_width, detail::fixed_block_size, input, scale, output, blocks, sign_values);
    }

private:
    void release() noexcept {
#if defined(PERNIX_USE_COMPILED_LIB)
        pernix_dispatcher_destroy(handle_);
        handle_ = nullptr;
#endif
    }

    Backend backend_ = Backend::Auto;
    bool configured_ = false;
#if defined(PERNIX_USE_COMPILED_LIB)
    pernix_dispatcher* handle_ = nullptr;
#endif
};
}  // namespace pernix

#endif  // PERNIX_HPP
