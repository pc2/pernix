#ifndef PERNIX_CPP_DISPATCHER_TEST_HELPERS_H
#define PERNIX_CPP_DISPATCHER_TEST_HELPERS_H

#include <pernix/pernix.hpp>

#include <array>
#include <cstddef>
#include <span>

namespace pernix_test {
struct CppDispatcherCache {
    std::array<pernix::Dispatcher, 9> dispatchers{};
    std::array<pernix::Status, 9> statuses{};

    CppDispatcherCache() {
        for (std::size_t i = 0; i < dispatchers.size(); ++i) {
            statuses[i] = pernix::Dispatcher::create(static_cast<pernix::Backend>(i), dispatchers[i]);
        }
    }
};

inline CppDispatcherCache& cpp_dispatcher_cache() {
    static CppDispatcherCache cache;
    return cache;
}

inline const pernix::Dispatcher* cpp_dispatcher(pernix::Backend backend, pernix::Status& status) {
    const auto index = static_cast<std::size_t>(backend);
    if (index >= cpp_dispatcher_cache().dispatchers.size()) {
        status = PERNIX_STATUS_UNSUPPORTED_BACKEND;
        return nullptr;
    }
    status = cpp_dispatcher_cache().statuses[index];
    return status == PERNIX_STATUS_OK ? &cpp_dispatcher_cache().dispatchers[index] : nullptr;
}

template <typename FloatT>
inline pernix::Status compress_block(pernix::Backend backend, u8 bit_width, u32 block_size, std::span<const FloatT> input,
                                     FloatT inverse_scale, std::span<u8> output) {
    pernix::Status status  = PERNIX_STATUS_OK;
    const auto* dispatcher = cpp_dispatcher(backend, status);
    return dispatcher == nullptr ? status : dispatcher->compress_block(bit_width, block_size, input, inverse_scale, output);
}

template <typename FloatT>
inline pernix::Status decompress_block(pernix::Backend backend, u8 bit_width, u32 block_size, std::span<const u8> input, FloatT scale,
                                       std::span<FloatT> output, bool sign_values = true) {
    pernix::Status status  = PERNIX_STATUS_OK;
    const auto* dispatcher = cpp_dispatcher(backend, status);
    return dispatcher == nullptr ? status : dispatcher->decompress_block(bit_width, block_size, input, scale, output, sign_values);
}

template <typename FloatT>
inline pernix::Status compress_blocks(pernix::Backend backend, u8 bit_width, u32 block_size, std::span<const FloatT> input,
                                      FloatT inverse_scale, std::span<u8> output, u32 blocks) {
    pernix::Status status  = PERNIX_STATUS_OK;
    const auto* dispatcher = cpp_dispatcher(backend, status);
    return dispatcher == nullptr ? status : dispatcher->compress_blocks(bit_width, block_size, input, inverse_scale, output, blocks);
}

template <typename FloatT>
inline pernix::Status decompress_blocks(pernix::Backend backend, u8 bit_width, u32 block_size, std::span<const u8> input, FloatT scale,
                                        std::span<FloatT> output, u32 blocks, bool sign_values = true) {
    pernix::Status status  = PERNIX_STATUS_OK;
    const auto* dispatcher = cpp_dispatcher(backend, status);
    return dispatcher == nullptr ? status : dispatcher->decompress_blocks(bit_width, block_size, input, scale, output, blocks, sign_values);
}
}  // namespace pernix_test

#endif  // PERNIX_CPP_DISPATCHER_TEST_HELPERS_H
