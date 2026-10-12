#ifndef PERNIX_DISPATCHER_TEST_HELPERS_H
#define PERNIX_DISPATCHER_TEST_HELPERS_H

#include <pernix/pernix.h>

#include <array>
#include <cstddef>

namespace pernix_test {
struct CDispatcherCache {
    std::array<pernix_dispatcher*, 9> dispatchers{};
    std::array<pernix_status, 9> statuses{};

    CDispatcherCache() {
        for (std::size_t i = 0; i < dispatchers.size(); ++i) {
            statuses[i] = pernix_dispatcher_create(static_cast<pernix_backend>(i), &dispatchers[i]);
        }
    }

    ~CDispatcherCache() {
        for (pernix_dispatcher* dispatcher : dispatchers) {
            pernix_dispatcher_destroy(dispatcher);
        }
    }
};

inline CDispatcherCache& c_dispatcher_cache() {
    static CDispatcherCache cache;
    return cache;
}

inline pernix_status c_dispatcher(pernix_backend backend, const pernix_dispatcher** dispatcher) {
    const auto index = static_cast<unsigned int>(backend);
    if (dispatcher == nullptr) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }
    if (index >= c_dispatcher_cache().dispatchers.size()) {
        *dispatcher = nullptr;
        return PERNIX_STATUS_UNSUPPORTED_BACKEND;
    }
    const auto& cache = c_dispatcher_cache();
    *dispatcher = cache.dispatchers[index];
    return cache.statuses[index];
}

inline pernix_status test_pernix_compress_block_f32(pernix_backend backend, u8 bit_width, u32 block_size, const void* input,
                                                    float inverse_scale, void* output) {
    const pernix_dispatcher* dispatcher = nullptr;
    const pernix_status status = c_dispatcher(backend, &dispatcher);
    return status == PERNIX_STATUS_OK ? pernix_compress_block_f32(dispatcher, bit_width, block_size, input, inverse_scale, output)
                                      : status;
}

inline pernix_status test_pernix_compress_blocks_f32(pernix_backend backend, u8 bit_width, u32 block_size, const void* input,
                                                     float inverse_scale, void* output, u32 blocks) {
    const pernix_dispatcher* dispatcher = nullptr;
    const pernix_status status = c_dispatcher(backend, &dispatcher);
    return status == PERNIX_STATUS_OK
               ? pernix_compress_blocks_f32(dispatcher, bit_width, block_size, input, inverse_scale, output, blocks)
               : status;
}

inline pernix_status test_pernix_decompress_block_f32(pernix_backend backend, u8 bit_width, u32 block_size, const void* input,
                                                      float scale, void* output, bool sign_values) {
    const pernix_dispatcher* dispatcher = nullptr;
    const pernix_status status = c_dispatcher(backend, &dispatcher);
    return status == PERNIX_STATUS_OK
               ? pernix_decompress_block_f32(dispatcher, bit_width, block_size, input, scale, output, sign_values)
               : status;
}

inline pernix_status test_pernix_decompress_blocks_f32(pernix_backend backend, u8 bit_width, u32 block_size, const void* input,
                                                       float scale, void* output, u32 blocks, bool sign_values) {
    const pernix_dispatcher* dispatcher = nullptr;
    const pernix_status status = c_dispatcher(backend, &dispatcher);
    return status == PERNIX_STATUS_OK
               ? pernix_decompress_blocks_f32(dispatcher, bit_width, block_size, input, scale, output, blocks, sign_values)
               : status;
}

inline pernix_status test_pernix_compress_block_f64(pernix_backend backend, u8 bit_width, u32 block_size, const void* input,
                                                    double inverse_scale, void* output) {
    const pernix_dispatcher* dispatcher = nullptr;
    const pernix_status status = c_dispatcher(backend, &dispatcher);
    return status == PERNIX_STATUS_OK ? pernix_compress_block_f64(dispatcher, bit_width, block_size, input, inverse_scale, output)
                                      : status;
}

inline pernix_status test_pernix_compress_blocks_f64(pernix_backend backend, u8 bit_width, u32 block_size, const void* input,
                                                     double inverse_scale, void* output, u32 blocks) {
    const pernix_dispatcher* dispatcher = nullptr;
    const pernix_status status = c_dispatcher(backend, &dispatcher);
    return status == PERNIX_STATUS_OK
               ? pernix_compress_blocks_f64(dispatcher, bit_width, block_size, input, inverse_scale, output, blocks)
               : status;
}

inline pernix_status test_pernix_decompress_block_f64(pernix_backend backend, u8 bit_width, u32 block_size, const void* input,
                                                      double scale, void* output, bool sign_values) {
    const pernix_dispatcher* dispatcher = nullptr;
    const pernix_status status = c_dispatcher(backend, &dispatcher);
    return status == PERNIX_STATUS_OK
               ? pernix_decompress_block_f64(dispatcher, bit_width, block_size, input, scale, output, sign_values)
               : status;
}

inline pernix_status test_pernix_decompress_blocks_f64(pernix_backend backend, u8 bit_width, u32 block_size, const void* input,
                                                       double scale, void* output, u32 blocks, bool sign_values) {
    const pernix_dispatcher* dispatcher = nullptr;
    const pernix_status status = c_dispatcher(backend, &dispatcher);
    return status == PERNIX_STATUS_OK
               ? pernix_decompress_blocks_f64(dispatcher, bit_width, block_size, input, scale, output, blocks, sign_values)
               : status;
}
}  // namespace pernix_test

#endif  // PERNIX_DISPATCHER_TEST_HELPERS_H
