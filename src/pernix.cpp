#include <pernix/detail/api.hpp>

#include <new>

struct pernix_dispatcher {
    pernix::Backend backend;
};

extern "C" {
u8 pernix_min_bit_width(void) {
    return pernix::detail::min_bit_width;
}

u8 pernix_max_bit_width(void) {
    return pernix::detail::max_bit_width;
}

bool pernix_is_valid_bit_width(u8 bit_width) {
    return pernix::detail::is_valid_bit_width(bit_width);
}

bool pernix_is_valid_block_size(u32 block_size) {
    return pernix::detail::is_valid_block_size(block_size);
}

u32 pernix_compressed_block_size(void) {
    return pernix::detail::fixed_block_size;
}

u32 pernix_elements_per_block(u8 bit_width) {
    return pernix::detail::elements_per_block(bit_width);
}

const char* pernix_status_string(pernix_status status) {
    return pernix::detail::status_string(status);
}

pernix_status pernix_decompression_scale_f32(float bmax, u8 bit_width, float* scale) {
    return pernix::detail::scale_from_bmax(bmax, bit_width, scale);
}

pernix_status pernix_decompression_scale_f64(double bmax, u8 bit_width, double* scale) {
    return pernix::detail::scale_from_bmax(bmax, bit_width, scale);
}

pernix_status pernix_inverse_scale_f32(float scale, float* inverse_scale) {
    return pernix::detail::inverse_scale(scale, inverse_scale);
}

pernix_status pernix_inverse_scale_f64(double scale, double* inverse_scale) {
    return pernix::detail::inverse_scale(scale, inverse_scale);
}

pernix_status pernix_compression_scale_f32(float bmax, u8 bit_width, float* inverse_scale) {
    return pernix::detail::compression_scale_from_bmax(bmax, bit_width, inverse_scale);
}

pernix_status pernix_compression_scale_f64(double bmax, u8 bit_width, double* inverse_scale) {
    return pernix::detail::compression_scale_from_bmax(bmax, bit_width, inverse_scale);
}

pernix_status pernix_dispatcher_create(pernix_backend backend, pernix_dispatcher** dispatcher) {
    if (dispatcher == nullptr) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }
    *dispatcher = nullptr;

    const int requested_value = static_cast<int>(backend);
    if (requested_value < PERNIX_BACKEND_AUTO || requested_value > PERNIX_BACKEND_CUDA) {
        return PERNIX_STATUS_UNSUPPORTED_BACKEND;
    }

    pernix::Backend resolved   = pernix::Backend::Fallback;
    const pernix_status status = pernix::internal::resolve_backend(static_cast<pernix::Backend>(backend), &resolved);
    if (status != PERNIX_STATUS_OK) {
        return status;
    }

    auto* created = new (std::nothrow) pernix_dispatcher{resolved};
    if (created == nullptr) {
        return PERNIX_STATUS_OUT_OF_MEMORY;
    }
    *dispatcher = created;
    return PERNIX_STATUS_OK;
}

void pernix_dispatcher_destroy(pernix_dispatcher* dispatcher) {
    delete dispatcher;
}

pernix_status pernix_dispatcher_backend(const pernix_dispatcher* dispatcher, pernix_backend* backend) {
    if (dispatcher == nullptr || backend == nullptr) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }
    *backend = static_cast<pernix_backend>(dispatcher->backend);
    return PERNIX_STATUS_OK;
}

pernix_status pernix_compress_block_f32(const pernix_dispatcher* dispatcher, u8 bit_width, u32 block_size, const void* input, float scale,
                                        void* output) {
    if (dispatcher == nullptr) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }
    return pernix::detail::compress_block(dispatcher->backend, bit_width, block_size, input, scale, output);
}

pernix_status pernix_compress_blocks_f32(const pernix_dispatcher* dispatcher, u8 bit_width, u32 block_size, const void* input, float scale,
                                         void* output, u32 blocks) {
    if (dispatcher == nullptr) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }
    return pernix::detail::compress_blocks(dispatcher->backend, bit_width, block_size, input, scale, output, blocks);
}

pernix_status pernix_decompress_block_f32(const pernix_dispatcher* dispatcher, u8 bit_width, u32 block_size, const void* input, float scale,
                                          void* output, bool sign_values) {
    if (dispatcher == nullptr) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }
    return pernix::detail::decompress_block(dispatcher->backend, bit_width, block_size, input, scale, output, sign_values);
}

pernix_status pernix_decompress_blocks_f32(const pernix_dispatcher* dispatcher, u8 bit_width, u32 block_size, const void* input,
                                           float scale, void* output, u32 blocks, bool sign_values) {
    if (dispatcher == nullptr) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }
    return pernix::detail::decompress_blocks(dispatcher->backend, bit_width, block_size, input, scale, output, blocks, sign_values);
}

pernix_status pernix_compress_block_f64(const pernix_dispatcher* dispatcher, u8 bit_width, u32 block_size, const void* input, double scale,
                                        void* output) {
    if (dispatcher == nullptr) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }
    return pernix::detail::compress_block(dispatcher->backend, bit_width, block_size, input, scale, output);
}

pernix_status pernix_compress_blocks_f64(const pernix_dispatcher* dispatcher, u8 bit_width, u32 block_size, const void* input, double scale,
                                         void* output, u32 blocks) {
    if (dispatcher == nullptr) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }
    return pernix::detail::compress_blocks(dispatcher->backend, bit_width, block_size, input, scale, output, blocks);
}

pernix_status pernix_decompress_block_f64(const pernix_dispatcher* dispatcher, u8 bit_width, u32 block_size, const void* input,
                                          double scale, void* output, bool sign_values) {
    if (dispatcher == nullptr) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }
    return pernix::detail::decompress_block(dispatcher->backend, bit_width, block_size, input, scale, output, sign_values);
}

pernix_status pernix_decompress_blocks_f64(const pernix_dispatcher* dispatcher, u8 bit_width, u32 block_size, const void* input,
                                           double scale, void* output, u32 blocks, bool sign_values) {
    if (dispatcher == nullptr) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }
    return pernix::detail::decompress_blocks(dispatcher->backend, bit_width, block_size, input, scale, output, blocks, sign_values);
}
}
