#ifndef PERNIX_H
#define PERNIX_H

#include <pernix/compat.h>
#include <cstdint>
#include <span>

#if defined(__cplusplus)
extern "C" {
#endif

typedef enum pernix_status {
    PERNIX_STATUS_OK = 0,
    PERNIX_STATUS_INVALID_ARGUMENT = -1,
    PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH = -2,
    PERNIX_STATUS_UNSUPPORTED_BACKEND = -3,
    PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE = -4
} pernix_status;

typedef enum pernix_backend {
    PERNIX_BACKEND_AUTO = 0,
    PERNIX_BACKEND_FALLBACK = 1,
    PERNIX_BACKEND_X86_AVX2 = 2,
    PERNIX_BACKEND_X86_BMI2 = 3,
    PERNIX_BACKEND_X86_AVX512_VBMI = 4,
    PERNIX_BACKEND_ARM64_NEON = 5,
    PERNIX_BACKEND_ARM64_SVE = 6
} pernix_backend;

PERNIX_API pernix_status pernix_compress_block_f32(pernix_backend backend, uint8_t bit_width, uint32_t block_size, const void* input,
                                                   float scale, void* output);

PERNIX_API pernix_status pernix_compress_blocks_f32(pernix_backend backend, uint8_t bit_width, uint32_t block_size, const void* input,
                                                    float scale, void* output, uint32_t blocks);

PERNIX_API pernix_status pernix_decompress_block_f32(pernix_backend backend, uint8_t bit_width, uint32_t block_size, const void* input,
                                                     float scale, void* output, bool sign_values);

PERNIX_API pernix_status pernix_decompress_blocks_f32(pernix_backend backend, uint8_t bit_width, uint32_t block_size, const void* input,
                                                      float scale, void* output, uint32_t blocks, bool sign_values);

PERNIX_API pernix_status pernix_compress_block_f64(pernix_backend backend, uint8_t bit_width, uint32_t block_size, const void* input,
                                                   double scale, void* output);

PERNIX_API pernix_status pernix_compress_blocks_f64(pernix_backend backend, uint8_t bit_width, uint32_t block_size, const void* input,
                                                    double scale, void* output, uint32_t blocks);

PERNIX_API pernix_status pernix_decompress_block_f64(pernix_backend backend, uint8_t bit_width, uint32_t block_size, const void* input,
                                                     double scale, void* output, bool sign_values);

PERNIX_API pernix_status pernix_decompress_blocks_f64(pernix_backend backend, uint8_t bit_width, uint32_t block_size, const void* input,
                                                      double scale, void* output, uint32_t blocks, bool sign_values);

#if defined(__cplusplus)
}
#endif

#endif //PERNIX_H
