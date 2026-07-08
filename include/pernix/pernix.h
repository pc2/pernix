#ifndef PERNIX_H
#define PERNIX_H

#include <pernix/compat.h>

#if !defined(__cplusplus)
#include <stdbool.h>
#endif

#if defined(__cplusplus)
extern "C" {
#endif

typedef enum pernix_status {
    PERNIX_STATUS_OK = 0,
    PERNIX_STATUS_INVALID_ARGUMENT = -1,
    PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH = -2,
    PERNIX_STATUS_UNSUPPORTED_BACKEND = -3,
    PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE = -4,
    PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION = -5,
    PERNIX_STATUS_UNSUPPORTED_IMPLEMENTAION = PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION
} pernix_status;

typedef enum pernix_backend {
    PERNIX_BACKEND_AUTO = 0,
    PERNIX_BACKEND_FALLBACK = 1,
    PERNIX_BACKEND_FALLBACK_SCALAR = PERNIX_BACKEND_FALLBACK,
    PERNIX_BACKEND_X86_AVX2 = 2,
    PERNIX_BACKEND_X86_BMI2 = 3,
    PERNIX_BACKEND_X86_AVX512_VBMI = 4,
    PERNIX_BACKEND_ARM64_NEON = 5,
    PERNIX_BACKEND_ARM64_SVE = 6,
    PERNIX_BACKEND_FALLBACK_STDPAR = 7,
    PERNIX_BACKEND_FALLBACK_SIMD = 8
} pernix_backend;

PERNIX_API u8 pernix_min_bit_width(void);

PERNIX_API u8 pernix_max_bit_width(void);

PERNIX_API bool pernix_is_valid_bit_width(u8 bit_width);

PERNIX_API bool pernix_is_valid_block_size(u32 block_size);

PERNIX_API u32 pernix_compressed_block_size(void);

PERNIX_API u32 pernix_elements_per_block(u8 bit_width);

PERNIX_API pernix_status pernix_scale_f32(float bmax, u8 bit_width, float* scale);

PERNIX_API pernix_status pernix_scale_f64(double bmax, u8 bit_width, double* scale);

PERNIX_API pernix_status pernix_compress_block_f32(pernix_backend backend, u8 bit_width, u32 block_size,
                                                   const void* input,
                                                   float scale, void* output);

PERNIX_API pernix_status pernix_compress_blocks_f32(pernix_backend backend, u8 bit_width, u32 block_size,
                                                    const void* input,
                                                    float scale, void* output, u32 blocks);

PERNIX_API pernix_status pernix_decompress_block_f32(pernix_backend backend, u8 bit_width, u32 block_size,
                                                     const void* input,
                                                     float scale, void* output, bool sign_values);

PERNIX_API pernix_status pernix_decompress_blocks_f32(pernix_backend backend, u8 bit_width, u32 block_size,
                                                      const void* input,
                                                      float scale, void* output, u32 blocks, bool sign_values);

PERNIX_API pernix_status pernix_compress_block_f64(pernix_backend backend, u8 bit_width, u32 block_size,
                                                   const void* input,
                                                   double scale, void* output);

PERNIX_API pernix_status pernix_compress_blocks_f64(pernix_backend backend, u8 bit_width, u32 block_size,
                                                    const void* input,
                                                    double scale, void* output, u32 blocks);

PERNIX_API pernix_status pernix_decompress_block_f64(pernix_backend backend, u8 bit_width, u32 block_size,
                                                     const void* input,
                                                     double scale, void* output, bool sign_values);

PERNIX_API pernix_status pernix_decompress_blocks_f64(pernix_backend backend, u8 bit_width, u32 block_size,
                                                      const void* input,
                                                      double scale, void* output, u32 blocks, bool sign_values);

#if defined(__cplusplus)
}
#endif

#endif //PERNIX_H
