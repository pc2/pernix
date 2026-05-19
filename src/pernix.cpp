#include <pernix/pernix.h>

#ifdef __cplusplus
namespace pernix {
extern "C" {
#endif

// Use the best available implementation based on detected CPU features at compile time
#if defined(PERNIX_BACKEND_X86) && defined(PERNIX_AVX2_ENABLED)
#ifdef PERNIX_AVX512_VBMI_ENABLED
int compress_block(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output) {
    return mm512_compress_block_avx512vbmi(bit_width, input, scale, output);
}

int compress_block_f64(const uint8_t bit_width, const double_t* __restrict__ input, const double_t scale, uint8_t* __restrict__ output) {
    return mm512_compress_block_f64_avx512vbmi(bit_width, input, scale, output);
}

int compress_blocks(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output,
                    const uint32_t blocks) {
    return mm512_compress_blocks_avx512vbmi(bit_width, input, scale, output, blocks);
}

int compress_blocks_f64(const uint8_t bit_width, const double_t* __restrict__ input, const double_t scale, uint8_t* __restrict__ output,
                        const uint32_t blocks) {
    return mm512_compress_blocks_f64_avx512vbmi(bit_width, input, scale, output, blocks);
}

int decompress_block(const uint8_t bit_width, const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output) {
    return mm512_decompress_block_avx512vbmi(bit_width, input, scale, output);
}

int decompress_block_f64(const uint8_t bit_width, const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output) {
    return mm512_decompress_block_f64_avx512vbmi(bit_width, input, scale, output);
}

int decompress_blocks(const uint8_t bit_width, const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output,
                      const uint32_t blocks) {
    return mm512_decompress_blocks_avx512vbmi(bit_width, input, scale, output, blocks);
}

int decompress_blocks_f64(const uint8_t bit_width, const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output,
                          const uint32_t blocks) {
    return mm512_decompress_blocks_f64_avx512vbmi(bit_width, input, scale, output, blocks);
}
#else
int compress_block(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output) {
    return mm256_compress_block_avx2(bit_width, input, scale, output);
}

int compress_block_f64(const uint8_t bit_width, const double_t* __restrict__ input, const double_t scale, uint8_t* __restrict__ output) {
    return mm256_compress_block_f64_avx2(bit_width, input, scale, output);
}

int compress_blocks(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output,
                    const uint32_t blocks) {
    return mm256_compress_blocks_avx2(bit_width, input, scale, output, blocks);
}

int compress_blocks_f64(const uint8_t bit_width, const double_t* __restrict__ input, const double_t scale, uint8_t* __restrict__ output,
                        const uint32_t blocks) {
    return mm256_compress_blocks_f64_avx2(bit_width, input, scale, output, blocks);
}

int decompress_block(const uint8_t bit_width, const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output) {
    return mm256_decompress_block_avx2(bit_width, input, scale, output);
}

int decompress_block_f64(const uint8_t bit_width, const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output) {
    return mm256_decompress_block_f64_avx2(bit_width, input, scale, output);
}

int decompress_blocks(const uint8_t bit_width, const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output,
                      const uint32_t blocks) {
    return mm256_decompress_blocks_avx2(bit_width, input, scale, output, blocks);
}

int decompress_blocks_f64(const uint8_t bit_width, const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output,
                          const uint32_t blocks) {
    return mm256_decompress_blocks_f64_avx2(bit_width, input, scale, output, blocks);
}
#endif
#elif defined(PERNIX_BACKEND_ARM64_NEON)
int compress_block(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output) {
    return neon_compress_block(bit_width, input, scale, output);
}

int compress_block_f64(const uint8_t bit_width, const double_t* __restrict__ input, const double_t scale, uint8_t* __restrict__ output) {
    return neon_compress_block_f64(bit_width, input, scale, output);
}

int compress_blocks(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output,
                    const uint32_t blocks) {
    return neon_compress_blocks(bit_width, input, scale, output, blocks);
}

int compress_blocks_f64(const uint8_t bit_width, const double_t* __restrict__ input, const double_t scale, uint8_t* __restrict__ output,
                        const uint32_t blocks) {
    return neon_compress_blocks_f64(bit_width, input, scale, output, blocks);
}

int decompress_block(const uint8_t bit_width, const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output) {
    return neon_decompress_block(bit_width, input, scale, output);
}

int decompress_block_f64(const uint8_t bit_width, const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output) {
    return neon_decompress_block_f64(bit_width, input, scale, output);
}

int decompress_blocks(const uint8_t bit_width, const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output,
                      const uint32_t blocks) {
    return neon_decompress_blocks(bit_width, input, scale, output, blocks);
}

int decompress_blocks_f64(const uint8_t bit_width, const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output,
                          const uint32_t blocks) {
    return neon_decompress_blocks_f64(bit_width, input, scale, output, blocks);
}
#elif defined(PERNIX_BACKEND_ARM64_SVE)
int compress_block(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output) {
    return sve_compress_block(bit_width, input, scale, output);
}

int compress_block_f64(const uint8_t bit_width, const double_t* __restrict__ input, const double_t scale, uint8_t* __restrict__ output) {
    return sve_compress_block_f64(bit_width, input, scale, output);
}

int compress_blocks(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output,
                    const uint32_t blocks) {
    return sve_compress_blocks(bit_width, input, scale, output, blocks);
}

int compress_blocks_f64(const uint8_t bit_width, const double_t* __restrict__ input, const double_t scale, uint8_t* __restrict__ output,
                        const uint32_t blocks) {
    return sve_compress_blocks_f64(bit_width, input, scale, output, blocks);
}

int decompress_block(const uint8_t bit_width, const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output) {
    return sve_decompress_block(bit_width, input, scale, output);
}

int decompress_block_f64(const uint8_t bit_width, const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output) {
    return sve_decompress_block_f64(bit_width, input, scale, output);
}

int decompress_blocks(const uint8_t bit_width, const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output,
                      const uint32_t blocks) {
    return sve_decompress_blocks(bit_width, input, scale, output, blocks);
}

int decompress_blocks_f64(const uint8_t bit_width, const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output,
                          const uint32_t blocks) {
    return sve_decompress_blocks_f64(bit_width, input, scale, output, blocks);
}
#elif defined(PERNIX_BACKEND_ARM64_SVE2)
int compress_block(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output) {
    return sve2_compress_block(bit_width, input, scale, output);
}

int compress_block_f64(const uint8_t bit_width, const double_t* __restrict__ input, const double_t scale, uint8_t* __restrict__ output) {
    return sve2_compress_block_f64(bit_width, input, scale, output);
}

int compress_blocks(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output,
                    const uint32_t blocks) {
    return sve2_compress_blocks(bit_width, input, scale, output, blocks);
}

int compress_blocks_f64(const uint8_t bit_width, const double_t* __restrict__ input, const double_t scale, uint8_t* __restrict__ output,
                        const uint32_t blocks) {
    return sve2_compress_blocks_f64(bit_width, input, scale, output, blocks);
}

int decompress_block(const uint8_t bit_width, const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output) {
    return sve2_decompress_block(bit_width, input, scale, output);
}

int decompress_block_f64(const uint8_t bit_width, const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output) {
    return sve2_decompress_block_f64(bit_width, input, scale, output);
}

int decompress_blocks(const uint8_t bit_width, const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output,
                      const uint32_t blocks) {
    return sve2_decompress_blocks(bit_width, input, scale, output, blocks);
}

int decompress_blocks_f64(const uint8_t bit_width, const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output,
                          const uint32_t blocks) {
    return sve2_decompress_blocks_f64(bit_width, input, scale, output, blocks);
}
#else
int compress_block(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output) {
    return compress_block_fallback(bit_width, input, scale, output);
}

int compress_block_f64(const uint8_t bit_width, const double_t* __restrict__ input, const double_t scale, uint8_t* __restrict__ output) {
    return compress_block_fallback_f64(bit_width, input, scale, output);
}

int compress_blocks(const uint8_t bit_width, const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output,
                    const uint32_t blocks) {
    return compress_blocks_fallback(bit_width, input, scale, output, blocks);
}

int compress_blocks_f64(const uint8_t bit_width, const double_t* __restrict__ input, const double_t scale, uint8_t* __restrict__ output,
                        const uint32_t blocks) {
    return compress_blocks_fallback_f64(bit_width, input, scale, output, blocks);
}

int decompress_block(const uint8_t bit_width, const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output) {
    return decompress_block_fallback(bit_width, input, scale, output);
}

int decompress_block_f64(const uint8_t bit_width, const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output) {
    return decompress_block_fallback_f64(bit_width, input, scale, output);
}

int decompress_blocks(const uint8_t bit_width, const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output,
                      const uint32_t blocks) {
    return decompress_blocks_fallback(bit_width, input, scale, output, blocks);
}

int decompress_blocks_f64(const uint8_t bit_width, const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output,
                          const uint32_t blocks) {
    return decompress_blocks_fallback_f64(bit_width, input, scale, output, blocks);
}
#endif

#ifdef __cplusplus
}
}  // namespace pernix
#endif  // __cplusplus
