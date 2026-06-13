#ifndef PERNIX_HPP
#define PERNIX_HPP
#include <pernix/pernix.h>

namespace pernix {
enum class Backend {
    Auto = PERNIX_BACKEND_AUTO,
    Fallback = PERNIX_BACKEND_FALLBACK,
    X86Avx2 = PERNIX_BACKEND_X86_AVX2,
    X86Bmi2 = PERNIX_BACKEND_X86_BMI2,
    X86Avx512Vbmi = PERNIX_BACKEND_X86_AVX512_VBMI,
    Arm64Neon = PERNIX_BACKEND_ARM64_NEON,
    Arm64Sve = PERNIX_BACKEND_ARM64_SVE
};

__always_inline int compress_block(Backend backend, const uint8_t bit_width, const uint32_t block_size,
                                   const std::span<const float> input, const float scale, std::span<uint8_t> output) {
    return pernix_compress_block_f32(static_cast<pernix_backend>(backend), bit_width, block_size, input.data(), scale, output.data());
}

__always_inline int compress_block(Backend backend, const uint8_t bit_width, const uint32_t block_size,
                                   const std::span<const double> input, const double scale, std::span<uint8_t> output) {
    return pernix_compress_block_f64(static_cast<pernix_backend>(backend), bit_width, block_size, input.data(), scale, output.data());
}

__always_inline int decompress_block(Backend backend, const uint8_t bit_width, const uint32_t block_size,
                                     const std::span<const uint8_t> input, const float scale, std::span<float> output,
                                     const bool sign_values = true) {
    return pernix_decompress_block_f32(static_cast<pernix_backend>(backend), bit_width, block_size, input.data(), scale, output.data(),
                                       sign_values);
}

__always_inline int decompress_block(Backend backend, const uint8_t bit_width, const uint32_t block_size,
                                     const std::span<const uint8_t> input, const double scale, std::span<double> output,
                                     const bool sign_values = true) {
    return pernix_decompress_block_f64(static_cast<pernix_backend>(backend), bit_width, block_size, input.data(), scale, output.data(),
                                       sign_values);
}

__always_inline int compress_blocks(Backend backend, const uint8_t bit_width, const uint32_t block_size,
                                    const std::span<const float> input, const float scale, std::span<uint8_t> output,
                                    const uint32_t blocks) {
    return pernix_compress_blocks_f32(static_cast<pernix_backend>(backend), bit_width, block_size, input.data(), scale, output.data(),
                                      blocks);
}

__always_inline int compress_blocks(Backend backend, const uint8_t bit_width, const uint32_t block_size,
                                    const std::span<const double> input, const double scale, std::span<uint8_t> output,
                                    const uint32_t blocks) {
    return pernix_compress_blocks_f64(static_cast<pernix_backend>(backend), bit_width, block_size, input.data(), scale, output.data(),
                                      blocks);
}

__always_inline int decompress_blocks(Backend backend, const uint8_t bit_width, const uint32_t block_size,
                                      const std::span<const uint8_t> input, const float scale, std::span<float> output,
                                      const uint32_t blocks, const bool sign_values = true) {
    return pernix_decompress_blocks_f32(static_cast<pernix_backend>(backend), bit_width, block_size, input.data(), scale, output.data(),
                                        blocks, sign_values);
}

__always_inline int decompress_blocks(Backend backend, const uint8_t bit_width, const uint32_t block_size,
                                      const std::span<const uint8_t> input, const double scale, std::span<double> output,
                                      const uint32_t blocks, const bool sign_values = true) {
    return pernix_decompress_blocks_f64(static_cast<pernix_backend>(backend), bit_width, block_size, input.data(), scale, output.data(),
                                        blocks, sign_values);
}

// convenience overloads without backend (defaults to Auto)
__always_inline int compress_block(const uint8_t bit_width, const uint32_t block_size, const std::span<const float> input,
                                   const float scale, const std::span<uint8_t> output) {
    return compress_block(Backend::Auto, bit_width, block_size, input, scale, output);
}

__always_inline int compress_block(const uint8_t bit_width, const uint32_t block_size, const std::span<const double> input,
                                   const double scale, const std::span<uint8_t> output) {
    return compress_block(Backend::Auto, bit_width, block_size, input, scale, output);
}

__always_inline int decompress_block(const uint8_t bit_width, const uint32_t block_size, const std::span<const uint8_t> input,
                                     const float scale, const std::span<float> output, const bool sign_values = true) {
    return decompress_block(Backend::Auto, bit_width, block_size, input, scale, output, sign_values);
}

__always_inline int decompress_block(const uint8_t bit_width, const uint32_t block_size, const std::span<const uint8_t> input,
                                     const double scale, const std::span<double> output, const bool sign_values = true) {
    return decompress_block(Backend::Auto, bit_width, block_size, input, scale, output, sign_values);
}

__always_inline int compress_blocks(const uint8_t bit_width, const uint32_t block_size, const std::span<const float> input,
                                    const float scale, const std::span<uint8_t> output, const uint32_t blocks) {
    return compress_blocks(Backend::Auto, bit_width, block_size, input, scale, output, blocks);
}

__always_inline int compress_blocks(const uint8_t bit_width, const uint32_t block_size, const std::span<const double> input,
                                    const double scale, const std::span<uint8_t> output, const uint32_t blocks) {
    return compress_blocks(Backend::Auto, bit_width, block_size, input, scale, output, blocks);
}

__always_inline int decompress_blocks(const uint8_t bit_width, const uint32_t block_size, const std::span<const uint8_t> input,
                                      const float scale, const std::span<float> output, const uint32_t blocks,
                                      const bool sign_values = true) {
    return decompress_blocks(Backend::Auto, bit_width, block_size, input, scale, output, blocks, sign_values);
}

__always_inline int decompress_blocks(const uint8_t bit_width, const uint32_t block_size, const std::span<const uint8_t> input,
                                      const double scale, const std::span<double> output, const uint32_t blocks,
                                      const bool sign_values = true) {
    return decompress_blocks(Backend::Auto, bit_width, block_size, input, scale, output, blocks, sign_values);
}

// convenience overloads without backend and without block_size (defaults to 64)
__always_inline int compress_block(const uint8_t bit_width, const std::span<const float> input, const float scale,
                                   const std::span<uint8_t> output) {
    return compress_block(Backend::Auto, bit_width, 64, input, scale, output);
}

__always_inline int compress_block(const uint8_t bit_width, const std::span<const double> input, const double scale,
                                   const std::span<uint8_t> output) {
    return compress_block(Backend::Auto, bit_width, 64, input, scale, output);
}

__always_inline int decompress_block(const uint8_t bit_width, const std::span<const uint8_t> input, const float scale,
                                     const std::span<float> output, const bool sign_values = true) {
    return decompress_block(Backend::Auto, bit_width, 64, input, scale, output, sign_values);
}

__always_inline int decompress_block(const uint8_t bit_width, const std::span<const uint8_t> input, const double scale,
                                     const std::span<double> output, const bool sign_values = true) {
    return decompress_block(Backend::Auto, bit_width, 64, input, scale, output, sign_values);
}

__always_inline int compress_blocks(const uint8_t bit_width, const std::span<const float> input, const float scale,
                                    const std::span<uint8_t> output, const uint32_t blocks) {
    return compress_blocks(Backend::Auto, bit_width, 64, input, scale, output, blocks);
}

__always_inline int compress_blocks(const uint8_t bit_width, const std::span<const double> input, const double scale,
                                    const std::span<uint8_t> output, const uint32_t blocks) {
    return compress_blocks(Backend::Auto, bit_width, 64, input, scale, output, blocks);
}

__always_inline int decompress_blocks(const uint8_t bit_width, const std::span<const uint8_t> input, const float scale,
                                      const std::span<float> output, const uint32_t blocks, const bool sign_values = true) {
    return decompress_blocks(Backend::Auto, bit_width, 64, input, scale, output, blocks, sign_values);
}

__always_inline int decompress_blocks(const uint8_t bit_width, const std::span<const uint8_t> input, const double scale,
                                      const std::span<double> output, const uint32_t blocks, const bool sign_values = true) {
    return decompress_blocks(Backend::Auto, bit_width, 64, input, scale, output, blocks, sign_values);
}
}

#endif //PERNIX_HPP
