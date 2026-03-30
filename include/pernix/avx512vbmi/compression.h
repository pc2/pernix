#ifndef PERNIX_AVX512VBMI_COMPRESSION_H
#define PERNIX_AVX512VBMI_COMPRESSION_H

#include <pernix/avx2/compression.h>
#include <pernix/avx512vbmi/packing.h>

#include <cstring>

namespace pernix {

namespace internal {
/**
 * @brief Quantize sixteen float values to 32-bit integers.
 */
[[gnu::always_inline]] inline __m512i mm512_quantize_ps_epi32(const __m512& input, const __m512& scale) {
    const __m512 scaled = _mm512_mul_ps(input, scale);
    return _mm512_cvtps_epi32(scaled);
}

[[gnu::always_inline]] inline __m256i mm512_quantize_pd_epi32(const __m512d& input, const __m512d& scale) {
    const __m512d scaled = _mm512_mul_pd(input, scale);
    return _mm512_cvtpd_epi32(scaled);
}

}  // namespace internal

/**
 * @brief Compress a single 512-bit block using AVX-512 and AVX-512-VBMI instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 *
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX-512 and AVX-512-VBMI support.
 */
template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm512_compress_block_avx512vbmi(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr uint32_t iterations_32 = elements_per_block / 32;
    constexpr uint32_t iterations_16 = (elements_per_block % 32) / 16;
    constexpr uint32_t iterations_8  = (elements_per_block % 16) / 8;
    constexpr uint8_t remaining      = elements_per_block - iterations_32 * 32 - iterations_16 * 16 - iterations_8 * 8;

    const __m512 scale_v    = _mm512_set1_ps(scale);
    const __m256 scale_v256 = _mm256_set1_ps(scale);
    if constexpr (iterations_32 > 0) {
#pragma GCC unroll 16
        for (uint32_t iter = 0; iter < iterations_32; iter++) {
            const __m512 source1     = _mm512_loadu_ps(input);
            const __m512 source2     = _mm512_loadu_ps(input + 16);
            const __m512i quantized1 = internal::mm512_quantize_ps_epi32(source1, scale_v);
            const __m512i quantized2 = internal::mm512_quantize_ps_epi32(source2, scale_v);
            const __m512i packed     = internal::mm1024_pack_epi32_avx512vbmi<BIT_WIDTH>(quantized1, quantized2);

            _mm512_mask_storeu_epi32(output, (1u << (BIT_WIDTH)) - 1u, packed);

            input += 32;
            output += 4u * BIT_WIDTH;
        }
    }

    if constexpr (iterations_16 > 0) {
        const __m512 source     = _mm512_loadu_ps(input);
        const __m512i quantized = internal::mm512_quantize_ps_epi32(source, scale_v);
        const __m512i packed    = internal::mm512_pack_epi32_avx512vbmi<BIT_WIDTH>(quantized);

        _mm512_mask_storeu_epi16(output, (1u << (BIT_WIDTH)) - 1u, packed);

        input += 16;
        output += 2 * BIT_WIDTH;
    }

    if constexpr (iterations_8 > 0) {
        const __m256 source     = _mm256_loadu_ps(input);
        const __m256i quantized = internal::mm256_quantize_ps_epi32(source, scale_v256);
        const __m256i packed    = internal::mm256_pack_epi32_avx512vbmi<BIT_WIDTH>(quantized);

        _mm256_mask_storeu_epi8(output, (1u << BIT_WIDTH) - 1u, packed);

        input += 8;
        output += BIT_WIDTH;
    }

    if constexpr (remaining > 0) {
        const __m256 source     = _mm256_maskz_loadu_ps((1u << remaining) - 1u, input);
        const __m256i quantized = internal::mm256_quantize_ps_epi32(source, scale_v256);
        const __m256i packed    = internal::mm256_pack_epi32_avx512vbmi<BIT_WIDTH>(quantized);

        constexpr uint32_t tail_bits  = remaining * BIT_WIDTH;
        constexpr uint32_t tail_bytes = (tail_bits + 7u) / 8u;
        const __mmask32 tail_mask     = (1u << tail_bytes) - 1u;

        _mm256_mask_storeu_epi8(output, tail_mask, packed);
    }

    return 0;
}

/**
 * @brief Compress a single block of double values using AVX-512 and AVX-512-VBMI instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @param input pointer to the start of the input double values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @return int status code.
 *
 * @note This overload is declared for parity with the float path.
 */
template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm512_compress_block_avx512vbmi(const double_t* __restrict__ input, const double_t scale, uint8_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    // Reuse the scalar fallback packer for widths not representable by the 8/16-lane VBMI register paths.
    constexpr uint32_t iterations_16 = elements_per_block / 16;
    constexpr uint32_t iterations_8  = (elements_per_block % 16) / 8;
    constexpr uint8_t remaining      = elements_per_block - iterations_16 * 16 - iterations_8 * 8;

    const __m512d scale_v = _mm512_set1_pd(scale);

#pragma GCC unroll 2
    for (uint32_t iter = 0; iter < iterations_16; iter++) {
        const __m512d source1    = _mm512_loadu_pd(input);
        const __m512d source2    = _mm512_loadu_pd(input + 8);
        const __m256i quantized1 = internal::mm512_quantize_pd_epi32(source1, scale_v);
        const __m256i quantized2 = internal::mm512_quantize_pd_epi32(source2, scale_v);
        __m512i combined         = _mm512_castsi256_si512(quantized1);
        combined                 = _mm512_inserti64x4(combined, quantized2, 1);
        const __m512i packed     = internal::mm512_pack_epi32_avx512vbmi<BIT_WIDTH>(combined);

        _mm512_mask_storeu_epi16(output, (1u << (BIT_WIDTH)) - 1u, packed);

        input += 16;
        output += 2 * BIT_WIDTH;
    }

    if constexpr (iterations_8 > 0) {
        const __m512d source    = _mm512_loadu_pd(input);
        const __m256i quantized = internal::mm512_quantize_pd_epi32(source, scale_v);
        const __m256i packed    = internal::mm256_pack_epi32_avx512vbmi<BIT_WIDTH>(quantized);

        _mm256_mask_storeu_epi8(output, (1u << BIT_WIDTH) - 1u, packed);

        input += 8;
        output += BIT_WIDTH;
    }

    if constexpr (remaining > 0) {
        const __m512d source    = _mm512_maskz_loadu_pd((1u << remaining) - 1u, input);
        const __m256i quantized = internal::mm512_quantize_pd_epi32(source, scale_v);
        const __m256i packed    = internal::mm256_pack_epi32_avx512vbmi<BIT_WIDTH>(quantized);

        constexpr uint32_t tail_bits  = remaining * BIT_WIDTH;
        constexpr uint32_t tail_bytes = (tail_bits + 7u) / 8u;
        const __mmask32 tail_mask     = (1u << tail_bytes) - 1u;

        _mm256_mask_storeu_epi8(output, tail_mask, packed);
    }

    return 0;
}

/**
 * @brief Compress multiple 512-bit blocks using AVX-512 and AVX-512-VBMI instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 *
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @param blocks number of 512-bit blocks to compress.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX-512 and AVX-512-VBMI support.
 */
template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm512_compress_blocks_avx512vbmi(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output,
                                     const uint32_t blocks) {
    const float_t* block_input = input;
    uint8_t* block_output      = output;

    for (uint32_t block = 0; block < blocks; block++) {
        mm512_compress_block_avx512vbmi<BIT_WIDTH, BLOCK_SIZE>(block_input, scale, block_output);
        block_input += (BLOCK_SIZE * 8) / BIT_WIDTH;
        block_output += BLOCK_SIZE;
    }

    return 0;
}

/**
 * @brief Compress multiple blocks of double values using AVX-512 and AVX-512-VBMI instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @param input pointer to the start of the input double values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @param blocks number of blocks to compress.
 * @return int status code.
 */
template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm512_compress_blocks_avx512vbmi(const double_t* __restrict__ input, const double_t scale, uint8_t* __restrict__ output,
                                     const uint32_t blocks) {
    const double_t* block_input = input;
    uint8_t* block_output       = output;

    for (uint32_t block = 0; block < blocks; block++) {
        mm512_compress_block_avx512vbmi<BIT_WIDTH, BLOCK_SIZE>(block_input, scale, block_output);
        block_input += (BLOCK_SIZE * 8) / BIT_WIDTH;
        block_output += BLOCK_SIZE;
    }

    return 0;
}

}  // namespace pernix

#ifdef __cplusplus
namespace pernix {
extern "C" {
#endif

/**
 * @brief Compress a single 512-bit block using AVX-512 and AVX-512-VBMI instructions.
 *
 * @param bit_width bit width per value in the packed representation (1 to 16).
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX-512 and AVX-512-VBMI support.
 */
int mm512_compress_block_avx512vbmi(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output);

/**
 * @brief Compress a single 512-bit block using AVX-512 and AVX-512-VBMI instructions.
 *
 * @param bit_width bit width per value in the packed representation (1 to 16).
 * @param input pointer to the start of the input double values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX-512 and AVX-512-VBMI support.
 */
int mm512_compress_block_f64_avx512vbmi(uint8_t bit_width, const double_t* __restrict__ input, double_t scale,
                                        uint8_t* __restrict__ output);

/**
 * @brief Compress multiple 512-bit blocks using AVX-512 and AVX-512-VBMI instructions.
 *
 * @param bit_width bit width per value in the packed representation (1 to 16).
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @param blocks number of 512-bit blocks to compress.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX-512 and AVX-512-VBMI support.
 */
int mm512_compress_blocks_avx512vbmi(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output,
                                     uint32_t blocks);

/**
 * @brief Compress multiple 512-bit blocks using AVX-512 and AVX-512-VBMI instructions.
 *
 * @param bit_width bit width per value in the packed representation (1 to 16).
 * @param input pointer to the start of the input double values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @param blocks number of 512-bit blocks to compress.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX-512 and AVX-512-VBMI support.
 */
int mm512_compress_blocks_f64_avx512vbmi(uint8_t bit_width, const double_t* __restrict__ input, double_t scale,
                                         uint8_t* __restrict__ output, uint32_t blocks);

#ifdef __cplusplus
}
}  // namespace pernix
#endif

#endif  // PERNIX_AVX512VBMI_COMPRESSION_H
