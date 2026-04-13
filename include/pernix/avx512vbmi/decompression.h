#ifndef PERNIX_AVX512VBMI_DECOMPRESSION_H
#define PERNIX_AVX512VBMI_DECOMPRESSION_H

#include <immintrin.h>
#include <pernix/avx2/decompression.h>
#include <pernix/avx512vbmi/unpacking.h>

#include <cmath>
#include <cstdint>

namespace pernix {

namespace internal {
template <uint32_t LANES>
    requires(LANES <= 8)
consteval __mmask8 lane_mask_8() {
    if constexpr (LANES == 8) {
        return static_cast<__mmask8>(0xFF);
    } else {
        return static_cast<__mmask8>((1u << LANES) - 1u);
    }
}

template <uint8_t BIT_WIDTH, uint32_t REMAINING>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
static constexpr __mmask32 tail_store_mask() {
    constexpr uint32_t tail_bits  = REMAINING * BIT_WIDTH;
    constexpr uint32_t tail_bytes = (tail_bits + 7u) / 8u;
    return (1u << tail_bytes) - 1u;
}

/**
 * @brief Dequantize up to four integer values under a mask.
 */
__always_inline __m128 mm_maskz_dequantize_epi32(const __mmask8& mask, const __m128i& input, const __m128& scale) {
    const __m128 converted = _mm_maskz_cvtepi32_ps(mask, input);
    return _mm_maskz_mul_ps(mask, converted, scale);
}

/**
 * @brief Dequantize up to eight integer values under a mask.
 */
__always_inline __m256 mm256_maskz_dequantize_epi32(const __mmask8& mask, const __m256i& input, const __m256& scale) {
    const __m256 converted = _mm256_maskz_cvtepi32_ps(mask, input);
    return _mm256_maskz_mul_ps(mask, converted, scale);
}

/**
 * @brief Dequantize sixteen integer values to floats.
 */
__always_inline __m512 mm512_dequantize_epi32(const __m512i& input, const __m512& scale) {
    const __m512 converted = _mm512_cvtepi32_ps(input);
    return _mm512_mul_ps(converted, scale);
}

/**
 * @brief Dequantize up to sixteen integer values under a mask.
 */
__always_inline __m512 mm512_maskz_dequantize_epi32(const __mmask8& mask, const __m512i& input, const __m512& scale) {
    const __m512 converted = _mm512_maskz_cvtepi32_ps(mask, input);
    return _mm512_maskz_mul_ps(mask, converted, scale);
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8) && (BLOCK_SIZE % 32 == 0)
[[gnu::always_inline]] inline int mm512_decompress_block_avx512vbmi_1to8(const uint8_t* __restrict__ input, const float_t scale,
                                                                         float_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    const uint32_t iterations_64      = elements_per_block / 64;
    const uint32_t iterations_32      = (elements_per_block % 64) / 32;
    const uint32_t iterations_16      = (elements_per_block % 32) / 16;
    const uint32_t remaining_elements = elements_per_block - iterations_64 * 64 - iterations_32 * 32 - iterations_16 * 16;

    const __m512 scale_v = _mm512_set1_ps(scale);

    if constexpr (iterations_64 > 0) {
        for (uint32_t i = 0; i < iterations_64; ++i) {
            const __m512i source   = _mm512_maskz_loadu_epi64((1ull << BIT_WIDTH) - 1ull, input);
            const __m512i unpacked = m512::mm512_unpack_epi8_avx512vbmi_1to8<BIT_WIDTH, SIGN_VALUES>(source);

            const __m512i converted1 = _mm512_cvtepi8_epi32(_mm512_castsi512_si128(unpacked));
            const __m512i converted2 = _mm512_cvtepi8_epi32(_mm512_extracti64x2_epi64(unpacked, 1));
            const __m512i converted3 = _mm512_cvtepi8_epi32(_mm512_extracti64x2_epi64(unpacked, 2));
            const __m512i converted4 = _mm512_cvtepi8_epi32(_mm512_extracti64x2_epi64(unpacked, 3));

            const __m512 dequantized1 = mm512_dequantize_epi32(converted1, scale_v);
            const __m512 dequantized2 = mm512_dequantize_epi32(converted2, scale_v);
            const __m512 dequantized3 = mm512_dequantize_epi32(converted3, scale_v);
            const __m512 dequantized4 = mm512_dequantize_epi32(converted4, scale_v);

            _mm512_storeu_ps(output, dequantized1);
            _mm512_storeu_ps(output + 16, dequantized2);
            _mm512_storeu_ps(output + 32, dequantized3);
            _mm512_storeu_ps(output + 48, dequantized4);

            output += 64;
            input += 8 * BIT_WIDTH;
        }
    }

    if constexpr (iterations_32 > 0) {
        const __m256i source   = _mm256_maskz_loadu_epi32((1ull << BIT_WIDTH) - 1ull, input);
        const __m256i unpacked = m256::mm256_unpack_epi8_avx512vbmi_1to8<BIT_WIDTH, SIGN_VALUES>(source);

        const __m512i converted1 = _mm512_cvtepi8_epi64(_mm256_castsi256_si128(unpacked));
        const __m512i converted2 = _mm512_cvtepi8_epi64(_mm256_extracti128_si256(unpacked, 1));

        const __m512 dequantized1 = mm512_dequantize_epi32(converted1, scale_v);
        const __m512 dequantized2 = mm512_dequantize_epi32(converted2, scale_v);

        _mm512_storeu_ps(output, dequantized1);
        _mm512_storeu_ps(output + 16, dequantized2);

        output += 32;
        input += 4 * BIT_WIDTH;
    }

    if constexpr (iterations_16 > 0) {
        const __m128i source   = _mm_maskz_loadu_epi16((1u << BIT_WIDTH) - 1u, input);
        const __m128i unpacked = m128::mm_unpack_epi8_avx512vbmi_1to8<BIT_WIDTH, SIGN_VALUES>(source);

        const __m512i converted = _mm512_cvtepi8_epi64(unpacked);

        const __m512 dequantized = mm512_dequantize_epi32(converted, scale_v);

        _mm512_storeu_ps(output, dequantized);

        output += 16;
        input += 2 * BIT_WIDTH;
    }

    if constexpr (remaining_elements > 0) {
        const __m128i source   = _mm_maskz_loadu_epi16(tail_store_mask<BIT_WIDTH, remaining_elements>(), input);
        const __m128i unpacked = m128::mm_unpack_epi8_avx512vbmi_1to8<BIT_WIDTH, SIGN_VALUES>(source);

        const __m512i converted = _mm512_cvtepi8_epi64(unpacked);

        const __m512 dequantized = mm512_dequantize_epi32(converted, scale_v);

        _mm512_mask_storeu_ps(output, (1u << remaining_elements) - 1u, dequantized);
    }

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8) && (BLOCK_SIZE % 32 == 0)
[[gnu::always_inline]] inline int mm512_decompress_block_avx512vbmi_1to8(const uint8_t* __restrict__ input, const double_t scale,
                                                                         double_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    const uint32_t iterations_64      = elements_per_block / 64;
    const uint32_t iterations_32      = (elements_per_block % 64) / 32;
    const uint32_t iterations_16      = (elements_per_block % 32) / 16;
    const uint32_t remaining_elements = elements_per_block - iterations_64 * 64 - iterations_32 * 32 - iterations_16 * 16;

    const __m512d scale_v = _mm512_set1_pd(scale);

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16) && (BLOCK_SIZE % 32 == 0)
[[gnu::always_inline]] inline int mm512_decompress_block_avx512vbmi_9to16(const uint8_t* __restrict__ input, const float_t scale,
                                                                          float_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr uint32_t iterations_32      = elements_per_block / 32;
    constexpr uint32_t iterations_16      = (elements_per_block % 32) / 16;
    constexpr uint32_t iterations_8       = (elements_per_block % 16) / 8;
    constexpr uint32_t remaining_elements = elements_per_block - iterations_32 * 32 - iterations_16 * 16 - iterations_8 * 8;

    const __m512 scale_v    = _mm512_set1_ps(scale);
    const __m256 scale_v256 = _mm256_set1_ps(scale);

    if constexpr (iterations_32 > 0) {
        const __m512i source   = _mm512_maskz_loadu_epi32((1ull << BIT_WIDTH) - 1ull, input);
        const __m512i unpacked = m512::mm512_unpack_epi16_avx512vbmi_9to16<BIT_WIDTH, SIGN_VALUES>(source);

        const __m512i converted1 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(unpacked));
        const __m512i converted2 = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(unpacked, 1));

        const __m512 dequantized1 = mm512_dequantize_epi32(converted1, scale_v);
        const __m512 dequantized2 = mm512_dequantize_epi32(converted2, scale_v);

        _mm512_storeu_ps(output, dequantized1);
        _mm512_storeu_ps(output + 16, dequantized2);

        output += 32;
        input += 4 * BIT_WIDTH;
    }

    if constexpr (iterations_16 > 0) {
        const __m256i source   = _mm256_maskz_loadu_epi16((1u << BIT_WIDTH) - 1u, input);
        const __m256i unpacked = m256::mm256_unpack_epi16_avx512vbmi_9to16<BIT_WIDTH, SIGN_VALUES>(source);

        const __m512i converted  = _mm512_cvtepi16_epi32(unpacked);
        const __m512 dequantized = mm512_dequantize_epi32(converted, scale_v);

        _mm512_storeu_ps(output, dequantized);

        output += 16;
        input += 2 * BIT_WIDTH;
    }

    if constexpr (iterations_8 > 0) {
        const __m128i source   = _mm_maskz_loadu_epi8((1u << BIT_WIDTH) - 1u, input);
        const __m128i unpacked = m128::mm_unpack_epi16_avx512vbmi_9to16<BIT_WIDTH, SIGN_VALUES>(source);

        const __m256i converted  = _mm256_cvtepi16_epi32(unpacked);
        const __m256 dequantized = mm256_dequantize_epi32(converted, scale_v256);

        _mm256_storeu_ps(output, dequantized);

        output += 8;
        input += BIT_WIDTH;
    }

    if (remaining_elements > 0) {
        const __m128i source   = _mm_maskz_loadu_epi8(tail_store_mask<BIT_WIDTH, remaining_elements>(), input);
        const __m128i unpacked = m128::mm_unpack_epi16_avx512vbmi_9to16<BIT_WIDTH, SIGN_VALUES>(source);

        const __m256i converted  = _mm256_cvtepi16_epi32(unpacked);
        const __m256 dequantized = mm256_dequantize_epi32(converted, scale_v256);

        _mm256_mask_storeu_ps(output, (1u << remaining_elements) - 1u, dequantized);
    }

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16) && (BLOCK_SIZE % 32 == 0)
[[gnu::always_inline]] inline int mm512_decompress_block_avx512vbmi_9to16(const uint8_t* __restrict__ input, const double_t scale,
                                                                          double_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr uint32_t iterations_32      = elements_per_block / 32;
    constexpr uint32_t iterations_16      = (elements_per_block % 32) / 16;
    constexpr uint32_t iterations_8       = (elements_per_block % 16) / 8;
    constexpr uint32_t remaining_elements = elements_per_block - iterations_32 * 32 - iterations_16 * 16 - iterations_8 * 8;

    const __m512d scale_v    = _mm512_set1_pd(scale);
    const __m256d scale_v256 = _mm256_set1_pd(scale);

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
[[gnu::always_inline]] inline int mm512_decompress_block_avx512vbmi_17to24(const uint8_t* __restrict__ input, const float_t scale,
                                                                           float_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr uint32_t iterations_16      = elements_per_block / 16;
    constexpr uint32_t iterations_8       = (elements_per_block % 16) / 8;
    constexpr uint32_t remaining_elements = elements_per_block - iterations_16 * 16 - iterations_8 * 8;

    const __m512 scale_v    = _mm512_set1_ps(scale);
    const __m256 scale_v256 = _mm256_set1_ps(scale);

    if constexpr (iterations_16 > 0) {
        const __m512i source   = _mm512_maskz_loadu_epi16((1ull << BIT_WIDTH) - 1ull, input);
        const __m512i unpacked = m512::mm512_unpack_epi32_avx512vbmi_17to24<BIT_WIDTH, SIGN_VALUES>(source);

        const __m512 dequantized = mm512_dequantize_epi32(unpacked, scale_v);

        _mm512_storeu_ps(output, dequantized);

        output += 16;
        input += 2 * BIT_WIDTH;
    }

    if constexpr (iterations_8 > 0) {
        const __m256i source   = _mm256_maskz_loadu_epi8((1u << BIT_WIDTH) - 1u, input);
        const __m256i unpacked = m256::mm256_unpack_epi32_avx512vbmi_17to24<BIT_WIDTH, SIGN_VALUES>(source);

        const __m256 dequantized = mm256_dequantize_epi32(unpacked, scale_v256);

        _mm256_storeu_ps(output, dequantized);

        output += 8;
        input += BIT_WIDTH;
    }

    if constexpr (remaining_elements > 0) {
        const __m256i source   = _mm256_maskz_loadu_epi8(tail_store_mask<BIT_WIDTH, remaining_elements>(), input);
        const __m256i unpacked = m256::mm256_unpack_epi32_avx512vbmi_17to24<BIT_WIDTH, SIGN_VALUES>(source);

        const __m256 dequantized = mm256_dequantize_epi32(unpacked, scale_v256);

        _mm256_mask_storeu_ps(output, (1u << remaining_elements) - 1u, dequantized);
    }

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
[[gnu::always_inline]] inline int mm512_decompress_block_avx512vbmi_17to24(const uint8_t* __restrict__ input, const double_t scale,
                                                                           double_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr uint32_t iterations_16      = elements_per_block / 16;
    constexpr uint32_t iterations_8       = (elements_per_block % 16) / 8;
    constexpr uint32_t remaining_elements = elements_per_block - iterations_16 * 16 - iterations_8 * 8;

    const __m512d scale_v    = _mm512_set1_pd(scale);
    const __m256d scale_v256 = _mm256_set1_pd(scale);

    return 0;
}
}  // namespace internal

/**
 * @brief Decompress a single 512\-bit block using AVX-512 and AVX-512-VBMI instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 *
 * @param input pointer to the start of the compressed block.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX-512 and AVX-512-VBMI support.
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_decompress_block_avx512vbmi(const uint8_t* __restrict__ input, const float_t scale,
                                                      float_t* __restrict__ output) {
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
        return internal::mm512_decompress_block_avx512vbmi_1to8<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
        return internal::mm512_decompress_block_avx512vbmi_9to16<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH >= 17 && BIT_WIDTH <= 24) {
        return internal::mm512_decompress_block_avx512vbmi_17to24<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    }
    return 0;
}

/**
 * @brief Decompress a single block to double values using AVX-512 and AVX-512-VBMI instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 * @param input pointer to the start of the compressed block.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed double values will be stored.
 * @return int status code.
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_decompress_block_avx512vbmi(const uint8_t* __restrict__ input, const double_t scale,
                                                      double_t* __restrict__ output) {
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
        return internal::mm512_decompress_block_avx512vbmi_1to8<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
        return internal::mm512_decompress_block_avx512vbmi_9to16<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH >= 17 && BIT_WIDTH <= 24) {
        return internal::mm512_decompress_block_avx512vbmi_17to24<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    }
    return 0;
}

/**
 * @brief Decompress multiple 512\-bit blocks using AVX-512 and AVX-512-VBMI instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 *
 * @param input pointer to the start of the compressed data.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @param blocks number of 512-bit blocks to decompress.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX-512 and AVX-512-VBMI support.
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm512_decompress_blocks_avx512vbmi(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output,
                                       const uint32_t blocks) {
    const uint8_t* block_input = input;
    float_t* block_output      = output;

    for (uint32_t block = 0; block < blocks; block++) {
        mm512_decompress_block_avx512vbmi<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(block_input, scale, block_output);
        block_input += BLOCK_SIZE;
        block_output += (BLOCK_SIZE * 8) / BIT_WIDTH;
    }

    return 0;
}

/**
 * @brief Decompress multiple blocks to double values using AVX-512 and AVX-512-VBMI instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 * @param input pointer to the start of the compressed data.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed double values will be stored.
 * @param blocks number of blocks to decompress.
 * @return int status code.
 */
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm512_decompress_blocks_avx512vbmi(const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output,
                                       const uint32_t blocks) {
    const uint8_t* block_input = input;
    double_t* block_output     = output;

    for (uint32_t block = 0; block < blocks; block++) {
        mm512_decompress_block_avx512vbmi<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(block_input, scale, block_output);
        block_input += BLOCK_SIZE;
        block_output += (BLOCK_SIZE * 8) / BIT_WIDTH;
    }
    return 0;
}
}  // namespace pernix

#ifdef __cplusplus
namespace pernix {
extern "C" {
#endif
/**
 * @brief Decompress a single 512-bit block using AVX-512 and AVX-512-VBMI instructions.
 *
 * @param bit_width bit width per value in the packed representation (1 to 24).
 * @param input pointer to the start of the compressed block.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX-512 and AVX-512-VBMI support.
 */
int mm512_decompress_block_avx512vbmi(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output);

/**
 * @brief Decompress a single 512-bit block using AVX-512 and AVX-512-VBMI instructions.
 *
 * @param bit_width bit width per value in the packed representation (1 to 24).
 * @param input pointer to the start of the compressed block.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed double values will be stored.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX-512 and AVX-512-VBMI support.
 */
int mm512_decompress_block_f64_avx512vbmi(uint8_t bit_width, const uint8_t* __restrict__ input, double_t scale,
                                          double_t* __restrict__ output);

/**
 * @brief Decompress multiple 512-bit blocks using AVX-512 and AVX-512-VBMI instructions.
 *
 * @param bit_width bit width per value in the packed representation (1 to 24).
 * @param input pointer to the start of the compressed data.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @param blocks number of 512-bit blocks to decompress.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX-512 and AVX-512-VBMI support.
 */
int mm512_decompress_blocks_avx512vbmi(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output,
                                       uint32_t blocks);

/**
 * @brief Decompress multiple 512-bit blocks using AVX-512 and AVX-512-VBMI instructions.
 *
 * @param bit_width bit width per value in the packed representation (1 to 24).
 * @param input pointer to the start of the compressed data.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed double values will be stored.
 * @param blocks number of 512-bit blocks to decompress.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX-512 and AVX-512-VBMI support.
 */
int mm512_decompress_blocks_f64_avx512vbmi(uint8_t bit_width, const uint8_t* __restrict__ input, double_t scale,
                                           double_t* __restrict__ output, uint32_t blocks);

#ifdef __cplusplus
}
}  // namespace pernix
#endif

#endif  // PERNIX_AVX512VBMI_DECOMPRESSION_H
