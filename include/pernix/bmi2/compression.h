#ifndef PERNIX_BMI2_COMPRESSION_H
#define PERNIX_BMI2_COMPRESSION_H

#include <immintrin.h>
#include <pernix/avx2/compression.h>
#include <pernix/fallback/compression.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>

namespace pernix {
namespace internal {
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 32)
static constexpr std::tuple<uint16_t, uint64_t, uint16_t, uint16_t> pack_avx2_bmi2_constants() {
    uint32_t mask = BIT_WIDTH == 32 ? std::numeric_limits<uint32_t>::max() : (1ULL << BIT_WIDTH) - 1U;
    uint64_t pext_mask;
    uint16_t shift1 = BIT_WIDTH * 4;
    uint16_t shift2 = 64 - shift1;

    if constexpr (BIT_WIDTH > 0 && BIT_WIDTH <= 8) {
        pext_mask = 0x0101010101010101ULL * mask;
    } else if constexpr (BIT_WIDTH > 8 && BIT_WIDTH <= 16) {
        pext_mask = 0x0001000100010001ULL * mask;
    } else {
        pext_mask = 0x0000000100000001ULL * mask;
    }

    return {
        mask,
        pext_mask,
        shift1,
        shift2,
    };
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 32)
static inline auto mm_pack_epi32_bmi2(const __m128i& input) -> __m128i {
    const auto [mask, pext_mask, shift1, shift2] = pack_avx2_bmi2_constants<BIT_WIDTH>();

    if constexpr (BIT_WIDTH > 0 && BIT_WIDTH <= 16) {
        const __m128i packed = _mm_packs_epi32(input, _mm_setzero_si128());
        const uint64_t value = _pext_u64(_mm_extract_epi64(packed, 0), pext_mask);

        const __m128i result = _mm_set_epi64x(0, value);
        return result;
    } else {
        alignas(16) uint64_t values[2];
        values[0] = _pext_u64(_mm_extract_epi64(input, 0), pext_mask);

        const uint64_t temp_combined = _pext_u64(_mm_extract_epi64(input, 1), pext_mask);
        values[1]                    = temp_combined >> shift2;
        values[0] |= (temp_combined << shift1);

        const __m128i result = _mm_set_epi64x(static_cast<int64_t>(values[1]), static_cast<int64_t>(values[0]));
        return result;
    }
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 16)
static inline auto mm256_pack_epi32_bmi2(const __m256i& input) -> __m256i {
    const auto [mask, pext_mask, shift1, shift2] = pack_avx2_bmi2_constants<BIT_WIDTH>();

    const __m256i packed16 = _mm256_packs_epi32(input, _mm256_setzero_si256());

    if constexpr (BIT_WIDTH > 0 && BIT_WIDTH <= 8) {
        const __m256i permuted = _mm256_permute4x64_epi64(packed16, _MM_SHUFFLE(3, 1, 2, 0));
        const __m256i packed8  = _mm256_packs_epi16(permuted, _mm256_setzero_si256());
        const uint64_t value   = _pext_u64(_mm256_extract_epi64(packed8, 0), pext_mask);

        const __m256i result = _mm256_setr_epi64x(value, 0, 0, 0);
        return result;
    } else {
        alignas(16) int64_t values[2] = {};
        values[0]                     = _pext_u64(_mm256_extract_epi64(packed16, 0), pext_mask);

        const uint64_t temp_combined = _pext_u64(_mm256_extract_epi64(packed16, 2), pext_mask);
        values[1]                    = temp_combined >> shift2;
        if (shift1 < 64) {
            values[0] |= (temp_combined << shift1);
        }

        const __m256i result = _mm256_setr_epi64x(values[0], values[1], 0, 0);
        return result;
    }
}
}  // namespace internal

/**
 * @brief Compress a single 512-bit block using AVX2 and BMI2 instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 16).
 *
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 and BMI2 support.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 16)
int mm256_compress_block_bmi2(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = 512 / BIT_WIDTH;
    constexpr uint32_t iterations_8       = elements_per_block / 8;
    constexpr uint8_t remaining           = elements_per_block - iterations_8 * 8;

    const __m256 scale_v = _mm256_set1_ps(scale);
#pragma GCC unroll 4
    for (uint32_t iter = 0; iter < iterations_8; iter++) {
        const __m256 source     = _mm256_loadu_ps(input);
        const __m256i quantized = internal::mm256_quantize_ps_epi32(source, scale_v);
        const __m256i packed    = internal::mm256_pack_epi32_bmi2<BIT_WIDTH>(quantized);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output), _mm256_castsi256_si128(packed));
        input += 8;
        output += BIT_WIDTH;
    }

    if constexpr (remaining) {
        std::vector<uint32_t> block_values(remaining);
#pragma GCC unroll 8
        for (uint32_t i = 0; i < remaining; i++) {
            block_values[i] = static_cast<uint32_t>(internal::quantize_ps_epi32(input[i], scale));
        }

        internal::pack_epi32_fallback<BIT_WIDTH>(block_values, output);
    }

    return 0;
}

/**
 * @brief Compress multiple 512-bit blocks using AVX2 and BMI2 instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 16).
 *
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @param blocks number of 512-bit blocks to compress.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 and BMI2 support.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 16)
int mm256_compress_blocks_bmi2(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output,
                               const uint32_t blocks) {
    const float_t* block_input = input;
    uint8_t* block_output      = output;

    for (uint32_t block = 0; block < blocks; block++) {
        mm256_compress_block_bmi2<BIT_WIDTH>(block_input, scale, block_output);
        block_input += 512 / BIT_WIDTH;
        block_output += 64;
    }

    return 0;
}

}  // namespace pernix

#ifdef __cplusplus
namespace pernix {
extern "C" {
#endif

/**
 * @brief Compress a single 512-bit block using AVX2 and BMI2 instructions.
 *
 * @param bit_width bit width per value in the packed representation (1 to 16).
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 and BMI2 support.
 */
int mm256_compress_block_bmi2(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output);

/**
 * @brief Compress multiple 512-bit blocks using AVX2 and BMI2 instructions.
 *
 * @param bit_width bit width per value in the packed representation (1 to 16).
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @param blocks number of 512-bit blocks to compress.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 and BMI2 support.
 */
int mm256_compress_blocks_bmi2(uint8_t bit_width, const float_t* __restrict__ input, float_t scale, uint8_t* __restrict__ output,
                               uint32_t blocks);

#ifdef __cplusplus
}
}  // namespace pernix
#endif

#endif  // PERNIX_BMI2_COMPRESSION_H
