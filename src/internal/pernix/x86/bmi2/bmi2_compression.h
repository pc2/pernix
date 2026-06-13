#ifndef PERNIX_BMI2_COMPRESSION_H
#define PERNIX_BMI2_COMPRESSION_H

#include <pernix/x86/avx2/avx2_compression.h>
#include <pernix/fallback/avx2_compression.h>
#include <pernix/simd_compat.h>

#include <cmath>
#include <limits>
#include <tuple>

namespace pernix {
namespace internal {
/**
* @brief Build the masks and shift constants used by the BMI2 packers.
*
* @tparam BIT_WIDTH bit width per packed value.
* @return std::tuple<uint16_t, uint64_t, uint16_t, uint16_t> mask tuple used by the BMI2 helpers.
*/
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

/**
* @brief Pack four 32-bit values with BMI2 extract instructions.
*
* @tparam BIT_WIDTH bit width per packed value.
* @param input SIMD register containing four quantized values.
* @return __m128i packed bitstream in the low bytes of the result.
*/
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
        values[0]                    |= (temp_combined << shift1);

        const __m128i result = _mm_set_epi64x(static_cast<int64_t>(values[1]), static_cast<int64_t>(values[0]));
        return result;
    }
}

/**
* @brief Pack eight 32-bit values with BMI2 extract instructions.
*
* @tparam BIT_WIDTH bit width per packed value.
* @param input SIMD register containing eight quantized values.
* @return __m256i packed bitstream in the low bytes of the result.
*/
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
static inline auto mm256_pack_epi32_bmi2(const __m256i& input) -> __m256i {
    const auto [mask, pext_mask, shift1, shift2] = pack_avx2_bmi2_constants<BIT_WIDTH>();

    if constexpr (BIT_WIDTH > 0 && BIT_WIDTH <= 8) {
        const __m256i packed16 = _mm256_packs_epi32(input, _mm256_setzero_si256());
        const __m256i permuted = _mm256_permute4x64_epi64(packed16, _MM_SHUFFLE(3, 1, 2, 0));
        const __m256i packed8  = _mm256_packs_epi16(permuted, _mm256_setzero_si256());
        const uint64_t value   = _pext_u64(_mm256_extract_epi64(packed8, 0), pext_mask);

        const __m256i result = _mm256_setr_epi64x(static_cast<int64_t>(value), 0, 0, 0);
        return result;
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
        const __m256i packed16        = _mm256_packs_epi32(input, _mm256_setzero_si256());
        alignas(16) int64_t values[2] = {};
        values[0]                     = _pext_u64(_mm256_extract_epi64(packed16, 0), pext_mask);

        const uint64_t temp_combined = _pext_u64(_mm256_extract_epi64(packed16, 2), pext_mask);
        values[1]                    = temp_combined >> shift2;
        if constexpr (BIT_WIDTH != 16) {
            values[0] |= static_cast<int64_t>(temp_combined << shift1);
        }

        const __m256i result = _mm256_setr_epi64x(values[0], values[1], 0, 0);
        return result;
    } else {
        constexpr uint32_t chunk_bits = BIT_WIDTH * 2; // bits extracted per 64-bit lane
        static_assert(chunk_bits < 64);

        constexpr uint64_t chunk_mask = (chunk_bits == 64) ? ~uint64_t{0} : ((uint64_t{1} << chunk_bits) - 1);

        const uint64_t x0 = _pext_u64(_mm256_extract_epi64(input, 0), pext_mask) & chunk_mask;
        const uint64_t x1 = _pext_u64(_mm256_extract_epi64(input, 1), pext_mask) & chunk_mask;
        const uint64_t x2 = _pext_u64(_mm256_extract_epi64(input, 2), pext_mask) & chunk_mask;
        const uint64_t x3 = _pext_u64(_mm256_extract_epi64(input, 3), pext_mask) & chunk_mask;

        uint64_t out0 = 0;
        uint64_t out1 = 0;
        uint64_t out2 = 0;

        auto append_bits = [&](uint64_t value, uint32_t bit_offset) {
            const uint32_t word = bit_offset >> 6; // / 64
            const uint32_t off  = bit_offset & 63; // % 64

            if (word == 0) {
                out0 |= value << off;
                if (off + chunk_bits > 64) {
                    out1 |= value >> (64 - off);
                }
            } else if (word == 1) {
                out1 |= value << off;
                if (off + chunk_bits > 64) {
                    out2 |= value >> (64 - off);
                }
            } else {
                out2 |= value << off;
            }
        };

        append_bits(x0, 0 * chunk_bits);
        append_bits(x1, 1 * chunk_bits);
        append_bits(x2, 2 * chunk_bits);
        append_bits(x3, 3 * chunk_bits);

        return _mm256_setr_epi64x(static_cast<int64_t>(out0), static_cast<int64_t>(out1),
                                  static_cast<int64_t>(out2), 0);
    }
}
} // namespace internal

/**
* @brief Compress a single 512-bit block using AVX2 and BMI2 instructions.
*
* @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
*
* @param input pointer to the start of the input float values.
* @param scale scaling factor used during quantization.
* @param output pointer to the output buffer where compressed bytes will be stored.
* @return int status code (0 for success).
*
* @note This function requires AVX2 and BMI2 support.
*/
template <uint8_t BIT_WIDTH, uint16_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_compress_block_bmi2(const void* __restrict__ input_ptr, const float_t scale,
                              void* __restrict__ output_ptr) {
    const auto* input = static_cast<const float*>(input_ptr);
    auto* output      = static_cast<uint8_t*>(output_ptr);

    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;
    constexpr uint32_t iterations_8       = elements_per_block / 8;
    constexpr uint8_t remaining           = elements_per_block - iterations_8 * 8;

    std::memset(output, 0, BLOCK_SIZE);

    const __m256 scale_v = _mm256_set1_ps(scale);
#pragma GCC unroll 4
    for (uint32_t iter = 0; iter < iterations_8; iter++) {
        const __m256 source        = _mm256_loadu_ps(input);
        const __m256i quantized    = internal::mm256_quantize_ps_epi32(source, scale_v);
        const __m256i packed_input = internal::mm256_clamp_signed_epi32<BIT_WIDTH>(quantized);
        const __m256i packed       = internal::mm256_pack_epi32_bmi2<BIT_WIDTH>(packed_input);
        std::memcpy(output, &packed, BIT_WIDTH);
        input  += 8;
        output += BIT_WIDTH;
    }

    if constexpr (remaining) {
        std::vector<uint32_t> block_values(remaining);
#pragma GCC unroll 8
        for (uint32_t i = 0; i < remaining; i++) {
            block_values[i] =
                static_cast<uint32_t>(internal::clamp_signed_quantized<BIT_WIDTH>(
                    internal::quantize_ps_epi32(input[i], scale)));
        }

        internal::pack_epi32_fallback<BIT_WIDTH>(block_values, output);
    }

    return 0;
}

/**
* @brief Compress a single block of double values using AVX2 and BMI2 instructions.
*
* @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
* @param input pointer to the start of the input double values.
* @param scale scaling factor used during quantization.
* @param output pointer to the output buffer where compressed bytes will be stored.
* @return int status code (0 for success).
*
* @note This function requires AVX2 and BMI2 support.
*/
template <uint8_t BIT_WIDTH, uint16_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_compress_block_bmi2(const void* __restrict__ input_ptr, const double_t scale,
                              void* __restrict__ output_ptr) {
    const auto* input = static_cast<const double*>(input_ptr);
    auto* output      = static_cast<uint8_t*>(output_ptr);

    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;
    constexpr uint32_t iterations_8       = elements_per_block / 8;
    constexpr uint8_t remaining           = elements_per_block - iterations_8 * 8;

    std::memset(output, 0, BLOCK_SIZE);

    const __m256d scale_v = _mm256_set1_pd(scale);
#pragma GCC unroll 4
    for (uint32_t iter = 0; iter < iterations_8; iter++) {
        const __m256d source1    = _mm256_loadu_pd(input);
        const __m256d source2    = _mm256_loadu_pd(input + 4);
        const __m128i quantized1 = internal::mm256_quantize_pd_epi32(source1, scale_v);
        const __m128i quantized2 = internal::mm256_quantize_pd_epi32(source2, scale_v);
        __m256i combined         = _mm256_castsi128_si256(quantized1);
        combined                 = _mm256_inserti128_si256(combined, quantized2, 1);
        const __m256i packed     = internal::mm256_pack_epi32_bmi2<BIT_WIDTH>(
            internal::mm256_clamp_signed_epi32<BIT_WIDTH>(combined));
        std::memcpy(output, &packed, BIT_WIDTH);
        input  += 8;
        output += BIT_WIDTH;
    }

    if constexpr (remaining) {
        std::vector<uint32_t> block_values(remaining);
#pragma GCC unroll 8
        for (uint32_t i = 0; i < remaining; i++) {
            block_values[i] =
                static_cast<uint32_t>(internal::clamp_signed_quantized<BIT_WIDTH>(
                    internal::quantize_pd_epi64(input[i], scale)));
        }

        internal::pack_epi32_fallback<BIT_WIDTH>(block_values, output);
    }
    return 0;
}

/**
* @brief Compress multiple 512-bit blocks using AVX2 and BMI2 instructions.
*
* @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
*
* @param input pointer to the start of the input float values.
* @param scale scaling factor used during quantization.
* @param output pointer to the output buffer where compressed bytes will be stored.
* @param blocks number of 512-bit blocks to compress.
* @return int status code (0 for success).
*
* @note This function requires AVX2 and BMI2 support.
*/
template <uint8_t BIT_WIDTH, uint16_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_compress_blocks_bmi2(const void* __restrict__ input_ptr, const float_t scale,
                               void* __restrict__ output_ptr, const uint32_t blocks) {
    const auto* input = static_cast<const float*>(input_ptr);
    auto* output      = static_cast<uint8_t*>(output_ptr);

    const float_t* block_input = input;
    uint8_t* block_output      = output;

    for (uint32_t block = 0; block < blocks; block++) {
        mm256_compress_block_bmi2<BIT_WIDTH, BLOCK_SIZE>(block_input, scale, block_output);
        block_input  += (BLOCK_SIZE * 8) / BIT_WIDTH;
        block_output += BLOCK_SIZE;
    }

    return 0;
}

/**
* @brief Compress multiple blocks of double values using AVX2 and BMI2 instructions.
*
* @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
* @param input pointer to the start of the input double values.
* @param scale scaling factor used during quantization.
* @param output pointer to the output buffer where compressed bytes will be stored.
* @param blocks number of blocks to compress.
* @return int status code (0 for success).
*
* @note This function requires AVX2 and BMI2 support.
*/
template <uint8_t BIT_WIDTH, uint16_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_compress_blocks_bmi2(const void* __restrict__ input_ptr, const double_t scale,
                               void* __restrict__ output_ptr, const uint32_t blocks) {
    const auto* input = static_cast<const double*>(input_ptr);
    auto* output      = static_cast<uint8_t*>(output_ptr);

    const double_t* block_input = input;
    uint8_t* block_output       = output;

    for (uint32_t block = 0; block < blocks; block++) {
        mm256_compress_block_bmi2<BIT_WIDTH, BLOCK_SIZE>(block_input, scale, block_output);
        block_input  += (BLOCK_SIZE * 8) / BIT_WIDTH;
        block_output += BLOCK_SIZE;
    }

    return 0;
}
} // namespace pernix

#endif  // PERNIX_BMI2_COMPRESSION_H
