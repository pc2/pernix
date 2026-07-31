#ifndef PERNIX_BMI2_COMPRESSION_H
#define PERNIX_BMI2_COMPRESSION_H

#include <pernix/fallback/scalar_compression.h>
#include <pernix/simd_compat.h>
#include <pernix/x86/avx2/avx2_compression.h>

#include <cmath>
#include <limits>
#include <tuple>

namespace pernix {
namespace internal {

template <int LANE>
    requires(LANE >= 0 && LANE < 4)
[[nodiscard]] __always_inline u64 mm256_extract_u64(const __m256i value) noexcept {
    return static_cast<u64>(_mm256_extract_epi64(value, LANE));
}

/**
 * @brief Build the masks and shift constants used by the BMI2 packers.
 *
 * @tparam BIT_WIDTH bit width per packed value.
 * @return std::tuple<u16, u64, u16, u16> mask tuple used by the BMI2 helpers.
 */
template <u8 BIT_WIDTH>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 32)
static constexpr std::tuple<u16, u64, u16, u16> pack_avx2_bmi2_constants() {
    const u32 mask = BIT_WIDTH == 32 ? std::numeric_limits<u32>::max() : (1ULL << BIT_WIDTH) - 1U;
    u64 pext_mask;
    const u16 shift1 = BIT_WIDTH * 4;
    const u16 shift2 = 64 - shift1;

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
template <u8 BIT_WIDTH>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 32)
static inline auto mm_pack_epi32_bmi2(__m128i input) -> __m128i {
    const auto [mask, pext_mask, shift1, shift2] = pack_avx2_bmi2_constants<BIT_WIDTH>();

    if constexpr (BIT_WIDTH > 0 && BIT_WIDTH <= 16) {
        const __m128i packed = _mm_packs_epi32(input, _mm_setzero_si128());
        const u64 value      = _pext_u64(_mm_extract_epi64(packed, 0), pext_mask);

        const __m128i result = _mm_set_epi64x(0, value);
        return result;
    } else {
        alignas(16) u64 values[2];
        values[0] = _pext_u64(_mm_extract_epi64(input, 0), pext_mask);

        const u64 temp_combined = _pext_u64(_mm_extract_epi64(input, 1), pext_mask);
        values[1]               = temp_combined >> shift2;
        values[0] |= (temp_combined << shift1);

        const __m128i result = _mm_set_epi64x(static_cast<i64>(values[1]), static_cast<i64>(values[0]));
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
template <u8 BIT_WIDTH>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
static inline auto mm256_pack_epi32_bmi2(__m256i input) -> __m256i {
    const auto [mask, pext_mask, shift1, shift2] = pack_avx2_bmi2_constants<BIT_WIDTH>();

    if constexpr (BIT_WIDTH > 0 && BIT_WIDTH <= 8) {
        const __m256i packed16 = _mm256_packs_epi32(input, _mm256_setzero_si256());
        const __m256i permuted = _mm256_permute4x64_epi64(packed16, _MM_SHUFFLE(3, 1, 2, 0));
        const __m256i packed8  = _mm256_packs_epi16(permuted, _mm256_setzero_si256());
        const u64 value        = _pext_u64(mm256_extract_u64<0>(packed8), pext_mask);

        const __m256i result = _mm256_setr_epi64x(static_cast<i64>(value), 0, 0, 0);
        return result;
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
        const __m256i packed16    = _mm256_packs_epi32(input, _mm256_setzero_si256());
        alignas(16) u64 values[2] = {};
        values[0]                 = _pext_u64(mm256_extract_u64<0>(packed16), pext_mask);

        const u64 temp_combined = _pext_u64(mm256_extract_u64<2>(packed16), pext_mask);
        values[1]               = temp_combined >> shift2;
        if constexpr (BIT_WIDTH != 16) {
            values[0] |= static_cast<u64>(temp_combined << shift1);
        }

        const __m256i result = _mm256_setr_epi64x(values[0], values[1], 0, 0);
        return result;
    } else {
        constexpr u32 chunk_bits = BIT_WIDTH * 2;  // bits extracted per 64-bit lane
        static_assert(chunk_bits < 64);

        constexpr u64 chunk_mask = (chunk_bits == 64) ? ~u64{0} : ((u64{1} << chunk_bits) - 1);

        const u64 x0 = _pext_u64(mm256_extract_u64<0>(input), pext_mask) & chunk_mask;
        const u64 x1 = _pext_u64(mm256_extract_u64<1>(input), pext_mask) & chunk_mask;
        const u64 x2 = _pext_u64(mm256_extract_u64<2>(input), pext_mask) & chunk_mask;
        const u64 x3 = _pext_u64(mm256_extract_u64<3>(input), pext_mask) & chunk_mask;

        u64 out0 = 0;
        u64 out1 = 0;
        u64 out2 = 0;

        auto append_bits = [&](u64 value, u32 bit_offset) {
            const u32 word = bit_offset >> 6;  // / 64
            const u32 off  = bit_offset & 63;  // % 64

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

        return _mm256_setr_epi64x(static_cast<i64>(out0), static_cast<i64>(out1), static_cast<i64>(out2), 0);
    }
}
}  // namespace internal

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
template <u8 BIT_WIDTH, u16 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_compress_block_bmi2(const void* __restrict__ input_ptr, const f32 scale, void* __restrict__ output_ptr) {
    const auto* input = static_cast<const float*>(input_ptr);
    auto* output      = static_cast<u8*>(output_ptr);

    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;
    constexpr u32 iterations_8       = elements_per_block / 8;
    constexpr u8 remaining           = elements_per_block - iterations_8 * 8;

    std::memset(output, 0, BLOCK_SIZE);

    const __m256 scale_v = _mm256_set1_ps(scale);
#pragma GCC unroll 4
    for (u32 iter = 0; iter < iterations_8; iter++) {
        const __m256 source        = _mm256_loadu_ps(input);
        const __m256i quantized    = internal::mm256_quantize_ps_epi32(source, scale_v);
        const __m256i packed_input = internal::mm256_clamp_signed_epi32<BIT_WIDTH>(quantized);
        const __m256i packed       = internal::mm256_pack_epi32_bmi2<BIT_WIDTH>(packed_input);
        std::memcpy(output, &packed, BIT_WIDTH);
        input += 8;
        output += BIT_WIDTH;
    }

    if constexpr (remaining) {
        constexpr u32 tail_bytes   = (BIT_WIDTH * remaining + 7) / 8;
        const __m256 source        = _mm256_maskload_ps(input, internal::mm256_tail_mask_epi32<remaining>());
        const __m256i quantized    = internal::mm256_quantize_ps_epi32(source, scale_v);
        const __m256i packed_input = internal::mm256_clamp_signed_epi32<BIT_WIDTH>(quantized);
        const __m256i packed       = internal::mm256_pack_epi32_bmi2<BIT_WIDTH>(packed_input);
        std::memcpy(output, &packed, tail_bytes);
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
template <u8 BIT_WIDTH, u16 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_compress_block_bmi2(const void* __restrict__ input_ptr, const f64 scale, void* __restrict__ output_ptr) {
    const auto* input = static_cast<const double*>(input_ptr);
    auto* output      = static_cast<u8*>(output_ptr);

    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;
    constexpr u32 iterations_8       = elements_per_block / 8;
    constexpr u8 remaining           = elements_per_block - iterations_8 * 8;

    std::memset(output, 0, BLOCK_SIZE);

    const __m256d scale_v = _mm256_set1_pd(scale);
#pragma GCC unroll 4
    for (u32 iter = 0; iter < iterations_8; iter++) {
        const __m256d source1    = _mm256_loadu_pd(input);
        const __m256d source2    = _mm256_loadu_pd(input + 4);
        const __m128i quantized1 = internal::mm256_quantize_pd_epi32(source1, scale_v);
        const __m128i quantized2 = internal::mm256_quantize_pd_epi32(source2, scale_v);
        __m256i combined         = _mm256_castsi128_si256(quantized1);
        combined                 = _mm256_inserti128_si256(combined, quantized2, 1);
        const __m256i packed     = internal::mm256_pack_epi32_bmi2<BIT_WIDTH>(internal::mm256_clamp_signed_epi32<BIT_WIDTH>(combined));
        std::memcpy(output, &packed, BIT_WIDTH);
        input += 8;
        output += BIT_WIDTH;
    }

    if constexpr (remaining) {
        constexpr usize tail_bytes   = ((BIT_WIDTH * remaining) + 7) / 8;
        constexpr usize first_lanes  = remaining < 4 ? remaining : 4;
        constexpr usize second_lanes = remaining > 4 ? remaining - 4 : 0;

        const __m256d source1 = _mm256_maskload_pd(input, internal::mm256_tail_mask_epi64<first_lanes>());
        const __m256d source2 = [&] {
            if constexpr (second_lanes > 0) {
                return _mm256_maskload_pd(input + 4, internal::mm256_tail_mask_epi64<second_lanes>());
            } else {
                return _mm256_setzero_pd();
            }
        }();
        const __m128i quantized1 = internal::mm256_quantize_pd_epi32(source1, scale_v);
        const __m128i quantized2 = internal::mm256_quantize_pd_epi32(source2, scale_v);
        __m256i combined         = _mm256_castsi128_si256(quantized1);
        combined                 = _mm256_inserti128_si256(combined, quantized2, 1);
        const __m256i packed     = internal::mm256_pack_epi32_bmi2<BIT_WIDTH>(internal::mm256_clamp_signed_epi32<BIT_WIDTH>(combined));
        std::memcpy(output, &packed, tail_bytes);
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
template <u8 BIT_WIDTH, u16 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_compress_blocks_bmi2(const void* __restrict__ input_ptr, const f32 scale, void* __restrict__ output_ptr, const u32 blocks) {
    const auto* input = static_cast<const float*>(input_ptr);
    auto* output      = static_cast<u8*>(output_ptr);

    const f32* block_input = input;
    u8* block_output       = output;

    for (u32 block = 0; block < blocks; block++) {
        mm256_compress_block_bmi2<BIT_WIDTH, BLOCK_SIZE>(block_input, scale, block_output);
        block_input += (BLOCK_SIZE * 8) / BIT_WIDTH;
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
template <u8 BIT_WIDTH, u16 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_compress_blocks_bmi2(const void* __restrict__ input_ptr, const f64 scale, void* __restrict__ output_ptr, const u32 blocks) {
    const auto* input = static_cast<const double*>(input_ptr);
    auto* output      = static_cast<u8*>(output_ptr);

    const f64* block_input = input;
    u8* block_output       = output;

    for (u32 block = 0; block < blocks; block++) {
        mm256_compress_block_bmi2<BIT_WIDTH, BLOCK_SIZE>(block_input, scale, block_output);
        block_input += (BLOCK_SIZE * 8) / BIT_WIDTH;
        block_output += BLOCK_SIZE;
    }

    return 0;
}
}  // namespace pernix

#endif  // PERNIX_BMI2_COMPRESSION_H
