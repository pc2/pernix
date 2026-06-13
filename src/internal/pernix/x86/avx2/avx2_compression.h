#ifndef PERNIX_AVX2_COMPRESSION_H
#define PERNIX_AVX2_COMPRESSION_H

#include <pernix/x86/avx2/avx2_tables.h>
#include <pernix/fallback/avx2_compression.h>
#include <pernix/simd_compat.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace pernix {
namespace internal {
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
__always_inline __m256i mm256_clamp_signed_epi32(__m256i input) {
    constexpr int32_t min_value = BIT_WIDTH == 1 ? 0 : -(1 << (BIT_WIDTH - 1));
    constexpr int32_t max_value = BIT_WIDTH == 1 ? 1 : ((1 << (BIT_WIDTH - 1)) - 1);
    return _mm256_min_epi32(_mm256_max_epi32(input, _mm256_set1_epi32(min_value)),
                            _mm256_set1_epi32(max_value));
}

/**
* @brief Quantize four float values into signed 32-bit integers.
*
* @param input source float lane values.
* @param scale per-lane scale factor.
* @return __m128i quantized values.
*/
__always_inline __m128i mm_quantize_ps_epi32(const __m128& input, const __m128& scale) {
    const __m128 scaled = _mm_mul_ps(input, scale);
    // const __m128 rounded = _mm_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return _mm_cvtps_epi32(scaled);
}

/**
* @brief Quantize two double values into a partially filled 128-bit integer register.
*
* @param input source double lane values.
* @param scale per-lane scale factor.
* @return __m128i quantized values in the low lanes.
*/
__always_inline __m128i mm_quantize_pd_epi32(const __m128d& input, const __m128d& scale) {
    const __m128d scaled = _mm_mul_pd(input, scale);
    return _mm_cvtpd_epi32(scaled);
}

/**
* @brief Quantize eight float values into signed 32-bit integers.
*
* @param input source float lane values.
* @param scale per-lane scale factor.
* @return __m256i quantized values.
*/
__always_inline __m256i mm256_quantize_ps_epi32(const __m256& input, const __m256& scale) {
    const __m256 scaled = _mm256_mul_ps(input, scale);
    // const __m256 rounded = _mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return _mm256_cvtps_epi32(scaled);
}

/**
* @brief Quantize four double values into signed 32-bit integers.
*
* @param input source double lane values.
* @param scale per-lane scale factor.
* @return __m128i quantized values in the low lanes.
*/
__always_inline __m128i mm256_quantize_pd_epi32(const __m256d& input, const __m256d& scale) {
    const __m256d scaled = _mm256_mul_pd(input, scale);
    return _mm256_cvtpd_epi32(scaled);
}

#ifndef PERNIX_USE_SIMDE
/**
* @brief Emulate per-16-bit left shifts on AVX2.
*
* @param a source values.
* @param count per-lane shift amounts.
* @return __m128i shifted values.
*/
__always_inline static __m128i _mm_sllv_epi16(const __m128i a, const __m128i count) {
    const __m128i mask      = _mm_set1_epi32(0xffff0000);
    const __m128i low_half  = _mm_sllv_epi32(a, _mm_andnot_si128(mask, count));
    const __m128i high_half = _mm_sllv_epi32(_mm_and_si128(mask, a), _mm_srli_epi32(count, 16));
    return _mm_blend_epi16(low_half, high_half, 0xaa);
}

/**
* @brief Emulate per-16-bit right shifts on AVX2.
*/
__always_inline static __m128i _mm_srlv_epi16(const __m128i a, const __m128i count) {
    const __m128i mask      = _mm_set1_epi32(0x0000ffff);
    const __m128i low_half  = _mm_srlv_epi32(_mm_and_si128(mask, a), _mm_and_si128(mask, count));
    const __m128i high_half = _mm_srlv_epi32(a, _mm_srli_epi32(count, 16));
    return _mm_blend_epi16(low_half, high_half, 0xaa);
}

/**
* @brief Emulate per-16-bit left shifts on 256-bit AVX2 vectors.
*/
__always_inline static __m256i _mm256_sllv_epi16(const __m256i a, const __m256i count) {
    const __m256i mask      = _mm256_set1_epi32(0xffff0000);
    const __m256i low_half  = _mm256_sllv_epi32(a, _mm256_andnot_si256(mask, count));
    const __m256i high_half = _mm256_sllv_epi32(_mm256_and_si256(mask, a), _mm256_srli_epi32(count, 16));
    return _mm256_blend_epi16(low_half, high_half, 0xaa);
}

/**
* @brief Emulate per-16-bit right shifts on 256-bit AVX2 vectors.
*/
__always_inline static __m256i _mm256_srlv_epi16(const __m256i a, const __m256i count) {
    const __m256i mask      = _mm256_set1_epi32(0x0000ffff);
    const __m256i low_half  = _mm256_srlv_epi32(_mm256_and_si256(mask, a), _mm256_and_si256(mask, count));
    const __m256i high_half = _mm256_srlv_epi32(a, _mm256_srli_epi32(count, 16));
    return _mm256_blend_epi16(low_half, high_half, 0xaa);
}

/**
* @brief Blend 8-bit lanes by expanding a scalar mask value.
*/
__always_inline static __m128i mm_blend_epi8(const __m128i X, const __m128i Y, const int8_t M) {
    return _mm_blendv_epi8(X, Y, _mm_set1_epi8(M));
}

/**
* @brief Blend 8-bit lanes in 256-bit vectors by expanding a scalar mask value.
*/
__always_inline static __m256i mm256_blend_epi8(const __m256i X, const __m256i Y, const int8_t M) {
    return _mm256_blendv_epi8(X, Y, _mm256_set1_epi8(M));
}

/**
* @brief Emulate per-byte left shifts on 128-bit vectors.
*/
__always_inline static __m128i _mm_sllv_epi8(const __m128i a, const __m128i count) {
    const __m128i mask      = _mm_set1_epi16(0xff00);
    const __m128i low_half  = _mm_sllv_epi16(a, _mm_andnot_si128(mask, count));
    const __m128i high_half = _mm_sllv_epi16(_mm_and_si128(mask, a), _mm_srli_epi16(count, 8));
    return mm_blend_epi8(low_half, high_half, 0xaa);
}

/**
* @brief Emulate per-byte right shifts on 128-bit vectors.
*/
__always_inline static __m128i _mm_srlv_epi8(const __m128i a, const __m128i count) {
    const __m128i mask      = _mm_set1_epi16(0x00ff);
    const __m128i low_half  = _mm_srlv_epi16(_mm_and_si128(mask, a), _mm_and_si128(mask, count));
    const __m128i high_half = _mm_srlv_epi16(a, _mm_srli_epi16(count, 8));
    return mm_blend_epi8(low_half, high_half, 0xaa);
}

/**
* @brief Emulate per-byte left shifts on 256-bit vectors.
*/
__always_inline static __m256i _mm256_sllv_epi8(const __m256i a, const __m256i count) {
    const __m256i mask      = _mm256_set1_epi16(0xff00);
    const __m256i low_half  = _mm256_sllv_epi16(a, _mm256_andnot_si256(mask, count));
    const __m256i high_half = _mm256_sllv_epi16(_mm256_and_si256(mask, a), _mm256_srli_epi16(count, 8));
    return mm256_blend_epi8(low_half, high_half, 0xaa);
}

/**
* @brief Emulate per-byte right shifts on 256-bit vectors.
*/
__always_inline static __m256i _mm256_srlv_epi8(const __m256i a, const __m256i count) {
    const __m256i mask      = _mm256_set1_epi16(0x00ff);
    const __m256i low_half  = _mm256_srlv_epi16(_mm256_and_si256(mask, a), _mm256_and_si256(mask, count));
    const __m256i high_half = _mm256_srlv_epi16(a, _mm256_srli_epi16(count, 8));
    return mm256_blend_epi8(low_half, high_half, 0xaa);
}
#endif

/**
* @brief Pack four 32-bit values for bit widths 1 through 3.
*/
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 3)
__always_inline auto mm_pack_epi32_avx2_1to3(__m128i& input) -> __m128i {
    constexpr uint32_t bitmask = (1U << BIT_WIDTH) - 1U;
    const __m128i masked       = _mm_and_si128(input, _mm_set1_epi32(static_cast<int32_t>(bitmask)));

    alignas(16) uint32_t lanes[4];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(lanes), masked);

    const uint32_t packed = (lanes[0] & bitmask) | ((lanes[1] & bitmask) << BIT_WIDTH) | (
                                (lanes[2] & bitmask) << (2 * BIT_WIDTH)) |
                            ((lanes[3] & bitmask) << (3 * BIT_WIDTH));

    return _mm_cvtsi32_si128(static_cast<int32_t>(packed));
}

/**
* @brief Pack eight 32-bit values for bit widths 1 through 3.
*/
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 3)
__always_inline __m256i mm256_pack_epi32_avx2_1to3(const __m256i& input) {
    constexpr uint32_t bitmask = (1u << BIT_WIDTH) - 1u;

    const __m256i masked = _mm256_and_si256(input, _mm256_set1_epi32(static_cast<int32_t>(bitmask)));

    const __m256i shifts = _mm256_setr_epi32(0 * BIT_WIDTH, 1 * BIT_WIDTH, 2 * BIT_WIDTH, 3 * BIT_WIDTH,
                                             4 * BIT_WIDTH, 5 * BIT_WIDTH,
                                             6 * BIT_WIDTH, 7 * BIT_WIDTH);

    const __m256i shifted = _mm256_sllv_epi32(masked, shifts);

    __m128i x = _mm_or_si128(_mm256_castsi256_si128(shifted), _mm256_extracti128_si256(shifted, 1));

    x = _mm_or_si128(x, _mm_srli_si128(x, 8));
    x = _mm_or_si128(x, _mm_srli_si128(x, 4));

    return _mm256_castsi128_si256(x);
}

__always_inline __m256i mm256_pack_epi32_avx2_4(const __m256i& input) {
    const __m256i zero = _mm256_setzero_si256();

    const __m256i packed16 = _mm256_packus_epi32(input, zero);
    const __m256i permuted = _mm256_permute4x64_epi64(packed16, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i packed8  = _mm256_packus_epi16(permuted, zero);

    const __m256i combined = _mm256_or_si256(packed8, _mm256_srli_epi16(packed8, 4));

    const __m256i shuffled = _mm256_shuffle_epi8(combined, _mm256_setr_epi8(
                                                     0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
                                                     0, 2,
                                                     4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1));

    return shuffled;
}

/**
* @brief Pack four 32-bit values for bit widths 9 through 16.
*/
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
__always_inline auto mm_pack_epi32_avx2_9to16(__m128i& input) -> __m128i {
    using tables               = pack_tables_avx2_16<BIT_WIDTH, __m128i>;
    constexpr uint16_t bitmask = (1 << BIT_WIDTH) - 1;
    const __m128i masked       = _mm_and_si128(input, _mm_set1_epi16(bitmask));
    __m128i combined;

    if constexpr (BIT_WIDTH == 12 || BIT_WIDTH == 14 || BIT_WIDTH == 15) {
        const __m128i shuffled1 = _mm_shuffle_epi8(masked, tables::get_permute1());
        const __m128i shuffled2 = _mm_shuffle_epi8(masked, tables::get_permute2());

        const __m128i shifted1 = _mm_sllv_epi16(shuffled1, tables::get_shift1());
        const __m128i shifted2 = _mm_srlv_epi16(shuffled2, tables::get_shift2());

        combined = _mm_or_si128(shifted1, shifted2);
    } else {
        const __m128i shuffled1 = _mm_shuffle_epi8(masked, tables::get_permute1());
        const __m128i shuffled2 = _mm_shuffle_epi8(masked, tables::get_permute2());
        const __m128i shuffled3 = _mm_shuffle_epi8(masked, tables::get_permute3());

        const __m128i shifted1 = _mm_sllv_epi16(shuffled1, tables::get_shift1());
        const __m128i shifted2 = _mm_sllv_epi16(shuffled2, tables::get_shift2());
        const __m128i shifted3 = _mm_srlv_epi16(shuffled3, tables::get_shift3());

        combined = _mm_or_si128(_mm_or_si128(shifted1, shifted2), shifted3);
    }
    return combined;
}

/**
* @brief Pack eight 32-bit values for bit widths 9 through 16.
*/
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
__always_inline auto mm256_pack_epi32_avx2_9to16(const __m256i& input) -> __m256i {
    using tables               = pack_tables_avx2_16<BIT_WIDTH, __m128i>;
    constexpr uint16_t bitmask = (1 << BIT_WIDTH) - 1;
    const __m128i packed       = _mm_packs_epi32(_mm256_castsi256_si128(input), _mm256_extracti128_si256(input, 1));
    const __m128i masked       = _mm_and_si128(packed, _mm_set1_epi16(bitmask));
    __m128i combined;

    if constexpr (BIT_WIDTH == 12 || BIT_WIDTH == 14 || BIT_WIDTH == 15 || BIT_WIDTH == 16) {
        const __m128i shuffled1 = _mm_shuffle_epi8(masked, tables::get_permute1());
        const __m128i shuffled2 = _mm_shuffle_epi8(masked, tables::get_permute2());

        const __m128i shifted1 = _mm_sllv_epi16(shuffled1, tables::get_shift1());
        const __m128i shifted2 = _mm_srlv_epi16(shuffled2, tables::get_shift2());

        combined = _mm_or_si128(shifted1, shifted2);
    } else {
        const __m128i shuffled1 = _mm_shuffle_epi8(masked, tables::get_permute1());
        const __m128i shuffled2 = _mm_shuffle_epi8(masked, tables::get_permute2());
        const __m128i shuffled3 = _mm_shuffle_epi8(masked, tables::get_permute3());

        const __m128i shifted1 = _mm_sllv_epi16(shuffled1, tables::get_shift1());
        const __m128i shifted2 = _mm_sllv_epi16(shuffled2, tables::get_shift2());
        const __m128i shifted3 = _mm_srlv_epi16(shuffled3, tables::get_shift3());

        combined = _mm_or_si128(_mm_or_si128(shifted1, shifted2), shifted3);
    }
    return _mm256_castsi128_si256(combined);
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 5 && BIT_WIDTH <= 7)
__always_inline auto mm256_pack_epi32_avx2_5to7(const __m256i& input) -> __m256i {
    const __m256i zero = _mm256_setzero_si256();

    const __m256i packed16 = _mm256_packus_epi32(input, zero);
    const __m256i permuted = _mm256_permute4x64_epi64(packed16, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i packed8  = _mm256_packus_epi16(permuted, zero);

    const __m256i even = _mm256_and_si256(packed8, _mm256_set1_epi16(0x00FF));
    const __m256i odd  = _mm256_and_si256(packed8, _mm256_set1_epi16(0xFF00));

    const __m256i pair16   = _mm256_or_si256(even, _mm256_srli_epi16(odd, 8 - BIT_WIDTH));
    const __m256i extended = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(pair16));

    return mm256_pack_epi32_avx2_9to16<2 * BIT_WIDTH>(extended);
}

/**
* @brief Pack eight 32-bit values for bit widths 17 through 24.
*/
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24)
__always_inline auto mm256_pack_epi32_avx2_17to24(const __m256i& input) -> __m256i {
    using tables               = pack_tables_avx2_24<BIT_WIDTH, __m256i>;
    constexpr uint32_t bitmask = (1 << BIT_WIDTH) - 1;
    const __m256i masked       = _mm256_and_si256(input, _mm256_set1_epi32(bitmask));
    __m256i combined;

    if constexpr (BIT_WIDTH == 24) {
        const __m256i shuffled1 = _mm256_permutevar8x32_epi32(masked, tables::get_permute1());
        const __m256i shuffled2 = _mm256_permutevar8x32_epi32(masked, tables::get_permute2());

        const __m256i shifted1 = _mm256_sllv_epi32(shuffled1, tables::get_shift1());
        const __m256i shifted2 = _mm256_srlv_epi32(shuffled2, tables::get_shift2());

        combined = _mm256_or_si256(shifted1, shifted2);
    } else {
        const __m256i shuffled1 = _mm256_permutevar8x32_epi32(masked, tables::get_permute1());
        const __m256i shuffled2 = _mm256_permutevar8x32_epi32(masked, tables::get_permute2());
        const __m256i shuffled3 = _mm256_permutevar8x32_epi32(masked, tables::get_permute3());

        const __m256i shifted1 = _mm256_sllv_epi32(shuffled1, tables::get_shift1());
        const __m256i shifted2 = _mm256_sllv_epi32(shuffled2, tables::get_shift2());
        const __m256i shifted3 = _mm256_srlv_epi32(shuffled3, tables::get_shift3());

        combined = _mm256_or_si256(_mm256_or_si256(shifted1, shifted2), shifted3);
    }

    return combined;
}

/**
* @brief Pack aligned 8-bit or 16-bit values from four 32-bit lanes.
*/
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH == 8 || BIT_WIDTH == 16)
auto mm_pack_aligned_epi32_avx2(__m128i& input) -> __m128i {
    if constexpr (BIT_WIDTH == 8) {
        return _mm_packus_epi16(_mm_packs_epi32(input, _mm_setzero_si128()), _mm_setzero_si128());
    } else {
        return _mm_packs_epi32(input, _mm_setzero_si128());
    }
}

/**
* @brief Dispatch to the appropriate 128-bit AVX2 packer for the selected bit width.
*/
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 16)
auto mm_pack_epi32_avx2(__m128i& input) -> __m128i {
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 3) {
        return internal::mm_pack_epi32_avx2_1to3<BIT_WIDTH>(input);
    } else if constexpr (BIT_WIDTH >= 4 && BIT_WIDTH <= 8) {
        // TODO: implementation for 4-8 bits
        return _mm_setzero_si128();
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
        return internal::mm_pack_epi32_avx2_9to16<BIT_WIDTH>(input);
    } else {
        return _mm_setzero_si128();
    }
}

/**
* @brief Pack aligned 8-bit or 16-bit values from eight 32-bit lanes.
*/
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH == 8 || BIT_WIDTH == 16)
__m256i mm256_pack_aligned_epi32_avx2(const __m256i& input) {
    if constexpr (BIT_WIDTH == 8) {
        const __m128i packed16 = _mm_packs_epi32(_mm256_castsi256_si128(input),
                                                 _mm256_extracti128_si256(input, 1));
        const __m128i packed8 = _mm_packs_epi16(packed16, _mm_setzero_si128());
        return _mm256_castsi128_si256(packed8);
    } else {
        return _mm256_castsi128_si256(
            _mm_packs_epi32(_mm256_castsi256_si128(input), _mm256_extracti128_si256(input, 1)));
    }
}

/**
* @brief Dispatch to the appropriate 256-bit AVX2 packer for the selected bit width.
*/
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
__m256i mm256_pack_epi32_avx2(const __m256i& input) {
    if constexpr (BIT_WIDTH == 8 || BIT_WIDTH == 16) {
        return internal::mm256_pack_aligned_epi32_avx2<BIT_WIDTH>(input);
    } else {
        if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 3) {
            return internal::mm256_pack_epi32_avx2_1to3<BIT_WIDTH>(input);
        } else if constexpr (BIT_WIDTH == 4) {
            return mm256_pack_epi32_avx2_4(input);
        } else if constexpr (BIT_WIDTH >= 5 && BIT_WIDTH <= 7) {
            return internal::mm256_pack_epi32_avx2_5to7<BIT_WIDTH>(input);
        } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 15) {
            return internal::mm256_pack_epi32_avx2_9to16<BIT_WIDTH>(input);
        } else if constexpr (BIT_WIDTH >= 17 && BIT_WIDTH <= 24) {
            return internal::mm256_pack_epi32_avx2_17to24<BIT_WIDTH>(input);
        }
    }
    return _mm256_setzero_si256();
}
} // namespace internal

/**
* @brief Compress a single block of float using AVX2 instructions.
*
* @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
* @tparam BLOCK_SIZE size of the block in bytes (must be a multiple of 32).
*
* @param input pointer to the start of the input float values.
* @param scale scaling factor used during quantization.
* @param output pointer to the output buffer where compressed bytes will be stored.
* @return int status code (0 for success).
*
* @note This function requires AVX2 support.
*/
template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_compress_block_avx2(const void* __restrict__ input_ptr, const float_t scale,
                              void* __restrict__ output_ptr) {
    const auto* input = static_cast<const float*>(input_ptr);
    auto* output      = static_cast<uint8_t*>(output_ptr);

    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;
    constexpr uint32_t iterations_8       = elements_per_block / 8;
    constexpr uint8_t remaining           = elements_per_block - iterations_8 * 8;

    std::memset(output, 0, BLOCK_SIZE);

    const __m256 scale_v = _mm256_set1_ps(scale);
#pragma GCC unroll 8
    for (uint32_t iter = 0; iter < iterations_8; iter++) {
        const __m256 source        = _mm256_loadu_ps(input);
        const __m256i quantized    = internal::mm256_quantize_ps_epi32(source, scale_v);
        const __m256i packed_input = internal::mm256_clamp_signed_epi32<BIT_WIDTH>(quantized);
        const __m256i packed       = internal::mm256_pack_epi32_avx2<BIT_WIDTH>(packed_input);
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
* @brief Compress a single block of double using AVX2 instructions.
*
* @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
* @tparam BLOCK_SIZE size of the block in bytes (must be a multiple of 32).
*
* @param input pointer to the start of the input double values.
* @param scale scaling factor used during quantization.
* @param output pointer to the output buffer where compressed bytes will be stored.
* @return int status code (0 for success).
*
* @note This function requires AVX2 support.
*/
template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_compress_block_avx2(const void* __restrict__ input_ptr, const double_t scale,
                              void* __restrict__ output_ptr) {
    const auto* input = static_cast<const double*>(input_ptr);
    auto* output      = static_cast<uint8_t*>(output_ptr);

    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;
    constexpr uint32_t iterations_8       = elements_per_block / 8;
    constexpr uint8_t remaining           = elements_per_block - iterations_8 * 8;

    std::memset(output, 0, BLOCK_SIZE);

    const __m256d scale_v = _mm256_set1_pd(scale);
#pragma GCC unroll 8
    for (uint32_t iter = 0; iter < iterations_8; iter++) {
        const __m256d source1    = _mm256_loadu_pd(input);
        const __m256d source2    = _mm256_loadu_pd(input + 4);
        const __m128i quantized1 = internal::mm256_quantize_pd_epi32(source1, scale_v);
        const __m128i quantized2 = internal::mm256_quantize_pd_epi32(source2, scale_v);
        __m256i combined         = _mm256_castsi128_si256(quantized1);
        combined                 = _mm256_inserti128_si256(combined, quantized2, 1);
        const __m256i packed     = internal::mm256_pack_epi32_avx2<BIT_WIDTH>(
            internal::mm256_clamp_signed_epi32<BIT_WIDTH>(combined));
        // _mm_storeu_si128(reinterpret_cast<__m128i*>(output), _mm256_castsi256_si128(packed));
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
* @brief Compress multiple blocks using AVX2 instructions.
*
* @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
* @tparam BLOCK_SIZE size of the block in bytes (must be a multiple of 32).
*
* @param input pointer to the start of the input float values.
* @param scale scaling factor used during quantization.
* @param output pointer to the output buffer where compressed bytes will be stored.
* @param blocks number of blocks to compress.
* @return int status code (0 for success).
*
* @note This function requires AVX2 support.
*/
template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_compress_blocks_avx2(const void* __restrict__ input_ptr, const float_t scale,
                               void* __restrict__ output_ptr,
                               const uint32_t blocks) {
    const auto* input = static_cast<const float*>(input_ptr);
    auto* output      = static_cast<uint8_t*>(output_ptr);

    const float_t* block_input = input;
    uint8_t* block_output      = output;

    for (uint32_t block = 0; block < blocks; block++) {
        mm256_compress_block_avx2<BIT_WIDTH, BLOCK_SIZE>(block_input, scale, block_output);
        block_input  += (BLOCK_SIZE * 8) / BIT_WIDTH;
        block_output += BLOCK_SIZE;
    }

    return 0;
}

/**
* @brief Compress multiple blocks using AVX2 instructions.
*
* @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
* @tparam BLOCK_SIZE size of the block in bytes (must be a multiple of 32).
*
* @param input pointer to the start of the input double values.
* @param scale scaling factor used during quantization.
* @param output pointer to the output buffer where compressed bytes will be stored.
* @param blocks number of blocks to compress.
* @return int status code (0 for success).
*
* @note This function requires AVX2 support.
*/
template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_compress_blocks_avx2(const void* __restrict__ input_ptr, const double_t scale,
                               void* __restrict__ output_ptr,
                               const uint32_t blocks) {
    const auto* input = static_cast<const double*>(input_ptr);
    auto* output      = static_cast<uint8_t*>(output_ptr);

    const double_t* block_input = input;
    uint8_t* block_output       = output;

    for (uint32_t block = 0; block < blocks; block++) {
        mm256_compress_block_avx2<BIT_WIDTH, BLOCK_SIZE>(block_input, scale, block_output);
        block_input  += (BLOCK_SIZE * 8) / BIT_WIDTH;
        block_output += BLOCK_SIZE;
    }

    return 0;
}
} // namespace pernix

#endif  // PERNIX_AVX2_COMPRESSION_H
