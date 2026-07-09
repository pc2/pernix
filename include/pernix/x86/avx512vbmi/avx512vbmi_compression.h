#ifndef PERNIX_AVX512VBMI_COMPRESSION_H
#define PERNIX_AVX512VBMI_COMPRESSION_H

#include <pernix/simd_compat.h>
#include <pernix/x86/avx2/avx2_compression.h>
#include <pernix/x86/avx512vbmi/compat.h>
#include <pernix/x86/avx512vbmi/packing.h>
#include <pernix/x86/utils.h>

#include <cstring>

using namespace pernix::x86::internal;

namespace pernix {
namespace internal {
template <u8 BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
static __always_inline __m512i mm512_clamp_signed_epi32(__m512i input) {
    constexpr i32 min_value = BIT_WIDTH == 1 ? 0 : -(1 << (BIT_WIDTH - 1));
    constexpr i32 max_value = BIT_WIDTH == 1 ? 1 : ((1 << (BIT_WIDTH - 1)) - 1);
    return _mm512_min_epi32(_mm512_max_epi32(input, _mm512_set1_epi32(min_value)), _mm512_set1_epi32(max_value));
}

template <u8 BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
static __always_inline __m256i mm256_clamp_signed_epi32_avx512(__m256i input) {
    constexpr i32 min_value = BIT_WIDTH == 1 ? 0 : -(1 << (BIT_WIDTH - 1));
    constexpr i32 max_value = BIT_WIDTH == 1 ? 1 : ((1 << (BIT_WIDTH - 1)) - 1);
    return _mm256_min_epi32(_mm256_max_epi32(input, _mm256_set1_epi32(min_value)), _mm256_set1_epi32(max_value));
}

template <u8 BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
static __always_inline __m512i mm512_clamp_signed_epi64(__m512i input) {
    constexpr i64 min_value = BIT_WIDTH == 1 ? 0 : -(i64{1} << (BIT_WIDTH - 1));
    constexpr i64 max_value = BIT_WIDTH == 1 ? 1 : ((i64{1} << (BIT_WIDTH - 1)) - 1);
    return _mm512_min_epi64(_mm512_max_epi64(input, _mm512_set1_epi64(min_value)), _mm512_set1_epi64(max_value));
}

/**
 * @brief Quantize sixteen float values to 32-bit integers.
 */

static __always_inline __m512i mm512_quantize_ps_epi32(const __m512& input, const __m512& scale) {
    const __m512 scaled = _mm512_mul_ps(input, scale);
    return _mm512_cvtps_epi32(scaled);
}

static __always_inline __m512i mm512_quantize_pd_epi64(const __m512d& input, const __m512d& scale) {
    const __m512d scaled = _mm512_mul_pd(input, scale);
    return _mm512_cvtpd_epi64(scaled);
}

static __always_inline __m256i mm512_quantize_pd_epi32(const __m512d& input, const __m512d& scale) {
    const __m512d scaled = _mm512_mul_pd(input, scale);
    return _mm512_cvtpd_epi32(scaled);
}

static __always_inline __m512i make_m512i_from_2x256(const __m256i a, const __m256i b) {
    __m512i result = _mm512_castsi256_si512(a);
    result         = _mm512_inserti64x4(result, b, 1);
    return result;
}

static __always_inline __m512i make_m512i_from_4x128(const __m128i a, const __m128i b, const __m128i c, const __m128i d) {
    __m512i result = _mm512_castsi128_si512(a);
    result         = _mm512_inserti64x2(result, b, 1);
    result         = _mm512_inserti64x2(result, c, 2);
    result         = _mm512_inserti64x2(result, d, 3);
    return result;
}

static __always_inline __m512i make_m512i_from_8x64(const __m128i a, const __m128i b, const __m128i c, const __m128i d, const __m128i e,
                                                    const __m128i f, const __m128i g, const __m128i h) {
    const __m128i ab = _mm_unpacklo_epi64(a, b);
    const __m128i cd = _mm_unpacklo_epi64(c, d);
    const __m128i ef = _mm_unpacklo_epi64(e, f);
    const __m128i gh = _mm_unpacklo_epi64(g, h);

    __m512i x = _mm512_castsi128_si512(ab);
    x         = _mm512_inserti32x4(x, cd, 1);
    x         = _mm512_inserti32x4(x, ef, 2);
    x         = _mm512_inserti32x4(x, gh, 3);
    return x;
}

static __always_inline __m256i make_m256i_from_2x128(const __m128i a, const __m128i b) {
    __m256i result = _mm256_castsi128_si256(a);
    result         = _mm256_inserti128_si256(result, b, 1);
    return result;
}

static __always_inline __m256i make_m256i_from_4x64(const __m128i a, const __m128i b, const __m128i c, const __m128i d) {
    const __m128i ab = _mm_unpacklo_epi64(a, b);
    const __m128i cd = _mm_unpacklo_epi64(c, d);

    __m256i x = _mm256_castsi128_si256(ab);
    x         = _mm256_inserti128_si256(x, cd, 1);
    return x;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_compress_block_avx512vbmi_1to8(const f32* __restrict__ input, const f32 scale, u8* __restrict__ output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr u32 iterations_64      = elements_per_block / 64;
    constexpr u32 iterations_32      = (elements_per_block % 64) / 32;
    constexpr u32 iterations_16      = (elements_per_block % 32) / 16;
    constexpr u32 remaining_elements = elements_per_block - iterations_64 * 64 - iterations_32 * 32 - iterations_16 * 16;

    const __m512 scale_v = _mm512_set1_ps(scale);

    if constexpr (iterations_64 > 0) {
#pragma GCC unroll 8
        for (u32 iter = 0; iter < iterations_64; ++iter) {
            const __m512 source1 = _mm512_loadu_ps(input);
            const __m512 source2 = _mm512_loadu_ps(input + 16);
            const __m512 source3 = _mm512_loadu_ps(input + 32);
            const __m512 source4 = _mm512_loadu_ps(input + 48);

            const __m512i quantized1 = mm512_clamp_signed_epi32<BIT_WIDTH>(mm512_quantize_ps_epi32(source1, scale_v));
            const __m512i quantized2 = mm512_clamp_signed_epi32<BIT_WIDTH>(mm512_quantize_ps_epi32(source2, scale_v));
            const __m512i quantized3 = mm512_clamp_signed_epi32<BIT_WIDTH>(mm512_quantize_ps_epi32(source3, scale_v));
            const __m512i quantized4 = mm512_clamp_signed_epi32<BIT_WIDTH>(mm512_quantize_ps_epi32(source4, scale_v));

            const __m128i converted1 = _mm512_cvtepi32_epi8(quantized1);
            const __m128i converted2 = _mm512_cvtepi32_epi8(quantized2);
            const __m128i converted3 = _mm512_cvtepi32_epi8(quantized3);
            const __m128i converted4 = _mm512_cvtepi32_epi8(quantized4);

            const __m512i packed =
                m512::mm512_pack_epi8_avx512vbmi_1to8<BIT_WIDTH>(make_m512i_from_4x128(converted1, converted2, converted3, converted4));

            mm512_storeu_elements_epi64(output, BIT_WIDTH, packed);

            input += 64;
            output += 8 * BIT_WIDTH;
        }
    }

    if constexpr (iterations_32 > 0) {
        const __m512 source1 = _mm512_loadu_ps(input);
        const __m512 source2 = _mm512_loadu_ps(input + 16);

        const __m512i quantized1 = mm512_clamp_signed_epi32<BIT_WIDTH>(mm512_quantize_ps_epi32(source1, scale_v));
        const __m512i quantized2 = mm512_clamp_signed_epi32<BIT_WIDTH>(mm512_quantize_ps_epi32(source2, scale_v));

        const __m128i converted1 = _mm512_cvtepi32_epi8(quantized1);
        const __m128i converted2 = _mm512_cvtepi32_epi8(quantized2);

        const __m256i packed = m256::mm256_pack_epi8_avx512vbmi_1to8<BIT_WIDTH>(make_m256i_from_2x128(converted1, converted2));
        mm256_storeu_elements_epi32(output, BIT_WIDTH, packed);

        input += 32;
        output += 4 * BIT_WIDTH;
    }

    if constexpr (iterations_16 > 0) {
        const __m512 source     = _mm512_loadu_ps(input);
        const __m512i quantized = mm512_clamp_signed_epi32<BIT_WIDTH>(mm512_quantize_ps_epi32(source, scale_v));
        const __m128i converted = _mm512_cvtepi32_epi8(quantized);

        const __m128i packed = m128::mm_pack_epi8_avx512vbmi_1to8<BIT_WIDTH>(converted);
        mm_storeu_elements_epi16(output, BIT_WIDTH, packed);

        input += 16;
        output += 2 * BIT_WIDTH;
    }

    if constexpr (remaining_elements > 0) {
        const __m512 source     = mm512_loadu_elements_ps(remaining_elements, input);
        const __m512i quantized = mm512_clamp_signed_epi32<BIT_WIDTH>(mm512_quantize_ps_epi32(source, scale_v));
        const __m128i converted = _mm512_cvtepi32_epi8(quantized);
        const __m128i packed    = m128::mm_pack_epi8_avx512vbmi_1to8<BIT_WIDTH>(converted);

        mm_storeu_elements_epi8(output, tail_bytes(BIT_WIDTH, remaining_elements), packed);
    }

    return 0;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_compress_block_avx512vbmi_9to16(const f32* __restrict__ input, const f32 scale, u8* __restrict__ output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr u32 iterations_32      = elements_per_block / 32;
    constexpr u32 iterations_16      = (elements_per_block % 32) / 16;
    constexpr u32 iterations_8       = (elements_per_block % 16) / 8;
    constexpr u32 remaining_elements = elements_per_block - iterations_32 * 32 - iterations_16 * 16 - iterations_8 * 8;

    const __m512 scale_v    = _mm512_set1_ps(scale);
    const __m256 scale_v256 = _mm256_set1_ps(scale);

    if constexpr (iterations_32 > 0) {
#pragma GCC unroll 4
        for (u32 iter = 0; iter < iterations_32; ++iter) {
            const __m512 source1 = _mm512_loadu_ps(input);
            const __m512 source2 = _mm512_loadu_ps(input + 16);

            const __m512i quantized1 = mm512_clamp_signed_epi32<BIT_WIDTH>(mm512_quantize_ps_epi32(source1, scale_v));
            const __m512i quantized2 = mm512_clamp_signed_epi32<BIT_WIDTH>(mm512_quantize_ps_epi32(source2, scale_v));

            const __m256i converted1 = _mm512_cvtepi32_epi16(quantized1);
            const __m256i converted2 = _mm512_cvtepi32_epi16(quantized2);

            const __m512i packed = m512::mm512_pack_epi16_avx512vbmi_9to16<BIT_WIDTH>(make_m512i_from_2x256(converted1, converted2));
            mm512_storeu_elements_epi32(output, BIT_WIDTH, packed);

            input += 32;
            output += 4 * BIT_WIDTH;
        }
    }

    if constexpr (iterations_16 > 0) {
        const __m512 source     = _mm512_loadu_ps(input);
        const __m512i quantized = mm512_clamp_signed_epi32<BIT_WIDTH>(mm512_quantize_ps_epi32(source, scale_v));
        const __m256i converted = _mm512_cvtepi32_epi16(quantized);

        const __m256i packed = m256::mm256_pack_epi16_avx512vbmi_9to16<BIT_WIDTH>(converted);
        mm256_storeu_elements_epi16(output, BIT_WIDTH, packed);

        input += 16;
        output += 2 * BIT_WIDTH;
    }

    if constexpr (iterations_8 > 0) {
        const __m256 source     = _mm256_loadu_ps(input);
        const __m256i quantized = mm256_clamp_signed_epi32_avx512<BIT_WIDTH>(mm256_quantize_ps_epi32(source, scale_v256));
        const __m128i converted = _mm256_cvtepi32_epi16(quantized);

        const __m128i packed = m128::mm_pack_epi16_avx512vbmi_9to16<BIT_WIDTH>(converted);
        mm_storeu_elements_epi8(output, BIT_WIDTH, packed);

        input += 8;
        output += BIT_WIDTH;
    }

    if constexpr (remaining_elements > 0) {
        const __m256 source     = mm256_loadu_elements_ps(remaining_elements, input);
        const __m256i quantized = mm256_clamp_signed_epi32_avx512<BIT_WIDTH>(mm256_quantize_ps_epi32(source, scale_v256));
        const __m128i converted = _mm256_cvtepi32_epi16(quantized);
        const __m128i packed    = m128::mm_pack_epi16_avx512vbmi_9to16<BIT_WIDTH>(converted);

        mm_storeu_elements_epi8(output, tail_bytes(BIT_WIDTH, remaining_elements), packed);
    }

    return 0;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_compress_block_avx512vbmi_17to24(const f32* __restrict__ input, const f32 scale, u8* __restrict__ output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr u32 iterations_16      = elements_per_block / 16;
    constexpr u32 iterations_8       = (elements_per_block % 16) / 8;
    constexpr u32 remaining_elements = elements_per_block - iterations_16 * 16 - iterations_8 * 8;

    const __m512 scale_v    = _mm512_set1_ps(scale);
    const __m256 scale_v256 = _mm256_set1_ps(scale);

    if constexpr (iterations_16 > 0) {
#pragma GCC unroll 2
        for (u32 i = 0; i < iterations_16; ++i) {
            const __m512 source        = _mm512_loadu_ps(input);
            const __m512i packed_input = mm512_clamp_signed_epi32<BIT_WIDTH>(mm512_quantize_ps_epi32(source, scale_v));

            const __m512i packed = m512::mm512_pack_epi32_avx512vbmi_17to24<BIT_WIDTH>(packed_input);
            mm512_storeu_elements_epi16(output, BIT_WIDTH, packed);
            input += 16;
            output += 2 * BIT_WIDTH;
        }
    }

    if constexpr (iterations_8 > 0) {
        const __m256 source        = _mm256_loadu_ps(input);
        const __m256i packed_input = mm256_clamp_signed_epi32_avx512<BIT_WIDTH>(mm256_quantize_ps_epi32(source, scale_v256));

        const __m256i packed = m256::mm256_pack_epi32_avx512vbmi_17to24<BIT_WIDTH>(packed_input);
        mm256_storeu_elements_epi8(output, BIT_WIDTH, packed);

        input += 8;
        output += BIT_WIDTH;
    }

    if constexpr (remaining_elements > 0) {
        const __m256 source        = mm256_loadu_elements_ps(remaining_elements, input);
        const __m256i packed_input = mm256_clamp_signed_epi32_avx512<BIT_WIDTH>(mm256_quantize_ps_epi32(source, scale_v256));
        const __m256i packed       = m256::mm256_pack_epi32_avx512vbmi_17to24<BIT_WIDTH>(packed_input);

        mm256_storeu_elements_epi8(output, tail_bytes(BIT_WIDTH, remaining_elements), packed);
    }

    return 0;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_compress_block_avx512vbmi_1to8(const f64* __restrict__ input, const f64 scale, u8* __restrict__ output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr u32 iterations_64      = elements_per_block / 64;
    constexpr u32 iterations_32      = (elements_per_block % 64) / 32;
    constexpr u32 iterations_16      = (elements_per_block % 32) / 16;
    constexpr u32 remaining_elements = elements_per_block - iterations_64 * 64 - iterations_32 * 32 - iterations_16 * 16;

    const __m512d scale_v = _mm512_set1_pd(scale);

    if constexpr (iterations_64 > 0) {
#pragma GCC unroll 8
        for (u32 iter = 0; iter < iterations_64; ++iter) {
            const __m512d source1 = _mm512_loadu_pd(input);
            const __m512d source2 = _mm512_loadu_pd(input + 8);
            const __m512d source3 = _mm512_loadu_pd(input + 16);
            const __m512d source4 = _mm512_loadu_pd(input + 24);
            const __m512d source5 = _mm512_loadu_pd(input + 32);
            const __m512d source6 = _mm512_loadu_pd(input + 40);
            const __m512d source7 = _mm512_loadu_pd(input + 48);
            const __m512d source8 = _mm512_loadu_pd(input + 56);

            const __m512i quantized1 = mm512_clamp_signed_epi64<BIT_WIDTH>(mm512_quantize_pd_epi64(source1, scale_v));
            const __m512i quantized2 = mm512_clamp_signed_epi64<BIT_WIDTH>(mm512_quantize_pd_epi64(source2, scale_v));
            const __m512i quantized3 = mm512_clamp_signed_epi64<BIT_WIDTH>(mm512_quantize_pd_epi64(source3, scale_v));
            const __m512i quantized4 = mm512_clamp_signed_epi64<BIT_WIDTH>(mm512_quantize_pd_epi64(source4, scale_v));
            const __m512i quantized5 = mm512_clamp_signed_epi64<BIT_WIDTH>(mm512_quantize_pd_epi64(source5, scale_v));
            const __m512i quantized6 = mm512_clamp_signed_epi64<BIT_WIDTH>(mm512_quantize_pd_epi64(source6, scale_v));
            const __m512i quantized7 = mm512_clamp_signed_epi64<BIT_WIDTH>(mm512_quantize_pd_epi64(source7, scale_v));
            const __m512i quantized8 = mm512_clamp_signed_epi64<BIT_WIDTH>(mm512_quantize_pd_epi64(source8, scale_v));

            const __m128i converted1 = _mm512_cvtepi64_epi8(quantized1);
            const __m128i converted2 = _mm512_cvtepi64_epi8(quantized2);
            const __m128i converted3 = _mm512_cvtepi64_epi8(quantized3);
            const __m128i converted4 = _mm512_cvtepi64_epi8(quantized4);
            const __m128i converted5 = _mm512_cvtepi64_epi8(quantized5);
            const __m128i converted6 = _mm512_cvtepi64_epi8(quantized6);
            const __m128i converted7 = _mm512_cvtepi64_epi8(quantized7);
            const __m128i converted8 = _mm512_cvtepi64_epi8(quantized8);

            const __m512i packed = m512::mm512_pack_epi8_avx512vbmi_1to8<BIT_WIDTH>(
                make_m512i_from_8x64(converted1, converted2, converted3, converted4, converted5, converted6, converted7, converted8));

            mm512_storeu_elements_epi64(output, BIT_WIDTH, packed);

            input += 64;
            output += 8 * BIT_WIDTH;
        }
    }

    if constexpr (iterations_32 > 0) {
        const __m512d source1 = _mm512_loadu_pd(input);
        const __m512d source2 = _mm512_loadu_pd(input + 8);
        const __m512d source3 = _mm512_loadu_pd(input + 16);
        const __m512d source4 = _mm512_loadu_pd(input + 24);

        const __m512i quantized1 = mm512_clamp_signed_epi64<BIT_WIDTH>(mm512_quantize_pd_epi64(source1, scale_v));
        const __m512i quantized2 = mm512_clamp_signed_epi64<BIT_WIDTH>(mm512_quantize_pd_epi64(source2, scale_v));
        const __m512i quantized3 = mm512_clamp_signed_epi64<BIT_WIDTH>(mm512_quantize_pd_epi64(source3, scale_v));
        const __m512i quantized4 = mm512_clamp_signed_epi64<BIT_WIDTH>(mm512_quantize_pd_epi64(source4, scale_v));

        const __m128i converted1 = _mm512_cvtepi64_epi8(quantized1);
        const __m128i converted2 = _mm512_cvtepi64_epi8(quantized2);
        const __m128i converted3 = _mm512_cvtepi64_epi8(quantized3);
        const __m128i converted4 = _mm512_cvtepi64_epi8(quantized4);

        const __m256i packed =
            m256::mm256_pack_epi8_avx512vbmi_1to8<BIT_WIDTH>(make_m256i_from_4x64(converted1, converted2, converted3, converted4));

        mm256_storeu_elements_epi32(output, BIT_WIDTH, packed);

        input += 32;
        output += 4 * BIT_WIDTH;
    }

    if constexpr (iterations_16 > 0) {
        const __m512d source1    = _mm512_loadu_pd(input);
        const __m512d source2    = _mm512_loadu_pd(input + 8);
        const __m512i quantized1 = mm512_clamp_signed_epi64<BIT_WIDTH>(mm512_quantize_pd_epi64(source1, scale_v));
        const __m512i quantized2 = mm512_clamp_signed_epi64<BIT_WIDTH>(mm512_quantize_pd_epi64(source2, scale_v));

        const __m128i converted1 = _mm512_cvtepi64_epi8(quantized1);
        const __m128i converted2 = _mm512_cvtepi64_epi8(quantized2);

        const __m128i packed = m128::mm_pack_epi8_avx512vbmi_1to8<BIT_WIDTH>(_mm_unpacklo_epi64(converted1, converted2));

        mm_storeu_elements_epi16(output, BIT_WIDTH, packed);

        input += 16;
        output += 2 * BIT_WIDTH;
    }

    if constexpr (remaining_elements > 0) {
        constexpr u32 source1_elements = remaining_elements > 8 ? 8 : remaining_elements;
        constexpr u32 source2_elements = remaining_elements > 8 ? remaining_elements - 8 : 0;

        const __m512d source1    = mm512_loadu_elements_pd(source1_elements, input);
        const __m512d source2    = source2_elements > 0 ? mm512_loadu_elements_pd(source2_elements, input + 8) : _mm512_setzero_pd();
        const __m512i quantized1 = mm512_clamp_signed_epi64<BIT_WIDTH>(mm512_quantize_pd_epi64(source1, scale_v));
        const __m512i quantized2 = mm512_clamp_signed_epi64<BIT_WIDTH>(mm512_quantize_pd_epi64(source2, scale_v));

        const __m128i converted1 = _mm512_cvtepi64_epi8(quantized1);
        const __m128i converted2 = _mm512_cvtepi64_epi8(quantized2);

        const __m128i packed = m128::mm_pack_epi8_avx512vbmi_1to8<BIT_WIDTH>(_mm_unpacklo_epi64(converted1, converted2));

        mm_storeu_elements_epi8(output, tail_bytes(BIT_WIDTH, remaining_elements), packed);
    }

    return 0;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_compress_block_avx512vbmi_9to16(const f64* __restrict__ input, const f64 scale, u8* __restrict__ output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr u32 iterations_32      = elements_per_block / 32;
    constexpr u32 iterations_16      = (elements_per_block % 32) / 16;
    constexpr u32 iterations_8       = (elements_per_block % 16) / 8;
    constexpr u32 remaining_elements = elements_per_block - iterations_32 * 32 - iterations_16 * 16 - iterations_8 * 8;

    const __m512d scale_v = _mm512_set1_pd(scale);

    if constexpr (iterations_32 > 0) {
#pragma GCC unroll 4
        for (u32 iter = 0; iter < iterations_32; ++iter) {
            const __m512d source1 = _mm512_loadu_pd(input);
            const __m512d source2 = _mm512_loadu_pd(input + 8);
            const __m512d source3 = _mm512_loadu_pd(input + 16);
            const __m512d source4 = _mm512_loadu_pd(input + 24);

            const __m512i quantized1 = mm512_clamp_signed_epi64<BIT_WIDTH>(mm512_quantize_pd_epi64(source1, scale_v));
            const __m512i quantized2 = mm512_clamp_signed_epi64<BIT_WIDTH>(mm512_quantize_pd_epi64(source2, scale_v));
            const __m512i quantized3 = mm512_clamp_signed_epi64<BIT_WIDTH>(mm512_quantize_pd_epi64(source3, scale_v));
            const __m512i quantized4 = mm512_clamp_signed_epi64<BIT_WIDTH>(mm512_quantize_pd_epi64(source4, scale_v));

            const __m128i converted1 = _mm512_cvtepi64_epi16(quantized1);
            const __m128i converted2 = _mm512_cvtepi64_epi16(quantized2);
            const __m128i converted3 = _mm512_cvtepi64_epi16(quantized3);
            const __m128i converted4 = _mm512_cvtepi64_epi16(quantized4);

            const __m512i packed =
                m512::mm512_pack_epi16_avx512vbmi_9to16<BIT_WIDTH>(make_m512i_from_4x128(converted1, converted2, converted3, converted4));

            mm512_storeu_elements_epi32(output, BIT_WIDTH, packed);

            input += 32;
            output += 4 * BIT_WIDTH;
        }
    }

    if constexpr (iterations_16 > 0) {
        const __m512d source1    = _mm512_loadu_pd(input);
        const __m512d source2    = _mm512_loadu_pd(input + 8);
        const __m512i quantized1 = mm512_clamp_signed_epi64<BIT_WIDTH>(mm512_quantize_pd_epi64(source1, scale_v));
        const __m512i quantized2 = mm512_clamp_signed_epi64<BIT_WIDTH>(mm512_quantize_pd_epi64(source2, scale_v));

        const __m128i converted1 = _mm512_cvtepi64_epi16(quantized1);
        const __m128i converted2 = _mm512_cvtepi64_epi16(quantized2);

        const __m256i packed = m256::mm256_pack_epi16_avx512vbmi_9to16<BIT_WIDTH>(make_m256i_from_2x128(converted1, converted2));

        mm256_storeu_elements_epi16(output, BIT_WIDTH, packed);

        input += 16;
        output += 2 * BIT_WIDTH;
    }

    if constexpr (iterations_8 > 0) {
        const __m512d source    = _mm512_loadu_pd(input);
        const __m512i quantized = mm512_clamp_signed_epi64<BIT_WIDTH>(mm512_quantize_pd_epi64(source, scale_v));
        const __m128i converted = _mm512_cvtepi64_epi16(quantized);

        const __m128i packed = m128::mm_pack_epi16_avx512vbmi_9to16<BIT_WIDTH>(converted);
        mm_storeu_elements_epi8(output, BIT_WIDTH, packed);

        input += 8;
        output += BIT_WIDTH;
    }

    if constexpr (remaining_elements > 0) {
        const __m512d source    = mm512_loadu_elements_pd(remaining_elements, input);
        const __m512i quantized = mm512_clamp_signed_epi64<BIT_WIDTH>(mm512_quantize_pd_epi64(source, scale_v));
        const __m128i converted = _mm512_cvtepi64_epi16(quantized);

        const __m128i packed = m128::mm_pack_epi16_avx512vbmi_9to16<BIT_WIDTH>(converted);
        mm_storeu_elements_epi8(output, tail_bytes(BIT_WIDTH, remaining_elements), packed);
    }

    return 0;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_compress_block_avx512vbmi_17to24(const f64* __restrict__ input, const f64 scale, u8* __restrict__ output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr u32 iterations_16      = elements_per_block / 16;
    constexpr u32 iterations_8       = (elements_per_block % 16) / 8;
    constexpr u32 remaining_elements = elements_per_block - iterations_16 * 16 - iterations_8 * 8;

    const __m512d scale_v = _mm512_set1_pd(scale);

    if constexpr (iterations_16 > 0) {
#pragma GCC unroll 2
        for (u32 i = 0; i < iterations_16; ++i) {
            const __m512d source1 = _mm512_loadu_pd(input);
            const __m512d source2 = _mm512_loadu_pd(input + 8);

            const __m256i quantized1 = mm256_clamp_signed_epi32_avx512<BIT_WIDTH>(mm512_quantize_pd_epi32(source1, scale_v));
            const __m256i quantized2 = mm256_clamp_signed_epi32_avx512<BIT_WIDTH>(mm512_quantize_pd_epi32(source2, scale_v));

            const __m512i packed = m512::mm512_pack_epi32_avx512vbmi_17to24<BIT_WIDTH>(make_m512i_from_2x256(quantized1, quantized2));
            mm512_storeu_elements_epi16(output, BIT_WIDTH, packed);

            input += 16;
            output += 2 * BIT_WIDTH;
        }
    }

    if constexpr (iterations_8 > 0) {
        const __m512d source    = _mm512_loadu_pd(input);
        const __m256i quantized = mm256_clamp_signed_epi32_avx512<BIT_WIDTH>(mm512_quantize_pd_epi32(source, scale_v));

        const __m256i packed = m256::mm256_pack_epi32_avx512vbmi_17to24<BIT_WIDTH>(quantized);
        mm256_storeu_elements_epi8(output, BIT_WIDTH, packed);

        input += 8;
        output += BIT_WIDTH;
    }

    if constexpr (remaining_elements > 0) {
        const __m512d source    = mm512_loadu_elements_pd(remaining_elements, input);
        const __m256i quantized = mm256_clamp_signed_epi32_avx512<BIT_WIDTH>(mm512_quantize_pd_epi32(source, scale_v));
        const __m256i packed    = m256::mm256_pack_epi32_avx512vbmi_17to24<BIT_WIDTH>(quantized);

        mm256_storeu_elements_epi8(output, tail_bytes(BIT_WIDTH, remaining_elements), packed);
    }

    return 0;
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
template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm512_compress_block_avx512vbmi(const void* __restrict__ input_ptr, const f32 scale, void* __restrict__ output_ptr) {
    const auto* input = static_cast<const float*>(input_ptr);
    auto* output      = static_cast<u8*>(output_ptr);

    std::memset(output, 0, BLOCK_SIZE);

    if constexpr (BIT_WIDTH <= 8) {
        return internal::mm512_compress_block_avx512vbmi_1to8<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH <= 16) {
        return internal::mm512_compress_block_avx512vbmi_9to16<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    } else {
        return internal::mm512_compress_block_avx512vbmi_17to24<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    }
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
template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm512_compress_block_avx512vbmi(const void* __restrict__ input_ptr, const f64 scale, void* __restrict__ output_ptr) {
    const auto* input = static_cast<const double*>(input_ptr);
    auto* output      = static_cast<u8*>(output_ptr);

    std::memset(output, 0, BLOCK_SIZE);

    if constexpr (BIT_WIDTH <= 8) {
        return internal::mm512_compress_block_avx512vbmi_1to8<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH <= 16) {
        return internal::mm512_compress_block_avx512vbmi_9to16<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    } else {
        return internal::mm512_compress_block_avx512vbmi_17to24<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    }
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
template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm512_compress_blocks_avx512vbmi(const void* __restrict__ input_ptr, const f32 scale, void* __restrict__ output_ptr, const u32 blocks) {
    const auto* input = static_cast<const float*>(input_ptr);
    auto* output      = static_cast<u8*>(output_ptr);

    const f32* block_input = input;
    u8* block_output       = output;

    for (u32 block = 0; block < blocks; ++block) {
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
template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm512_compress_blocks_avx512vbmi(const void* __restrict__ input_ptr, const f64 scale, void* __restrict__ output_ptr, const u32 blocks) {
    const auto* input = static_cast<const double*>(input_ptr);
    auto* output      = static_cast<u8*>(output_ptr);

    const f64* block_input = input;
    u8* block_output       = output;

    for (u32 block = 0; block < blocks; ++block) {
        mm512_compress_block_avx512vbmi<BIT_WIDTH, BLOCK_SIZE>(block_input, scale, block_output);
        block_input += (BLOCK_SIZE * 8) / BIT_WIDTH;
        block_output += BLOCK_SIZE;
    }

    return 0;
}
}  // namespace pernix

#endif  // PERNIX_AVX512VBMI_COMPRESSION_H
