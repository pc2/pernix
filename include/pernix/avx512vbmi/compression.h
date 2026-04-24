#ifndef PERNIX_AVX512VBMI_COMPRESSION_H
#define PERNIX_AVX512VBMI_COMPRESSION_H

#include <pernix/avx2/compression.h>
#include <pernix/avx512vbmi/packing.h>
#include <pernix/simd_compat.h>

namespace pernix {

namespace internal {
/**
 * @brief Quantize sixteen float values to 32-bit integers.
 */
[[gnu::always_inline]]
static inline __m512i mm512_quantize_ps_epi32(const __m512& input, const __m512& scale) {
    const __m512 scaled = _mm512_mul_ps(input, scale);
    return _mm512_cvtps_epi32(scaled);
}

[[gnu::always_inline]]
static inline __m512i mm512_quantize_pd_epi64(const __m512d& input, const __m512d& scale) {
    const __m512d scaled = _mm512_mul_pd(input, scale);
    return _mm512_cvtpd_epi64(scaled);
}

[[gnu::always_inline]]
static inline __m256i mm512_quantize_pd_epi32(const __m512d& input, const __m512d& scale) {
    const __m512d scaled = _mm512_mul_pd(input, scale);
    return _mm512_cvtpd_epi32(scaled);
}

[[gnu::always_inline]] static inline __m512i make_m512i_from_2x256(const __m256i a, const __m256i b) {
    __m512i result = _mm512_castsi256_si512(a);
    result         = _mm512_inserti64x4(result, b, 1);
    return result;
}

[[gnu::always_inline]]
static inline __m512i make_m512i_from_4x128(const __m128i a, const __m128i b, const __m128i c, const __m128i d) {
    __m512i result = _mm512_castsi128_si512(a);
    result         = _mm512_inserti64x2(result, b, 1);
    result         = _mm512_inserti64x2(result, c, 2);
    result         = _mm512_inserti64x2(result, d, 3);
    return result;
}

[[gnu::always_inline]]
static inline __m512i make_m512i_from_8x64(const __m128i a, const __m128i b, const __m128i c, const __m128i d, const __m128i e,
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

[[gnu::always_inline]]
static inline __m256i make_m256i_from_2x128(const __m128i a, const __m128i b) {
    __m256i result = _mm256_castsi128_si256(a);
    result         = _mm256_inserti128_si256(result, b, 1);
    return result;
}

[[gnu::always_inline]]
static inline __m256i make_m256i_from_4x64(const __m128i a, const __m128i b, const __m128i c, const __m128i d) {
    const __m128i ab = _mm_unpacklo_epi64(a, b);
    const __m128i cd = _mm_unpacklo_epi64(c, d);

    __m256i x = _mm256_castsi128_si256(ab);
    x         = _mm256_inserti128_si256(x, cd, 1);
    return x;
}

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8) && (BLOCK_SIZE % 32 == 0)
[[gnu::always_inline]] inline int mm512_compress_block_avx512vbmi_1to8(const float_t* __restrict__ input, const float_t scale,
                                                                       uint8_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr uint32_t iterations_64      = elements_per_block / 64;
    constexpr uint32_t iterations_32      = (elements_per_block % 64) / 32;
    constexpr uint32_t iterations_16      = (elements_per_block % 32) / 16;
    constexpr uint32_t remaining_elements = elements_per_block - iterations_64 * 64 - iterations_32 * 32 - iterations_16 * 16;

    const __m512 scale_v = _mm512_set1_ps(scale);

    if constexpr (iterations_64 > 0) {
#pragma GCC unroll 8
        for (uint32_t iter = 0; iter < iterations_64; ++iter) {
            const __m512 source1 = _mm512_loadu_ps(input);
            const __m512 source2 = _mm512_loadu_ps(input + 16);
            const __m512 source3 = _mm512_loadu_ps(input + 32);
            const __m512 source4 = _mm512_loadu_ps(input + 48);

            const __m512i quantized1 = mm512_quantize_ps_epi32(source1, scale_v);
            const __m512i quantized2 = mm512_quantize_ps_epi32(source2, scale_v);
            const __m512i quantized3 = mm512_quantize_ps_epi32(source3, scale_v);
            const __m512i quantized4 = mm512_quantize_ps_epi32(source4, scale_v);

            const __m128i converted1 = _mm512_cvtepi32_epi8(quantized1);
            const __m128i converted2 = _mm512_cvtepi32_epi8(quantized2);
            const __m128i converted3 = _mm512_cvtepi32_epi8(quantized3);
            const __m128i converted4 = _mm512_cvtepi32_epi8(quantized4);

            const __m512i packed =
                m512::mm512_pack_epi8_avx512vbmi_1to8<BIT_WIDTH>(make_m512i_from_4x128(converted1, converted2, converted3, converted4));

            _mm512_mask_storeu_epi64(output, (1u << BIT_WIDTH) - 1u, packed);

            input += 64;
            output += 8 * BIT_WIDTH;
        }
    }

    if constexpr (iterations_32 > 0) {
        const __m512 source1 = _mm512_loadu_ps(input);
        const __m512 source2 = _mm512_loadu_ps(input + 16);

        const __m512i quantized1 = mm512_quantize_ps_epi32(source1, scale_v);
        const __m512i quantized2 = mm512_quantize_ps_epi32(source2, scale_v);

        const __m128i converted1 = _mm512_cvtepi32_epi8(quantized1);
        const __m128i converted2 = _mm512_cvtepi32_epi8(quantized2);

        const __m256i packed = m256::mm256_pack_epi8_avx512vbmi_1to8<BIT_WIDTH>(make_m256i_from_2x128(converted1, converted2));
        _mm256_mask_storeu_epi32(output, (1u << BIT_WIDTH) - 1u, packed);

        input += 32;
        output += 4 * BIT_WIDTH;
    }

    if constexpr (iterations_16 > 0) {
        const __m512 source     = _mm512_loadu_ps(input);
        const __m512i quantized = mm512_quantize_ps_epi32(source, scale_v);
        const __m128i converted = _mm512_cvtepi32_epi8(quantized);

        const __m128i packed = m128::mm_pack_epi8_avx512vbmi_1to8<BIT_WIDTH>(converted);
        _mm_mask_storeu_epi16(output, (1u << BIT_WIDTH) - 1u, packed);

        input += 16;
        output += 2 * BIT_WIDTH;
    }

    if constexpr (remaining_elements > 0) {
        const __m512 source     = _mm512_maskz_loadu_ps((1u << remaining_elements) - 1u, input);
        const __m512i quantized = mm512_quantize_ps_epi32(source, scale_v);
        const __m128i converted = _mm512_cvtepi32_epi8(quantized);
        const __m128i packed    = m128::mm_pack_epi8_avx512vbmi_1to8<BIT_WIDTH>(converted);

        _mm_mask_storeu_epi8(output, tail_mask<__mmask16>(BIT_WIDTH, remaining_elements), packed);
    }

    return 0;
}

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16) && (BLOCK_SIZE % 32 == 0)
[[gnu::always_inline]] inline int mm512_compress_block_avx512vbmi_9to16(const float_t* __restrict__ input, const float_t scale,
                                                                        uint8_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr uint32_t iterations_32      = elements_per_block / 32;
    constexpr uint32_t iterations_16      = (elements_per_block % 32) / 16;
    constexpr uint32_t iterations_8       = (elements_per_block % 16) / 8;
    constexpr uint32_t remaining_elements = elements_per_block - iterations_32 * 32 - iterations_16 * 16 - iterations_8 * 8;

    const __m512 scale_v    = _mm512_set1_ps(scale);
    const __m256 scale_v256 = _mm256_set1_ps(scale);

    if constexpr (iterations_32 > 0) {
#pragma GCC unroll 4
        for (uint32_t iter = 0; iter < iterations_32; ++iter) {
            const __m512 source1 = _mm512_loadu_ps(input);
            const __m512 source2 = _mm512_loadu_ps(input + 16);

            const __m512i quantized1 = mm512_quantize_ps_epi32(source1, scale_v);
            const __m512i quantized2 = mm512_quantize_ps_epi32(source2, scale_v);

            const __m256i converted1 = _mm512_cvtepi32_epi16(quantized1);
            const __m256i converted2 = _mm512_cvtepi32_epi16(quantized2);

            const __m512i packed = m512::mm512_pack_epi16_avx512vbmi_9to16<BIT_WIDTH>(make_m512i_from_2x256(converted1, converted2));
            _mm512_mask_storeu_epi32(output, (1u << BIT_WIDTH) - 1u, packed);

            input += 32;
            output += 4 * BIT_WIDTH;
        }
    }

    if constexpr (iterations_16 > 0) {
        const __m512 source     = _mm512_loadu_ps(input);
        const __m512i quantized = mm512_quantize_ps_epi32(source, scale_v);
        const __m256i converted = _mm512_cvtepi32_epi16(quantized);

        const __m256i packed = m256::mm256_pack_epi16_avx512vbmi_9to16<BIT_WIDTH>(converted);
        _mm256_mask_storeu_epi16(output, (1u << BIT_WIDTH) - 1u, packed);

        input += 16;
        output += 2 * BIT_WIDTH;
    }

    if constexpr (iterations_8 > 0) {
        const __m256 source     = _mm256_loadu_ps(input);
        const __m256i quantized = mm256_quantize_ps_epi32(source, scale_v256);
        const __m128i converted = _mm256_cvtepi32_epi16(quantized);

        const __m128i packed = m128::mm_pack_epi16_avx512vbmi_9to16<BIT_WIDTH>(converted);
        _mm_mask_storeu_epi8(output, (1u << BIT_WIDTH) - 1u, packed);

        input += 8;
        output += BIT_WIDTH;
    }

    if constexpr (remaining_elements > 0) {
        const __m256 source     = _mm256_maskz_loadu_ps((1u << remaining_elements) - 1u, input);
        const __m256i quantized = mm256_quantize_ps_epi32(source, scale_v256);
        const __m128i converted = _mm256_cvtepi32_epi16(quantized);
        const __m128i packed    = m128::mm_pack_epi16_avx512vbmi_9to16<BIT_WIDTH>(converted);

        _mm_mask_storeu_epi8(output, tail_mask<__mmask16>(BIT_WIDTH, remaining_elements), packed);
    }

    return 0;
}

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
[[gnu::always_inline]] inline int mm512_compress_block_avx512vbmi_17to24(const float_t* __restrict__ input, const float_t scale,
                                                                         uint8_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr uint32_t iterations_16      = elements_per_block / 16;
    constexpr uint32_t iterations_8       = (elements_per_block % 16) / 8;
    constexpr uint32_t remaining_elements = elements_per_block - iterations_16 * 16 - iterations_8 * 8;

    const __m512 scale_v    = _mm512_set1_ps(scale);
    const __m256 scale_v256 = _mm256_set1_ps(scale);

    if constexpr (iterations_16 > 0) {
#pragma GCC unroll 2
        for (uint32_t i = 0; i < iterations_16; ++i) {
            const __m512 source        = _mm512_loadu_ps(input);
            const __m512i quantized    = mm512_quantize_ps_epi32(source, scale_v);
            const __m512i packed_input = [&]() {
                if constexpr (BIT_WIDTH == 24) {
                    constexpr int32_t min_value = -(1 << (BIT_WIDTH - 1));
                    constexpr int32_t max_value = (1 << (BIT_WIDTH - 1)) - 1;
                    return _mm512_min_epi32(_mm512_max_epi32(quantized, _mm512_set1_epi32(min_value)), _mm512_set1_epi32(max_value));
                } else {
                    return quantized;
                }
            }();

            const __m512i packed = m512::mm512_pack_epi32_avx512vbmi_17to24<BIT_WIDTH>(packed_input);
            _mm512_mask_storeu_epi16(output, (1ull << BIT_WIDTH) - 1ull, packed);
            input += 16;
            output += 2 * BIT_WIDTH;
        }
    }

    if constexpr (iterations_8 > 0) {
        const __m256 source        = _mm256_loadu_ps(input);
        const __m256i quantized    = mm256_quantize_ps_epi32(source, scale_v256);
        const __m256i packed_input = [&]() {
            if constexpr (BIT_WIDTH == 24) {
                constexpr int32_t min_value = -(1 << (BIT_WIDTH - 1));
                constexpr int32_t max_value = (1 << (BIT_WIDTH - 1)) - 1;
                return _mm256_min_epi32(_mm256_max_epi32(quantized, _mm256_set1_epi32(min_value)), _mm256_set1_epi32(max_value));
            } else {
                return quantized;
            }
        }();

        const __m256i packed = m256::mm256_pack_epi32_avx512vbmi_17to24<BIT_WIDTH>(packed_input);
        _mm256_mask_storeu_epi8(output, (1u << BIT_WIDTH) - 1u, packed);

        input += 8;
        output += BIT_WIDTH;
    }

    if constexpr (remaining_elements > 0) {
        const __m256 source        = _mm256_maskz_loadu_ps((1u << remaining_elements) - 1u, input);
        const __m256i quantized    = mm256_quantize_ps_epi32(source, scale_v256);
        const __m256i packed_input = [&]() {
            if constexpr (BIT_WIDTH == 24) {
                constexpr int32_t min_value = -(1 << (BIT_WIDTH - 1));
                constexpr int32_t max_value = (1 << (BIT_WIDTH - 1)) - 1;
                return _mm256_min_epi32(_mm256_max_epi32(quantized, _mm256_set1_epi32(min_value)), _mm256_set1_epi32(max_value));
            } else {
                return quantized;
            }
        }();
        const __m256i packed = m256::mm256_pack_epi32_avx512vbmi_17to24<BIT_WIDTH>(packed_input);

        _mm256_mask_storeu_epi8(output, tail_mask<__mmask32>(BIT_WIDTH, remaining_elements), packed);
    }

    return 0;
}

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8) && (BLOCK_SIZE % 32 == 0)
[[gnu::always_inline]] inline int mm512_compress_block_avx512vbmi_1to8(const double_t* __restrict__ input, const double_t scale,
                                                                       uint8_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr uint32_t iterations_64      = elements_per_block / 64;
    constexpr uint32_t iterations_32      = (elements_per_block % 64) / 32;
    constexpr uint32_t iterations_16      = (elements_per_block % 32) / 16;
    constexpr uint32_t remaining_elements = elements_per_block - iterations_64 * 64 - iterations_32 * 32 - iterations_16 * 16;

    const __m512d scale_v = _mm512_set1_pd(scale);

    if constexpr (iterations_64 > 0) {
#pragma GCC unroll 8
        for (uint32_t iter = 0; iter < iterations_64; ++iter) {
            const __m512d source1 = _mm512_loadu_pd(input);
            const __m512d source2 = _mm512_loadu_pd(input + 8);
            const __m512d source3 = _mm512_loadu_pd(input + 16);
            const __m512d source4 = _mm512_loadu_pd(input + 24);
            const __m512d source5 = _mm512_loadu_pd(input + 32);
            const __m512d source6 = _mm512_loadu_pd(input + 40);
            const __m512d source7 = _mm512_loadu_pd(input + 48);
            const __m512d source8 = _mm512_loadu_pd(input + 56);

            const __m512i quantized1 = mm512_quantize_pd_epi64(source1, scale_v);
            const __m512i quantized2 = mm512_quantize_pd_epi64(source2, scale_v);
            const __m512i quantized3 = mm512_quantize_pd_epi64(source3, scale_v);
            const __m512i quantized4 = mm512_quantize_pd_epi64(source4, scale_v);
            const __m512i quantized5 = mm512_quantize_pd_epi64(source5, scale_v);
            const __m512i quantized6 = mm512_quantize_pd_epi64(source6, scale_v);
            const __m512i quantized7 = mm512_quantize_pd_epi64(source7, scale_v);
            const __m512i quantized8 = mm512_quantize_pd_epi64(source8, scale_v);

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

            _mm512_mask_storeu_epi64(output, (1u << BIT_WIDTH) - 1u, packed);

            input += 64;
            output += 8 * BIT_WIDTH;
        }
    }

    if constexpr (iterations_32 > 0) {
        const __m512d source1 = _mm512_loadu_pd(input);
        const __m512d source2 = _mm512_loadu_pd(input + 8);
        const __m512d source3 = _mm512_loadu_pd(input + 16);
        const __m512d source4 = _mm512_loadu_pd(input + 24);

        const __m512i quantized1 = mm512_quantize_pd_epi64(source1, scale_v);
        const __m512i quantized2 = mm512_quantize_pd_epi64(source2, scale_v);
        const __m512i quantized3 = mm512_quantize_pd_epi64(source3, scale_v);
        const __m512i quantized4 = mm512_quantize_pd_epi64(source4, scale_v);

        const __m128i converted1 = _mm512_cvtepi64_epi8(quantized1);
        const __m128i converted2 = _mm512_cvtepi64_epi8(quantized2);
        const __m128i converted3 = _mm512_cvtepi64_epi8(quantized3);
        const __m128i converted4 = _mm512_cvtepi64_epi8(quantized4);

        const __m256i packed =
            m256::mm256_pack_epi8_avx512vbmi_1to8<BIT_WIDTH>(make_m256i_from_4x64(converted1, converted2, converted3, converted4));

        _mm256_mask_storeu_epi32(output, (1u << BIT_WIDTH) - 1u, packed);

        input += 32;
        output += 4 * BIT_WIDTH;
    }

    if constexpr (iterations_16 > 0) {
        const __m512d source1    = _mm512_loadu_pd(input);
        const __m512d source2    = _mm512_loadu_pd(input + 8);
        const __m512i quantized1 = mm512_quantize_pd_epi64(source1, scale_v);
        const __m512i quantized2 = mm512_quantize_pd_epi64(source2, scale_v);

        const __m128i converted1 = _mm512_cvtepi64_epi8(quantized1);
        const __m128i converted2 = _mm512_cvtepi64_epi8(quantized2);

        const __m128i packed = m128::mm_pack_epi8_avx512vbmi_1to8<BIT_WIDTH>(_mm_unpacklo_epi64(converted1, converted2));

        _mm_mask_storeu_epi16(output, (1u << BIT_WIDTH) - 1u, packed);

        input += 16;
        output += 2 * BIT_WIDTH;
    }

    if constexpr (remaining_elements > 0) {
        constexpr __mmask16 load_mask = (1u << remaining_elements) - 1u;

        const __m512d source1    = _mm512_maskz_loadu_pd(load_mask, input);
        const __m512d source2    = _mm512_maskz_loadu_pd(load_mask >> 8, input + 8);
        const __m512i quantized1 = mm512_quantize_pd_epi64(source1, scale_v);
        const __m512i quantized2 = mm512_quantize_pd_epi64(source2, scale_v);

        const __m128i converted1 = _mm512_cvtepi64_epi8(quantized1);
        const __m128i converted2 = _mm512_cvtepi64_epi8(quantized2);

        const __m128i packed = m128::mm_pack_epi8_avx512vbmi_1to8<BIT_WIDTH>(_mm_unpacklo_epi64(converted1, converted2));

        _mm_mask_storeu_epi8(output, tail_mask<__mmask16>(BIT_WIDTH, remaining_elements), packed);
    }

    return 0;
}

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16) && (BLOCK_SIZE % 32 == 0)
[[gnu::always_inline]] inline int mm512_compress_block_avx512vbmi_9to16(const double_t* __restrict__ input, const double_t scale,
                                                                        uint8_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr uint32_t iterations_32      = elements_per_block / 32;
    constexpr uint32_t iterations_16      = (elements_per_block % 32) / 16;
    constexpr uint32_t iterations_8       = (elements_per_block % 16) / 8;
    constexpr uint32_t remaining_elements = elements_per_block - iterations_32 * 32 - iterations_16 * 16 - iterations_8 * 8;

    const __m512d scale_v = _mm512_set1_pd(scale);

    if constexpr (iterations_32 > 0) {
#pragma GCC unroll 4
        for (uint32_t iter = 0; iter < iterations_32; ++iter) {
            const __m512d source1 = _mm512_loadu_pd(input);
            const __m512d source2 = _mm512_loadu_pd(input + 8);
            const __m512d source3 = _mm512_loadu_pd(input + 16);
            const __m512d source4 = _mm512_loadu_pd(input + 24);

            const __m512i quantized1 = mm512_quantize_pd_epi64(source1, scale_v);
            const __m512i quantized2 = mm512_quantize_pd_epi64(source2, scale_v);
            const __m512i quantized3 = mm512_quantize_pd_epi64(source3, scale_v);
            const __m512i quantized4 = mm512_quantize_pd_epi64(source4, scale_v);

            const __m128i converted1 = _mm512_cvtepi64_epi16(quantized1);
            const __m128i converted2 = _mm512_cvtepi64_epi16(quantized2);
            const __m128i converted3 = _mm512_cvtepi64_epi16(quantized3);
            const __m128i converted4 = _mm512_cvtepi64_epi16(quantized4);

            const __m512i packed =
                m512::mm512_pack_epi16_avx512vbmi_9to16<BIT_WIDTH>(make_m512i_from_4x128(converted1, converted2, converted3, converted4));

            _mm512_mask_storeu_epi32(output, (1u << BIT_WIDTH) - 1u, packed);

            input += 32;
            output += 4 * BIT_WIDTH;
        }
    }

    if constexpr (iterations_16 > 0) {
        const __m512d source1    = _mm512_loadu_pd(input);
        const __m512d source2    = _mm512_loadu_pd(input + 8);
        const __m512i quantized1 = mm512_quantize_pd_epi64(source1, scale_v);
        const __m512i quantized2 = mm512_quantize_pd_epi64(source2, scale_v);

        const __m128i converted1 = _mm512_cvtepi64_epi16(quantized1);
        const __m128i converted2 = _mm512_cvtepi64_epi16(quantized2);

        const __m256i packed = m256::mm256_pack_epi16_avx512vbmi_9to16<BIT_WIDTH>(make_m256i_from_2x128(converted1, converted2));

        _mm256_mask_storeu_epi16(output, (1u << BIT_WIDTH) - 1u, packed);

        input += 16;
        output += 2 * BIT_WIDTH;
    }

    if constexpr (iterations_8 > 0) {
        const __m512d source    = _mm512_loadu_pd(input);
        const __m512i quantized = mm512_quantize_pd_epi64(source, scale_v);
        const __m128i converted = _mm512_cvtepi64_epi16(quantized);

        const __m128i packed = m128::mm_pack_epi16_avx512vbmi_9to16<BIT_WIDTH>(converted);
        _mm_mask_storeu_epi8(output, (1u << BIT_WIDTH) - 1u, packed);

        input += 8;
        output += BIT_WIDTH;
    }

    if constexpr (remaining_elements > 0) {
        const __m512d source    = _mm512_maskz_loadu_pd((1u << remaining_elements) - 1u, input);
        const __m512i quantized = mm512_quantize_pd_epi64(source, scale_v);
        const __m128i converted = _mm512_cvtepi64_epi16(quantized);

        const __m128i packed = m128::mm_pack_epi16_avx512vbmi_9to16<BIT_WIDTH>(converted);
        _mm_mask_storeu_epi8(output, tail_mask<__mmask16>(BIT_WIDTH, remaining_elements), packed);
    }

    return 0;
}

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
[[gnu::always_inline]] inline int mm512_compress_block_avx512vbmi_17to24(const double_t* __restrict__ input, const double_t scale,
                                                                         uint8_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr uint32_t iterations_16      = elements_per_block / 16;
    constexpr uint32_t iterations_8       = (elements_per_block % 16) / 8;
    constexpr uint32_t remaining_elements = elements_per_block - iterations_16 * 16 - iterations_8 * 8;

    const __m512d scale_v = _mm512_set1_pd(scale);

    if constexpr (iterations_16 > 0) {
#pragma GCC unroll 2
        for (uint32_t i = 0; i < iterations_16; ++i) {
            const __m512d source1 = _mm512_loadu_pd(input);
            const __m512d source2 = _mm512_loadu_pd(input + 8);

            const __m256i quantized1 = mm512_quantize_pd_epi32(source1, scale_v);
            const __m256i quantized2 = mm512_quantize_pd_epi32(source2, scale_v);

            const __m512i packed = m512::mm512_pack_epi32_avx512vbmi_17to24<BIT_WIDTH>(make_m512i_from_2x256(quantized1, quantized2));
            _mm512_mask_storeu_epi16(output, (1ull << BIT_WIDTH) - 1ull, packed);

            input += 16;
            output += 2 * BIT_WIDTH;
        }
    }

    if constexpr (iterations_8 > 0) {
        const __m512d source    = _mm512_loadu_pd(input);
        const __m256i quantized = mm512_quantize_pd_epi32(source, scale_v);

        const __m256i packed = m256::mm256_pack_epi32_avx512vbmi_17to24<BIT_WIDTH>(quantized);
        _mm256_mask_storeu_epi8(output, (1u << BIT_WIDTH) - 1u, packed);

        input += 8;
        output += BIT_WIDTH;
    }

    if constexpr (remaining_elements > 0) {
        const __m512d source    = _mm512_maskz_loadu_pd((1u << remaining_elements) - 1u, input);
        const __m256i quantized = mm512_quantize_pd_epi32(source, scale_v);
        const __m256i packed    = m256::mm256_pack_epi32_avx512vbmi_17to24<BIT_WIDTH>(quantized);

        _mm256_mask_storeu_epi8(output, tail_mask<__mmask32>(BIT_WIDTH, remaining_elements), packed);
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
template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm512_compress_block_avx512vbmi(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output) {
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
template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm512_compress_block_avx512vbmi(const double_t* __restrict__ input, const double_t scale, uint8_t* __restrict__ output) {
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
template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm512_compress_blocks_avx512vbmi(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output,
                                     const uint32_t blocks) {
    const float_t* block_input = input;
    uint8_t* block_output      = output;

    for (uint32_t block = 0; block < blocks; ++block) {
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

    for (uint32_t block = 0; block < blocks; ++block) {
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
