#ifndef PERNIX_AVX512VBMI_COMPRESSION_H
#define PERNIX_AVX512VBMI_COMPRESSION_H

#include <pernix/simd_compat.h>
#include <pernix/x86/avx2/avx2_compression.h>
#include <pernix/x86/avx512vbmi/compat.h>
#include <pernix/x86/avx512vbmi/packing.h>
#include <pernix/x86/utils.h>

#include <cstddef>
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

/**
 * @brief Quantize sixteen float values to 32-bit integers.
 */

static __always_inline __m512i mm512_quantize_ps_epi32(__m512 input, __m512 scale) {
    const __m512 scaled = _mm512_mul_ps(input, scale);
    return _mm512_cvtps_epi32(scaled);
}

static __always_inline __m512i mm512_quantize_pd_epi64(__m512d input, __m512d scale) {
    const __m512d scaled = _mm512_mul_pd(input, scale);
    return _mm512_cvtpd_epi64(scaled);
}

static __always_inline __m256i mm512_quantize_pd_epi32(__m512d input, __m512d scale) {
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

static __always_inline __m256i make_m256i_from_2x128(const __m128i a, const __m128i b) {
    __m256i result = _mm256_castsi128_si256(a);
    result         = _mm256_inserti128_si256(result, b, 1);
    return result;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
__always_inline void clear_unused_packed_bits(u8* output) {
    constexpr u32 used_bits       = ((BLOCK_SIZE * 8) / BIT_WIDTH) * BIT_WIDTH;
    constexpr u32 complete_bytes  = used_bits / 8;
    constexpr u32 remaining_bits  = used_bits % 8;
    constexpr u32 first_zero_byte = complete_bytes + (remaining_bits != 0 ? 1 : 0);

    if constexpr (remaining_bits != 0) {
        output[complete_bytes] &= static_cast<u8>((1U << remaining_bits) - 1U);
    }
    if constexpr (first_zero_byte < BLOCK_SIZE) {
        std::memset(output + first_zero_byte, 0, BLOCK_SIZE - first_zero_byte);
    }
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 7) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_compress_block_avx512vbmi_1to7(const f32* __restrict__ input, const f32 scale, u8* __restrict__ output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr u32 iterations_64      = elements_per_block / 64;
    constexpr u32 iterations_32      = (elements_per_block % 64) / 32;
    constexpr u32 iterations_16      = (elements_per_block % 32) / 16;
    constexpr u32 remaining_elements = elements_per_block - (iterations_64 * 64) - (iterations_32 * 32) - (iterations_16 * 16);

    const __m512 scale_v      = _mm512_set1_ps(scale);
    const auto process_single = [scale_v](const f32* source) {
        const __m512 values     = _mm512_loadu_ps(source);
        const __m512i quantized = mm512_clamp_signed_epi32<BIT_WIDTH>(mm512_quantize_ps_epi32(values, scale_v));
        return _mm512_cvtepi32_epi8(quantized);
    };
    const auto process_pair = [&process_single](const f32* source) {
        return make_m256i_from_2x128(process_single(source), process_single(source + 16));
    };

    if constexpr (iterations_64 > 0) {
#pragma GCC unroll 8
        for (u32 iter = 0; iter < iterations_64; ++iter) {
            const __m512i packed_input = make_m512i_from_2x256(process_pair(input), process_pair(input + 32));
            const __m512i packed       = m512::mm512_pack_epi8_avx512vbmi_1to8<BIT_WIDTH>(packed_input);

            mm512_storeu_elements_epi64(output, BIT_WIDTH, packed);

            input += 64;
            output += static_cast<ptrdiff_t>(8 * BIT_WIDTH);
        }
    }

    if constexpr (iterations_32 > 0) {
        const __m256i packed = m256::mm256_pack_epi8_avx512vbmi_1to8<BIT_WIDTH>(process_pair(input));
        mm256_storeu_elements_epi32(output, BIT_WIDTH, packed);

        input += 32;
        output += static_cast<ptrdiff_t>(4 * BIT_WIDTH);
    }

    if constexpr (iterations_16 > 0) {
        const __m128i packed = m128::mm_pack_epi8_avx512vbmi_1to8<BIT_WIDTH>(process_single(input));
        mm_storeu_elements_epi16(output, BIT_WIDTH, packed);

        input += 16;
        output += static_cast<ptrdiff_t>(2 * BIT_WIDTH);
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
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 15) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_compress_block_avx512vbmi_9to15(const f32* __restrict__ input, const f32 scale, u8* __restrict__ output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr u32 iterations_32      = elements_per_block / 32;
    constexpr u32 iterations_16      = (elements_per_block % 32) / 16;
    constexpr u32 iterations_8       = (elements_per_block % 16) / 8;
    constexpr u32 remaining_elements = elements_per_block - (iterations_32 * 32) - (iterations_16 * 16) - (iterations_8 * 8);

    const __m512 scale_v      = _mm512_set1_ps(scale);
    const __m256 scale_v256   = _mm256_set1_ps(scale);
    const auto process_single = [scale_v](const f32* source) {
        const __m512 values     = _mm512_loadu_ps(source);
        const __m512i quantized = mm512_clamp_signed_epi32<BIT_WIDTH>(mm512_quantize_ps_epi32(values, scale_v));
        return _mm512_cvtepi32_epi16(quantized);
    };
    const auto process_pair = [&process_single](const f32* source) {
        return make_m512i_from_2x256(process_single(source), process_single(source + 16));
    };

    if constexpr (iterations_32 > 0) {
#pragma GCC unroll 4
        for (u32 iter = 0; iter < iterations_32; ++iter) {
            const __m512i packed = m512::mm512_pack_epi16_avx512vbmi_9to16<BIT_WIDTH>(process_pair(input));
            mm512_storeu_elements_epi32(output, BIT_WIDTH, packed);

            input += 32;
            output += static_cast<ptrdiff_t>(4 * BIT_WIDTH);
        }
    }

    if constexpr (iterations_16 > 0) {
        const __m256i packed = m256::mm256_pack_epi16_avx512vbmi_9to16<BIT_WIDTH>(process_single(input));
        mm256_storeu_elements_epi16(output, BIT_WIDTH, packed);

        input += 16;
        output += static_cast<ptrdiff_t>(2 * BIT_WIDTH);
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
    constexpr u32 remaining_elements = elements_per_block - (iterations_16 * 16) - (iterations_8 * 8);

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
            output += static_cast<ptrdiff_t>(2 * BIT_WIDTH);
        }
    }

    if constexpr (iterations_8 > 0) {
        const __m256 source        = _mm256_loadu_ps(input);
        const __m256i packed_input = mm256_clamp_signed_epi32_avx512<BIT_WIDTH>(mm256_quantize_ps_epi32(source, scale_v256));

        const __m256i packed = m256::mm256_pack_epi32_avx512vbmi_17to24<BIT_WIDTH>(packed_input);
        mm256_storeu_elements_epi8(output, BIT_WIDTH, packed);

        input += 8;
        output += static_cast<ptrdiff_t>(BIT_WIDTH);
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
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 7) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_compress_block_avx512vbmi_1to7(const f64* __restrict__ input, const f64 scale, u8* __restrict__ output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr u32 iterations_64      = elements_per_block / 64;
    constexpr u32 iterations_32      = (elements_per_block % 64) / 32;
    constexpr u32 iterations_16      = (elements_per_block % 32) / 16;
    constexpr u32 remaining_elements = elements_per_block - (iterations_64 * 64) - (iterations_32 * 32) - (iterations_16 * 16);

    const __m512d scale_v     = _mm512_set1_pd(scale);
    const auto process_single = [scale_v](const f64* source, const u32 elements = 8) {
        const __m512d values    = mm512_loadu_elements_pd(elements, source);
        const __m256i quantized = mm256_clamp_signed_epi32_avx512<BIT_WIDTH>(mm512_quantize_pd_epi32(values, scale_v));
        return _mm256_cvtepi32_epi8(quantized);
    };
    const auto process_pair = [&process_single](const f64* source) {
        return _mm_unpacklo_epi64(process_single(source), process_single(source + 8));
    };

    if constexpr (iterations_64 > 0) {
#pragma GCC unroll 8
        for (u32 iter = 0; iter < iterations_64; ++iter) {
            const __m512i packed_input =
                make_m512i_from_4x128(process_pair(input), process_pair(input + 16), process_pair(input + 32), process_pair(input + 48));
            const __m512i packed = m512::mm512_pack_epi8_avx512vbmi_1to8<BIT_WIDTH>(packed_input);

            mm512_storeu_elements_epi64(output, BIT_WIDTH, packed);

            input += 64;
            output += static_cast<ptrdiff_t>(8 * BIT_WIDTH);
        }
    }

    if constexpr (iterations_32 > 0) {
        const __m256i packed_input = make_m256i_from_2x128(process_pair(input), process_pair(input + 16));
        const __m256i packed       = m256::mm256_pack_epi8_avx512vbmi_1to8<BIT_WIDTH>(packed_input);

        mm256_storeu_elements_epi32(output, BIT_WIDTH, packed);

        input += 32;
        output += static_cast<ptrdiff_t>(4 * BIT_WIDTH);
    }

    if constexpr (iterations_16 > 0) {
        const __m128i packed = m128::mm_pack_epi8_avx512vbmi_1to8<BIT_WIDTH>(process_pair(input));

        mm_storeu_elements_epi16(output, BIT_WIDTH, packed);

        input += 16;
        output += static_cast<ptrdiff_t>(2 * BIT_WIDTH);
    }

    if constexpr (remaining_elements > 0) {
        constexpr u32 source1_elements = remaining_elements > 8 ? 8 : remaining_elements;
        constexpr u32 source2_elements = remaining_elements > 8 ? remaining_elements - 8 : 0;

        const __m128i packed_input =
            _mm_unpacklo_epi64(process_single(input, source1_elements), process_single(input + 8, source2_elements));
        const __m128i packed = m128::mm_pack_epi8_avx512vbmi_1to8<BIT_WIDTH>(packed_input);

        mm_storeu_elements_epi8(output, tail_bytes(BIT_WIDTH, remaining_elements), packed);
    }

    return 0;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 15) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_compress_block_avx512vbmi_9to15(const f64* __restrict__ input, const f64 scale, u8* __restrict__ output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr u32 iterations_32      = elements_per_block / 32;
    constexpr u32 iterations_16      = (elements_per_block % 32) / 16;
    constexpr u32 iterations_8       = (elements_per_block % 16) / 8;
    constexpr u32 remaining_elements = elements_per_block - (iterations_32 * 32) - (iterations_16 * 16) - (iterations_8 * 8);

    const __m512d scale_v     = _mm512_set1_pd(scale);
    const auto process_single = [scale_v](const f64* source, const u32 elements = 8) {
        const __m512d values    = mm512_loadu_elements_pd(elements, source);
        const __m256i quantized = mm256_clamp_signed_epi32_avx512<BIT_WIDTH>(mm512_quantize_pd_epi32(values, scale_v));
        return _mm256_cvtepi32_epi16(quantized);
    };
    const auto process_pair = [&process_single](const f64* source) {
        return make_m256i_from_2x128(process_single(source), process_single(source + 8));
    };

    if constexpr (iterations_32 > 0) {
#pragma GCC unroll 4
        for (u32 iter = 0; iter < iterations_32; ++iter) {
            const __m512i packed_input = make_m512i_from_2x256(process_pair(input), process_pair(input + 16));
            const __m512i packed       = m512::mm512_pack_epi16_avx512vbmi_9to16<BIT_WIDTH>(packed_input);

            mm512_storeu_elements_epi32(output, BIT_WIDTH, packed);

            input += 32;
            output += static_cast<ptrdiff_t>(4 * BIT_WIDTH);
        }
    }

    if constexpr (iterations_16 > 0) {
        const __m256i packed = m256::mm256_pack_epi16_avx512vbmi_9to16<BIT_WIDTH>(process_pair(input));

        mm256_storeu_elements_epi16(output, BIT_WIDTH, packed);

        input += 16;
        output += static_cast<ptrdiff_t>(2 * BIT_WIDTH);
    }

    if constexpr (iterations_8 > 0) {
        const __m128i packed = m128::mm_pack_epi16_avx512vbmi_9to16<BIT_WIDTH>(process_single(input));
        mm_storeu_elements_epi8(output, BIT_WIDTH, packed);

        input += 8;
        output += static_cast<ptrdiff_t>(BIT_WIDTH);
    }

    if constexpr (remaining_elements > 0) {
        const __m128i packed = m128::mm_pack_epi16_avx512vbmi_9to16<BIT_WIDTH>(process_single(input, remaining_elements));
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
    constexpr u32 remaining_elements = elements_per_block - (iterations_16 * 16) - (iterations_8 * 8);

    const __m512d scale_v     = _mm512_set1_pd(scale);
    const auto process_single = [scale_v](const f64* source) {
        const __m512d values = _mm512_loadu_pd(source);
        return mm256_clamp_signed_epi32_avx512<BIT_WIDTH>(mm512_quantize_pd_epi32(values, scale_v));
    };
    const auto process_pair = [&process_single](const f64* source) {
        return make_m512i_from_2x256(process_single(source), process_single(source + 8));
    };

    if constexpr (iterations_16 > 0) {
#pragma GCC unroll 2
        for (u32 i = 0; i < iterations_16; ++i) {
            const __m512i packed = m512::mm512_pack_epi32_avx512vbmi_17to24<BIT_WIDTH>(process_pair(input));
            mm512_storeu_elements_epi16(output, BIT_WIDTH, packed);

            input += 16;
            output += static_cast<ptrdiff_t>(2 * BIT_WIDTH);
        }
    }

    if constexpr (iterations_8 > 0) {
        const __m256i packed = m256::mm256_pack_epi32_avx512vbmi_17to24<BIT_WIDTH>(process_single(input));
        mm256_storeu_elements_epi8(output, BIT_WIDTH, packed);

        input += 8;
        output += static_cast<ptrdiff_t>(BIT_WIDTH);
    }

    if constexpr (remaining_elements > 0) {
        const __m512d source    = mm512_loadu_elements_pd(remaining_elements, input);
        const __m256i quantized = mm256_clamp_signed_epi32_avx512<BIT_WIDTH>(mm512_quantize_pd_epi32(source, scale_v));
        const __m256i packed    = m256::mm256_pack_epi32_avx512vbmi_17to24<BIT_WIDTH>(quantized);

        mm256_storeu_elements_epi8(output, tail_bytes(BIT_WIDTH, remaining_elements), packed);
    }

    return 0;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH == 8) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_compress_block_avx512vbmi_8(const f32* __restrict__ input, const f32 scale, u8* __restrict__ output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;
    constexpr u32 iterations         = elements_per_block / 16;

    const __m512 scale_v = _mm512_set1_ps(scale);

#pragma GCC unroll 32
    for (u32 i = 0; i < iterations; ++i) {
        const __m512 source     = _mm512_loadu_ps(input);
        const __m512i quantized = mm512_quantize_ps_epi32(source, scale_v);
        const __m128i converted = _mm512_cvtepi32_epi8(quantized);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output), converted);

        input += 16;
        output += 16;
    }

    return 0;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH == 8) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_compress_block_avx512vbmi_8(const f64* __restrict__ input, const f64 scale, u8* __restrict__ output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;
    constexpr u32 iterations         = elements_per_block / 8;

    const __m512d scale_v = _mm512_set1_pd(scale);

#pragma GCC unroll 32
    for (u32 i = 0; i < iterations; ++i) {
        const __m512d source    = _mm512_loadu_pd(input);
        const __m256i quantized = mm512_quantize_pd_epi32(source, scale_v);
        const __m128i converted = _mm256_cvtepi32_epi8(quantized);
        _mm_storeu_si64(output, converted);

        input += 8;
        output += 8;
    }

    return 0;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH == 16) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_compress_block_avx512vbmi_16(const f32* __restrict__ input, const f32 scale, u8* __restrict__ output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;
    constexpr u32 iterations         = elements_per_block / 16;

    const __m512 scale_v = _mm512_set1_ps(scale);

#pragma GCC unroll 16
    for (u32 i = 0; i < iterations; ++i) {
        const __m512 source     = _mm512_loadu_ps(input);
        const __m512i quantized = mm512_quantize_ps_epi32(source, scale_v);
        const __m256i converted = _mm512_cvtepi32_epi16(quantized);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output), converted);

        input += 16;
        output += 32;
    }

    return 0;
}

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH == 16) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm512_compress_block_avx512vbmi_16(const f64* __restrict__ input, const f64 scale, u8* __restrict__ output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;
    constexpr u32 iterations         = elements_per_block / 8;

    const __m512d scale_v = _mm512_set1_pd(scale);

#pragma GCC unroll 16
    for (u32 i = 0; i < iterations; ++i) {
        const __m512d source    = _mm512_loadu_pd(input);
        const __m512i quantized = mm512_quantize_pd_epi64(source, scale_v);
        const __m128i converted = _mm512_cvtepi64_epi16(quantized);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output), converted);

        input += 8;
        output += 16;
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

    int status;
    if constexpr (BIT_WIDTH <= 7) {
        status = internal::mm512_compress_block_avx512vbmi_1to7<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH == 8) {
        status = internal::mm512_compress_block_avx512vbmi_8<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH <= 15) {
        status = internal::mm512_compress_block_avx512vbmi_9to15<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH == 16) {
        status = internal::mm512_compress_block_avx512vbmi_16<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    } else {
        status = internal::mm512_compress_block_avx512vbmi_17to24<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    }

    internal::clear_unused_packed_bits<BIT_WIDTH, BLOCK_SIZE>(output);
    return status;
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

    int status;
    if constexpr (BIT_WIDTH <= 7) {
        status = internal::mm512_compress_block_avx512vbmi_1to7<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH == 8) {
        status = internal::mm512_compress_block_avx512vbmi_8<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH <= 15) {
        status = internal::mm512_compress_block_avx512vbmi_9to15<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH == 16) {
        status = internal::mm512_compress_block_avx512vbmi_16<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    } else {
        status = internal::mm512_compress_block_avx512vbmi_17to24<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
    }

    internal::clear_unused_packed_bits<BIT_WIDTH, BLOCK_SIZE>(output);
    return status;
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
