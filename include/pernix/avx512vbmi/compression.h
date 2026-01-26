#ifndef PERNIX_AVX512VBMI_COMPRESSION_H
#define PERNIX_AVX512VBMI_COMPRESSION_H

#include <pernix/avx2/compression.h>
#include <pernix/avx512vbmi/tables.h>

namespace pernix {

namespace internal {
__always_inline __m128i mm_maskz_quantize_ps_epi32(const __mmask8& mask, const __m128& input, const __m128& scale) {
    const __m128 scaled  = _mm_maskz_mul_ps(mask, input, scale);
    const __m128 rounded = _mm_maskz_roundscale_ps(mask, scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return _mm_maskz_cvtps_epi32(mask, rounded);
}

__always_inline __m256i mm256_maskz_quantize_ps_epi32(const __mmask8& mask, const __m256& input, const __m256& scale) {
    const __m256 scaled  = _mm256_maskz_mul_ps(mask, input, scale);
    const __m256 rounded = _mm256_maskz_roundscale_ps(mask, scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return _mm256_maskz_cvtps_epi32(mask, rounded);
}

__always_inline __m512i mm512_quantize_ps_epi32(const __m512& input, const __m512& scale) {
    const __m512 scaled = _mm512_mul_ps(input, scale);
    return _mm512_cvt_roundps_epi32(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__always_inline __m512i mm512_maskz_quantize_ps_epi32(const __mmask16& mask, const __m512& input, const __m512& scale) {
    const __m512 scaled = _mm512_maskz_mul_ps(mask, input, scale);
    return _mm512_maskz_cvt_roundps_epi32(mask, scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
__always_inline auto mm256_pack_epi32_avx512vbmi_9to15(const __m256i& input) -> __m128i {
    using tables = pack_tables_avx512<BIT_WIDTH, __m128i>;

    constexpr uint16_t bit_mask = (1u << BIT_WIDTH) - 1u;

    const __m128i packed = _mm256_cvtepi32_epi16(input);
    const __m128i maskv  = _mm_set1_epi16(static_cast<int16_t>(bit_mask));
    const __m128i masked = _mm_and_si128(packed, maskv);

    const __m128i p1 = tables::get_permute1();
    const __m128i p2 = tables::get_permute2();
    const __m128i s1 = tables::get_shift1();
    const __m128i s2 = tables::get_shift2();

    if constexpr (BIT_WIDTH == 12 || BIT_WIDTH == 14 || BIT_WIDTH == 15) {
        const __m128i permuted1 = _mm_permutexvar_epi16(p1, masked);
        const __m128i permuted2 = _mm_permutexvar_epi16(p2, masked);

        const __m128i shifted1 = _mm_sllv_epi16(permuted1, s1);
        const __m128i shifted2 = _mm_srlv_epi16(permuted2, s2);

        return _mm_or_si128(shifted1, shifted2);
    } else {
        const __m128i p3 = tables::get_permute3();
        const __m128i s3 = tables::get_shift3();

        const auto [mask1, mask2, mask3] = tables::get_permute_masks();

        const __m128i permuted1 = _mm_maskz_permutexvar_epi16(mask1, p1, masked);
        const __m128i permuted2 = _mm_maskz_permutexvar_epi16(mask2, p2, masked);
        const __m128i permuted3 = _mm_maskz_permutexvar_epi16(mask3, p3, masked);

        const __m128i shifted1 = _mm_sllv_epi16(permuted1, s1);
        const __m128i shifted2 = _mm_sllv_epi16(permuted2, s2);
        const __m128i shifted3 = _mm_srlv_epi16(permuted3, s3);

        return _mm_or_si128(_mm_or_si128(shifted1, shifted2), shifted3);
    }
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
__always_inline auto mm_pack_epi32_avx512vbmi_9to15(const __m128i& input) -> __m128i {
    return mm256_pack_epi32_avx512vbmi_9to15<BIT_WIDTH>(_mm256_castsi128_si256(input));
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
__always_inline auto mm512_pack_epi32_avx512vbmi_9to15(const __m512i& input) -> __m256i {
    using tables = pack_tables_avx512<BIT_WIDTH, __m256i>;

    constexpr uint16_t bit_mask = (1u << BIT_WIDTH) - 1u;

    const __m256i converted = _mm512_cvtepi32_epi16(input);
    const __m256i maskv     = _mm256_set1_epi16(static_cast<int16_t>(bit_mask));
    const __m256i masked    = _mm256_and_si256(converted, maskv);

    if constexpr (BIT_WIDTH == 12 || BIT_WIDTH == 14 || BIT_WIDTH == 15) {
        const __m256i permuted1 = _mm256_permutexvar_epi16(tables::get_permute1(), masked);
        const __m256i permuted2 = _mm256_permutexvar_epi16(tables::get_permute2(), masked);

        const __m256i shifted1 = _mm256_sllv_epi16(permuted1, tables::get_shift1());
        const __m256i shifted2 = _mm256_srlv_epi16(permuted2, tables::get_shift2());

        return _mm256_or_si256(shifted1, shifted2);
    } else {
        const auto [mask1, mask2, mask3] = tables::get_permute_masks();

        const __m256i permuted1 = _mm256_maskz_permutexvar_epi16(mask1, tables::get_permute1(), masked);
        const __m256i permuted2 = _mm256_maskz_permutexvar_epi16(mask2, tables::get_permute2(), masked);
        const __m256i permuted3 = _mm256_maskz_permutexvar_epi16(mask3, tables::get_permute3(), masked);

        const __m256i shifted1 = _mm256_sllv_epi16(permuted1, tables::get_shift1());
        const __m256i shifted2 = _mm256_sllv_epi16(permuted2, tables::get_shift2());
        const __m256i shifted3 = _mm256_srlv_epi16(permuted3, tables::get_shift3());

        return _mm256_or_si256(_mm256_or_si256(shifted1, shifted2), shifted3);
    }
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
__always_inline auto mm1024_pack_epi32_avx512vbmi_9to15(const __m512i& input) -> __m512i {
    using tables                = pack_tables_avx512<BIT_WIDTH, __m512i>;
    constexpr uint16_t bit_mask = (1u << BIT_WIDTH) - 1u;

    const __m512i maskv  = _mm512_set1_epi16(static_cast<int16_t>(bit_mask));
    const __m512i masked = _mm512_and_si512(input, maskv);

    if constexpr (BIT_WIDTH == 12 || BIT_WIDTH == 14 || BIT_WIDTH == 15) {
        const __m512i permuted1 = _mm512_permutexvar_epi16(tables::get_permute1(), masked);
        const __m512i permuted2 = _mm512_permutexvar_epi16(tables::get_permute2(), masked);

        const __m512i shifted1 = _mm512_sllv_epi16(permuted1, tables::get_shift1());
        const __m512i shifted2 = _mm512_srlv_epi16(permuted2, tables::get_shift2());

        return _mm512_or_si512(shifted1, shifted2);
    } else {
        const auto [mask1, mask2, mask3] = tables::get_permute_masks();

        const __m512i permuted1 = _mm512_maskz_permutexvar_epi16(mask1, tables::get_permute1(), masked);
        const __m512i permuted2 = _mm512_maskz_permutexvar_epi16(mask2, tables::get_permute2(), masked);
        const __m512i permuted3 = _mm512_maskz_permutexvar_epi16(mask3, tables::get_permute3(), masked);

        const __m512i shifted1 = _mm512_sllv_epi16(permuted1, tables::get_shift1());
        const __m512i shifted2 = _mm512_sllv_epi16(permuted2, tables::get_shift2());
        const __m512i shifted3 = _mm512_srlv_epi16(permuted3, tables::get_shift3());

        return _mm512_or_si512(_mm512_or_si512(shifted1, shifted2), shifted3);
    }
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH == 8 || BIT_WIDTH == 16)
__always_inline auto mm_pack_aligned_epi32_avx512(const __m128i& input) -> __m128i {
    if constexpr (BIT_WIDTH == 8) {
        return _mm_cvtepi32_epi8(input);
    } else {
        return _mm_cvtepi32_epi16(input);
    }
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH == 8 || BIT_WIDTH == 16)
__always_inline auto mm256_pack_aligned_epi32_avx512(const __m256i& input) -> __m128i {
    if constexpr (BIT_WIDTH == 8) {
        return _mm256_cvtepi32_epi8(input);
    } else {
        return _mm256_cvtepi32_epi16(input);
    }
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH == 8 || BIT_WIDTH == 16)
__always_inline auto mm512_pack_aligned_epi32_avx512(const __m512i& input) -> __m256i {
    if constexpr (BIT_WIDTH == 8) {
        return _mm256_castsi128_si256(_mm512_cvtepi32_epi8(input));
    } else {
        return _mm512_cvtepi32_epi16(input);
    }
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH == 8 || BIT_WIDTH == 16)
__always_inline auto mm1024_pack_aligned_epi32_avx512(const __m512i& input) -> __m512i {
    if constexpr (BIT_WIDTH == 8) {
        return _mm512_castsi256_si512(_mm512_cvtepi16_epi8(input));
    } else {
        return input;
    }
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
__always_inline auto mm_pack_epi32_avx512vbmi(const __m128i& input) -> __m128i {
    if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 15) {
        return mm_pack_epi32_avx512vbmi_9to15<BIT_WIDTH>(input);
    } else {
        return mm_pack_aligned_epi32_avx512<BIT_WIDTH>(input);
    }
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
__always_inline auto mm256_pack_epi32_avx512vbmi(const __m256i& input) -> __m128i {
    if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 15) {
        return mm256_pack_epi32_avx512vbmi_9to15<BIT_WIDTH>(input);
    } else {
        return mm256_pack_aligned_epi32_avx512<BIT_WIDTH>(input);
    }
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
__always_inline auto mm512_pack_epi32_avx512vbmi(const __m512i& input) -> __m256i {
    if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 15) {
        return mm512_pack_epi32_avx512vbmi_9to15<BIT_WIDTH>(input);
    } else {
        return mm512_pack_aligned_epi32_avx512<BIT_WIDTH>(input);
    }
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
__always_inline auto mm1024_pack_epi32_avx512vbmi(const __m512i& input1, const __m512i& input2) -> __m512i {
    const __m256i converted1 = _mm512_cvtepi32_epi16(input1);
    const __m256i converted2 = _mm512_cvtepi32_epi16(input2);

    __m512i packed = _mm512_castsi256_si512(converted1);
    packed         = _mm512_inserti64x4(packed, converted2, 1);

    if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 15) {
        return mm1024_pack_epi32_avx512vbmi_9to15<BIT_WIDTH>(packed);
    } else {
        return mm1024_pack_aligned_epi32_avx512<BIT_WIDTH>(packed);
    }
}

}  // namespace internal

/**
 * @brief Compress a single 512-bit block using AVX-512 and AVX-512-VBMI instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (8 to 16).
 *
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX-512 and AVX-512-VBMI support.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 8 && BIT_WIDTH <= 16)
int mm512_compress_block_avx512vbmi(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = 512 / BIT_WIDTH;
    constexpr uint32_t iterations_32      = elements_per_block / 32;
    constexpr uint32_t iterations_16      = (elements_per_block % 32) / 16;
    constexpr uint32_t iterations_8       = (elements_per_block % 16) / 8;
    constexpr uint8_t remaining           = elements_per_block - iterations_32 * 32 - iterations_16 * 16 - iterations_8 * 8;

    const __m512 scale_v    = _mm512_set1_ps(scale);
    const __m256 scale_v256 = _mm256_set1_ps(scale);
#pragma GCC unroll 2
    for (uint32_t iter = 0; iter < iterations_32; iter++) {
        const __m512 source1     = _mm512_loadu_ps(input);
        const __m512 source2     = _mm512_loadu_ps(input + 16);
        const __m512i quantized1 = internal::mm512_quantize_ps_epi32(source1, scale_v);
        const __m512i quantized2 = internal::mm512_quantize_ps_epi32(source2, scale_v);
        const __m512i packed     = internal::mm1024_pack_epi32_avx512vbmi<BIT_WIDTH>(quantized1, quantized2);
        if constexpr (BIT_WIDTH == 8) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(output), _mm512_castsi512_si256(packed));
        } else {
            _mm512_storeu_si512(output, packed);
        }

        input += 32;
        output += 4 * BIT_WIDTH;
    }

    if constexpr (iterations_16 > 0) {
        const __m512 source     = _mm512_loadu_ps(input);
        const __m512i quantized = internal::mm512_quantize_ps_epi32(source, scale_v);
        const __m256i packed    = internal::mm512_pack_epi32_avx512vbmi<BIT_WIDTH>(quantized);
        _mm256_storeu_epi8(output, packed);
        input += 16;
        output += 2 * BIT_WIDTH;
    }

    if constexpr (iterations_8 > 0) {
        const __m256 source     = _mm256_loadu_ps(input);
        const __m256i quantized = internal::mm256_quantize_ps_epi32(source, scale_v256);
        const __m128i packed    = internal::mm256_pack_epi32_avx512vbmi<BIT_WIDTH>(quantized);
        _mm_storeu_epi8(output, packed);
        input += 8;
        output += BIT_WIDTH;
    }

    if constexpr (remaining > 0) {
        constexpr __mmask16 store_mask = (1U << (remaining * BIT_WIDTH) / 8) - 1;
        const __m256 source            = _mm256_loadu_ps(input);
        const __m256i quantized        = internal::mm256_quantize_ps_epi32(source, scale_v256);
        const __m128i packed           = internal::mm256_pack_epi32_avx512vbmi<BIT_WIDTH>(quantized);
        _mm_mask_storeu_epi8(output, store_mask, packed);
    }

    return 0;
}

/**
 * @brief Compress multiple 512-bit blocks using AVX-512 and AVX-512-VBMI instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 16).
 *
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @param blocks number of 512-bit blocks to compress.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX-512 and AVX-512-VBMI support.
 */
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 16)
int mm512_compress_blocks_avx512vbmi(const float_t* __restrict__ input, const float_t scale, uint8_t* __restrict__ output,
                                     const uint32_t blocks) {
    const float_t* block_input = input;
    uint8_t* block_output      = output;

    for (uint32_t block = 0; block < blocks; block++) {
        mm512_compress_block_avx512vbmi<BIT_WIDTH>(block_input, scale, block_output);
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

#ifdef __cplusplus
}
}  // namespace pernix
#endif

#endif  // PERNIX_AVX512VBMI_COMPRESSION_H
