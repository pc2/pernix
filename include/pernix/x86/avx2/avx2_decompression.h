#ifndef PERNIX_AVX2_DECOMPRESSION_H
#define PERNIX_AVX2_DECOMPRESSION_H

#include <pernix/fallback/scalar_decompression.h>
#include <pernix/simd_compat.h>
#include <pernix/x86/avx2/avx2_tables.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace pernix {
namespace internal {
/**
 * @brief Convert an 8-lane mask to the lane representation used by AVX2 float masked stores.
 */
__always_inline __m256i mm256_convert_vmask_epi32(const __mmask8 mask8) {
    return _mm256_setr_epi32((mask8 & 0x1) ? -1 : 0, (mask8 & 0x2) ? -1 : 0, (mask8 & 0x4) ? -1 : 0, (mask8 & 0x8) ? -1 : 0,
                             (mask8 & 0x10) ? -1 : 0, (mask8 & 0x20) ? -1 : 0, (mask8 & 0x40) ? -1 : 0, (mask8 & 0x80) ? -1 : 0);
}

/**
 * @brief Convert a 4-lane mask to the lane representation used by AVX2 double masked stores.
 */
__always_inline __m256i mm256_convert_vmask_epi64(const __mmask8 mask8) {
    return _mm256_setr_epi64x((mask8 & 0x1) ? -1 : 0, (mask8 & 0x2) ? -1 : 0, (mask8 & 0x4) ? -1 : 0, (mask8 & 0x8) ? -1 : 0);
}

/**
 * @brief Dequantize four 32-bit integers to floats.
 */
__always_inline __m128 mm_dequantize_epi32(__m128i input, __m128 scale) {
    const __m128 converted = _mm_cvtepi32_ps(input);
    return _mm_mul_ps(converted, scale);
}

/* https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx */
__always_inline __m128d convert_epi64_pd(const __m128i v) {
    __m128i xH       = _mm_srai_epi32(v, 16);
    xH               = _mm_blend_epi16(xH, _mm_setzero_si128(), 0x33);
    xH               = _mm_add_epi64(xH, _mm_castpd_si128(_mm_set1_pd(442721857769029238784.)));     //  3*2^67
    const __m128i xL = _mm_blend_epi16(v, _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)), 0x88);  //  2^52
    const __m128d f  = _mm_sub_pd(_mm_castsi128_pd(xH), _mm_set1_pd(442726361368656609280.));        //  3*2^67 + 2^52
    return _mm_add_pd(f, _mm_castsi128_pd(xL));
}

/**
 * @brief Dequantize two 64-bit integers to doubles.
 */
__always_inline __m128d mm_dequantize_epi64_pd(__m128i input, __m128d scale) {
    const __m128d converted = convert_epi64_pd(input);
    return _mm_mul_pd(converted, scale);
}

/**
 * @brief Dequantize eight 32-bit integers to floats.
 */
__always_inline __m256 mm256_dequantize_epi32(__m256i input, __m256 scale) {
    const __m256 converted = _mm256_cvtepi32_ps(input);
    return _mm256_mul_ps(converted, scale);
}

/* https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx */
__always_inline __m256d convert_epi64_pd(__m256i v) {
    __m256i xH       = _mm256_srai_epi32(v, 16);
    xH               = _mm256_blend_epi16(xH, _mm256_setzero_si256(), 0x33);
    xH               = _mm256_add_epi64(xH, _mm256_castpd_si256(_mm256_set1_pd(442721857769029238784.)));
    const __m256i xL = _mm256_blend_epi16(v, _mm256_castpd_si256(_mm256_set1_pd(0x0010000000000000)), 0x88);
    const __m256d f  = _mm256_sub_pd(_mm256_castsi256_pd(xH), _mm256_set1_pd(442726361368656609280.));
    return _mm256_add_pd(f, _mm256_castsi256_pd(xL));
}

/**
 * @brief Dequantize four 64-bit integers to doubles.
 */
__always_inline __m256d mm256_dequantize_epi64_pd(__m256i input, __m256d scale) {
    const __m256d converted = convert_epi64_pd(input);
    return _mm256_mul_pd(converted, scale);
}

/**
 * @brief Unpack four aligned 8-bit or 16-bit values directly from the input buffer.
 */
template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH == 8 || BIT_WIDTH == 16)
__m128i mm_unpack_aligned_epi32_avx2(const u8* __restrict__ input) {
    if constexpr (BIT_WIDTH == 8) {
        const __m128i source = _mm_loadu_si32(input);
        if constexpr (SIGN_VALUES) {
            return _mm_cvtepi8_epi32(source);
        } else {
            return _mm_cvtepu8_epi32(source);
        }
    } else {
        const __m128i source = _mm_loadu_si64(input);
        if constexpr (SIGN_VALUES) {
            return _mm_cvtepi16_epi32(source);
        } else {
            return _mm_cvtepu16_epi32(source);
        }
    }
}

/**
 * @brief Unpack four values using the table-driven AVX2 shuffle path.
 */
template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
__m128i mm_unpack_epi32_avx2(const u8* __restrict__ input) {
    using unpack_table           = unpack_tables_avx2<BIT_WIDTH, __m128i>;
    constexpr usize packed_bytes = (4 * BIT_WIDTH + 7) / 8;

    __m128i source = _mm_setzero_si128();
    std::memcpy(&source, input, packed_bytes);

    const __m128i shuffled = _mm_shuffle_epi8(source, unpack_table::get_shuffle());

    constexpr u16 shift = 32 - BIT_WIDTH;
    __m128i shifted     = _mm_sllv_epi32(shuffled, unpack_table::get_shift());
    if constexpr (SIGN_VALUES && BIT_WIDTH > 1) {
        shifted = _mm_srai_epi32(shifted, shift);
    } else {
        shifted = _mm_srli_epi32(shifted, shift);
    }

    return shifted;
}

/**
 * @brief Unpack eight aligned 8-bit or 16-bit values directly from the input buffer.
 */
template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH == 8 || BIT_WIDTH == 16)
__m256i mm256_unpack_aligned_epi32_avx2(const u8* __restrict__ input) {
    if constexpr (BIT_WIDTH == 8) {
        const __m128i source = _mm_loadu_si64(input);
        if constexpr (SIGN_VALUES) {
            return _mm256_cvtepi8_epi32(source);
        } else {
            return _mm256_cvtepu8_epi32(source);
        }
    } else {
        const __m128i source = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input));
        if constexpr (SIGN_VALUES) {
            return _mm256_cvtepi16_epi32(source);
        } else {
            return _mm256_cvtepu16_epi32(source);
        }
    }
}

/**
 * @brief Unpack eight values using the table-driven AVX2 shuffle path.
 */
template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
__m256i mm256_unpack_epi32_avx2(const __m256i source) {
    using unpack_table     = unpack_tables_avx2<BIT_WIDTH, __m256i>;
    const __m256i permuted = _mm256_permutevar8x32_epi32(source, unpack_table::get_permute());
    const __m256i shuffled = _mm256_shuffle_epi8(permuted, unpack_table::get_shuffle());

    constexpr u16 shift = 32 - BIT_WIDTH;
    __m256i shifted     = _mm256_sllv_epi32(shuffled, unpack_table::get_shift());
    if constexpr (SIGN_VALUES && BIT_WIDTH > 1) {
        shifted = _mm256_srai_epi32(shifted, shift);
    } else {
        shifted = _mm256_srli_epi32(shifted, shift);
    }

    return shifted;
}

/**
 * @brief Load and unpack eight values using the table-driven AVX2 shuffle path.
 */
template <u8 BIT_WIDTH, bool SIGN_VALUES = true>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
__m256i mm256_unpack_epi32_avx2(const u8* __restrict__ input) {
    __m256i source = _mm256_setzero_si256();
    std::memcpy(&source, input, BIT_WIDTH);
    return mm256_unpack_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(source);
}
}  // namespace internal

/**
 * @brief Decompress a single block to float using AVX2 instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 *
 * @param input pointer to the start of the compressed block.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 support.
 */
template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_decompress_block_avx2(const void* __restrict__ input_ptr, const f32 scale, void* __restrict__ output_ptr) {
    const auto* input = static_cast<const u8*>(input_ptr);
    auto* output      = static_cast<f32*>(output_ptr);

    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;
    constexpr u32 iterations_8       = elements_per_block / 8;
    constexpr u8 remaining           = elements_per_block - iterations_8 * 8;

    const __m256 scale_v = _mm256_set1_ps(scale);
#pragma GCC unroll 4
    for (u32 iter = 0; iter < iterations_8; iter++) {
        const __m256i unpacked   = internal::mm256_unpack_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m256 dequantized = internal::mm256_dequantize_epi32(unpacked, scale_v);
        _mm256_storeu_ps(output, dequantized);
        input += BIT_WIDTH;
        output += 8;
    }

    if constexpr (remaining > 0) {
        constexpr u32 tail_bytes = (BIT_WIDTH * remaining + 7) / 8;
        __m256i source           = _mm256_setzero_si256();
        std::memcpy(&source, input, tail_bytes);

        const __m256i unpacked   = internal::mm256_unpack_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(source);
        const __m256 dequantized = internal::mm256_dequantize_epi32(unpacked, scale_v);
        std::memcpy(output, &dequantized, remaining * sizeof(f32));
    }

    return 0;
}

/**
 * @brief Decompress a single block to double using AVX2 instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 *
 * @param input pointer to the start of the compressed block.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed double values will be stored.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 support.
 */
template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_decompress_block_avx2(const void* __restrict__ input_ptr, const f64 scale, void* __restrict__ output_ptr) {
    const auto* input = static_cast<const u8*>(input_ptr);
    auto* output      = static_cast<f64*>(output_ptr);

    constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;
    constexpr u32 iterations_8       = elements_per_block / 8;
    constexpr u8 remaining           = elements_per_block - iterations_8 * 8;
    const __m256d scale_v            = _mm256_set1_pd(scale);
#pragma GCC unroll 4
    for (u32 iter = 0; iter < iterations_8; iter++) {
        const __m256i unpacked = internal::mm256_unpack_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(input);
        const __m256i extend1  = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(unpacked));
        const __m256i extend2  = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(unpacked, 1));

        const __m256d dequantized1 = internal::mm256_dequantize_epi64_pd(extend1, scale_v);
        const __m256d dequantized2 = internal::mm256_dequantize_epi64_pd(extend2, scale_v);

        _mm256_storeu_pd(output, dequantized1);
        _mm256_storeu_pd(output + 4, dequantized2);

        input += BIT_WIDTH;
        output += 8;
    }

    if constexpr (remaining > 0) {
        constexpr u32 tail_bytes = (BIT_WIDTH * remaining + 7) / 8;
        __m256i source           = _mm256_setzero_si256();
        std::memcpy(&source, input, tail_bytes);

        const __m256i unpacked = internal::mm256_unpack_epi32_avx2<BIT_WIDTH, SIGN_VALUES>(source);
        const __m256i extend1  = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(unpacked));
        const __m256i extend2  = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(unpacked, 1));

        const __m256d dequantized1 = internal::mm256_dequantize_epi64_pd(extend1, scale_v);
        const __m256d dequantized2 = internal::mm256_dequantize_epi64_pd(extend2, scale_v);

        constexpr u32 first_elements = remaining < 4 ? remaining : 4;
        std::memcpy(output, &dequantized1, first_elements * sizeof(f64));
        if constexpr (remaining > 4) {
            std::memcpy(output + 4, &dequantized2, (remaining - 4) * sizeof(f64));
        }
    }
    return 0;
}

/**
 * @brief Decompress multiple blocks to float using AVX2 instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 *
 * @param input pointer to the start of the compressed data.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @param blocks number of blocks to decompress.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 support.
 */
template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_decompress_blocks_avx2(const void* __restrict__ input_ptr, const f32 scale, void* __restrict__ output_ptr, const u32 blocks) {
    const auto* input = static_cast<const u8*>(input_ptr);
    auto* output      = static_cast<f32*>(output_ptr);

    const u8* block_input = input;
    f32* block_output     = output;

    for (u32 block = 0; block < blocks; block++) {
        mm256_decompress_block_avx2<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(block_input, scale, block_output);
        block_input += BLOCK_SIZE;
        block_output += (BLOCK_SIZE * 8) / BIT_WIDTH;
    }

    return 0;
}

/**
 * @brief Decompress multiple blocks to double using AVX2 instructions.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam SIGN_VALUES whether the values are signed or unsigned.
 *
 * @param input pointer to the start of the compressed data.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where decompressed float values will be stored.
 * @param blocks number of blocks to decompress.
 * @return int status code (0 for success).
 *
 * @note This function requires AVX2 support.
 */
template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_decompress_blocks_avx2(const void* __restrict__ input_ptr, const f64 scale, void* __restrict__ output_ptr, const u32 blocks) {
    const auto* input = static_cast<const u8*>(input_ptr);
    auto* output      = static_cast<f64*>(output_ptr);

    const u8* block_input = input;
    f64* block_output     = output;

    for (u32 block = 0; block < blocks; block++) {
        mm256_decompress_block_avx2<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(block_input, scale, block_output);
        block_input += BLOCK_SIZE;
        block_output += (BLOCK_SIZE * 8) / BIT_WIDTH;
    }

    return 0;
}
}  // namespace pernix

#endif  // PERNIX_AVX2_DECOMPRESSION_H
