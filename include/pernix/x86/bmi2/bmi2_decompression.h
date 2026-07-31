#ifndef PERNIX_BMI2_DECOMPRESSION_H
#define PERNIX_BMI2_DECOMPRESSION_H

#include <pernix/simd_compat.h>
#include <pernix/x86/avx2/avx2_decompression.h>
#include <pernix/x86/bmi2/bmi2_unpacking.h>

#include <cstring>
#include <span>
#include <type_traits>

namespace pernix {
namespace internal {
/**
 * @brief Dequantize and store eight 32-bit values as floats.
 */
__always_inline void mm256_process_bmi2(const __m256i unpacked, const __m256 scale, f32* destination) {
    const __m256 dequantized = mm256_dequantize_epi32(unpacked, scale);
    _mm256_storeu_ps(destination, dequantized);
}

/**
 * @brief Dequantize and store eight 32-bit values as doubles.
 */
__always_inline void mm256_process_bmi2(const __m256i unpacked, const __m256d scale, f64* destination) {
    const __m256i low  = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(unpacked));
    const __m256i high = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(unpacked, 1));

    const __m256d dequantized_low  = mm256_dequantize_epi64_pd(low, scale);
    const __m256d dequantized_high = mm256_dequantize_epi64_pd(high, scale);

    _mm256_storeu_pd(destination, dequantized_low);
    _mm256_storeu_pd(destination + 4, dequantized_high);
}

/**
 * @brief Dequantize and store a partial group of 32-bit values as floats.
 */
template <u32 ELEMENTS>
    requires(ELEMENTS > 0 && ELEMENTS < 8)
__always_inline void mm256_process_bmi2_tail(const __m256i unpacked, const __m256 scale, f32* destination) {
    const __m256 dequantized = mm256_dequantize_epi32(unpacked, scale);
    std::memcpy(destination, &dequantized, ELEMENTS * sizeof(f32));
}

/**
 * @brief Dequantize and store a partial group of 32-bit values as doubles.
 */
template <u32 ELEMENTS>
    requires(ELEMENTS > 0 && ELEMENTS < 8)
__always_inline void mm256_process_bmi2_tail(const __m256i unpacked, const __m256d scale, f64* destination) {
    const __m256i low  = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(unpacked));
    const __m256i high = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(unpacked, 1));

    const __m256d dequantized_low  = mm256_dequantize_epi64_pd(low, scale);
    const __m256d dequantized_high = mm256_dequantize_epi64_pd(high, scale);

    constexpr u32 low_elements = ELEMENTS < 4U ? ELEMENTS : 4U;
    std::memcpy(destination, &dequantized_low, low_elements * sizeof(f64));
    if constexpr (ELEMENTS > 4U) {
        std::memcpy(destination + 4, &dequantized_high, (ELEMENTS - 4U) * sizeof(f64));
    }
}

}  // namespace internal

/**
 * @brief Decompress a single BMI2 block into a typed output span.
 */
template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE, typename FloatT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0) && (std::is_floating_point_v<FloatT>)
__always_inline int mm256_decompress_block_bmi2(const std::span<const u8> input, const FloatT scale, const std::span<FloatT> output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8U) / BIT_WIDTH;
    constexpr u32 groups_8           = elements_per_block / 8U;
    constexpr u32 remaining_elements = elements_per_block % 8U;

    const auto scale_v = [scale] {
        if constexpr (std::is_same_v<FloatT, f32>) {
            return _mm256_set1_ps(scale);
        } else {
            return _mm256_set1_pd(scale);
        }
    }();

#pragma GCC unroll 4
    for (u32 group = 0; group < groups_8; ++group) {
        const auto packed = input.subspan(static_cast<usize>(group) * BIT_WIDTH).first<BIT_WIDTH>();

        const __m256i unpacked = internal::bmi2::mm256_unpack_epi32<BIT_WIDTH, SIGN_VALUES>(packed);
        internal::mm256_process_bmi2(unpacked, scale_v, output.data() + (static_cast<usize>(group) * 8U));
    }

    if constexpr (remaining_elements > 0) {
        constexpr u32 remaining_bytes = ((remaining_elements * BIT_WIDTH) + 7U) / 8U;
        const auto packed             = input.subspan(static_cast<usize>(groups_8) * BIT_WIDTH).first<remaining_bytes>();

        const __m256i unpacked = internal::bmi2::mm256_unpack_epi32<BIT_WIDTH, SIGN_VALUES, remaining_elements>(packed);
        internal::mm256_process_bmi2_tail<remaining_elements>(unpacked, scale_v, output.data() + (static_cast<usize>(groups_8) * 8U));
    }

    return 0;
}

/**
 * @brief Decompress a single block to floats using AVX2 and BMI2 instructions.
 */
template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm256_decompress_block_bmi2(const void* __restrict__ input_ptr, const f32 scale, void* __restrict__ output_ptr) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8U) / BIT_WIDTH;
    const auto input                 = std::span<const u8>(static_cast<const u8*>(input_ptr), BLOCK_SIZE);
    const auto output                = std::span<f32>(static_cast<f32*>(output_ptr), elements_per_block);
    return mm256_decompress_block_bmi2<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
}

/**
 * @brief Decompress a single block to doubles using AVX2 and BMI2 instructions.
 */
template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm256_decompress_block_bmi2(const void* __restrict__ input_ptr, const f64 scale, void* __restrict__ output_ptr) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8U) / BIT_WIDTH;
    const auto input                 = std::span<const u8>(static_cast<const u8*>(input_ptr), BLOCK_SIZE);
    const auto output                = std::span<f64>(static_cast<f64*>(output_ptr), elements_per_block);
    return mm256_decompress_block_bmi2<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
}

/**
 * @brief Decompress multiple BMI2 blocks into a typed output span.
 */
template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE, typename FloatT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0) && std::is_floating_point_v<FloatT>
int mm256_decompress_blocks_bmi2(const std::span<const u8> input, const FloatT scale, const std::span<FloatT> output, const u32 blocks) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8U) / BIT_WIDTH;

    for (u32 block = 0; block < blocks; ++block) {
        const auto block_input  = input.subspan(static_cast<usize>(block) * BLOCK_SIZE, BLOCK_SIZE);
        const auto block_output = output.subspan(static_cast<usize>(block) * elements_per_block, elements_per_block);
        mm256_decompress_block_bmi2<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(block_input, scale, block_output);
    }
    return 0;
}

/**
 * @brief Decompress multiple blocks to floats using AVX2 and BMI2 instructions.
 */
template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_decompress_blocks_bmi2(const void* __restrict__ input_ptr, const f32 scale, void* __restrict__ output_ptr, const u32 blocks) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8U) / BIT_WIDTH;
    const auto input                 = std::span<const u8>(static_cast<const u8*>(input_ptr), static_cast<usize>(blocks) * BLOCK_SIZE);
    const auto output                = std::span<f32>(static_cast<f32*>(output_ptr), static_cast<usize>(blocks) * elements_per_block);
    return mm256_decompress_blocks_bmi2<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output, blocks);
}

/**
 * @brief Decompress multiple blocks to doubles using AVX2 and BMI2 instructions.
 */
template <u8 BIT_WIDTH, bool SIGN_VALUES = true, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_decompress_blocks_bmi2(const void* __restrict__ input_ptr, const f64 scale, void* __restrict__ output_ptr, const u32 blocks) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8U) / BIT_WIDTH;
    const auto input                 = std::span<const u8>(static_cast<const u8*>(input_ptr), static_cast<usize>(blocks) * BLOCK_SIZE);
    const auto output                = std::span<f64>(static_cast<f64*>(output_ptr), static_cast<usize>(blocks) * elements_per_block);
    return mm256_decompress_blocks_bmi2<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output, blocks);
}
}  // namespace pernix

#endif  // PERNIX_BMI2_DECOMPRESSION_H
