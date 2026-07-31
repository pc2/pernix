#ifndef PERNIX_BMI2_COMPRESSION_H
#define PERNIX_BMI2_COMPRESSION_H

#include <pernix/simd_compat.h>
#include <pernix/x86/avx2/avx2_compression.h>
#include <pernix/x86/bmi2/bmi2_packing.h>

#include <array>
#include <cstring>
#include <span>
#include <type_traits>

namespace pernix {
namespace internal {

__always_inline __m256i mm256_quantize_bmi2(const std::span<const f32, 8> input, const __m256 scale) noexcept {
    const __m256 source = _mm256_loadu_ps(input.data());
    return mm256_quantize_ps_epi32(source, scale);
}

__always_inline __m256i mm256_quantize_bmi2(const std::span<const f64, 8> input, const __m256d scale) noexcept {
    const __m128i low  = mm256_quantize_pd_epi32(_mm256_loadu_pd(input.data()), scale);
    const __m128i high = mm256_quantize_pd_epi32(_mm256_loadu_pd(input.data() + 4), scale);

    const __m256i result = _mm256_castsi128_si256(low);
    return _mm256_inserti128_si256(result, high, 1);
}

template <u32 Elements>
    requires(Elements > 0 && Elements < 8)
__always_inline __m256i mm256_quantize_bmi2_tail(const std::span<const f32> input, const __m256 scale) noexcept {
    const __m256 source = _mm256_maskload_ps(input.data(), mm256_tail_mask_epi32<Elements>());
    return mm256_quantize_ps_epi32(source, scale);
}

template <u32 Elements>
    requires(Elements > 0 && Elements < 8)
__always_inline __m256i mm256_quantize_bmi2_tail(const std::span<const f64> input, const __m256d scale) noexcept {
    constexpr u32 low_elements  = Elements < 4U ? Elements : 4U;
    constexpr u32 high_elements = Elements > 4U ? Elements - 4U : 0U;

    const __m256d low_source = _mm256_maskload_pd(input.data(), mm256_tail_mask_epi64<low_elements>());
    const __m128i low        = mm256_quantize_pd_epi32(low_source, scale);
    const __m128i high       = [&] {
        if constexpr (high_elements > 0) {
            const __m256d high_source = _mm256_maskload_pd(input.data() + 4, mm256_tail_mask_epi64<high_elements>());
            return mm256_quantize_pd_epi32(high_source, scale);
        } else {
            return _mm_setzero_si128();
        }
    }();

    __m256i const result = _mm256_castsi128_si256(low);
    return _mm256_inserti128_si256(result, high, 1);
}

}  // namespace internal

/**
 * @brief Compress a single BMI2 block from a typed input span.
 */
template <u8 BIT_WIDTH, u32 BLOCK_SIZE, typename FloatT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0) && (std::is_floating_point_v<FloatT>)
__always_inline int mm256_compress_block_bmi2(const std::span<const FloatT> input, const FloatT scale, const std::span<u8> output) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8U) / BIT_WIDTH;
    constexpr u32 groups_8           = elements_per_block / 8U;
    constexpr u32 remaining_elements = elements_per_block % 8U;

    std::memset(output.data(), 0, BLOCK_SIZE);

    const auto scale_v = [scale] {
        if constexpr (std::is_same_v<FloatT, f32>) {
            return _mm256_set1_ps(scale);
        } else {
            return _mm256_set1_pd(scale);
        }
    }();

#pragma GCC unroll 4
    for (u32 group = 0; group < groups_8; ++group) {
        const auto group_input  = input.subspan(static_cast<usize>(group) * 8U).template first<8>();
        const auto group_output = output.subspan(static_cast<usize>(group) * BIT_WIDTH).first<BIT_WIDTH>();

        const __m256i quantized = internal::mm256_quantize_bmi2(group_input, scale_v);
        const __m256i clamped   = internal::mm256_clamp_signed_epi32<BIT_WIDTH>(quantized);
        internal::bmi2::mm256_pack_epi32<BIT_WIDTH>(clamped, group_output);
    }

    if constexpr (remaining_elements > 0) {
        constexpr u32 remaining_bytes = ((remaining_elements * BIT_WIDTH) + 7U) / 8U;
        const auto tail_input         = input.subspan(static_cast<usize>(groups_8) * 8U, remaining_elements);
        const auto tail_output        = output.subspan(static_cast<usize>(groups_8) * BIT_WIDTH).first<remaining_bytes>();

        const __m256i quantized = internal::mm256_quantize_bmi2_tail<remaining_elements>(tail_input, scale_v);
        const __m256i clamped   = internal::mm256_clamp_signed_epi32<BIT_WIDTH>(quantized);
        internal::bmi2::mm256_pack_epi32<BIT_WIDTH, remaining_elements>(clamped, tail_output);
    }

    return 0;
}

/**
 * @brief Compress a single block of floats using AVX2 and BMI2 instructions.
 */
template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm256_compress_block_bmi2(const void* __restrict__ input_ptr, const f32 scale, void* __restrict__ output_ptr) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8U) / BIT_WIDTH;
    const auto input                 = std::span<const f32>(static_cast<const f32*>(input_ptr), elements_per_block);
    const auto output                = std::span<u8>(static_cast<u8*>(output_ptr), BLOCK_SIZE);
    return mm256_compress_block_bmi2<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
}

/**
 * @brief Compress a single block of doubles using AVX2 and BMI2 instructions.
 */
template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int mm256_compress_block_bmi2(const void* __restrict__ input_ptr, const f64 scale, void* __restrict__ output_ptr) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8U) / BIT_WIDTH;
    const auto input                 = std::span<const f64>(static_cast<const f64*>(input_ptr), elements_per_block);
    const auto output                = std::span<u8>(static_cast<u8*>(output_ptr), BLOCK_SIZE);
    return mm256_compress_block_bmi2<BIT_WIDTH, BLOCK_SIZE>(input, scale, output);
}

/**
 * @brief Compress multiple BMI2 blocks from and into typed spans.
 */
template <u8 BIT_WIDTH, u32 BLOCK_SIZE, typename FloatT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0) && (std::is_same_v<FloatT, f32> || std::is_same_v<FloatT, f64>)
int mm256_compress_blocks_bmi2(const std::span<const FloatT> input, const FloatT scale, const std::span<u8> output, const u32 blocks) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8U) / BIT_WIDTH;

    for (u32 block = 0; block < blocks; ++block) {
        const auto block_input  = input.subspan(static_cast<usize>(block) * elements_per_block, elements_per_block);
        const auto block_output = output.subspan(static_cast<usize>(block) * BLOCK_SIZE, BLOCK_SIZE);
        mm256_compress_block_bmi2<BIT_WIDTH, BLOCK_SIZE>(block_input, scale, block_output);
    }

    return 0;
}

/**
 * @brief Compress multiple blocks of floats using AVX2 and BMI2 instructions.
 */
template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_compress_blocks_bmi2(const void* __restrict__ input_ptr, const f32 scale, void* __restrict__ output_ptr, const u32 blocks) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8U) / BIT_WIDTH;
    const auto input  = std::span<const f32>(static_cast<const f32*>(input_ptr), static_cast<usize>(blocks) * elements_per_block);
    const auto output = std::span<u8>(static_cast<u8*>(output_ptr), static_cast<usize>(blocks) * BLOCK_SIZE);
    return mm256_compress_blocks_bmi2<BIT_WIDTH, BLOCK_SIZE>(input, scale, output, blocks);
}

/**
 * @brief Compress multiple blocks of doubles using AVX2 and BMI2 instructions.
 */
template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int mm256_compress_blocks_bmi2(const void* __restrict__ input_ptr, const f64 scale, void* __restrict__ output_ptr, const u32 blocks) {
    constexpr u32 elements_per_block = (BLOCK_SIZE * 8U) / BIT_WIDTH;
    const auto input  = std::span<const f64>(static_cast<const f64*>(input_ptr), static_cast<usize>(blocks) * elements_per_block);
    const auto output = std::span<u8>(static_cast<u8*>(output_ptr), static_cast<usize>(blocks) * BLOCK_SIZE);
    return mm256_compress_blocks_bmi2<BIT_WIDTH, BLOCK_SIZE>(input, scale, output, blocks);
}

}  // namespace pernix

#endif  // PERNIX_BMI2_COMPRESSION_H
