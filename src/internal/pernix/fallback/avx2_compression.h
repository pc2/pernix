#ifndef PERNIX_FALLBACK_COMPRESSION_H
#define PERNIX_FALLBACK_COMPRESSION_H

#include <pernix/simd_compat.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace pernix {
    namespace internal {
        /**
 * @brief Quantize a single float value to i32 using the provided scale.
 *
 * @param input input float value to be quantized.
 * @param scale scaling factor used during quantization.
 * @return i32 quantized integer value.
 */
__always_inline i32 quantize_ps_epi32(const float input, const float scale) {
            return static_cast<i32>(std::lroundf(input * scale));
        }

        /**
 * @brief Quantize a single double value to i64 using the provided scale.
 *
 * @param input input double value to be quantized.
 * @param scale scaling factor used during quantization.
 * @return i64 quantized integer value.
 */
__always_inline i64 quantize_pd_epi64(const f64 input, const f64 scale) {
            return std::llround(input * scale);
        }

        /**
 * @brief Quantize and clamp without narrowing through an out-of-range integer type.
 */
        template<u8 BIT_WIDTH, typename T>
            requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<T>)
__always_inline i32 quantize_clamped(const T input, const T scale) {
            constexpr i64 min_value = BIT_WIDTH == 1 ? 0 : -(i64{1} << (BIT_WIDTH - 1));
            constexpr i64 max_value = BIT_WIDTH == 1 ? 1 : ((i64{1} << (BIT_WIDTH - 1)) - 1);

            const long double scaled = static_cast<long double>(input) * static_cast<long double>(scale);
            if (std::isnan(scaled)) {
                return 0;
            }
            if (scaled <= static_cast<long double>(min_value)) {
                return static_cast<i32>(min_value);
            }
            if (scaled >= static_cast<long double>(max_value)) {
                return static_cast<i32>(max_value);
            }

            return static_cast<i32>(std::llround(scaled));
        }

        /**
 * @brief Clamp a signed quantized value to the representable range of BIT_WIDTH bits.
 */
        template<u8 BIT_WIDTH>
            requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
__always_inline i32 clamp_signed_quantized(const i64 value) {
            if constexpr (BIT_WIDTH == 1) {
                // 1-bit fallback is treated as binary quantization (0/1).
                return static_cast<i32>(std::clamp<i64>(value, 0, 1));
            }

            constexpr i32 min_value = -(1 << (BIT_WIDTH - 1));
            constexpr i32 max_value = (1 << (BIT_WIDTH - 1)) - 1;
            return static_cast<i32>(std::clamp<i64>(value, min_value, max_value));
        }

        /**
 * @brief Append packed scalar values into an output buffer using the selected
 * storage width.
 *
 * @tparam T unsigned integer type used as the packing word.
 * @tparam BIT_WIDTH bit width per value in the packed representation.
 * @param input vector of quantized values to pack.
 * @param bit_offset starting bit offset in the destination buffer.
 * @param destination pointer to the output buffer.
 */
        template<typename T, u8 BIT_WIDTH>
            requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_integral_v<T> && std::is_unsigned_v<T>)
        void pack_epi32_fallback_inner(const std::vector<u32> &input, const u8 bit_offset,
                                       u8 * __restrict__ destination) {
            constexpr u32 bits_in_type = sizeof(T) * 8;
            constexpr u32 bitmask = BIT_WIDTH == bits_in_type
                                        ? std::numeric_limits<T>::max()
                                        : (1U << BIT_WIDTH) - 1U;

            std::size_t idx = 0;
            std::size_t bits_in_buffer = bit_offset;
            u64 buffer = bit_offset ? static_cast<u64>(destination[0] & ((1U << bit_offset) - 1U)) : 0;

#pragma GCC unroll 64
            for (u32 raw_value: input) {
                const u32 next_value = raw_value & bitmask;

                buffer |= static_cast<u64>(next_value) << bits_in_buffer;
                bits_in_buffer += BIT_WIDTH;

                while (bits_in_buffer >= 8) {
                    destination[idx++] = static_cast<u8>(buffer & 0xFFU);
                    buffer >>= 8;
                    bits_in_buffer -= 8;
                }
            }

            if (bits_in_buffer > 0) {
                destination[idx] = static_cast<u8>(buffer & 0xFFU);
            }
        }

        /**
 * @brief Pack a vector of u32 values into a compact byte representation using fallback scalar implementation.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam BLOCK_SIZE size of each block in bytes (default 64 for 512 bits).
 *
 * @param input vector of u32 values to be packed.
 * @param destination pointer to the output buffer where packed bytes will be stored.
 */
        template<u8 BIT_WIDTH>
            requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
        void pack_epi32_fallback(const std::vector<u32> &input, u8 * __restrict__ destination) {
            if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
                return internal::pack_epi32_fallback_inner<u8, BIT_WIDTH>(input, 0, destination);
            } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
                return internal::pack_epi32_fallback_inner<u16, BIT_WIDTH>(input, 0, destination);
            } else {
                return internal::pack_epi32_fallback_inner<u32, BIT_WIDTH>(input, 0, destination);
            }
        }
    } // namespace internal

    /**
 * @brief Compress a single 512-bit block using fallback scalar implementation.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam BLOCK_SIZE size of each block in bytes (default 64 for 512 bits).
 *
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @return int status code (0 for success).
 */
    template<u8 BIT_WIDTH, u32 BLOCK_SIZE>
        requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
    int compress_block_fallback(const void * __restrict__ input_ptr, const f32 scale,
                                void * __restrict__ output_ptr) {
        const auto *input = static_cast<const float *>(input_ptr);
        auto *output = static_cast<u8 *>(output_ptr);

        constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

        std::memset(output, 0, BLOCK_SIZE);

        std::vector<u32> block_values(elements_per_block);
#pragma GCC unroll 64
        for (u32 i = 0; i < elements_per_block; i++) {
            const i32 quantized = internal::quantize_clamped<BIT_WIDTH>(input[i], scale);
            block_values[i] = static_cast<u32>(quantized);
        }

        internal::pack_epi32_fallback<BIT_WIDTH>(block_values, output);
        return 0;
    }

    /**
 * @brief Compress a single block of double values using the fallback scalar implementation.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam BLOCK_SIZE size of each block in bytes (default 64 for 512 bits).
 * @param input pointer to the start of the input double values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @return int status code (0 for success).
 */
    template<u8 BIT_WIDTH, u32 BLOCK_SIZE>
        requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
    int compress_block_fallback(const void * __restrict__ input_ptr, const f64 scale,
                                void * __restrict__ output_ptr) {
        const auto *input = static_cast<const double *>(input_ptr);
        auto *output = static_cast<u8 *>(output_ptr);

        constexpr u32 elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

        std::memset(output, 0, BLOCK_SIZE);

        std::vector<u32> block_values(elements_per_block);
#pragma GCC unroll 32
        for (u32 i = 0; i < elements_per_block; i++) {
            const i32 quantized = internal::quantize_clamped<BIT_WIDTH>(input[i], scale);
            block_values[i] = static_cast<u32>(quantized);
        }

        internal::pack_epi32_fallback<BIT_WIDTH>(block_values, output);
        return 0;
    }

    /**
 * @brief Compress multiple 512-bit blocks using fallback scalar implementation.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam BLOCK_SIZE size of each block in bytes (default 64 for 512 bits).
 *
 * @param input pointer to the start of the input float values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @param blocks number of 512-bit blocks to compress.
 * @return int status code (0 for success).
 */
    template<u8 BIT_WIDTH, u32 BLOCK_SIZE>
        requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
    int compress_blocks_fallback(const void * __restrict__ input_ptr, float scale, void * __restrict__ output_ptr,
                                 u32 blocks) {
        const auto *input = static_cast<const float *>(input_ptr);
        auto *output = static_cast<u8 *>(output_ptr);

        const f32 *block_input = input;
        u8 *block_output = output;

        for (u32 block = 0; block < blocks; block++) {
            compress_block_fallback<BIT_WIDTH, BLOCK_SIZE>(block_input, scale, block_output);
            block_input += (BLOCK_SIZE * 8) / BIT_WIDTH;
            block_output += BLOCK_SIZE;
        }

        return 0;
    }

    /**
 * @brief Compress multiple blocks of double values using the fallback scalar implementation.
 *
 * @tparam BIT_WIDTH bit width per value in the packed representation (1 to 24).
 * @tparam BLOCK_SIZE size of each block in bytes (default 64 for 512 bits).
 * @param input pointer to the start of the input double values.
 * @param scale scaling factor used during quantization.
 * @param output pointer to the output buffer where compressed bytes will be stored.
 * @param blocks number of blocks to compress.
 * @return int status code (0 for success).
 */
    template<u8 BIT_WIDTH, u32 BLOCK_SIZE>
        requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
    int compress_blocks_fallback(const void * __restrict__ input_ptr, const f64 scale,
                                 void * __restrict__ output_ptr,
                                 const unsigned int blocks) {
        const auto *input = static_cast<const double *>(input_ptr);
        auto *output = static_cast<u8 *>(output_ptr);

        const f64 *block_input = input;
        u8 *block_output = output;

        for (u32 block = 0; block < blocks; block++) {
            compress_block_fallback<BIT_WIDTH, BLOCK_SIZE>(block_input, scale, block_output);
            block_input += (BLOCK_SIZE * 8) / BIT_WIDTH;
            block_output += BLOCK_SIZE;
        }
        return 0;
    }
} // namespace pernix
#endif  // PERNIX_FALLBACK_COMPRESSION_H
