#ifndef PERNIX_ARM64_NEON_DECOMPRESSION_H
#define PERNIX_ARM64_NEON_DECOMPRESSION_H

#include <pernix/simd_compat.h>
#include <pernix/arm64/neon/unpacking.h>
#include <pernix/arm64/neon/common.h>

#include <cmath>
#include <cstdint>

namespace pernix::arm64::neon {
namespace internal {
template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8) && (BLOCK_SIZE % 32 == 0)
__always_inline int neon_decompress_block_1to8(const uint8_t* __restrict__ input, const float_t scale,
                                               float_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr uint32_t iterations_16      = elements_per_block / 16;
    constexpr uint32_t remaining_elements = elements_per_block - iterations_16 * 16;

    const float32x4_t scale_v = vdupq_n_f32(scale);

    for (uint32_t i = 0; i < iterations_16; ++i) {
        const uint8x16_t source  = vld1q_u8(input);
        const int8x16_t unpacked = b128::neon_unpack_epi8_1to8<BIT_WIDTH, SIGN_VALUES>(source);

        const int32x4x4_t converted     = neon_convert_int8x16_int32x4x4(unpacked);
        const float32x4x4_t dequantized = neon_dequantize_epi32(converted, scale_v);

        for (uint32_t j = 0; j < 4; ++j) {
            vst1q_f32(output, dequantized.val[j]);
            output += 4;
        }

        input += 2 * BIT_WIDTH;
    }

    if constexpr (remaining_elements > 0) {
        const uint8x16_t tail_source  = neon_load_tail_elements_int8(input, tail_bytes(BIT_WIDTH, remaining_elements));
        const int8x16_t tail_unpacked = b128::neon_unpack_epi8_1to8<BIT_WIDTH, SIGN_VALUES>(tail_source);

        const int32x4x4_t tail_converted     = neon_convert_int8x16_int32x4x4(tail_unpacked);
        const float32x4x4_t tail_dequantized = neon_dequantize_epi32(tail_converted, scale_v);

        neon_store_tail_elements_f32(output, tail_dequantized, remaining_elements);
    }

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16) && (BLOCK_SIZE % 32 == 0)
__always_inline int neon_decompress_block_9to16(const uint8_t* __restrict__ input, const float_t scale,
                                                float_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr uint32_t iterations_8       = elements_per_block / 8;
    constexpr uint32_t remaining_elements = elements_per_block - iterations_8 * 8;

    const float32x4_t scale_v = vdupq_n_f32(scale);

    for (uint32_t i = 0; i < iterations_8; ++i) {
        const uint16x8_t source  = vld1q_u16(reinterpret_cast<const uint16_t*>(input));
        const int16x8_t unpacked = b128::neon_unpack_epi8_9to16<BIT_WIDTH, SIGN_VALUES>(source);

        const int32x4x2_t converted     = neon_convert_int16x8_int32x4x2(unpacked);
        const float32x4x2_t dequantized = neon_dequantize_epi32(converted, scale_v);

        for (uint32_t j = 0; j < 2; ++j) {
            vst1q_f32(output, dequantized.val[j]);
            output += 4;
        }

        input += BIT_WIDTH;
    }

    if constexpr (remaining_elements > 0) {
        const uint16x8_t tail_source  = neon_load_tail_elements_int16(input, tail_bytes(BIT_WIDTH, remaining_elements));
        const int16x8_t tail_unpacked = b128::neon_unpack_epi8_9to16<BIT_WIDTH, SIGN_VALUES>(tail_source);

        const int32x4x2_t tail_converted     = neon_convert_int16x8_int32x4x2(tail_unpacked);
        const float32x4x2_t tail_dequantized = neon_dequantize_epi32(tail_converted, scale_v);

        neon_store_tail_elements_f32(output, tail_dequantized, remaining_elements);
    }

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int neon_decompress_block_17to24(const uint8_t* __restrict__ input, const float_t scale,
                                                 float_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr uint32_t iterations_4       = elements_per_block / 4;
    constexpr uint32_t remaining_elements = elements_per_block - iterations_4 * 4;

    const float32x4_t scale_v = vdupq_n_f32(scale);

    for (uint32_t i = 0; i < iterations_4; ++i) {
        const uint32_t group_bit_start = i * 4u * BIT_WIDTH;
        const uint8_t* group_input     = input + group_bit_start / 8u;
        const uint32x4_t source        = vld1q_u32(reinterpret_cast<const uint32_t*>(group_input));

        int32x4_t unpacked;
        if constexpr (BIT_WIDTH % 2 == 0) {
            unpacked = b128::neon_unpack_epi8_17to24<BIT_WIDTH, SIGN_VALUES, 0>(source);
        } else {
            if (i % 2 == 0) {
                unpacked = b128::neon_unpack_epi8_17to24<BIT_WIDTH, SIGN_VALUES, 0>(source);
            } else {
                unpacked = b128::neon_unpack_epi8_17to24<BIT_WIDTH, SIGN_VALUES, 4>(source);
            }
        }

        const float32x4_t dequantized = neon_dequantize_epi32(unpacked, scale_v);

        vst1q_f32(output, dequantized);

        output += 4;
    }

    if constexpr (remaining_elements > 0) {
        constexpr uint32_t tail_bit_start = iterations_4 * 4u * BIT_WIDTH;
        constexpr uint32_t tail_bit_offset = tail_bit_start % 8u;
        const uint8_t* tail_input = input + tail_bit_start / 8u;

        constexpr uint32_t tail_bytes_count = (tail_bit_offset + remaining_elements * BIT_WIDTH + 7u) / 8u;
        const uint32x4_t tail_source = neon_load_tail_elements_int32(tail_input, tail_bytes_count);

        int32x4_t tail_unpacked;
        if constexpr (tail_bit_offset == 0) {
            tail_unpacked = b128::neon_unpack_epi8_17to24<BIT_WIDTH, SIGN_VALUES, 0>(tail_source);
        } else {
            tail_unpacked = b128::neon_unpack_epi8_17to24<BIT_WIDTH, SIGN_VALUES, tail_bit_offset>(tail_source);
        }

        const float32x4_t tail_dequantized = neon_dequantize_epi32(tail_unpacked, scale_v);

        neon_store_tail_elements_f32(output, tail_dequantized, remaining_elements);
    }

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8) && (BLOCK_SIZE % 32 == 0)
__always_inline int neon_decompress_block_1to8(const uint8_t* __restrict__ input, const double_t scale,
                                               double_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr uint32_t iterations_8       = elements_per_block / 8;
    constexpr uint32_t remaining_elements = elements_per_block - iterations_8 * 8;

    for (uint32_t i = 0; i < iterations_8; ++i) {
        static_assert(true, "Not yet implemented");
    }

    if constexpr (remaining_elements > 0) {
        static_assert(true, "Not yet implemented");
    }

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16) && (BLOCK_SIZE % 32 == 0)
__always_inline int neon_decompress_block_9to16(const uint8_t* __restrict__ input, const double_t scale,
                                                double_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr uint32_t iterations_4       = elements_per_block / 4;
    constexpr uint32_t remaining_elements = elements_per_block - iterations_4 * 4;

    for (uint32_t i = 0; i < iterations_4; ++i) {
        static_assert(true, "Not yet implemented");
    }

    if constexpr (remaining_elements > 0) {
        static_assert(true, "Not yet implemented");
    }

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int neon_decompress_block_17to24(const uint8_t* __restrict__ input, const double_t scale,
                                                 double_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    constexpr uint32_t iterations_2       = elements_per_block / 2;
    constexpr uint32_t remaining_elements = elements_per_block - iterations_2 * 2;

    for (uint32_t i = 0; i < iterations_2; ++i) {
        static_assert(true, "Not yet implemented");
    }

    if constexpr (remaining_elements > 0) {
        static_assert(true, "Not yet implemented");
    }

    return 0;
}
} // namespace internal

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int neon_decompress_block(const uint8_t* __restrict__ input, const float_t scale,
                                          float_t* __restrict__ output) {
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
        return internal::neon_decompress_block_1to8<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
        return internal::neon_decompress_block_9to16<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH >= 17 && BIT_WIDTH <= 24) {
        return internal::neon_decompress_block_17to24<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    }
    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int neon_decompress_block(const uint8_t* __restrict__ input, const double_t scale,
                                          double_t* __restrict__ output) {
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
        return internal::neon_decompress_block_1to8<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
        return internal::neon_decompress_block_9to16<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    } else if constexpr (BIT_WIDTH >= 17 && BIT_WIDTH <= 24) {
        return internal::neon_decompress_block_17to24<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(input, scale, output);
    }
    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int neon_decompress_blocks(const uint8_t* __restrict__ input, const float_t scale, float_t* __restrict__ output,
                           const uint32_t blocks) {
    const uint8_t* block_input = input;
    float_t* block_output      = output;

    for (uint32_t block = 0; block < blocks; ++block) {
        neon_decompress_block<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(block_input, scale, block_output);
        block_input  += BLOCK_SIZE;
        block_output += (BLOCK_SIZE * 8) / BIT_WIDTH;
    }

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE = 64>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int neon_decompress_blocks(const uint8_t* __restrict__ input, const double_t scale, double_t* __restrict__ output,
                           const uint32_t blocks) {
    const uint8_t* block_input = input;
    double_t* block_output     = output;

    for (uint32_t block = 0; block < blocks; ++block) {
        neon_decompress_block<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(block_input, scale, block_output);
        block_input  += BLOCK_SIZE;
        block_output += (BLOCK_SIZE * 8) / BIT_WIDTH;
    }
    return 0;
}

#ifdef __cplusplus
extern "C" {
#endif

int neon_decompress_block(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output);


int neon_decompress_block_f64(uint8_t bit_width, const uint8_t* __restrict__ input, double_t scale,
                              double_t* __restrict__ output);

int neon_decompress_blocks(uint8_t bit_width, const uint8_t* __restrict__ input, float_t scale, float_t* __restrict__ output,
                           uint32_t blocks);

int neon_decompress_blocks_f64(uint8_t bit_width, const uint8_t* __restrict__ input, double_t scale,
                               double_t* __restrict__ output, uint32_t blocks);

#ifdef __cplusplus
}
#endif
} // namespace pernix::arm64::neon

#endif  // PERNIX_ARM64_NEON_DECOMPRESSION_H
