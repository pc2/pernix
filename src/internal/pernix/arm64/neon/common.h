#ifndef PERNIX_ARM64_NEON_COMMON_H
#define PERNIX_ARM64_NEON_COMMON_H

#include <pernix/simd_compat.h>

#include <cstring>

namespace pernix::arm64::neon::internal {
    struct float64x2x8_t {
        float64x2_t val[8];
    };

    static constexpr uint32_t tail_bytes(const uint8_t bit_width, const uint32_t remaining_elements) {
        const uint32_t tail_bits = remaining_elements * bit_width;
        const uint32_t tail_bytes = (tail_bits + 7u) / 8u;
        return tail_bytes;
    }

__always_inline int32x4x4_t neon_convert_int8x16_int32x4x4(const int8x16_t &input) {
        const int16x8_t s16_lo = vmovl_s8(vget_low_s8(input));
        const int16x8_t s16_hi = vmovl_s8(vget_high_s8(input));

        return {
            {
                vmovl_s16(vget_low_s16(s16_lo)),
                vmovl_s16(vget_high_s16(s16_lo)),
                vmovl_s16(vget_low_s16(s16_hi)),
                vmovl_s16(vget_high_s16(s16_hi)),
            }
        };
    }

__always_inline int32x4x2_t neon_convert_int16x8_int32x4x2(const int16x8_t &input) {
        return {
            {
                vmovl_s16(vget_low_s16(input)),
                vmovl_s16(vget_high_s16(input)),
            }
        };
    }

__always_inline float32x4x4_t neon_dequantize_epi32(const int32x4x4_t &input, const float32x4_t &scale) {
        return {
            {
                vmulq_f32(vcvtq_f32_s32(input.val[0]), scale),
                vmulq_f32(vcvtq_f32_s32(input.val[1]), scale),
                vmulq_f32(vcvtq_f32_s32(input.val[2]), scale),
                vmulq_f32(vcvtq_f32_s32(input.val[3]), scale),
            }
        };
    }

__always_inline float32x4x2_t neon_dequantize_epi32(const int32x4x2_t &input, const float32x4_t &scale) {
        return {
            {
                vmulq_f32(vcvtq_f32_s32(input.val[0]), scale),
                vmulq_f32(vcvtq_f32_s32(input.val[1]), scale),
            }
        };
    }

__always_inline float32x4_t neon_dequantize_epi32(const int32x4_t &input, const float32x4_t &scale) {
        return vmulq_f32(vcvtq_f32_s32(input), scale);
    }

__always_inline float64x2_t neon_dequantize_epi32_f64(const int32x2_t &input, const float64x2_t &scale) {
        return vmulq_f64(vcvtq_f64_s64(vmovl_s32(input)), scale);
    }

__always_inline float64x2x2_t neon_dequantize_epi32_f64(const int32x4_t &input, const float64x2_t &scale) {
        return {
            {
                neon_dequantize_epi32_f64(vget_low_s32(input), scale),
                neon_dequantize_epi32_f64(vget_high_s32(input), scale),
            }
        };
    }

__always_inline float64x2x4_t neon_dequantize_epi32_f64(const int32x4x2_t &input, const float64x2_t &scale) {
        const float64x2x2_t dequantized_low = neon_dequantize_epi32_f64(input.val[0], scale);
        const float64x2x2_t dequantized_high = neon_dequantize_epi32_f64(input.val[1], scale);

        return {
            {
                dequantized_low.val[0],
                dequantized_low.val[1],
                dequantized_high.val[0],
                dequantized_high.val[1],
            }
        };
    }

__always_inline float64x2x8_t neon_dequantize_epi32_f64(const int32x4x4_t &input, const float64x2_t &scale) {
        const float64x2x2_t dequantized0 = neon_dequantize_epi32_f64(input.val[0], scale);
        const float64x2x2_t dequantized1 = neon_dequantize_epi32_f64(input.val[1], scale);
        const float64x2x2_t dequantized2 = neon_dequantize_epi32_f64(input.val[2], scale);
        const float64x2x2_t dequantized3 = neon_dequantize_epi32_f64(input.val[3], scale);

        return {
            {
                dequantized0.val[0],
                dequantized0.val[1],
                dequantized1.val[0],
                dequantized1.val[1],
                dequantized2.val[0],
                dequantized2.val[1],
                dequantized3.val[0],
                dequantized3.val[1],
            }
        };
    }

__always_inline uint8x16_t neon_load_tail_elements_int8(const uint8_t *input, const uint32_t tail_bytes_count) {
        uint8_t buffer[16] = {0};
        std::memcpy(buffer, input, tail_bytes_count);
        return vld1q_u8(buffer);
    }

__always_inline uint16x8_t neon_load_tail_elements_int16(const uint8_t *input, const uint32_t tail_bytes_count) {
        uint16_t buffer[8] = {0};
        std::memcpy(buffer, input, tail_bytes_count);
        return vld1q_u16(buffer);
    }

__always_inline uint32x4_t neon_load_tail_elements_int32(const uint8_t *input, const uint32_t tail_bytes_count) {
        uint32_t buffer[4] = {0};
        std::memcpy(buffer, input, tail_bytes_count);
        return vld1q_u32(buffer);
    }

__always_inline float32x4_t neon_load_tail_elements_f32(const uint8_t *input, const uint32_t tail_elements) {
        float32_t buffer[4] = {0.0f};
        std::memcpy(buffer, input, tail_elements * sizeof(float32_t));
        return vld1q_f32(buffer);
    }

__always_inline float64x2_t neon_load_tail_elements_f64(const uint8_t *input, const uint32_t tail_elements) {
        float64_t buffer[2] = {0.0};
        std::memcpy(buffer, input, tail_elements * sizeof(float64_t));
        return vld1q_f64(buffer);
    }

__always_inline void neon_store_tail_elements_int8(uint8_t *output, const uint8x16x4_t &data,
                                                   const uint32_t tail_elements) {
        uint8_t buffer[16 * 4];
        for (uint32_t i = 0; i < 4; ++i) {
            vst1q_u8(buffer + i * 16, data.val[i]);
        }
        std::memcpy(output, buffer, tail_elements * sizeof(uint8_t));
    }

__always_inline void neon_store_tail_elements_int16(uint16_t *output, const uint16x8x4_t &data,
                                                    const uint32_t tail_elements) {
        uint16_t buffer[8 * 4];
        for (uint32_t i = 0; i < 4; ++i) {
            vst1q_u16(buffer + i * 8, data.val[i]);
        }
        std::memcpy(output, buffer, tail_elements * sizeof(uint16_t));
    }

__always_inline void neon_store_tail_elements_int32(uint32_t *output, const uint32x4x4_t &data,
                                                    const uint32_t tail_elements) {
        uint32_t buffer[4 * 4];
        for (uint32_t i = 0; i < 4; ++i) {
            vst1q_u32(buffer + i * 4, data.val[i]);
        }
        std::memcpy(output, buffer, tail_elements * sizeof(uint32_t));
    }

__always_inline void neon_store_tail_elements_f32(float32_t *output, const float32x4x4_t &data,
                                                  const uint32_t tail_elements) {
        float32_t buffer[16 * 4];
        for (uint32_t i = 0; i < 4; ++i) {
            vst1q_f32(buffer + i * 4, data.val[i]);
        }
        std::memcpy(output, buffer, tail_elements * sizeof(float32_t));
    }

__always_inline void neon_store_tail_elements_f32(float32_t *output, const float32x4x2_t &data,
                                                  const uint32_t tail_elements) {
        float32_t buffer[8 * 2];
        for (uint32_t i = 0; i < 2; ++i) {
            vst1q_f32(buffer + i * 4, data.val[i]);
        }
        std::memcpy(output, buffer, tail_elements * sizeof(float32_t));
    }

__always_inline void neon_store_tail_elements_f32(float32_t *output, const float32x4_t &data,
                                                  const uint32_t tail_elements) {
        float32_t buffer[4];
        vst1q_f32(buffer, data);
        std::memcpy(output, buffer, tail_elements * sizeof(float32_t));
    }

__always_inline void neon_store_tail_elements_f64(float64_t *output, const float64x2x4_t &data,
                                                  const uint32_t tail_elements) {
        float64_t buffer[2 * 4];
        for (uint32_t i = 0; i < 4; ++i) {
            vst1q_f64(buffer + i * 2, data.val[i]);
        }
        std::memcpy(output, buffer, tail_elements * sizeof(float64_t));
    }

__always_inline void neon_store_tail_elements_f64(float64_t *output, const float64x2x2_t &data,
                                                  const uint32_t tail_elements) {
        float64_t buffer[2 * 2];
        for (uint32_t i = 0; i < 2; ++i) {
            vst1q_f64(buffer + i * 2, data.val[i]);
        }
        std::memcpy(output, buffer, tail_elements * sizeof(float64_t));
    }

__always_inline void neon_store_tail_elements_f64(float64_t *output, const float64x2x8_t &data,
                                                  const uint32_t tail_elements) {
        float64_t buffer[2 * 8];
        for (uint32_t i = 0; i < 8; ++i) {
            vst1q_f64(buffer + i * 2, data.val[i]);
        }
        std::memcpy(output, buffer, tail_elements * sizeof(float64_t));
    }
} // namespace pernix::arm64::neon::internal

#endif  // PERNIX_ARM64_NEON_COMMON_H
