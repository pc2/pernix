#ifndef PERNIX_FALLBACK_COMMON_H
#define PERNIX_FALLBACK_COMMON_H

#include <pernix/compat.h>

namespace pernix::internal {
__always_inline f32 dequantize_epi32_ps(const i32 input, const f32 scale) {
    return static_cast<f32>(input) * scale;
}

__always_inline f64 dequantize_epi32_pd(const i64 input, const f64 scale) {
    return static_cast<f64>(input) * scale;
}

__always_inline i32 quantize_ps_epi32(const f32 input, const f32 scale) {
    return static_cast<i32>(std::lroundf(input * scale));
}

__always_inline i64 quantize_pd_epi64(const f64 input, const f64 scale) {
    return std::llround(input * scale);
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 32)
__always_inline i32 sign_extend(const u32 value) {
    if constexpr (BIT_WIDTH == 1) {
        return static_cast<i32>(value & 1U);
    }

    constexpr u32 sign_bit = 1 << (BIT_WIDTH - 1);
    constexpr u32 mask     = (1 << BIT_WIDTH) - 1U;

    const u32 masked = value & mask;

    return static_cast<i32>(
        static_cast<int64_t>(masked ^ sign_bit) - static_cast<int64_t>(sign_bit)
    );
}
}

#endif //PERNIX_FALLBACK_COMMON_H
