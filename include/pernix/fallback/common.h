#ifndef PERNIX_FALLBACK_COMMON_H
#define PERNIX_FALLBACK_COMMON_H

#include <pernix/compat.h>

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace pernix::internal {

template <u8 BIT_WIDTH, typename FloatT, typename T>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<FloatT> && std::is_integral_v<T>)
__always_inline T quantize_clamped(const FloatT input, const FloatT scale) {
    constexpr T min_value = BIT_WIDTH == 1 ? 0 : -(1 << (BIT_WIDTH - 1));

    constexpr T max_value = BIT_WIDTH == 1 ? 1 : (1 << (BIT_WIDTH - 1)) - 1;

    const FloatT scaled = input * scale;

    if (std::isnan(scaled)) [[unlikely]] {
        return 0;
    }

    const FloatT clamped = std::clamp(scaled, static_cast<FloatT>(min_value), static_cast<FloatT>(max_value));

    return static_cast<T>(std::lrint(clamped));
}

template <typename T, typename FloatT>
    requires(std::is_integral_v<T> && std::is_floating_point_v<FloatT>)
__always_inline FloatT dequantize(const T input, const FloatT scale) {
    return static_cast<FloatT>(input) * scale;
}

template <u8 BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
__always_inline constexpr i32 sign_extend(const u32 value) noexcept {
    if constexpr (BIT_WIDTH == 1) {
        return static_cast<i32>(value & 1);
    }

    constexpr u32 sign_bit = 1 << (BIT_WIDTH - 1);
    constexpr u32 mask     = (1 << BIT_WIDTH) - 1;

    const u32 masked = value & mask;

    return static_cast<i32>(static_cast<i64>(masked ^ sign_bit) - static_cast<i64>(sign_bit));
}
}  // namespace pernix::internal

#endif  // PERNIX_FALLBACK_COMMON_H
