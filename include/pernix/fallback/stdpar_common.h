#ifndef PERNIX_FALLBACK_STDPAR_COMMON_H
#define PERNIX_FALLBACK_STDPAR_COMMON_H

#include <pernix/compat.h>

#include <array>
#include <cmath>
#include <execution>
#include <type_traits>

namespace pernix::internal {
template <class T, u8 CAPACITY = 8>
struct stdpar_buffer {
    std::array<T, CAPACITY> data{};
    usize size = 0;
};

template <u8 BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
__always_inline i32 sign_extend_stdpar(const u32 value) {
    if constexpr (BIT_WIDTH == 1) {
        return static_cast<i32>(value & 1U);
    }

    constexpr u32 sign_bit = u32{1} << (BIT_WIDTH - 1);
    constexpr u32 mask     = (u32{1} << BIT_WIDTH) - 1U;
    const u32 masked       = value & mask;
    return static_cast<i32>((static_cast<i64>(masked ^ sign_bit)) - static_cast<i64>(sign_bit));
}

template <u8 GROUP_SIZE, u8 BIT_WIDTH>
    requires(GROUP_SIZE >= 1 && GROUP_SIZE <= 8 && BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
inline constexpr usize packed_group_bytes_v = (static_cast<usize>(GROUP_SIZE) * BIT_WIDTH + 7U) / 8U;

template <typename ScaleType>
    requires(std::is_floating_point_v<ScaleType>)
__always_inline ScaleType dequantize_stdpar_value(const i32 input, const ScaleType scale) {
    if constexpr (std::is_same_v<ScaleType, f32>) {
        return static_cast<f32>(input) * scale;
    } else {
        return static_cast<f64>(input) * scale;
    }
}

template <u8 BIT_WIDTH, typename ScaleType>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<ScaleType>)
__always_inline u32 quantize_stdpar_value(const ScaleType input, const ScaleType scale) {
    constexpr i64 min_value = BIT_WIDTH == 1 ? 0 : -(i64{1} << (BIT_WIDTH - 1));
    constexpr i64 max_value = BIT_WIDTH == 1 ? 1 : ((i64{1} << (BIT_WIDTH - 1)) - 1);

    const long double scaled = static_cast<long double>(input) * static_cast<long double>(scale);
    if (std::isnan(scaled)) {
        return 0;
    }
    if (scaled <= static_cast<long double>(min_value)) {
        return static_cast<u32>(static_cast<i32>(min_value));
    }
    if (scaled >= static_cast<long double>(max_value)) {
        return static_cast<u32>(static_cast<i32>(max_value));
    }

    return static_cast<u32>(static_cast<i32>(std::llround(scaled)));
}

template <typename T>
class counting_iterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using iterator_concept  = std::random_access_iterator_tag;
    using value_type        = T;
    using difference_type   = std::ptrdiff_t;
    using pointer           = void;
    using reference         = T;

    constexpr explicit counting_iterator(T value) noexcept : value_(value) {}

    constexpr T operator*() const noexcept { return value_; }

    constexpr T operator[](difference_type n) const noexcept { return value_ + n; }

    constexpr counting_iterator& operator++() noexcept {
        ++value_;
        return *this;
    }

    constexpr counting_iterator operator++(i32) noexcept {
        counting_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    constexpr counting_iterator& operator--() noexcept {
        --value_;
        return *this;
    }

    constexpr counting_iterator operator--(i32) noexcept {
        counting_iterator tmp = *this;
        --(*this);
        return tmp;
    }

    constexpr counting_iterator& operator+=(difference_type n) noexcept {
        value_ += static_cast<T>(n);
        return *this;
    }

    constexpr counting_iterator& operator-=(difference_type n) noexcept {
        value_ -= static_cast<T>(n);
        return *this;
    }

    friend constexpr auto operator+(counting_iterator it, difference_type n) noexcept -> counting_iterator {
        it += n;
        return it;
    }

    friend constexpr auto operator+(difference_type n, counting_iterator it) noexcept -> counting_iterator {
        it += n;
        return it;
    }

    friend constexpr auto operator-(counting_iterator it, difference_type n) noexcept -> counting_iterator {
        it -= n;
        return it;
    }

    friend constexpr difference_type operator-(counting_iterator lhs, counting_iterator rhs) noexcept {
        return static_cast<difference_type>(lhs.value_) - static_cast<difference_type>(rhs.value_);
    }

    friend constexpr bool operator==(counting_iterator lhs, counting_iterator rhs) noexcept { return lhs.value_ == rhs.value_; }

    friend constexpr bool operator!=(counting_iterator lhs, counting_iterator rhs) noexcept { return !(lhs == rhs); }

    friend constexpr bool operator<(counting_iterator lhs, counting_iterator rhs) noexcept { return lhs.value_ < rhs.value_; }

    friend constexpr bool operator<=(counting_iterator lhs, counting_iterator rhs) noexcept { return lhs.value_ <= rhs.value_; }

    friend constexpr bool operator>(counting_iterator lhs, counting_iterator rhs) noexcept { return lhs.value_ > rhs.value_; }

    friend constexpr bool operator>=(counting_iterator lhs, counting_iterator rhs) noexcept { return lhs.value_ >= rhs.value_; }

private:
    T value_;
};
}  // namespace pernix::internal

#endif  // PERNIX_FALLBACK_STDPAR_COMMON_H
