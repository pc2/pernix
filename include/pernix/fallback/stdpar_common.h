#ifndef PERNIX_FALLBACK_STDPAR_COMMON_H
#define PERNIX_FALLBACK_STDPAR_COMMON_H

#include <pernix/compat.h>

#include <algorithm>
#include <cstddef>
#if defined(PERNIX_STDPAR_USE_PARALLEL_POLICY)
#include <execution>
#endif
#include <iterator>
#include <utility>

namespace pernix::internal {
inline constexpr u32 stdpar_group_size_v = 8;

template <u8 GROUP_SIZE, u8 BIT_WIDTH>
    requires(GROUP_SIZE >= 1 && GROUP_SIZE <= 8 && BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
inline constexpr usize packed_group_bytes_v = (static_cast<usize>(GROUP_SIZE) * BIT_WIDTH + 7U) / 8U;

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

template <typename Func>
__always_inline void for_each_index_stdpar(const u32 count, Func&& func) {
    auto first = counting_iterator<u32>{0};
#if defined(PERNIX_STDPAR_USE_PARALLEL_POLICY)
    std::for_each_n(std::execution::par_unseq, first, count, std::forward<Func>(func));
#else
    std::for_each_n(first, count, std::forward<Func>(func));
#endif
}
}  // namespace pernix::internal

#endif  // PERNIX_FALLBACK_STDPAR_COMMON_H
