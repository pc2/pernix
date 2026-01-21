#ifndef PERNIX_PACKING_FALLBACK_H
#define PERNIX_PACKING_FALLBACK_H

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace pernix::bitpacking {
namespace internal {
template <typename T, uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_integral_v<T> && std::is_unsigned_v<T>)
void pack_epi32_fallback_inner(const std::vector<uint32_t>& input, const uint8_t bit_offset, uint8_t* __restrict__ destination) {
    constexpr uint32_t bits_in_type = sizeof(T) * 8;
    constexpr uint32_t typemask     = std::numeric_limits<T>::max();
    constexpr T bitmask             = BIT_WIDTH == bits_in_type ? std::numeric_limits<T>::max() : (1U << BIT_WIDTH) - 1U;

    std::span<T> destination_span(reinterpret_cast<T*>(destination),
                                  (input.size() * BIT_WIDTH + bit_offset + bits_in_type - 1) / bits_in_type);

    std::size_t idx            = 0;
    std::size_t bits_in_buffer = bit_offset;
    uint64_t buffer            = 0;

    if (bit_offset) {
        buffer = static_cast<uint64_t>(destination_span[0] & (1ULL << bit_offset) - 1ULL);
        idx++;
    }

    // #pragma GCC unroll 512
    for (uint32_t raw_value : input) {
        const uint32_t next_value = raw_value & bitmask;

        buffer |= static_cast<uint64_t>(next_value) << bits_in_buffer;
        bits_in_buffer += BIT_WIDTH;

        if (bits_in_buffer >= bits_in_type) {
            destination_span[idx++] = buffer & typemask;
            buffer >>= bits_in_type;
            bits_in_buffer -= bits_in_type;
        }
    }

    if (bits_in_buffer > 0) {
        destination_span[idx] = buffer & typemask;
    }
}
}  // namespace internal

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
void pack_epi32_fallback(const std::vector<uint32_t>& input, uint8_t* __restrict__ destination) {
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
        return internal::pack_epi32_fallback_inner<uint8_t, BIT_WIDTH>(input, 0, destination);
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
        return internal::pack_epi32_fallback_inner<uint16_t, BIT_WIDTH>(input, 0, destination);
    } else {
        return internal::pack_epi32_fallback_inner<uint32_t, BIT_WIDTH>(input, 0, destination);
    }
}

void pack_epi32_fallback(uint8_t bit_width, const std::vector<uint32_t>& input, uint8_t* __restrict__ destination);
}  // namespace pernix::bitpacking

#endif  // PERNIX_PACKING_FALLBACK_H