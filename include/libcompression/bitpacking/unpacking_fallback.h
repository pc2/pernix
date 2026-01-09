#ifndef LIBCOMPRESSION_UNPACKING_FALLBACK_H
#define LIBCOMPRESSION_UNPACKING_FALLBACK_H

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace libcompression::bitpacking {
    namespace internal {
        template<uint8_t BIT_WIDTH>
            requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
        auto sign_extend(const uint32_t value) -> int32_t {
            constexpr uint32_t shift = 32 - BIT_WIDTH;
            return (static_cast<int32_t>(value) << shift) >> shift;
        }

        template<typename T, uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
            requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_integral_v<T> && std::is_unsigned_v<T>)
        auto unpack_epi32_fallback_inner(const T *__restrict__ input, const uint8_t bit_offset,
                                         const std::size_t elements)
            -> std::vector<int32_t> {
            constexpr uint32_t bits_in_type = sizeof(T) * 8;
            constexpr T bitmask = BIT_WIDTH == bits_in_type ? std::numeric_limits<T>::max() : (1U << BIT_WIDTH) - 1U;

            std::span<const T> input_span(input, (elements * BIT_WIDTH + bit_offset + bits_in_type - 1) / bits_in_type);
            std::vector<int32_t> output(elements);

            std::size_t idx = 0;
            uint8_t bits_in_buffer = bits_in_type - bit_offset;
            uint64_t buffer = static_cast<uint64_t>(input_span[idx++]) >> bit_offset;

            // #pragma GCC unroll 65534
            for (uint32_t i = 0; i < elements; i++) {
                if (BIT_WIDTH > bits_in_buffer) {
                    const auto next_value = static_cast<uint64_t>(input_span[idx++]) << bits_in_buffer;
                    buffer |= next_value;
                    bits_in_buffer += bits_in_type;
                }

                const T raw_value = static_cast<T>(buffer & bitmask);
                if constexpr (SIGN_VALUES) {
                    output[i] = sign_extend<BIT_WIDTH>(raw_value);
                } else {
                    output[i] = raw_value;
                }

                buffer >>= BIT_WIDTH;
                bits_in_buffer -= BIT_WIDTH;
            }

            return output;
        }
    } // namespace internal

    template<uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
        requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
    auto unpack_epi32_fallback(const uint8_t *__restrict__ input, const std::size_t elements) -> std::vector<int32_t> {
        if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
            return internal::unpack_epi32_fallback_inner<uint8_t, BIT_WIDTH, SIGN_VALUES>(
                reinterpret_cast<const uint8_t *>(input), 0, elements);
        } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
            return internal::unpack_epi32_fallback_inner<uint16_t, BIT_WIDTH, SIGN_VALUES>(
                reinterpret_cast<const uint16_t *>(input), 0, elements);
        } else {
            return internal::unpack_epi32_fallback_inner<uint32_t, BIT_WIDTH, SIGN_VALUES>(
                reinterpret_cast<const uint32_t *>(input), 0,
                elements);
        }
    }

    auto unpack_epi32_fallback(uint8_t bit_width, const uint8_t *__restrict__ input,
                               std::size_t elements) -> std::vector<int32_t>;
} // namespace libcompression::bitpacking

#endif  // LIBCOMPRESSION_UNPACKING_FALLBACK_H
