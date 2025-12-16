#ifndef LIBCOMPRESSION_UNPACKING_AVX2_BMI2_H
#define LIBCOMPRESSION_UNPACKING_AVX2_BMI2_H

#include <libcompression/helper.h>

#ifdef LIBCOMPRESSION_AVX2_ENABLED

#include <immintrin.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace libcompression::bitpacking {
    namespace internal {
        template<uint8_t BIT_WIDTH>
            requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
        __m128i mm_sign_extend32(__m128i source) {
            constexpr uint16_t shift = 32 - BIT_WIDTH;
            source = _mm_slli_epi32(source, shift);
            return _mm_srai_epi32(source, shift);
        }

        template<uint8_t BIT_WIDTH>
            requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
        __m256i mm256_sign_extend32(__m256i source) {
            constexpr uint16_t shift = 32 - BIT_WIDTH;
            source = _mm256_slli_epi32(source, shift);
            return _mm256_srai_epi32(source, shift);
        }
    } // namespace internal

    template<uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
        requires(BIT_WIDTH >= 1 && BIT_WIDTH > 0 && BIT_WIDTH <= 24)
    __m128i mm_unpack_epi32_bmi2(const uint8_t *__restrict__ input) {
        constexpr uint32_t mask = BIT_WIDTH == 32 ? std::numeric_limits<uint32_t>::max() : (1ULL << BIT_WIDTH) - 1U;

        __m128i result;
        if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
            constexpr uint64_t pdep_mask = 0x0101010101010101ULL * mask;

            uint32_t temp_value;
            std::memcpy(&temp_value, input, sizeof(uint32_t));

            const int32_t value = _pdep_u32(temp_value, static_cast<uint32_t>(pdep_mask));
            const __m128i source = _mm_insert_epi32(_mm_setzero_si128(), value, 0);

            result = _mm_cvtepi8_epi32(source);
        } else if constexpr (BIT_WIDTH > 8 && BIT_WIDTH <= 16) {
            constexpr uint64_t pdep_mask = 0x0001000100010001ULL * mask;

            uint64_t temp_value;
            std::memcpy(&temp_value, input, sizeof(uint64_t));

            const int64_t value = _pdep_u64(temp_value, pdep_mask);
            const __m128i source = _mm_insert_epi64(_mm_setzero_si128(), value, 0);

            result = _mm_cvtepi16_epi32(source);
        } else {
            constexpr uint64_t pdep_mask = 0x0000000100000001ULL * mask;
            constexpr uint32_t shift1 = BIT_WIDTH * 2;
            constexpr uint32_t shift2 = 64 - shift1;

            alignas(16) uint64_t temp_values[2];
            std::memcpy(temp_values, input, 2 * sizeof(uint64_t));

            alignas(16) int64_t values[2];
            values[0] = _pdep_u64(temp_values[0], pdep_mask);
            values[1] = _pdep_u64((temp_values[0] >> shift1) | (temp_values[1] << shift2), pdep_mask);

            result = _mm_set_epi64x(values[1], values[0]);
        }

        if constexpr (SIGN_VALUES) {
            result = internal::mm_sign_extend32<BIT_WIDTH>(result);
        }
        return result;
    }

    template<uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
        requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
    __m256i mm256_unpack_epi32_bmi2(const uint8_t *__restrict__ input) {
        constexpr uint32_t mask = BIT_WIDTH == 32 ? std::numeric_limits<uint32_t>::max() : (1ULL << BIT_WIDTH) - 1U;

        __m256i result;
        if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
            constexpr uint64_t pdep_mask = 0x0101010101010101ULL * mask;

            uint64_t temp_value;
            std::memcpy(&temp_value, input, sizeof(uint64_t));

            const int64_t value = _pdep_u64(temp_value, pdep_mask);
            const __m128i source = _mm_insert_epi64(_mm_setzero_si128(), value, 0);

            result = _mm256_cvtepi8_epi32(source);
        } else if constexpr (BIT_WIDTH > 8 && BIT_WIDTH <= 16) {
            constexpr uint64_t pdep_mask = 0x0001000100010001ULL * mask;
            constexpr uint32_t shift1 = BIT_WIDTH * 4;
            constexpr uint32_t shift2 = 64 - shift1;

            alignas(16) uint64_t temp_values[2];
            std::memcpy(temp_values, input, 2 * sizeof(uint64_t));

            alignas(16) int64_t values[2];
            values[0] = _pdep_u64(temp_values[0], pdep_mask);
            values[1] = _pdep_u64((temp_values[0] >> shift1) | (temp_values[1] << shift2), pdep_mask);

            const __m128i source = _mm_set_epi64x(values[1], values[0]);
            result = _mm256_cvtepi16_epi32(source);
        } else {
            constexpr uint64_t pdep_mask = 0x0000000100000001ULL * mask;
            constexpr uint32_t shift1 = BIT_WIDTH * 2;
            constexpr uint32_t shift2 = 64 - shift1;

            alignas(16) uint64_t temp_values[4];
            std::memcpy(temp_values, input, 2 * sizeof(uint64_t));
            std::memcpy(temp_values + 2, input + BIT_WIDTH / 2, 2 * sizeof(uint64_t));

            alignas(16) uint64_t values[4];
            values[0] = _pdep_u64((temp_values[0]), pdep_mask);
            values[1] = _pdep_u64((temp_values[0] >> shift1) | (temp_values[1] << shift2), pdep_mask);
            values[2] = _pdep_u64((temp_values[2]), pdep_mask);
            values[3] = _pdep_u64((temp_values[2] >> shift1) | (temp_values[3] << shift2), pdep_mask);

            result = _mm256_set_epi64x(static_cast<int64_t>(values[3]), static_cast<int64_t>(values[2]),
                                       static_cast<int64_t>(values[1]),
                                       static_cast<int64_t>(values[0]));
        }

        if constexpr (SIGN_VALUES) {
            result = internal::mm256_sign_extend32<BIT_WIDTH>(result);
        }
        return result;
    }

    auto mm_unpack_epi32_bmi2(uint8_t bit_width, const uint8_t *__restrict__ input) -> __m128i;

    auto mm256_unpack_epi32_bmi2(uint8_t bit_width, const uint8_t *__restrict__ input) -> __m256i;
} // namespace libcompression::bitpacking
#endif  // LIBCOMPRESSION_AVX2_ENABLED
#endif  // LIBCOMPRESSION_UNPACKING_AVX2_BMI2_H
