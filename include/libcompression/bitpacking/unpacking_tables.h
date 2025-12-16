#ifndef LIBCOMPRESSION_TABLES_H
#define LIBCOMPRESSION_TABLES_H

#include <immintrin.h>
#include <libcompression/helper.h>

#include <array>
#include <cstdint>

#ifdef LIBCOMPRESSION_AVX2_ENABLED
namespace libcompression::bitpacking::internal {
template <uint8_t BIT_WIDTH, typename T>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24 && (std::is_same_v<T, __m128i> || std::is_same_v<T, __m256i>))
struct unpack_tables_avx2 {
    alignas(32) inline static constexpr std::array<int32_t, 8> permute = [] {
        // clang-format off
        if constexpr (BIT_WIDTH == 1 || BIT_WIDTH == 2 || BIT_WIDTH == 3 || BIT_WIDTH == 4) {
            return std::array{0, -1, -1, -1, 0, -1, -1, -1};
        } else if constexpr (BIT_WIDTH == 5 || BIT_WIDTH == 6 || BIT_WIDTH == 7) {
            return std::array{0, -1, -1, -1, 0, 1, -1, -1};
        } else if constexpr (BIT_WIDTH == 8) {
            return std::array{0, -1, -1, -1, 1, 0, -1, -1};
        } else if constexpr (BIT_WIDTH == 9 || BIT_WIDTH == 10 || BIT_WIDTH == 11 || BIT_WIDTH == 12) {
            return std::array{0, 1, -1, -1, 1, 2, -1, -1};
        } else if constexpr (BIT_WIDTH == 13 || BIT_WIDTH == 14 || BIT_WIDTH == 15) {
            return std::array{0, 1, -1, -1, 1, 2, 3, -1};
        } else if constexpr (BIT_WIDTH == 16) {
            return std::array{0, 1, -1, -1, 2, 3, -1, -1};
        } else if constexpr (BIT_WIDTH == 17 || BIT_WIDTH == 18 || BIT_WIDTH == 19 || BIT_WIDTH == 20) {
            return std::array{0, 1, 2, -1, 2, 3, 4, -1};
        } else if constexpr (BIT_WIDTH == 21 || BIT_WIDTH == 22 || BIT_WIDTH == 23) {
            return std::array{0, 1, 2, -1, 2, 3, 4, 5};
        } else if constexpr (BIT_WIDTH == 24) {
            return std::array{0, 1, 2, -1, 3, 4, 5, -1};
        }
        // clang-format on
        return std::array<int32_t, 8>{};
    }();

    alignas(32) inline static constexpr std::array<int8_t, 32> shuffle = [] {
        // clang-format off
        if constexpr (BIT_WIDTH == 1) {
            return std::array<int8_t, 32>{
                0, -1, -1, -1,
                0, -1, -1, -1,
                0, -1, -1, -1,
                0, -1, -1, -1,

                0, -1, -1, -1,
                0, -1, -1, -1,
                0, -1, -1, -1,
                0, -1, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 2) {
            return std::array<int8_t, 32>{
                0, -1, -1, -1,
                0, -1, -1, -1,
                0, -1, -1, -1,
                0, -1, -1, -1,

                1, -1, -1, -1,
                1, -1, -1, -1,
                1, -1, -1, -1,
                1, -1, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 3) {
            return std::array<int8_t, 32>{
                0, -1, -1, -1,
                0, -1, -1, -1,
                0, 1, -1, -1,
                1, -1, -1, -1,

                1, -1, -1, -1,
                1, 2, -1, -1,
                2, -1, -1, -1,
                2, -1, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 4) {
            return std::array<int8_t, 32>{
                0, -1, -1, -1,
                0, -1, -1, -1,
                1, -1, -1, -1,
                1, -1, -1, -1,

                2, -1, -1, -1,
                2, -1, -1, -1,
                3, -1, -1, -1,
                3, -1, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 5) {
            return std::array<int8_t, 32>{
                0, -1, -1, -1,
                0, 1, -1, -1,
                1, -1, -1, -1,
                1, 2, -1, -1,

                2, 3, -1, -1,
                3, -1, -1, -1,
                3, 4, -1, -1,
                4, 5, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 6) {
            return std::array<int8_t, 32>{
                0, -1, -1, -1,
                0, 1, -1, -1,
                1, 2, -1, -1,
                2, -1, -1, -1,

                3, -1, -1, -1,
                3, 4, -1, -1,
                4, 5, -1, -1,
                5, -1, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 7) {
            return std::array<int8_t, 32>{
                0, -1, -1, -1,
                0, 1, -1, -1,
                1, 2, -1, -1,
                2, 3, -1, -1,

                3, 4, -1, -1,
                4, 5, -1, -1,
                5, 6, -1, -1,
                6, -1, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 8) {
            return std::array<int8_t, 32>{
                0, -1, -1, -1,
                1, -1, -1, -1,
                2, -1, -1, -1,
                3, -1, -1, -1,

                0, -1, -1, -1,
                1, -1, -1, -1,
                2, -1, -1, -1,
                3, -1, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 9) {
            return std::array<int8_t, 32>{
                0, 1, -1, -1,
                1, 2, -1, -1,
                2, 3, -1, -1,
                3, 4, -1, -1,

                0, 1, -1, -1,
                1, 2, -1, -1,
                2, 3, -1, -1,
                3, 4, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 10) {
            return std::array<int8_t, 32>{
                0, 1, -1, -1,
                1, 2, -1, -1,
                2, 3, -1, -1,
                3, 4, -1, -1,

                1, 2, -1, -1,
                2, 3, -1, -1,
                3, 4, -1, -1,
                4, 5, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 11) {
            return std::array<int8_t, 32>{
                0, 1, -1, -1,
                1, 2, -1, -1,
                2, 3, 4, -1,
                4, 5, -1, -1,

                1, 2, -1, -1,
                2, 3, 4, -1,
                4, 5, -1, -1,
                5, 6, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 12) {
            return std::array<int8_t, 32>{
                0, 1, -1, -1,
                1, 2, -1, -1,
                3, 4, -1, -1,
                4, 5, -1, -1,

                2, 3, -1, -1,
                3, 4, -1, -1,
                5, 6, -1, -1,
                6, 7, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 13) {
            return std::array<int8_t, 32>{
                0, 1, -1, -1,
                1, 2, 3, -1,
                3, 4, -1, -1,
                4, 5, 6, -1,

                2, 3, 4, -1,
                4, 5, -1, -1,
                5, 6, 7, -1,
                7, 8, -1, -1,
            };
        } else if constexpr (BIT_WIDTH == 14) {
            return std::array<int8_t, 32>{
                0, 1, -1, -1,
                1, 2, 3, -1,
                3, 4, 5, -1,
                5, 6, -1, -1,

                3, 4, -1, -1,
                4, 5, 6, -1,
                6, 7, 8, -1,
                8, 9, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 15) {
            return std::array<int8_t, 32>{
                0, 1, -1, -1,
                1, 2, 3, -1,
                3, 4, 5, -1,
                5, 6, 7, -1,

                3, 4, 5, -1,
                5, 6, 7, -1,
                7, 8, 9, -1,
                9, 10, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 16) {
            return std::array<int8_t, 32>{
                0, 1, -1, -1,
                2, 3, -1, -1,
                4, 5, -1, -1,
                6, 7, -1, -1,

                0, 1, -1, -1,
                2, 3, -1, -1,
                4, 5, -1, -1,
                6, 7, -1, -1,
            };
        } else if constexpr (BIT_WIDTH == 17) {
            return std::array<int8_t, 32>{
                0, 1, 2, -1,
                2, 3, 4, -1,
                4, 5, 6, -1,
                6, 7, 8, -1,

                0, 1, 2, -1,
                2, 3, 4, -1,
                4, 5, 6, -1,
                6, 7, 8, -1,
            };
        } else if constexpr (BIT_WIDTH == 18) {
            return std::array<int8_t, 32>{
                0, 1, 2, -1,
                2, 3, 4, -1,
                4, 5, 6, -1,
                6, 7, 8, -1,

                1, 2, 3, -1,
                3, 4, 5, -1,
                5, 6, 7, -1,
                7, 8, 9, -1,
            };
        } else if constexpr (BIT_WIDTH == 19) {
            return std::array<int8_t, 32>{
                0, 1, 2, -1,
                2, 3, 4, -1,
                4, 5, 6,  7,
                7, 8, 9, -1,

                1, 2, 3, -1,
                3, 4, 5, 6,
                6, 7, 8, -1,
                8, 9, 10, -1,
            };
        } else if constexpr (BIT_WIDTH == 20) {
            return std::array<int8_t, 32>{
                0, 1, 2, -1,
                2, 3, 4, -1,
                5, 6, 7, -1,
                7, 8, 9, -1,

                2, 3, 4, -1,
                4, 5, 6, -1,
                7, 8, 9, -1,
                9, 10, 11, -1,
            };
        } else if constexpr (BIT_WIDTH == 21) {
            return std::array<int8_t, 32>{
                0, 1, 2, -1,
                2, 3, 4,  5,
                5, 6, 7, -1,
                7, 8, 9, 10,

                2, 3, 4, 5,
                5, 6, 7, -1,
                7, 8, 9, 10,
                10, 11, 12, -1,
            };
        } else if constexpr (BIT_WIDTH == 22) {
            return std::array<int8_t, 32>{
                0, 1, 2, -1,
                2, 3, 4, 5,
                5, 6, 7, 8,
                8, 9, 10, 11,

                3, 4, 5, -1,
                5, 6, 7, 8,
                8, 9, 10, 11,
                11, 12, 13, -1,
            };
        } else if constexpr (BIT_WIDTH == 23) {
            return std::array<int8_t, 32>{
                0, 1,  2, -1,
                2, 3,  4,  5,
                5, 6,  7,  8,
                8, 9, 10, 11,

                3, 4,  5,  6,
                6, 7,  8,  9,
                9, 10, 11, 12,
                12,13, 14, -1,
            };
        } else if constexpr (BIT_WIDTH == 24) {
            return std::array<int8_t, 32>{
                0,  1,  2, -1,
                3,  4,  5, -1,
                6,  7,  8, -1,
                9, 10, 11, -1,

                0,  1,  2, -1,
                3,  4,  5, -1,
                6,  7,  8, -1,
                9, 10, 11, -1,
            };
        }
        // clang-format on
        return std::array<int8_t, 32>{};
    }();

    alignas(64) inline static constexpr std::array<int32_t, 8> shift = [] {
        // clang-format off
        if constexpr (BIT_WIDTH == 1) {
            return std::array{
                31, 30, 29, 28, 27, 26, 25, 24,
            };
        } else if constexpr (BIT_WIDTH == 2) {
            return std::array{
                30, 28, 26, 24, 30, 28, 26, 24,
            };
        } else if constexpr (BIT_WIDTH == 3) {
            return std::array{
                29, 26, 23, 28, 25, 22, 27, 24,
            };
        } else if constexpr (BIT_WIDTH == 4) {
            return std::array{
                28, 24, 28, 24, 28, 24, 28, 24,
            };
        } else if constexpr (BIT_WIDTH == 5) {
            return std::array{
                27, 22, 25, 20, 23, 26, 21, 24,
            };
        } else if constexpr (BIT_WIDTH == 6) {
            return std::array{
                26, 20, 22, 24, 26, 20, 22, 24,
            };
        } else if constexpr (BIT_WIDTH == 7) {
            return std::array{
                25, 18, 19, 20, 21, 22, 23, 24,
            };
        } else if constexpr (BIT_WIDTH == 8) {
            return std::array{
                24, 24, 24, 24, 24, 24, 24, 24,
            };
        } else if constexpr (BIT_WIDTH == 9) {
            return std::array{
                23, 22, 21, 20, 19, 18, 17, 16,
            };
        } else if constexpr (BIT_WIDTH == 10) {
            return std::array{
                22, 20, 18, 16, 22, 20, 18, 16,
            };
        } else if constexpr (BIT_WIDTH == 11) {
            return std::array{
                21, 18, 15, 20, 17, 14, 19, 16,
            };
        } else if constexpr (BIT_WIDTH == 12) {
            return std::array{
                20, 16, 20, 16, 20, 16, 20, 16,
            };
        } else if constexpr (BIT_WIDTH == 13) {
            return std::array{
                19, 14, 17, 12, 15, 18, 13, 16,
            };
        } else if constexpr (BIT_WIDTH == 14) {
            return std::array{
                18, 12, 14, 16, 18, 12, 14, 16,
            };
        } else if constexpr (BIT_WIDTH == 15) {
            return std::array{
                17, 10, 11, 12, 13, 14, 15, 16,
            };
        } else if constexpr (BIT_WIDTH == 16) {
            return std::array{
                16, 16, 16, 16, 16, 16, 16, 16,
            };
        } else if constexpr (BIT_WIDTH == 17) {
            return std::array{
                15, 14, 13, 12, 11, 10, 9, 8,
            };
        } else if constexpr (BIT_WIDTH == 18) {
            return std::array{
                14, 12, 10, 8, 14, 12, 10, 8,
            };
        } else if constexpr (BIT_WIDTH == 19) {
            return std::array{
                13, 10, 7, 12, 9, 6, 11, 8,
            };
        } else if constexpr (BIT_WIDTH == 20) {
            return std::array{
                12, 8, 12, 8, 12, 8, 12, 8,
            };
        } else if constexpr (BIT_WIDTH == 21) {
            return std::array{
                11, 6, 9, 4, 7, 10, 5, 8,
            };
        } else if constexpr (BIT_WIDTH == 22) {
            return std::array{
                10, 4, 6, 8, 10, 4, 6, 8,
            };
        } else if constexpr (BIT_WIDTH == 23) {
            return std::array{
                9, 2, 3, 4, 5, 6, 7, 8,
            };
        } else if constexpr (BIT_WIDTH == 24) {
            return std::array{
                8, 8, 8, 8, 8, 8, 8, 8,
            };
        }
        // clang-format on
        return std::array<int32_t, 8>{};
    }();

    __always_inline static __m256i get_permute() { return _mm256_load_si256(reinterpret_cast<const __m256i*>(permute.data())); }

    __always_inline static T get_shuffle() {
        if constexpr (std::is_same_v<T, __m128i>) {
            return _mm_load_si128(reinterpret_cast<const __m128i*>(shuffle.data()));
        } else {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(shuffle.data()));
        }
    }

    __always_inline static T get_shift() {
        if constexpr (std::is_same_v<T, __m128i>) {
            return _mm_load_si128(reinterpret_cast<const __m128i*>(shift.data()));
        } else {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(shift.data()));
        }
    }
};
} // namespace libcompression::bitpacking::internal
#endif  // LIBCOMPRESSION_AVX2_ENABLED

#ifdef LIBCOMPRESSION_AVX512_ENABLED
namespace libcompression::bitpacking::internal {
template <uint8_t BIT_WIDTH, typename T>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 8 && (std::is_same_v<T, __m512i> || std::is_same_v<T, __m256i> || std::is_same_v<T, __m128i>))
struct unpack_tables_avx512_8 {
    alignas(64) inline static constexpr std::array<int8_t, 32> shuffle = [] {
        // clang-format off
        if constexpr (BIT_WIDTH == 1) {
            return std::array<int8_t, 32>{
                0, -1, 0, -1,
                0, -1, 0, -1,
                0, -1, 0, -1,
                0, -1, 0, -1,

                1, -1, 1, -1,
                1, -1, 1, -1,
                1, -1, 1, -1,
                1, -1, 1, -1
            };
        } else if constexpr (BIT_WIDTH == 2) {
            return std::array<int8_t, 32>{
                0, -1, 0, -1,
                0, -1, 0, -1,
                1, -1, 1, -1,
                1, -1, 1, -1,

                2, -1, 2, -1,
                2, -1, 2, -1,
                3, -1, 3, -1,
                3, -1, 3, -1
            };
        } else if constexpr (BIT_WIDTH == 3) {
            return std::array<int8_t, 32>{
                0, -1, 0, -1,
                0, 1, 1, -1,
                1, -1, 1, 2,
                2, -1, 2, -1,

                3, -1, 3, -1,
                3, 4, 4, -1,
                4, -1, 4, 5,
                5, -1, 5, -1
            };
        } else if constexpr (BIT_WIDTH == 4) {
            return std::array<int8_t, 32>{
                0, -1, 0, -1,
                1, -1, 1, -1,
                2, -1, 2, -1,
                3, -1, 3, -1,

                4, -1, 4, -1,
                5, -1, 5, -1,
                6, -1, 6, -1,
                7, -1, 7, -1
            };
        } else if constexpr (BIT_WIDTH == 5) {
            return std::array<int8_t, 32>{
                0, -1, 0, 1,
                1, -1, 1, 2,
                2, 3, 3, -1,
                3, 4, 4, -1,

                5, -1, 5, 6,
                6, -1, 6, 7,
                7, 8, 8, -1,
                8, 9, 9, -1
            };
        } else if constexpr (BIT_WIDTH == 6) {
            return std::array<int8_t, 32>{
                0, -1, 0, 1,
                1, 2, 2, -1,
                3, -1, 3, 4,
                4, 5, 5, -1,

                6, -1, 6, 7,
                7, 8, 8, -1,
                9, -1, 9, 10,
                10, 11, 11, -1
            };
        } else if constexpr (BIT_WIDTH == 7) {
            return std::array<int8_t, 32>{
                0, -1, 0, 1,
                1, 2, 2, 3,
                3, 4, 4, 5,
                5, 6, 6, -1,

                7, -1, 7, 8,
                8, 9, 9, 10,
                10, 11, 11, 12,
                12, 13, 13, -1
            };
        } else if constexpr (BIT_WIDTH == 8) {
            return std::array<int8_t, 32>{
                0, -1, 1, -1,
                2, -1, 3, -1,
                4, -1, 5, -1,
                6, -1, 7, -1,

                8, -1, 9, -1,
                10, -1, 11, -1,
                12, -1, 13, -1,
                14, -1, 15, -1
            };
        }
        // clang-format on
        return std::array<int8_t, 32>{};
    }();

    alignas(64) inline static constexpr std::array<int16_t, 16> shift = [] {
        // clang-format off
        if constexpr (BIT_WIDTH == 1) {
            return std::array<int16_t, 16>{
                15, 14, 13, 12, 11, 10, 9, 8,
                15, 14, 13, 12, 11, 10, 9, 8,
            };
        } else if constexpr (BIT_WIDTH == 2) {
            return std::array<int16_t, 16>{
                14, 12, 10, 8, 14, 12, 10, 8,
                14, 12, 10, 8, 14, 12, 10, 8,
            };
        } else if constexpr (BIT_WIDTH == 3) {
            return std::array<int16_t, 16>{
                13, 10, 7, 12, 9, 6, 11, 8,
                13, 10, 7, 12, 9, 6, 11, 8,
            };
        } else if constexpr (BIT_WIDTH == 4) {
            return std::array<int16_t, 16>{
                12, 8, 12, 8, 12, 8, 12, 8,
                12, 8, 12, 8, 12, 8, 12, 8,
            };
        } else if constexpr (BIT_WIDTH == 5) {
            return std::array<int16_t, 16>{
                11, 6, 9, 4, 7, 10, 5, 8,
                11, 6, 9, 4, 7, 10, 5, 8,
            };
        } else if constexpr (BIT_WIDTH == 6) {
            return std::array<int16_t, 16>{
                10, 4, 6, 8, 10, 4, 6, 8,
                10, 4, 6, 8, 10, 4, 6, 8,
            };
        } else if constexpr (BIT_WIDTH == 7) {
            return std::array<int16_t, 16>{
                9, 2, 3, 4, 5, 6, 7, 8,
                9, 2, 3, 4, 5, 6, 7, 8,
            };
        } else if constexpr (BIT_WIDTH == 8) {
            return std::array<int16_t, 16>{
                8, 8, 8, 8, 8, 8, 8, 8,
                8, 8, 8, 8, 8, 8, 8, 8,
            };
        }
        // clang-format on
        return std::array<int16_t, 16>{};
    }();

    __always_inline static T get_shuffle() {
        if constexpr (std::is_same_v<T, __m128i>) {
            return _mm_load_si128(reinterpret_cast<const __m128i*>(shuffle.data()));
        } else if constexpr (std::is_same_v<T, __m256i>) {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(shuffle.data()));
        } else if constexpr (std::is_same_v<T, __m512i>) {
            return _mm512_load_si512(shuffle.data());
        } else {
            return {};
            // This line will never be reached, but is needed to satisfy the compiler.
        }
    }

    __always_inline static T get_shift() {
        if constexpr (std::is_same_v<T, __m128i>) {
            return _mm_load_si128(reinterpret_cast<const __m128i*>(shift.data()));
        } else if constexpr (std::is_same_v<T, __m256i>) {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(shift.data()));
        } else if constexpr (std::is_same_v<T, __m512i>) {
            return _mm512_load_si512(shift.data());
        } else {
            return {};
            // This line will never be reached, but is needed to satisfy the compiler.
        }
    }
};

template <uint8_t BIT_WIDTH, typename T>
    requires(BIT_WIDTH > 8 && BIT_WIDTH <= 24 && (std::is_same_v<T, __m512i> || std::is_same_v<T, __m256i> || std::is_same_v<T, __m128i>))
struct unpack_tables_avx512_24 {
    alignas(64) inline static constexpr std::array<int8_t, 64> shuffle = [] {
        // clang-format off
        if constexpr (BIT_WIDTH == 9) {
            return std::array<int8_t, 64>{
                0, 1, -1, -1,
                1, 2, -1, -1,
                2, 3, -1, -1,
                3, 4, -1, -1,

                4, 5, -1, -1,
                5, 6, -1, -1,
                6, 7, -1, -1,
                7, 8, -1, -1,

                9, 10, -1, -1,
                10, 11, -1, -1,
                11, 12, -1, -1,
                12, 13, -1, -1,

                13, 14, -1, -1,
                14, 15, -1, -1,
                15, 16, -1, -1,
                16, 17, -1, -1,
            };
        } else if constexpr (BIT_WIDTH == 10) {
            return std::array<int8_t, 64>{
                0, 1, -1, -1,
                1, 2, -1, -1,
                2, 3, -1, -1,
                3, 4, -1, -1,

                5, 6, -1, -1,
                6, 7, -1, -1,
                7, 8, -1, -1,
                8, 9, -1, -1,

                10, 11, -1, -1,
                11, 12, -1, -1,
                12, 13, -1, -1,
                13, 14, -1, -1,

                15, 16, -1, -1,
                16, 17, -1, -1,
                17, 18, -1, -1,
                18, 19, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 11) {
            return std::array<int8_t, 64>{
                0, 1, -1, -1,
                1, 2, -1, -1,
                2, 3, 4, -1,
                4, 5, -1, -1,

                5, 6, -1, -1,
                6, 7, 8, -1,
                8, 9, -1, -1,
                9, 10, -1, -1,

                11, 12, -1, -1,
                12, 13, -1, -1,
                13, 14, 15, -1,
                15, 16, -1, -1,

                16, 17, -1, -1,
                17, 18, 19, -1, 19, 20, -1, -1, 20, 21, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 12) {
            return std::array<int8_t, 64>{
                0, 1, -1, -1,
                1, 2, -1, -1,
                3, 4, -1, -1,
                4, 5, -1, -1,

                6, 7, -1, -1,
                7, 8, -1, -1,
                9, 10, -1, -1,
                10, 11, -1, -1,

                12, 13, -1, -1,
                13, 14, -1, -1,
                15, 16, -1, -1,
                16, 17, -1, -1,

                18, 19, -1, -1,
                19, 20, -1, -1,
                21, 22, -1, -1,
                22, 23, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 13) {
            return std::array<int8_t, 64>{
                0, 1, -1, -1,
                1, 2, 3, -1,
                3, 4, -1, -1,
                4, 5, 6, -1,

                6, 7, 8, -1,
                8, 9, -1, -1,
                9, 10, 11, -1,
                11, 12, -1, -1,

                13, 14, -1, -1,
                14, 15, 16, -1,
                16, 17, -1, -1,
                17, 18, 19, -1,

                19, 20, 21, -1,
                21, 22, -1, -1,
                22, 23, 24, -1,
                24, 25, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 14) {
            return std::array<int8_t, 64>{
                0, 1, -1, -1,
                1, 2, 3, -1,
                3, 4, 5, -1,
                5, 6, -1, -1,

                7, 8, -1, -1,
                8, 9, 10, -1,
                10, 11, 12, -1,
                12, 13, -1, -1,

                14, 15, -1, -1,
                15, 16, 17, -1,
                17, 18, 19, -1,
                19, 20, -1, -1,

                21, 22, -1, -1,
                22, 23, 24, -1,
                24, 25, 26, -1,
                26, 27, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 15) {
            return std::array<int8_t, 64>{
                0, 1, -1, -1,
                1, 2, 3, -1, 3,
                4, 5, -1, 5, 6,
                7, -1,

                7, 8, 9, -1,
                9, 10, 11, -1,
                11, 12, 13, -1,
                13, 14, -1, -1,

                15, 16, -1, -1,
                16, 17, 18, -1,
                18, 19, 20, -1,
                20, 21, 22, -1,

                22, 23, 24, -1,
                24, 25, 26, -1,
                26, 27, 28, -1,
                28, 29, -1, -1,
            };
        } else if constexpr (BIT_WIDTH == 16) {
            return std::array<int8_t, 64>{
                0, 1, -1, -1,
                2, 3, -1, -1,
                4, 5, -1, -1,
                6, 7, -1, -1,

                8, 9, -1, -1,
                10, 11, -1, -1,
                12, 13, -1, -1,
                14, 15, -1, -1,

                16, 17, -1, -1,
                18, 19, -1, -1,
                20, 21, -1, -1,
                22, 23, -1, -1,

                24, 25, -1, -1,
                26, 27, -1, -1,
                28, 29, -1, -1,
                30, 31, -1, -1,
            };
        } else if constexpr (BIT_WIDTH == 17) {
            return std::array<int8_t, 64>{
                0, 1, 2, -1,
                2, 3, 4, -1,
                4, 5, 6, -1,
                6, 7, 8, -1,

                8, 9, 10, -1,
                10,11,12, -1,
                12,13,14, -1,
                14,15,16, -1,

                17, 18,19, -1,
                19, 20,21, -1,
                21, 22,23, -1,
                23, 24,25, -1,

                25, 26,27, -1,
                27, 28,29, -1,
                29, 30,31, -1,
                31, 32,33, -1,
            };
        } else if constexpr (BIT_WIDTH == 18) {
            return std::array<int8_t, 64>{
                0, 1, 2, -1,
                2, 3, 4, -1,
                4, 5, 6, -1,
                6, 7, 8, -1,

                9, 10, 11, -1,
                11,12,13, -1,
                13,14,15, -1,
                15,16,17, -1,

                18, 19,20, -1,
                20, 21,22, -1,
                22, 23,24, -1,
                24, 25,26, -1,

                27, 28,29, -1,
                29, 30,31, -1,
                31, 32,33, -1,
                33, 34,35, -1,
            };
        } else if constexpr (BIT_WIDTH == 18) {
            return std::array<int8_t, 64>{
                0, 1, 2, -1,
                2, 3, 4, -1,
                4, 5, 6, -1,
                6, 7, 8, -1,

                9, 10, 11, -1,
                11,12,13, -1,
                13,14,15, -1,
                15,16,17, -1,

                18, 19,20, -1,
                20, 21,22, -1,
                22, 23,24, -1,
                24, 25,26, -1,

                27, 28,29, -1,
                29, 30,31, -1,
                31, 32,33, -1,
                33, 34,35, -1,
            };
        } else if constexpr (BIT_WIDTH == 19) {
            return std::array<int8_t, 64>{
                0, 1, 2, -1,
                2, 3, 4, -1,
                4, 5, 6, 7,
                7, 8, 9, -1,

                9, 10,11, -1,
                11,12,13,14,
                14,15,16, -1,
                16,17,18, -1,

                19, 20,21, -1,
                21, 22,23, -1,
                23, 24,25,26,
                26,27,28, -1,

                28, 29,30, -1,
                30, 31,32, 33,
                33,34,35, -1,
                35,36,37, -1,
            };
        } else if constexpr (BIT_WIDTH == 20) {
            return std::array<int8_t, 64>{
                0, 1, 2, -1,
                2, 3, 4, -1,
                5, 6, 7, -1,
                7, 8, 9, -1,

                10, 11,12, -1,
                12,13,14, -1,
                15,16,17, -1,
                17,18,19, -1,

                20, 21,22, -1,
                22,23,24, -1,
                25,26,27, -1,
                27,28,29, -1,

                30, 31,32, -1,
                32,33,34, -1,
                35,36,37, -1,
                37,38,39, -1,
            };
        } else if constexpr (BIT_WIDTH == 21) {
            return std::array<int8_t, 64>{
                0, 1, 2, -1,
                2, 3, 4, 5,
                5, 6, 7, -1,
                7, 8, 9, 10,

                10, 11,12,13,
                13,14,15, -1,
                15,16,17,18,
                18,19,20, -1,

                21, 22,23, -1,
                23, 24,25,26,
                26,27,28, -1,
                28, 29,30,31,

                31, 32,33,34,
                34,35,36, -1,
                36,37,38,39,
                39,40,41, -1,
            };
        } else if constexpr (BIT_WIDTH == 22) {
            return std::array<int8_t, 64>{
                0, 1, 2, -1,
                2, 3, 4, 5,
                5, 6, 7, 8,
                8, 9, 10, -1,

                11, 12,13, -1,
                13,14,15,16,
                16,17,18,19,
                19,20,21, -1,

                22, 23,24, -1,
                24,25,26,27,
                27,28,29,30,
                30,31,32, -1,

                33, 34,35, -1,
                35,36,37,38,
                38,39,40,41,
                41,42,43, -1,
            };
        } else if constexpr (BIT_WIDTH == 23) {
            return std::array<int8_t, 64>{
                0, 1, 2, -1,
                2, 3, 4, 5,
                5, 6, 7, 8,
                8, 9, 10,11,

                11,12,13,14,
                14,15,16,17,
                17,18,19,20,
                20,21,22, -1,

                23, 24,25, -1,
                25,26,27,28,
                28,29,30,31,
                31,32,33,34,

                34,35,36,37,
                37,38,39,40,
                40,41,42,43,
                43,44,45, -1,
            };
        } else if constexpr (BIT_WIDTH == 24) {
            return std::array<int8_t, 64>{
                0, 1, 2, -1,
                3, 4, 5, -1,
                6, 7, 8, -1,
                9, 10,11, -1,

                12,13,14, -1,
                15,16,17, -1,
                18,19,20, -1,
                21,22,23, -1,

                24, 25,26, -1,
                27, 28,29, -1,
                30, 31,32, -1,
                33, 34,35, -1,

                36, 37,38, -1,
                39, 40,41, -1,
                42, 43,44, -1,
                45, 46,47, -1,
            };
        }
        // clang-format on
        return std::array<int8_t, 64>{};
    }();

    alignas(64) inline static constexpr std::array<int32_t, 16> shift = [] {
        // clang-format off
        if constexpr (BIT_WIDTH == 9) {
            return std::array{
                23, 22, 21, 20, 19, 18, 17, 16,
                23, 22, 21, 20, 19, 18, 17, 16,
            };
        } else if constexpr (BIT_WIDTH == 10) {
            return std::array{
                22, 20, 18, 16, 22, 20, 18, 16,
                22, 20, 18, 16, 22, 20, 18, 16,
            };
        } else if constexpr (BIT_WIDTH == 11) {
            return std::array{
                21, 18, 15, 20, 17, 14, 19, 16,
                21, 18, 15, 20, 17, 14, 19, 16,
            };
        } else if constexpr (BIT_WIDTH == 12) {
            return std::array{
                20, 16, 20, 16, 20, 16, 20, 16,
                20, 16, 20, 16, 20, 16, 20, 16,
            };
        } else if constexpr (BIT_WIDTH == 13) {
            return std::array{
                19, 14, 17, 12, 15, 18, 13, 16,
                19, 14, 17, 12, 15, 18, 13, 16,
            };
        } else if constexpr (BIT_WIDTH == 14) {
            return std::array{
                18, 12, 14, 16, 18, 12, 14, 16,
                18, 12, 14, 16, 18, 12, 14, 16,
            };
        } else if constexpr (BIT_WIDTH == 15) {
            return std::array{
                17, 10, 11, 12, 13, 14, 15, 16,
                17, 10, 11, 12, 13, 14, 15, 16,
            };
        } else if constexpr (BIT_WIDTH == 16) {
            return std::array{
                16, 16, 16, 16, 16, 16, 16, 16,
                16, 16, 16, 16, 16, 16, 16, 16,
            };
        } else if constexpr (BIT_WIDTH == 17) {
            return std::array{
                15, 14, 13, 12, 11, 10, 9, 8,
                15, 14, 13, 12, 11, 10, 9, 8,
            };
        } else if constexpr (BIT_WIDTH == 18) {
            return std::array{
                14, 12, 10, 8, 14, 12, 10, 8,
                14, 12, 10, 8, 14, 12, 10, 8,
            };
        } else if constexpr (BIT_WIDTH == 19) {
            return std::array{
                13, 10, 7, 12, 9, 6, 11, 8,
                13, 10, 7, 12, 9, 6, 11, 8,
            };
        } else if constexpr (BIT_WIDTH == 20) {
            return std::array{
                12, 8, 12, 8, 12, 8, 12, 8,
                12, 8, 12, 8, 12, 8, 12, 8,
            };
        } else if constexpr (BIT_WIDTH == 21) {
            return std::array{
                11, 6, 9, 4, 7, 10, 5, 8,
                11, 6, 9, 4, 7, 10, 5, 8,
            };
        } else if constexpr (BIT_WIDTH == 22) {
            return std::array{
                10, 4, 6, 8, 10, 4, 6, 8,
                10, 4, 6, 8, 10, 4, 6, 8,
            };
        } else if constexpr (BIT_WIDTH == 23) {
            return std::array{
                9, 2, 3, 4, 5, 6, 7, 8,
                9, 2, 3, 4, 5, 6, 7, 8,
            };
        } else if constexpr (BIT_WIDTH == 24) {
            return std::array{
                8, 8, 8, 8, 8, 8, 8, 8,
                8, 8, 8, 8, 8, 8, 8, 8,
            };
        }
        // clang-format on
        return std::array<int32_t, 16>{};
    }();

    __always_inline static T get_shuffle() {
        if constexpr (std::is_same_v<T, __m128i>) {
            return _mm_load_si128(reinterpret_cast<const __m128i*>(shuffle.data()));
        } else if constexpr (std::is_same_v<T, __m256i>) {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(shuffle.data()));
        } else if constexpr (std::is_same_v<T, __m512i>) {
            return _mm512_load_si512(shuffle.data());
        } else {
            return {};
            // This line will never be reached, but is needed to satisfy the compiler.
        }
    }

    __always_inline static T get_shift() {
        if constexpr (std::is_same_v<T, __m128i>) {
            return _mm_load_si128(reinterpret_cast<const __m128i*>(shift.data()));
        } else if constexpr (std::is_same_v<T, __m256i>) {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(shift.data()));
        } else if constexpr (std::is_same_v<T, __m512i>) {
            return _mm512_load_si512(shift.data());
        } else {
            return {};
            // This line will never be reached, but is needed to satisfy the compiler.
        }
    }
};
} // namespace libcompression::bitpacking::internal
#endif  // LIBCOMPRESSION_AVX512_ENABLED
#endif  // LIBCOMPRESSION_TABLES_H
