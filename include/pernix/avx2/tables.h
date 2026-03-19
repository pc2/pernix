#ifndef PERNIX_AVX2_TABLES_H
#define PERNIX_AVX2_TABLES_H

#include <immintrin.h>

#include <array>
#include <cstdint>
#include <type_traits>

namespace pernix::internal {
// Lookup tables for the AVX2 shuffle/permute-based bit packers and unpackers.
// The values are generated as compile-time constants so the hot paths only need
// aligned loads when materializing the masks.
template <__uint8_t BIT_WIDTH, typename T>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16 && (std::is_same_v<T, __m128i> || std::is_same_v<T, __m256i>))
struct pack_tables_avx2_16 {
    alignas(64) inline static constexpr std::array<int8_t, 32> permute1 = [] {
        // clang-format off
        if constexpr (BIT_WIDTH == 9) {
            return std::array<int8_t, 32>{
                0, 1, 4, 5,
                8, 9, 12, 13,
                -1, -1, -1, -1,
                -1, -1, -1, -1,

                0, 1, 4, 5,
                8, 9, 12, 13,
                -1, -1, -1, -1,
                -1, -1, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 10) {
            return std::array<int8_t, 32>{
                0, 1, 4, 5,
                -1, -1, 10, 11,
                14, 15, -1, -1,
                -1, -1, -1, -1,

                0, 1, 4, 5,
                -1, -1, 10, 11,
                14, 15, -1, -1,
                -1, -1, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 11) {
            return std::array<int8_t, 32>{
                0, 1, 4, 5,
                6, 7, 10, 11,
                12, 13, -1, -1,
                -1, -1, -1, -1,

                0, 1, 4, 5,
                6, 7, 10, 11,
                12, 13, -1, -1,
                -1, -1, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 12) {
            return std::array<int8_t, 32>{
                2, 3, 4, 5,
                6, 7, 10, 11,
                12, 13, 14, 15,
                -1, -1, -1, -1,

                2, 3, 4, 5,
                6, 7, 10, 11,
                12, 13, 14, 15,
                -1, -1, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 13) {
            return std::array<int8_t, 32>{
                0, 1, -1, -1,
                6, 7, 8, 9,
                10, 11, -1, -1,
                -1, -1, -1, -1,

                0, 1, -1, -1,
                6, 7, 8, 9,
                10, 11, -1, -1,
                -1, -1, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 14) {
            return std::array<int8_t, 32>{
                2, 3, 4, 5,
                6, 7, 8, 9,
                10, 11, 12, 13,
                14, 15, -1, -1,

                2, 3, 4, 5,
                6, 7, 8, 9,
                10, 11, 12, 13,
                14, 15, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 15) {
            return std::array<int8_t, 32>{
                2, 3, 4, 5,
                6, 7, 8, 9,
                10, 11, 12, 13,
                14, 15, -1, -1,

                2, 3, 4, 5,
                6, 7, 8, 9,
                10, 11, 12, 13,
                14, 15, -1, -1
            };
        }
        return std::array<int8_t, 32>{};
        // clang-format on
    }();

    alignas(64) inline static constexpr std::array<int8_t, 32> permute2 = [] {
        // clang-format off
        if constexpr (BIT_WIDTH == 9) {
            return std::array<int8_t, 32>{
                2, 3, 6, 7,
                10, 11, 14, 15,
                -1, -1, -1, -1,
                -1, -1, -1, -1,

                2, 3, 6, 7,
                10, 11, 14, 15,
                -1, -1, -1, -1,
                -1, -1, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 10) {
            return std::array<int8_t, 32>{
                2, 3, 6, 7,
                8, 9, 12, 13,
                -1, -1, -1, -1,
                -1, -1, -1, -1,

                2, 3, 6, 7,
                8, 9, 12, 13,
                -1, -1, -1, -1,
                -1, -1, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 11) {
            return std::array<int8_t, 32>{
                2, 3, -1, -1,
                8, 9, -1, -1,
                14, 15, -1, -1,
                -1, -1, -1, -1,

                2, 3, -1, -1,
                8, 9, -1, -1,
                14, 15, -1, -1,
                -1, -1, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 12) {
            return std::array<int8_t, 32>{
                0, 1, 2, 3,
                4, 5, 8, 9,
                10, 11, 12, 13,
                -1, -1, -1, -1,

                0, 1, 2, 3,
                4, 5, 8, 9,
                10, 11, 12, 13,
                -1, -1, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 13) {
            return std::array<int8_t, 32>{
                2, 3, 4, 5,
                -1, -1, -1, -1,
                12, 13, 14, 15,
                -1, -1, -1, -1,

                2, 3, 4, 5,
                -1, -1, -1, -1,
                12, 13, 14, 15,
                -1, -1, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 14) {
            return std::array<int8_t, 32>{
                0, 1, 2, 3,
                4, 5, 6, 7,
                8, 9, 10, 11,
                12, 13, -1, -1,

                0, 1, 2, 3,
                4, 5, 6, 7,
                8, 9, 10, 11,
                12, 13, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 15) {
            return std::array<int8_t, 32>{
                0, 1, 2, 3,
                4, 5, 6, 7,
                8, 9, 10, 11,
                12, 13, 14, 15,

                0, 1, 2, 3,
                4, 5, 6, 7,
                8, 9, 10, 11,
                12, 13, 14, 15
            };
        }
        return std::array<int8_t, 32>{};
        // clang-format on
    }();

    alignas(64) inline static constexpr std::array<int8_t, 32> permute3 = [] {
        // clang-format off
        if constexpr (BIT_WIDTH == 9) {
            return std::array<int8_t, 32>{
                -1, -1, 2, 3,
                6, 7, 10, 11,
                14, 15, -1, -1,
                -1, -1, -1, -1,

                -1, -1, 2, 3,
                6, 7, 10, 11,
                14, 15, -1, -1,
                -1, -1, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 10) {
            return std::array<int8_t, 32>{
                -1, -1, 2, 3,
                6, 7, 8, 9,
                12, 13, -1, -1,
                -1, -1, -1, -1,

                -1, -1, 2, 3,
                6, 7, 8, 9,
                12, 13, -1, -1,
                -1, -1, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 11) {
            return std::array<int8_t, 32>{
                -1, -1, 2, 3,
                4, 5, 8, 9,
                10, 11, 14, 15,
                -1, -1, -1, -1,

                -1, -1, 2, 3,
                4, 5, 8, 9,
                10, 11, 14, 15,
                -1, -1, -1, -1
            };
        } else if constexpr (BIT_WIDTH == 13) {
            return std::array<int8_t, 32>{
                -1, -1, 2, 3,
                4, 5, 6, 7,
                8, 9, 12, 13,
                14, 15, -1, -1,

                -1, -1, 2, 3,
                4, 5, 6, 7,
                8, 9, 12, 13,
                14, 15, -1, -1
            };
        }
        return std::array<int8_t, 32>{};
        // clang-format on
    }();

    alignas(64) inline static constexpr std::array<uint16_t, 16> shift1 = [] {
        // clang-format off
        if constexpr (BIT_WIDTH == 9) {
            return std::array<uint16_t, 16>{
                0, 2, 4, 6, 0, 1, 3, 5,
                0, 2, 4, 6, 0, 1, 3, 5
            };
        } else if constexpr (BIT_WIDTH == 10) {
            return std::array<uint16_t, 16>{
                0, 4, 0, 2, 6, 0, 4, 0,
                0, 4, 0, 2, 6, 0, 4, 0
            };
        } else if constexpr (BIT_WIDTH == 11) {
            return std::array<uint16_t, 16>{
                0, 6, 1, 7, 2, 0, 3, 0,
                0, 6, 1, 7, 2, 0, 3, 0
            };
        } else if constexpr (BIT_WIDTH == 12) {
            return std::array<uint16_t, 16>{
                12, 8, 4, 12, 8, 4, 12, 8,
                12, 8, 4, 12, 8, 4, 12, 8
            };
        } else if constexpr (BIT_WIDTH == 13) {
            return std::array<uint16_t, 16>{
                0, 0, 7, 4, 1, 0, 0, 5,
                0, 0, 7, 4, 1, 0, 0, 5
            };
        } else if constexpr (BIT_WIDTH == 14) {
            return std::array<uint16_t, 16>{
                14, 12, 10, 8, 6, 4, 2, 14,
                14, 12, 10, 8, 6, 4, 2, 14
            };
        } else if constexpr (BIT_WIDTH == 15) {
            return std::array<uint16_t, 16>{
                15, 14, 13, 12, 11, 10, 9, 8,
                15, 14, 13, 12, 11, 10, 9, 8
            };
        }
        return std::array<uint16_t, 16>{};
        // clang-format on
    }();

    alignas(64) inline static constexpr std::array<uint16_t, 16> shift2 = [] {
        // clang-format off
        if constexpr (BIT_WIDTH == 9) {
            return std::array<uint16_t, 16>{
                9, 11, 13, 15, 8, 10, 12, 14,
                9, 11, 13, 15, 8, 10, 12, 14,
            };
        } else if constexpr (BIT_WIDTH == 10) {
            return std::array<uint16_t, 16>{
                10, 14, 8, 12, 0, 10, 14, 8,
                10, 14, 8, 12, 0, 10, 14, 8,
            };
        } else if constexpr (BIT_WIDTH == 11) {
            return std::array<uint16_t, 16>{
                11, 8, 12, 8, 13, 8, 14, 9,
                11, 8, 12, 8, 13, 8, 14, 9,
            };
        } else if constexpr (BIT_WIDTH == 12) {
            return std::array<uint16_t, 16>{
                0, 4, 8, 0, 4, 8, 0, 4,
                0, 4, 8, 0, 4, 8, 0, 4,
            };
        } else if constexpr (BIT_WIDTH == 13) {
            return std::array<uint16_t, 16>{
                13, 10, 0, 0, 14, 11, 8, 0,
                13, 10, 0, 0, 14, 11, 8, 0,
            };
        } else if constexpr (BIT_WIDTH == 14) {
            return std::array<uint16_t, 16>{
                0, 2, 4, 6, 8, 10, 12, 0,
                0, 2, 4, 6, 8, 10, 12, 0,
            };
        } else if constexpr (BIT_WIDTH == 15) {
        return std::array<uint16_t, 16>{
                0, 1, 2, 3, 4, 5, 6, 7,
                0, 1, 2, 3, 4, 5, 6, 7,
            };
        }
        return std::array<uint16_t, 16>{};
        // clang-format on
    }();

    alignas(64) inline static constexpr std::array<uint16_t, 16> shift3 = [] {
        // clang-format off
        if constexpr (BIT_WIDTH == 9) {
            return std::array<uint16_t, 16>{
                0, 7, 5, 3, 1, 8, 6, 4,
                0, 7, 5, 3, 1, 8, 6, 4
            };
        } else if constexpr (BIT_WIDTH == 10) {
            return std::array<uint16_t, 16>{
                0, 6, 2, 8, 4, 0, 6, 2,
                0, 6, 2, 8, 4, 0, 6, 2,
            };
        } else if constexpr (BIT_WIDTH == 11) {
            return std::array<uint16_t, 16>{
                0, 5, 10, 4, 9, 3, 8, 2,
                0, 5, 10, 4, 9, 3, 8, 2,
            };
        } else if constexpr (BIT_WIDTH == 13) {
            return std::array<uint16_t, 16>{
                0, 3, 6, 9, 12, 2, 5, 8,
                0, 3, 6, 9, 12, 2, 5, 8,
            };
        }
        return std::array<uint16_t, 16>{};
        // clang-format on
    }();

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
    __always_inline static T get_permute1() {
        if constexpr (std::is_same_v<T, __m256i>) {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(permute1.data()));
        } else if constexpr (std::is_same_v<T, __m128i>) {
            return _mm_load_si128(reinterpret_cast<const __m128i*>(permute1.data()));
        }
        return T{};
    }

    __always_inline static T get_permute2() {
        if constexpr (std::is_same_v<T, __m256i>) {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(permute2.data()));
        } else if constexpr (std::is_same_v<T, __m128i>) {
            return _mm_load_si128(reinterpret_cast<const __m128i*>(permute2.data()));
        }
        return T{};
    }

    __always_inline static T get_permute3() {
        if constexpr (std::is_same_v<T, __m256i>) {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(permute3.data()));
        } else if constexpr (std::is_same_v<T, __m128i>) {
            return _mm_load_si128(reinterpret_cast<const __m128i*>(permute3.data()));
        }
        return T{};
    }

    __always_inline static T get_shift1() {
        if constexpr (std::is_same_v<T, __m256i>) {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(shift1.data()));
        } else if constexpr (std::is_same_v<T, __m128i>) {
            return _mm_load_si128(reinterpret_cast<const __m128i*>(shift1.data()));
        }
        return T{};
    }

    __always_inline static T get_shift2() {
        if constexpr (std::is_same_v<T, __m256i>) {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(shift2.data()));
        } else if constexpr (std::is_same_v<T, __m128i>) {
            return _mm_load_si128(reinterpret_cast<const __m128i*>(shift2.data()));
        }
        return T{};
    }

    __always_inline static T get_shift3() {
        if constexpr (std::is_same_v<T, __m256i>) {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(shift3.data()));
        } else if constexpr (std::is_same_v<T, __m128i>) {
            return _mm_load_si128(reinterpret_cast<const __m128i*>(shift3.data()));
        }
        return T{};
    }
#pragma GCC diagnostic pop
};

template <__uint8_t BIT_WIDTH, typename T>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24 && (std::is_same_v<T, __m128i> || std::is_same_v<T, __m256i>))
struct pack_tables_avx2_32 {
    alignas(64) inline static constexpr std::array<int32_t, 8> permute1 = [] {
        // clang-format off
        if constexpr (BIT_WIDTH == 17) {
            return std::array<int32_t, 8>{
                0, 2, 4, 6, 8, 10, 12, 14
            };
        } else if constexpr (BIT_WIDTH == 18) {
            return std::array<int32_t, 8>{
                0, 2, 4, 6, 0, 9, 11, 13
            };
        } else if constexpr (BIT_WIDTH == 19) {
            return std::array<int32_t, 8>{
                0, 2, 4, 0, 7, 9, 0, 12
            };
        } else if constexpr (BIT_WIDTH == 20) {
            return std::array<int32_t, 8>{
                0, 2, 0, 5, 7, 8, 10, 0
            };
        } else if constexpr (BIT_WIDTH == 21) {
            return std::array<int32_t, 8>{
                0, 2, 0, 5, 0, 8, 0, 11
            };
        } else if constexpr (BIT_WIDTH == 22) {
            return std::array<int32_t, 8>{
                0, 2, 3, 5, 6, 0, 9, 0
            };
        } else if constexpr (BIT_WIDTH == 23) {
            return std::array<int32_t, 8>{
                0, 2, 3, 0, 6, 7, 9, 10
            };
        } else if constexpr (BIT_WIDTH == 24) {
            return std::array<int32_t, 8>{
                1, 2, 3, 5, 6, 7, -1, -1,
            };
        }
        return std::array<int32_t, 8>{};
        // clang-format on
    }();

    alignas(64) inline static constexpr std::array<int32_t, 8> permute2 = [] {
        // clang-format off
        if constexpr (BIT_WIDTH == 17) {
            return std::array<int32_t, 8>{
                1, 3, 5, 7, 9, 11, 13, 15
            };
        } else if constexpr (BIT_WIDTH == 18) {
            return std::array<int32_t, 8>{
                1, 3, 5, 7, 8, 10, 12, 14
            };
        } else if constexpr (BIT_WIDTH == 19) {
            return std::array<int32_t, 8>{
                1, 3, 5, 6, 8, 10, 11, 13
            };
        } else if constexpr (BIT_WIDTH == 20) {
            return std::array<int32_t, 8>{
                1, 3, 4, 6, 0, 9, 11, 12
            };
        } else if constexpr (BIT_WIDTH == 21) {
            return std::array<int32_t, 8>{
                1, 3, 4, 6, 7, 9, 10, 12
            };
        } else if constexpr (BIT_WIDTH == 22) {
            return std::array<int32_t, 8>{
                1, 0, 4, 0, 7, 8, 10, 11
            };
        } else if constexpr (BIT_WIDTH == 23) {
            return std::array<int32_t, 8>{
                1, 0, 4, 5, 0, 8, 0, 11
            };
        } else if constexpr (BIT_WIDTH == 24) {
            return std::array<int32_t, 8>{
                0, 1, 2, 4, 5, 6, -1, -1,
            };
        }
        return std::array<int32_t, 8>{};
        // clang-format on
    }();

    alignas(64) inline static constexpr std::array<int32_t, 8> permute3 = [] {
        // clang-format off
        if constexpr (BIT_WIDTH == 17) {
            return std::array<int32_t, 8>{
                0, 1, 3, 5, 7, 9, 11, 13
            };
        } else if constexpr (BIT_WIDTH == 18) {
            return std::array<int32_t, 8>{
                0, 1, 3, 5, 7, 8, 10, 12
            };
        } else if constexpr (BIT_WIDTH == 19) {
            return std::array<int32_t, 8>{
                0, 1, 3, 5, 6, 8, 10, 11
            };
        } else if constexpr (BIT_WIDTH == 20) {
            return std::array<int32_t, 8>{
                0, 1, 3, 4, 6, 0, 9, 11
            };
        } else if constexpr (BIT_WIDTH == 21) {
            return std::array<int32_t, 8>{
                0, 1, 3, 4, 6, 7, 9, 10
            };
        } else if constexpr (BIT_WIDTH == 22) {
            return std::array<int32_t, 8>{
                0, 1, 2, 4, 5, 7, 8, 10
            };
        } else if constexpr (BIT_WIDTH == 23) {
            return std::array<int32_t, 8>{
                0, 1, 2, 4, 5, 6, 8, 9
            };
        }
        return std::array<int32_t, 8>{};
        // clang-format on
    }();

    alignas(64) inline static constexpr std::array<uint32_t, 8> shift1 = [] {
        // clang-format off
        if constexpr (BIT_WIDTH == 17) {
            return std::array<uint32_t, 8>{
                0, 2, 4, 6, 8, 10, 12, 14
            };
        } else if constexpr (BIT_WIDTH == 18) {
            return std::array<uint32_t, 8>{
                0, 4, 8, 12, 0, 2, 6, 10
            };
        } else if constexpr (BIT_WIDTH == 19) {
            return std::array<uint32_t, 8>{
                0, 6, 12, 0, 5, 11, 0, 4
            };
        } else if constexpr (BIT_WIDTH == 20) {
            return std::array<uint32_t, 8>{
                0, 8, 0, 4, 12, 0, 8, 0
            };
        } else if constexpr (BIT_WIDTH == 21) {
            return std::array<uint32_t, 8>{
                0, 10, 0, 9, 0, 8, 0, 7
            };
        } else if constexpr (BIT_WIDTH == 22) {
            return std::array<uint32_t, 8>{
                0, 12, 2, 14, 4, 0, 6, 0
            };
        } else if constexpr (BIT_WIDTH == 23) {
            return std::array<uint32_t, 8>{
                0, 14, 5, 0, 10, 1, 15, 6
            };
        } else if constexpr (BIT_WIDTH == 24) {
            return std::array<uint32_t, 8>{
                24, 16, 8, 24, 16, 8, 24, 16
            };
        }
        return std::array<uint32_t, 8>{};
        // clang-format on
    }();

    alignas(64) inline static constexpr std::array<uint32_t, 8> shift2 = [] {
        // clang-format off
        if constexpr (BIT_WIDTH == 17) {
            return std::array<uint32_t, 8>{
                17, 19, 21, 23, 25, 27, 29, 31
            };
        } else if constexpr (BIT_WIDTH == 18) {
            return std::array<uint32_t, 8>{
                18, 22, 26, 30, 16, 20, 24, 28
            };
        } else if constexpr (BIT_WIDTH == 19) {
            return std::array<uint32_t, 8>{
                19, 25, 31, 18, 24, 30, 17, 23
            };
        } else if constexpr (BIT_WIDTH == 20) {
            return std::array<uint32_t, 8>{
                20, 28, 16, 24, 0, 20, 28, 16
            };
        } else if constexpr (BIT_WIDTH == 21) {
            return std::array<uint32_t, 8>{
                21, 31, 20, 30, 19, 29, 18, 28
            };
        } else if constexpr (BIT_WIDTH == 22) {
            return std::array<uint32_t, 8>{
                22, 0, 24, 0, 26, 16, 28, 18
            };
        } else if constexpr (BIT_WIDTH == 23) {
            return std::array<uint32_t, 8>{
                23, 0, 28, 19, 0, 24, 0, 29
            };
        } else if constexpr (BIT_WIDTH == 24) {
            return std::array<uint32_t, 8>{
                0, 8, 16, 0, 8, 16, 0, 8,
            };
        }
        return std::array<uint32_t, 8>{};
        // clang-format on
    }();

    alignas(64) inline static constexpr std::array<uint32_t, 8> shift3 = [] {
        // clang-format off
        if constexpr (BIT_WIDTH == 17) {
            return std::array<uint32_t, 8>{
                0, 15, 13, 11, 9, 7, 5, 3
            };
        } else if constexpr (BIT_WIDTH == 18) {
            return std::array<uint32_t, 8>{
                0, 14, 10, 6, 2, 16, 12, 8
            };
        } else if constexpr (BIT_WIDTH == 19) {
            return std::array<uint32_t, 8>{
                0, 13, 7, 1, 14, 8, 2, 15
            };
        } else if constexpr (BIT_WIDTH == 20) {
            return std::array<uint32_t, 8>{
                0, 12, 4, 16, 8, 0, 12, 4
            };
        } else if constexpr (BIT_WIDTH == 21) {
            return std::array<uint32_t, 8>{
                0, 11, 1, 12, 2, 13, 3, 14
            };
        } else if constexpr (BIT_WIDTH == 22) {
            return std::array<uint32_t, 8>{
                0, 10, 20, 8, 18, 6, 16, 4
            };
        } else if constexpr (BIT_WIDTH == 23) {
            return std::array<uint32_t, 8>{
                0, 9, 18, 4, 13, 22, 8, 17
            };
        }
        return std::array<uint32_t, 8>{};
        // clang-format on
    }();

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
    __always_inline static T get_permute1() {
        if constexpr (std::is_same_v<T, __m256i>) {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(permute1.data()));
        } else if constexpr (std::is_same_v<T, __m128i>) {
            return _mm_load_si128(reinterpret_cast<const __m128i*>(permute1.data()));
        }
        return T{};
    }

    __always_inline static T get_permute2() {
        if constexpr (std::is_same_v<T, __m256i>) {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(permute2.data()));
        } else if constexpr (std::is_same_v<T, __m128i>) {
            return _mm_load_si128(reinterpret_cast<const __m128i*>(permute2.data()));
        }
        return T{};
    }

    __always_inline static T get_permute3() {
        if constexpr (std::is_same_v<T, __m256i>) {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(permute3.data()));
        } else if constexpr (std::is_same_v<T, __m128i>) {
            return _mm_load_si128(reinterpret_cast<const __m128i*>(permute3.data()));
        }
        return T{};
    }

    __always_inline static T get_shift1() {
        if constexpr (std::is_same_v<T, __m256i>) {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(shift1.data()));
        } else if constexpr (std::is_same_v<T, __m128i>) {
            return _mm_load_si128(reinterpret_cast<const __m128i*>(shift1.data()));
        }
        return T{};
    }

    __always_inline static T get_shift2() {
        if constexpr (std::is_same_v<T, __m256i>) {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(shift2.data()));
        } else if constexpr (std::is_same_v<T, __m128i>) {
            return _mm_load_si128(reinterpret_cast<const __m128i*>(shift2.data()));
        }
        return T{};
    }

    __always_inline static T get_shift3() {
        if constexpr (std::is_same_v<T, __m256i>) {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(shift3.data()));
        } else if constexpr (std::is_same_v<T, __m128i>) {
            return _mm_load_si128(reinterpret_cast<const __m128i*>(shift3.data()));
        }
        return T{};
    }
#pragma GCC diagnostic pop
};

template <uint8_t BIT_WIDTH, typename T>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24 && (std::is_same_v<T, __m128i> || std::is_same_v<T, __m256i>))
struct unpack_tables_avx2 {
    alignas(32) inline static constexpr std::array<int32_t, 8> permute = [] {
        // clang-format off
        if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
            return std::array{0, -1, -1, -1, 0, 1, -1, -1};
        } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
            return std::array{0, 1, -1, -1, 1, 2, 3, -1};
        } else if constexpr (BIT_WIDTH >= 17 && BIT_WIDTH <= 24) {
            return std::array{0, 1, 2, -1, 2, 3, 4, 5};
        }
        // clang-format on
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

                4, -1, -1, -1,
                5, -1, -1, -1,
                6, -1, -1, -1,
                7, -1, -1, -1
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

                4, 5, -1, -1,
                6, 7, -1, -1,
                8, 9, -1, -1,
                10, 11, -1, -1,
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

                4,  5,  6, -1,
                7,  8,  9, -1,
                10, 11, 12, -1,
                13, 14, 15, -1,
            };
        }
        // clang-format on
    }();

    alignas(64) inline static constexpr std::array<int32_t, 8> shift = [] {
        std::array<int32_t, 8> shifts{};

        for (std::size_t lane = 0; lane < 8; ++lane) {
            const int bit_offset  = lane * BIT_WIDTH;
            const int bit_in_byte = bit_offset % 8;
            const int left_shift  = 32 - BIT_WIDTH - bit_in_byte;
            shifts[lane]          = left_shift;
        }

        return shifts;
    }();

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
    // The table storage is laid out as plain arrays to keep the constant
    // expressions readable; these helpers materialize them as vector registers.
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
#pragma GCC diagnostic pop
};

}  // namespace pernix::internal

#endif  // PERNIX_AVX2_TABLES_H
