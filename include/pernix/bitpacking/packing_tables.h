#ifndef PERNIX_PACKING_TABLES_H
#define PERNIX_PACKING_TABLES_H

#include <pernix/helper.h>

#include <array>
#include <cstdint>
#include <tuple>

#ifdef PERNIX_AVX2_ENABLED
namespace pernix::bitpacking::internal {
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
};

template <__uint8_t N, typename T>
    requires(N >= 8 && N <= 16)
struct pack_tables_avx512 {
    alignas(64) inline static constexpr std::array<int16_t, 32> permute1 = [] {
        // clang-format off
if constexpr (N == 9) {
return std::array<int16_t, 32>{
0, 2, 4, 6,
-1, 9, 11, 13,
15, 16, 18, 20,
22, -1, 25, 27,

29, 31, 32, 34,
36, 38, -1, 41,
43, 45, 47, -1,
-1, -1, -1, -1
};
} else if constexpr (N == 10) {
return std::array<int16_t, 32>{
0, 2, -1, 5,
7, 8, 10, -1,
13, 15, 16, 18,
-1, 21, 23, 24,

26, -1, 29, 31,
32, 34, -1, 37,
39, 40, 42, -1,
45, 47, -1, -1
};
} else if constexpr (N == 11) {
return std::array<int16_t, 32>{
0, 2, 3, 5,
6, -1, 9, -1,
12, -1, 15, 16,
18, 19, 21, 22,

-1, 25, -1, 28,
-1, 31, 32, 34,
35, 37, 38, -1,
-1, -1, -1, -1
};
} else if constexpr (N == 12) {
return std::array<int16_t, 32>{
1, 2, 3, 5,
6, 7, 9, 10,
11, 13, 14, 15,
17, 18, 19, 21,

22, 23, 25, 26,
27, 29, 30, 31,
33, 34, 35, 37,
38, 39, 41, 42
};
} else if constexpr (N == 13) {
return std::array<int16_t, 32>{
0, -1, 3, 4,
5, -1, -1, 9,
10, -1, -1, 14,
15, 16, -1, 19,

20, 21, -1, -1,
25, 26, -1, -1,
30, 31, -1, -1,
-1, -1, -1, -1
};
} else if constexpr (N == 14) {
return std::array<int16_t, 32>{
1, 2, 3, 4,
5, 6, 7, 9,
10, 11, 12, 13,
14, 15, 17, 18,

19, 20, 21, 22,
23, 25, 26, 27,
28, 29, 30, 31,
-1, -1, -1, -1
};
} else if constexpr (N == 15) {
return std::array<int16_t, 32>{
1, 2, 3, 4,
5, 6, 7, 8,
9, 10, 11, 12,
13, 14, 15, 17,

18, 19, 20, 21,
22, 23, 24, 25,
26, 27, 28, 29,
30, 31, -1, -1
};
}
        // clang-format on
    }();

    alignas(64) inline static constexpr std::array<int16_t, 32> permute2 = [] {
        // clang-format off
if constexpr (N == 9) {
return std::array<int16_t, 32>{
1, 3, 5, 7,
8, 10, 12, 14,
-1, 17, 19, 21,
23, 24, 26, 28,

30, -1, 33, 35,
37, 39, 40, 42,
44, 46, -1, -1,
-1, -1, -1, -1
};
} else if constexpr (N == 10) {
return std::array<int16_t, 32>{
1, 3, 4, 6,
-1, 9, 11, 12,
14, -1, 17, 19,
20, 22, -1, 25,

27, 28, 30, -1,
33, 35, 36, 38,
-1, 41, 43, 44,
46, -1, -1, -1
};
} else if constexpr (N == 11) {
return std::array<int16_t, 32>{
1, -1, 4, -1,
7, 8, 10, 11,
13, 14, -1, 17,
-1, 20, -1, 23,

24, 26, 27, 29,
30, -1, 33, -1,
36, -1, 39, 40,
-1, -1, -1, -1
};
} else if constexpr (N == 12) {
return std::array<int16_t, 32>{
0, 1, 2, 4,
5, 6, 8, 9,
10, 12, 13, 14,
16, 17, 18, 20,

21, 22, 24, 25,
26, 28, 29, 30,
32, 33, 34, 36,
37, 38, 40, 41
};
} else if constexpr (N == 13) {
return std::array<int16_t, 32>{
1, 2, -1, -1,
6, 7, 8, -1,
11, 12, 13, -1,
-1, 17, 18, -1,

-1, 22, 23, 24,
-1, 27, 28, 29,
-1, -1, -1, -1,
-1, -1, -1, -1
};
} else if constexpr (N == 14) {
return std::array<int16_t, 32>{
0, 1, 2, 3,
4, 5, 6, 8,
9, 10, 11, 12,
13, 14, 16, 17,

18, 19, 20, 21,
22, 24, 25, 26,
27, 28, 29, 30,
-1, -1, -1, -1
};
} else if constexpr (N == 15) {
return std::array<int16_t, 32>{
0, 1, 2, 3,
4, 5, 6, 7,
8, 9, 10, 11,
12, 13, 14, 16,

17, 18, 19, 20,
21, 22, 23, 24,
25, 26, 27, 28,
29, 30, -1, -1
};
}
        // clang-format on
    }();

    alignas(64) inline static constexpr std::array<int16_t, 32> permute3 = [] {
        // clang-format off
if constexpr (N == 9) {
return std::array<int16_t, 32>{
-1, 1, 3, 5,
7, 8, 10, 12,
14, -1, 17, 19,
21, 23, 24, 26,

28, 30, -1, 33,
35, 37, 39, 40,
42, 44, 46, -1,
-1, -1, -1, -1
};
} else if constexpr (N == 10) {
return std::array<int16_t, 32>{
-1, 1, 3, 4,
6, -1, 9, 11,
12, 14, -1, 17,
19, 20, 22, -1,

25, 27, 28, 30,
-1, 33, 35, 36,
38, -1, 41, 43,
44, 46, -1, -1
};
} else if constexpr (N == 11) {
return std::array<int16_t, 32>{
-1, 1, 2, 4,
5, 7, 8, 10,
11, 13, 14, -1,
17, 18, 20, 21,

23, 24, 26, 27,
29, 30, -1, 33,
34, 36, 37, 39,
-1, -1, -1, -1
};
} else if constexpr (N == 13) {
return std::array<int16_t, 32>{
-1, 1, 2, 3,
4, 6, 7, 8,
9, 11, 12, 13,
14, -1, 17, 18,

19, 20, 22, 23,
24, 25, 27, 28,
29, 30, -1, -1,
-1, -1, -1, -1
};
}
return std::array<int16_t, 32>{};
        // clang-format on
    }();

    alignas(64) inline static constexpr std::array<int16_t, 32> shift1 = [] {
        // clang-format off
if constexpr (N == 9) {
return std::array<int16_t, 32>{
0, 2, 4, 6,
0, 1, 3, 5,
7, 0, 2, 4,

6, 0, 1, 3,
5, 7, 0, 2,
4, 6, 0, 1,
3, 5, 7, -1,
-1, -1, -1, -1
};
} else if constexpr (N == 10) {
return std::array<int16_t, 32>{
0, 4, 0, 2,
6, 0, 4, 0,
2, 6, 0, 4,
0, 2, 6, 0,

4, 0, 2, 6,
0, 4, 0, 2,
6, 0, 4, 0,
2, 6, -1, -1
};
} else if constexpr (N == 11) {
return std::array<int16_t, 32>{
0, 6, 1, 7,
2, 0, 3, 0,
4, 0, 5, 0,
6, 1, 7, 2,

0, 3, 0, 4,
0, 5, 0, 6,
1, 7, 2, 0,
-1, -1, -1, -1
};
} else if constexpr (N == 12) {
return std::array<int16_t, 32>{
12, 8, 4, 12,
8, 4, 12, 8,
4, 12, 8, 4,
12, 8, 4, 12,

8, 4, 12, 8,
4, 12, 8, 4,
12, 8, 4, 12,
8, 4, 12, 8
};
} else if constexpr (N == 13) {
return std::array<int16_t, 32>{
0, 0, 7, 4,
1, 0, 0, 5,
2, 0, 0, 6,
3, 0, 0, 7,

4, 1, 0, 0,
5, 2, 0, 0,
6, 3, -1, -1,
-1, -1, -1, -1
};
} else if constexpr (N == 14) {
return std::array<int16_t, 32>{
14, 12, 10, 8,
6, 4, 2, 14,
12, 10, 8, 6,
4, 2, 14, 12,

10, 8, 6, 4,
2, 14, 12, 10,
8, 6, 4, 2,
-1, -1, -1, -1
};
} else if constexpr (N == 15) {
return std::array<int16_t, 32>{
15, 14, 13, 12,
11, 10, 9, 8,
7, 6, 5, 4,
3, 2, 1, 15,

14, 13, 12, 11,
10, 9, 8, 7,
6, 5, 4, 3,
2, 1, -1, -1
};
}
        // clang-format on
    }();

    alignas(64) inline static constexpr std::array<int16_t, 32> shift2 = [] {
        // clang-format off
if constexpr (N == 9) {
return std::array<int16_t, 32>{
9, 11, 13, 15,
8, 10, 12, 14,
8, 9, 11, 13,
15, 8, 10, 12,

14, 8, 9, 11,
13, 15, 8, 10,
12, 14, 8, -1,
-1, -1, -1, -1
};
} else if constexpr (N == 10) {
return std::array<int16_t, 32>{
10, 14, 8, 12,
0, 10, 14, 8,
12, 0, 10, 14,
8, 12, 0, 10,

14, 8, 12, 0,
10, 14, 8, 12,
0, 10, 14, 8,
12, 0, -1, -1
};
} else if constexpr (N == 11) {
return std::array<int16_t, 32>{
11, 8, 12, 8,
13, 8, 14, 9,
15, 10, 8, 11,

8, 12, 8, 13,
8, 14, 9, 15,
10, 8, 11, 8,
12, 8, 13, 8,
-1, -1, -1, -1
};
} else if constexpr (N == 12) {
return std::array<int16_t, 32>{
0, 4, 8, 0,
4, 8, 0, 4,
8, 0, 4, 8,
0, 4, 8, 0,

4, 8, 0, 4,
8, 0, 4, 8,
0, 4, 8, 0,
4, 8, 0, 4
};
} else if constexpr (N == 13) {
return std::array<int16_t, 32>{
13, 10, 0, 0,
14, 11, 8, 0,
15, 12, 9, 0,
0, 13, 10, 0,

0, 14, 11, 8,
0, 15, 12, 9,
0, 0, -1, -1,
-1, -1, -1, -1
};
} else if constexpr (N == 14) {
return std::array<int16_t, 32>{
0, 2, 4, 6,
8, 10, 12, 0,
2, 4, 6, 8,
10, 12, 0, 2,

4, 6, 8, 10,
12, 0, 2, 4,
6, 8, 10, 12,
-1, -1, -1, -1
};
} else if constexpr (N == 15) {
return std::array<int16_t, 32>{
0, 1, 2, 3,
4, 5, 6, 7,
8, 9, 10, 11,
12, 13, 14, 0,

1, 2, 3, 4,
5, 6, 7, 8,
9, 10, 11, 12,
13, 14, -1, -1
};
}
        // clang-format on
    }();

    alignas(64) inline static constexpr std::array<int16_t, 32> shift3 = [] {
        // clang-format off
if constexpr (N == 9) {
return std::array<int16_t, 32>{
0, 7, 5, 3,
1, 8, 6, 4,
2, 0, 7, 5,
3, 1, 8, 6,

4, 2, 0, 7,
5, 3, 1, 8,
6, 4, 2, -1,
-1, -1, -1, -1
};
} else if constexpr (N == 10) {
return std::array<int16_t, 32>{
0, 6, 2, 8,
4, 0, 6, 2,
8, 4, 0, 6,
2, 8, 4, 0,

6, 2, 8, 4,
0, 6, 2, 8,
4, 0, 6, 2,
8, 4, -1, -1
};
} else if constexpr (N == 11) {
return std::array<int16_t, 32>{
0, 5, 10, 4,
9, 3, 8, 2,
7, 1, 6, 0,
5, 10, 4, 9,

3, 8, 2, 7,
1, 6, 0, 5,
10, 4, 9, 3,
-1, -1, -1, -1
};
} else if constexpr (N == 13) {
return std::array<int16_t, 32>{
0, 3, 6, 9,
12, 2, 5, 8,
11, 1, 4, 7,
10, 0, 3, 6,

9, 12, 2, 5,
8, 11, 1, 4,
7, 10, -1, -1,
-1, -1, -1, -1
};
}
        // clang-format on
    }();

    __always_inline static constexpr std::tuple<__mmask32, __mmask32, __mmask32> get_permute_masks() {
        // clang-format off
if constexpr (N == 9) {
return {
0x07BFDFEF,
0x03FDFEFF,
0x07FBFDFE
};
} else if constexpr (N == 10) {
return {
0x37BDEF7B,
0x1EF7BDEF,
0x3DEF7BDE
};
} else if constexpr (N == 11) {
return {
0x07EAFD5F,
0x0D5FABF5,
0x0FBFF7FE
};
} else if constexpr (N == 13) {
return {
0x0333B99D,
0x00EE6773,
0x03FFDFFE
};
}
return {0, 0, 0};
        // clang-format on
    }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
    __always_inline static T get_permute1() {
        if constexpr (std::is_same_v<T, __m512i>) {
            return _mm512_load_si512(permute1.data());
        } else if constexpr (std::is_same_v<T, __m256i>) {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(permute1.data()));
        } else if constexpr (std::is_same_v<T, __m128i>) {
            return _mm_load_si128(reinterpret_cast<const __m128i*>(permute1.data()));
        }
        return T{};
    }

    __always_inline static T get_permute2() {
        if constexpr (std::is_same_v<T, __m512i>) {
            return _mm512_load_si512(permute2.data());
        } else if constexpr (std::is_same_v<T, __m256i>) {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(permute2.data()));
        } else if constexpr (std::is_same_v<T, __m128i>) {
            return _mm_load_si128(reinterpret_cast<const __m128i*>(permute2.data()));
        }
        return T{};
    }

    __always_inline static T get_permute3() {
        if constexpr (std::is_same_v<T, __m512i>) {
            return _mm512_load_si512(permute3.data());
        } else if constexpr (std::is_same_v<T, __m256i>) {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(permute3.data()));
        } else if constexpr (std::is_same_v<T, __m128i>) {
            return _mm_load_si128(reinterpret_cast<const __m128i*>(permute3.data()));
        }
        return T{};
    }

    __always_inline static T get_shift1() {
        if constexpr (std::is_same_v<T, __m512i>) {
            return _mm512_load_si512(shift1.data());
        } else if constexpr (std::is_same_v<T, __m256i>) {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(shift1.data()));
        } else if constexpr (std::is_same_v<T, __m128i>) {
            return _mm_load_si128(reinterpret_cast<const __m128i*>(shift1.data()));
        }
        return T{};
    }

    __always_inline static T get_shift2() {
        if constexpr (std::is_same_v<T, __m512i>) {
            return _mm512_load_si512(shift2.data());
        } else if constexpr (std::is_same_v<T, __m256i>) {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(shift2.data()));
        } else if constexpr (std::is_same_v<T, __m128i>) {
            return _mm_load_si128(reinterpret_cast<const __m128i*>(shift2.data()));
        }
        return T{};
    }

    __always_inline static T get_shift3() {
        if constexpr (std::is_same_v<T, __m512i>) {
            return _mm512_load_si512(shift3.data());
        } else if constexpr (std::is_same_v<T, __m256i>) {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(shift3.data()));
        } else if constexpr (std::is_same_v<T, __m128i>) {
            return _mm_load_si128(reinterpret_cast<const __m128i*>(shift3.data()));
        }
        return T{};
    }
#pragma GCC diagnostic pop
};
}  // namespace pernix::bitpacking::internal
#endif  // PERNIX_AVX2_ENABLED
#endif  // PERNIX_PACKING_TABLES_H