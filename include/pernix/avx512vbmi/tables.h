#ifndef PERNIX_AVX512VBMI_TABLES_H
#define PERNIX_AVX512VBMI_TABLES_H

#include <pernix/simd_compat.h>

#include <algorithm>
#include <array>
#include <ranges>
#include <tuple>

namespace pernix::internal {

template <typename Vec, typename U, std::size_t N>
[[gnu::always_inline]] static inline Vec load_table(const std::array<U, N>& table) {
    static_assert(sizeof(table) >= sizeof(Vec), "table is smaller than requested SIMD vector");
    if constexpr (std::is_same_v<Vec, __m512i>) {
        return _mm512_load_si512(static_cast<const void*>(table.data()));
    } else if constexpr (std::is_same_v<Vec, __m256i>) {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(table.data()));
    } else {
        return _mm_load_si128(reinterpret_cast<const __m128i*>(table.data()));
    }
}

template <uint8_t N, typename Vec>
    requires(N >= 9 && N <= 15)
struct pack_tables_avx512_16 {
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
        return std::array<int16_t, 32>{};
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
        return std::array<int16_t, 32>{};
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
        return std::array<int16_t, 32>{};
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
        return std::array<int16_t, 32>{};
        // clang-format on
    }();

    inline static constexpr std::tuple<__mmask32, __mmask32, __mmask32> get_permute_masks() {
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

    [[gnu::always_inline]] static inline Vec get_permute1() { return load_table<Vec>(permute1); }
    [[gnu::always_inline]] static inline Vec get_permute2() { return load_table<Vec>(permute2); }
    [[gnu::always_inline]] static inline Vec get_permute3() { return load_table<Vec>(permute3); }

    [[gnu::always_inline]] static inline Vec get_shift1() { return load_table<Vec>(shift1); }
    [[gnu::always_inline]] static inline Vec get_shift2() { return load_table<Vec>(shift2); }
    [[gnu::always_inline]] static inline Vec get_shift3() { return load_table<Vec>(shift3); }
};

template <std::uint8_t BIT_WIDTH, typename Vec>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24 &&
             (std::is_same_v<Vec, __m128i> || std::is_same_v<Vec, __m256i> || std::is_same_v<Vec, __m512i>))
struct pack_tables_avx512_24 {
private:
    struct word_plan {
        int32_t left_index1  = -1;
        int32_t left_index2  = -1;
        int32_t right_index  = -1;
        uint32_t left_shift1 = 32;
        uint32_t left_shift2 = 32;
        uint32_t right_shift = 32;
    };

    static constexpr word_plan create_plan(const uint32_t idx) {
        word_plan plan{};

        const uint32_t word_start = idx * 32u;
        const uint32_t word_end   = word_start + 32u;

        uint32_t left_slot = 0;
        for (uint32_t input_lane = 0; input_lane < 16; ++input_lane) {
            const uint32_t input_start = input_lane * BIT_WIDTH;
            const uint32_t input_end   = input_start + BIT_WIDTH;

            const uint32_t overlap_start = std::max(word_start, input_start);
            const uint32_t overlap_end   = std::min(word_end, input_end);
            if (overlap_start >= overlap_end) {
                continue;
            }

            const auto output_bit = static_cast<int32_t>(overlap_start - word_start);
            const auto input_bit  = static_cast<int32_t>(overlap_start - input_start);
            const int32_t delta   = output_bit - input_bit;

            if (delta >= 0) {
                if (left_slot == 0) {
                    plan.left_index1 = static_cast<int32_t>(input_lane);
                    plan.left_shift1 = static_cast<uint32_t>(delta);
                    ++left_slot;
                } else {
                    plan.left_index2 = static_cast<int32_t>(input_lane);
                    plan.left_shift2 = static_cast<uint32_t>(delta);
                }
            } else {
                plan.right_index = static_cast<int32_t>(input_lane);
                plan.right_shift = static_cast<uint32_t>(-delta);
            }
        }

        return plan;
    }

    static inline constexpr std::array<word_plan, 16> word_plans = [] {
        std::array<word_plan, 16> plans{};
        for (uint32_t i = 0; i < 16; ++i) {
            plans[i] = create_plan(i);
        }
        return plans;
    }();

    template <typename U, typename Getter>
    [[gnu::always_inline]] static constexpr std::array<U, 16> make_table(Getter getter) {
        std::array<U, 16> values{};
        for (uint32_t i = 0; i < 16; ++i) {
            values[i] = getter(word_plans[i]);
        }
        return values;
    }

    alignas(64) static inline constexpr auto permute1 = make_table<int32_t>([](const word_plan& p) { return p.left_index1; });

    alignas(64) static inline constexpr auto permute2 = make_table<int32_t>([](const word_plan& p) { return p.left_index2; });

    alignas(64) static inline constexpr auto permute3 = make_table<int32_t>([](const word_plan& p) { return p.right_index; });

    alignas(64) static inline constexpr auto shift1 = make_table<uint32_t>([](const word_plan& p) { return p.left_shift1; });

    alignas(64) static inline constexpr auto shift2 = make_table<uint32_t>([](const word_plan& p) { return p.left_shift2; });

    alignas(64) static inline constexpr auto shift3 = make_table<uint32_t>([](const word_plan& p) { return p.right_shift; });

public:
    [[gnu::always_inline]] static inline Vec get_permute1() { return load_table<Vec>(permute1); }
    [[gnu::always_inline]] static inline Vec get_permute2() { return load_table<Vec>(permute2); }
    [[gnu::always_inline]] static inline Vec get_permute3() { return load_table<Vec>(permute3); }

    [[gnu::always_inline]] static inline Vec get_shift1() { return load_table<Vec>(shift1); }
    [[gnu::always_inline]] static inline Vec get_shift2() { return load_table<Vec>(shift2); }
    [[gnu::always_inline]] static inline Vec get_shift3() { return load_table<Vec>(shift3); }
};

template <uint8_t BIT_WIDTH, typename Vec>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8 &&
             (std::is_same_v<Vec, __m512i> || std::is_same_v<Vec, __m256i> || std::is_same_v<Vec, __m128i>))
struct unpack_tables_avx512_8 {
private:
    alignas(64) inline static constexpr std::array<int8_t, 64> permute1 = [] {
        std::array<int8_t, 64> table{};
        std::ranges::fill(table, -1);
        for (size_t entry = 0; entry < 64; ++entry) {
            const size_t bit_start  = entry * BIT_WIDTH;
            const size_t first_byte = bit_start / 8;

            table[entry] = static_cast<int8_t>(first_byte);
        }

        return table;
    }();

    alignas(64) inline static constexpr std::array<int8_t, 64> permute2 = [] {
        std::array<int8_t, 64> table{};
        std::ranges::fill(table, -1);

        for (size_t entry = 0; entry < 64; ++entry) {
            const size_t bit_start  = entry * BIT_WIDTH;
            const size_t first_byte = bit_start / 8;
            const size_t bit_offset = bit_start % 8;

            if (bit_offset + BIT_WIDTH > 8) {
                table[entry] = static_cast<int8_t>(first_byte + 1);
            }
        }

        return table;
    }();

    alignas(64) inline static constexpr std::array<int8_t, 64> shift1 = [] {
        std::array<int8_t, 64> table{};

        for (size_t entry = 0; entry < 64; ++entry) {
            const size_t bit_start  = entry * BIT_WIDTH;
            const size_t bit_offset = bit_start % 8;

            table[entry] = static_cast<int8_t>(bit_offset);
        }

        return table;
    }();

    alignas(64) inline static constexpr std::array<int8_t, 64> shift2 = [] {
        std::array<int8_t, 64> table{};

        for (size_t entry = 0; entry < 64; ++entry) {
            const size_t bit_start  = entry * BIT_WIDTH;
            const size_t bit_offset = bit_start % 8u;
            const size_t spill_bits = (bit_offset + BIT_WIDTH > 8u) ? (bit_offset + BIT_WIDTH - 8u) : 0u;

            table[entry] = spill_bits ? static_cast<int8_t>(8 - bit_offset) : 0;
        }

        return table;
    }();

public:
    [[gnu::always_inline]] static inline Vec get_permute1() { return load_table<Vec>(permute1); }
    [[gnu::always_inline]] static inline Vec get_permute2() { return load_table<Vec>(permute2); }

    [[gnu::always_inline]] static inline Vec get_shift1() { return load_table<Vec>(shift1); }
    [[gnu::always_inline]] static inline Vec get_shift2() { return load_table<Vec>(shift2); }
};

template <uint8_t BIT_WIDTH, typename Vec>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16 &&
             (std::is_same_v<Vec, __m512i> || std::is_same_v<Vec, __m256i> || std::is_same_v<Vec, __m128i>))
struct unpack_tables_avx512_16 {
private:
    alignas(64) inline static constexpr std::array<int8_t, 64> permute1 = [] {
        std::array<int8_t, 64> table{};
        std::ranges::fill(table, -1);

        for (size_t entry = 0; entry < 32; ++entry) {
            const size_t bit_start  = entry * BIT_WIDTH;
            const size_t first_byte = bit_start / 8;
            const size_t base       = entry * 2;

            table[base]     = static_cast<int8_t>(first_byte);
            table[base + 1] = static_cast<int8_t>(first_byte + 1);
        }

        return table;
    }();

    alignas(64) inline static constexpr std::array<int8_t, 64> permute2 = [] {
        std::array<int8_t, 64> table{};
        std::ranges::fill(table, -1);

        for (size_t entry = 0; entry < 32; ++entry) {
            const size_t bit_start  = entry * BIT_WIDTH;
            const size_t first_byte = bit_start / 8;
            const size_t bit_offset = bit_start % 8;
            const size_t base       = entry * 2;

            if (bit_offset + BIT_WIDTH > 16) {
                table[base] = static_cast<int8_t>(first_byte + 2);
            }
        }

        return table;
    }();

    alignas(64) inline static constexpr std::array<int16_t, 32> shift1 = [] {
        std::array<int16_t, 32> table{};

        for (size_t entry = 0; entry < 32; ++entry) {
            const size_t bit_start  = entry * BIT_WIDTH;
            const size_t bit_offset = bit_start % 8u;

            // Right-shift the 16-bit chunk so the value starts at bit 0.
            table[entry] = static_cast<int16_t>(bit_offset);
        }

        return table;
    }();

    alignas(64) inline static constexpr std::array<int16_t, 32> shift2 = [] {
        std::array<int16_t, 32> table{};
        for (size_t entry = 0; entry < 32; ++entry) {
            const size_t bit_start  = entry * BIT_WIDTH;
            const size_t bit_offset = bit_start % 8u;
            const size_t spill_bits = (bit_offset + BIT_WIDTH > 16u) ? (bit_offset + BIT_WIDTH - 16u) : 0u;

            // Move spill bits from byte3 to their final bit positions before merge.
            table[entry] = spill_bits ? static_cast<int16_t>(16u - bit_offset) : 0;
        }

        return table;
    }();

public:
    [[gnu::always_inline]] static inline Vec get_permute1() { return load_table<Vec>(permute1); }
    [[gnu::always_inline]] static inline Vec get_permute2() { return load_table<Vec>(permute2); }

    [[gnu::always_inline]] static inline Vec get_shift1() { return load_table<Vec>(shift1); }
    [[gnu::always_inline]] static inline Vec get_shift2() { return load_table<Vec>(shift2); }
};

template <uint8_t BIT_WIDTH, typename Vec>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24 &&
             (std::is_same_v<Vec, __m512i> || std::is_same_v<Vec, __m256i> || std::is_same_v<Vec, __m128i>))
struct unpack_tables_avx512_24 {
private:
    alignas(64) inline static constexpr std::array<int8_t, 64> permute = [] {
        std::array<int8_t, 64> table{};
        std::ranges::fill(table, -1);

        for (size_t entry = 0; entry < 16; ++entry) {
            const size_t bit_start = entry * BIT_WIDTH;
            const size_t bit_end   = bit_start + BIT_WIDTH - 1;

            const size_t first_byte = bit_start / 8;
            const size_t last_byte  = bit_end / 8;

            const size_t base = entry * 4;

            for (size_t byte = first_byte; byte <= last_byte; ++byte) {
                table[base + (byte - first_byte)] = static_cast<int8_t>(byte);
            }
        }

        return table;
    }();

    alignas(64) inline static constexpr std::array<int32_t, 16> shift = [] {
        std::array<int32_t, 16> table{};

        for (size_t entry = 0; entry < 16; ++entry) {
            const size_t bit_start = entry * BIT_WIDTH;
            table[entry]           = static_cast<int32_t>(32u - BIT_WIDTH - (bit_start % 8u));
        }

        return table;
    }();

public:
    [[gnu::always_inline]] static inline Vec get_permute() { return load_table<Vec>(permute); }
    [[gnu::always_inline]] static inline Vec get_shift() { return load_table<Vec>(shift); }
};

}  // namespace pernix::internal

#endif  // PERNIX_AVX512VBMI_TABLES_H
