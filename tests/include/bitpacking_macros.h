#ifndef PERNIX_BITPACKING_MACROS_H
#define PERNIX_BITPACKING_MACROS_H

// clang-format off
#define TEST_UNPACK_MM_EPI32(method)                                                    \
void test_## method(const uint8_t bit_width) {                                           \
    constexpr uint32_t test_sets = 1 << 8;                                              \
                                                                                        \
    for (uint32_t i = 0; i < test_sets; i++) {                                          \
        constexpr uint32_t size = 4;                                                    \
        const TestSet<true, size> test_set(bit_width);                                  \
        std::array<int32_t, size> result{};                                             \
                                                                                        \
        const auto unpacked = method(bit_width, test_set.get_packed_data().data());     \
        _mm_storeu_si128(reinterpret_cast<__m128i*>(result.data()), unpacked);          \
                                                                                        \
        test_set.validate_unpacked(result);                                             \
    }                                                                                   \
}

#define TEST_UNPACK_MM256_EPI32(method)                                                                             \
void test_## method(const uint8_t bit_width) {                                                                       \
    constexpr uint32_t test_sets = 1 << 8;                                                                          \
                                                                                                                    \
    for (uint32_t i = 0; i < test_sets; i++) {                                                                      \
        constexpr uint32_t size = 64 * 8;                                                                           \
        const TestSet<true, size> test_set(bit_width);                                                              \
        std::array<int32_t, size> result{};                                                                         \
                                                                                                                    \
        for (uint32_t offset = 0; offset < size / 8; offset++) {                                                    \
            const auto unpacked = method(bit_width, test_set.get_packed_data().data() + (offset * bit_width));      \
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(result.data() + (8 * offset)), unpacked);                \
        }                                                                                                           \
                                                                                                                    \
        test_set.validate_unpacked(result);                                                                         \
    }                                                                                                               \
}

#define TEST_UNPACK_MM512_EPI32(method)                                                                             \
void test_## method(const uint8_t bit_width) {                                                                       \
    constexpr uint32_t test_sets = 1 << 8;                                                                          \
                                                                                                                    \
    for (uint32_t i = 0; i < test_sets; i++) {                                                                      \
        constexpr uint32_t size = 64 * 8;                                                                           \
        const TestSet<true, size> test_set(bit_width);                                                              \
        std::array<int32_t, size> result{};                                                                         \
                                                                                                                    \
        for (uint32_t offset = 0; offset < size / 16; offset++) {                                                   \
            const auto unpacked = method(bit_width, test_set.get_packed_data().data() + (2 * offset * bit_width));  \
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(result.data() + (16 * offset)), unpacked);               \
        }                                                                                                           \
                                                                                                                    \
        test_set.validate_unpacked(result);                                                                         \
    }                                                                                                               \
}
// clang-format on

#endif  // PERNIX_BITPACKING_MACROS_H