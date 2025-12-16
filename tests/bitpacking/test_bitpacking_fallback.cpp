#include <libcompression/helper.h>
#ifdef LIBCOMPRESSION_AVX2_ENABLED

#include <gtest/gtest.h>
#include <libcompression/bitpacking/unpacking_fallback.h>
#include <libcompression/bitpacking/packing_fallback.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>

#include "bitpacking_testset.h"
#include "bitpacking_macros.h"

using namespace libcompression::bitpacking;

using FallbackUnpacking = testing::TestWithParam<int>;
using FallbackPacking   = testing::TestWithParam<int>;

void test_unpack_epi32_fallback(const uint8_t bit_width) {
    constexpr std::size_t ELEMENTS = 64;
    const TestSet<true, ELEMENTS> test_set(bit_width);

    const auto unpacked_data = unpack_epi32_fallback(bit_width, test_set.get_packed_data().data(), ELEMENTS);

    std::array<int32_t, ELEMENTS> unpacked_array{};
    std::ranges::copy(unpacked_data, unpacked_array.begin());

    test_set.validate_unpacked(unpacked_array);
}

TEST_P(FallbackUnpacking, test_unpack_epi32_fallback) {
    test_unpack_epi32_fallback(static_cast<uint8_t>(GetParam()));
}

void test_pack_epi32_fallback(const uint8_t bit_width) {
    constexpr std::size_t ELEMENTS = 64;
    const TestSet<true, ELEMENTS> test_set(bit_width);
    const auto total_bits = ELEMENTS * bit_width;
    const auto num_bytes  = (total_bits + 7) / 8;

    std::vector<uint32_t> unpacked_vec(ELEMENTS);
    std::ranges::copy(test_set.get_unpacked_data(), unpacked_vec.begin());

    std::vector<uint8_t> packed_vector(num_bytes, 0);

    pack_epi32_fallback(bit_width, unpacked_vec, packed_vector.data());

    test_set.validate_packed(packed_vector);
}

TEST_P(FallbackPacking, test_pack_epi32_fallback) {
    test_pack_epi32_fallback(static_cast<uint8_t>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(Fallback, FallbackUnpacking, testing::Range(1, 25));
INSTANTIATE_TEST_SUITE_P(Fallback, FallbackPacking, testing::Range(1, 25));
#endif  // LIBCOMPRESSION_AVX2_ENABLED)
