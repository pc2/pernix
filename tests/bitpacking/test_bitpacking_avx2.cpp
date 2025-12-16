#include <libcompression/helper.h>
#ifdef LIBCOMPRESSION_AVX2_ENABLED

#include <gtest/gtest.h>
#include <libcompression/bitpacking/unpacking_avx2.h>
#include <libcompression/bitpacking/packing_avx2.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>

#include "bitpacking_testset.h"
#include "bitpacking_macros.h"

using namespace libcompression::bitpacking;

using AVX2Unpacking        = testing::TestWithParam<int>;
using AVX2UnpackingAligned = testing::TestWithParam<int>;

TEST_UNPACK_MM_EPI32(mm_unpack_aligned_epi32_avx2)
TEST_UNPACK_MM_EPI32(mm_unpack_epi32_avx2)

TEST_UNPACK_MM256_EPI32(mm256_unpack_aligned_epi32_avx2)
TEST_UNPACK_MM256_EPI32(mm256_unpack_epi32_avx2)

TEST_P(AVX2UnpackingAligned, test_mm_unpack_aligned_epi32_avx2) {
    test_mm_unpack_aligned_epi32_avx2(static_cast<uint8_t>(GetParam()));
}

TEST_P(AVX2Unpacking, test_mm_unpack_epi32_avx2) {
    test_mm_unpack_epi32_avx2(static_cast<uint8_t>(GetParam()));
}

TEST_P(AVX2UnpackingAligned, test_mm256_unpack_aligned_epi32_avx2) {
    test_mm256_unpack_aligned_epi32_avx2(static_cast<uint8_t>(GetParam()));
}

TEST_P(AVX2Unpacking, test_mm256_unpack_epi32_avx2) {
    test_mm256_unpack_epi32_avx2(static_cast<uint8_t>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(AVX2, AVX2Unpacking, testing::Range(1, 25));
INSTANTIATE_TEST_SUITE_P(AVX2, AVX2UnpackingAligned, testing::Values(8, 16));


using AVX2Packing        = testing::TestWithParam<int>;
using AVX2PackingAligned = testing::TestWithParam<int>;


// TEST_PACK_MM_EPI32 (mm_pack_aligned_epi32_avx2)
// TEST_PACK_MM_EPI32 (mm_pack_epi32_avx2)
//
// TEST_PACK_MM256_EPI32 (mm256_pack_aligned_epi32_avx2)
// TEST_PACK_MM256_EPI32 (mm256_pack_epi32_avx2)

void test_mm_pack_epi32_avx2(const uint8_t bit_width) {
    constexpr uint32_t test_sets = 1 << 8;

    for (uint32_t i = 0; i < test_sets; i++) {
        constexpr uint32_t size = 4;
        const TestSet<true, size> test_set(bit_width);
        std::vector<uint8_t> result(test_set.get_packed_data().size(), 0);

        auto input_data     = _mm_loadu_si128(reinterpret_cast<const __m128i*>(test_set.get_unpacked_data().data()));
        const auto unpacked = mm_pack_epi32_avx2(bit_width, input_data);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(result.data()), unpacked);

        test_set.validate_packed(result);
    }
}

void test_mm256_pack_epi32_avx2(const uint8_t bit_width) {
    constexpr uint32_t test_sets = 1 << 8;

    for (uint32_t i = 0; i < test_sets; i++) {
        constexpr uint32_t size = 64 * 8;
        const TestSet<true, size> test_set(bit_width);
        std::vector<uint8_t> result(test_set.get_packed_data().size());

        for (uint32_t offset = 0; offset < size / 8; offset++) {
            auto input_data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(test_set.get_unpacked_data().data() + (8 * offset)));

            const auto packed = mm256_pack_epi32_avx2(bit_width, input_data);
            _mm256_maskstore_epi32(reinterpret_cast<int*>(result.data() + (offset * bit_width)),
                                   libcompression::internal::mm256_convert_vmask_epi32((1 << (32 - bit_width)) - 1), packed);

            // _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_ptr), _mm256_castsi256_si128(unpacked));
            // _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_ptr + N), _mm256_extracti128_si256(unpacked, 1));
        }

        test_set.validate_packed(result);
    }
}

// TEST_P(AVX2PackingAligned, test_mm_pack_aligned_epi32_avx2) {
//     test_mm_pack_aligned_epi32_avx2(static_cast<uint8_t>(GetParam()));
// }

TEST_P(AVX2Packing, test_mm_pack_epi32_avx2) {
    test_mm_pack_epi32_avx2(static_cast<uint8_t>(GetParam()));
}

// TEST_P(AVX2PackingAligned, test_mm256_pack_aligned_epi32_avx2) {
//     test_mm256_pack_aligned_epi32_avx2(static_cast<uint8_t>(GetParam()));
// }

TEST_P(AVX2Packing, test_mm256_pack_epi32_avx2) {
    test_mm256_pack_epi32_avx2(static_cast<uint8_t>(GetParam()));
}


INSTANTIATE_TEST_SUITE_P(AVX2, AVX2Packing, testing::Range(17, 25));
INSTANTIATE_TEST_SUITE_P(AVX2, AVX2PackingAligned, testing::Values(8, 16));
#endif  // LIBCOMPRESSION_AVX2_ENABLED)
