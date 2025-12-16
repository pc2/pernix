#include <libcompression/helper.h>
#ifdef LIBCOMPRESSION_AVX512_ENABLED

#include <gtest/gtest.h>
#include <libcompression/bitpacking/unpacking_avx512vbmi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <cstring>

#include "bitpacking_testset.h"
#include "bitpacking_macros.h"

using namespace libcompression::bitpacking;

using AVX512VBMIUnpacking        = testing::TestWithParam<int>;
using AVX512VBMIUnpackingAligned = testing::TestWithParam<int>;

TEST_UNPACK_MM_EPI32 (mm_unpack_epi32_avx512vbmi)

TEST_UNPACK_MM256_EPI32 (mm256_unpack_epi32_avx512vbmi)

TEST_UNPACK_MM512_EPI32 (mm512_unpack_epi32_avx512vbmi)
TEST_UNPACK_MM512_EPI32 (mm512_unpack_aligned_epi32_avx512vbmi)

TEST_P(AVX512VBMIUnpacking, test_mm_unpack_epi32_avx512vbmi) {
    test_mm_unpack_epi32_avx512vbmi(static_cast<uint8_t>(GetParam()));
}

TEST_P(AVX512VBMIUnpacking, test_mm256_unpack_epi32_avx512vbmi) {
    test_mm256_unpack_epi32_avx512vbmi(static_cast<uint8_t>(GetParam()));
}

TEST_P(AVX512VBMIUnpacking, test_mm512_unpack_epi32_avx512vbmi) {
    test_mm512_unpack_epi32_avx512vbmi(static_cast<uint8_t>(GetParam()));
}

TEST_P(AVX512VBMIUnpackingAligned, test_mm512_unpack_aligned_epi32_avx512vbmi) {
    test_mm512_unpack_aligned_epi32_avx512vbmi(static_cast<uint8_t>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(AVX512VBMI, AVX512VBMIUnpacking, testing::Range(1, 25));
INSTANTIATE_TEST_SUITE_P(AVX512VBMI, AVX512VBMIUnpackingAligned, testing::Values(8, 16));
#endif  // LIBCOMPRESSION_AVX512_ENABLED)
