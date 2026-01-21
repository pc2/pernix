#include <pernix/helper.h>
#ifdef PERNIX_AVX2_ENABLED

#include <gtest/gtest.h>
#include <pernix/bitpacking/unpacking_bmi2.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>

#include "bitpacking_macros.h"
#include "bitpacking_testset.h"

using namespace pernix::bitpacking;

using BMI2Unpacking = testing::TestWithParam<int>;

TEST_UNPACK_MM_EPI32(mm_unpack_epi32_bmi2)
TEST_UNPACK_MM256_EPI32(mm256_unpack_epi32_bmi2)

TEST_P(BMI2Unpacking, test_mm_unpack_epi32_bmi2) {
    test_mm_unpack_epi32_bmi2(static_cast<uint8_t>(GetParam()));
}

TEST_P(BMI2Unpacking, test_mm256_unpack_epi32_bmi2) {
    test_mm256_unpack_epi32_bmi2(static_cast<uint8_t>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(BMI2, BMI2Unpacking, testing::Range(1, 25));
#endif  // PERNIX_AVX2_ENABLED)
