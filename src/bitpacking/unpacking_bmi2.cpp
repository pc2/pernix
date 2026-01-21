#include <pernix/bitpacking/unpacking_bmi2.h>

#ifdef PERNIX_AVX2_ENABLED

namespace pernix::bitpacking {
auto mm_unpack_epi32_bmi2(const uint8_t bit_width, const uint8_t* __restrict__ input) -> __m128i {
    switch (bit_width) {
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_bmi2, 1, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_bmi2, 2, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_bmi2, 3, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_bmi2, 4, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_bmi2, 5, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_bmi2, 6, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_bmi2, 7, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_bmi2, 8, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_bmi2, 9, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_bmi2, 10, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_bmi2, 11, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_bmi2, 12, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_bmi2, 13, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_bmi2, 14, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_bmi2, 15, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_bmi2, 16, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_bmi2, 17, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_bmi2, 18, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_bmi2, 19, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_bmi2, 20, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_bmi2, 21, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_bmi2, 22, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_bmi2, 23, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_bmi2, 24, true)
        default:
            throw std::invalid_argument("Unsupported bit width for BMI2 unpacking");
    }
}

auto mm256_unpack_epi32_bmi2(const uint8_t bit_width, const uint8_t* __restrict__ input) -> __m256i {
    switch (bit_width) {
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_bmi2, 1, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_bmi2, 2, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_bmi2, 3, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_bmi2, 4, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_bmi2, 5, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_bmi2, 6, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_bmi2, 7, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_bmi2, 8, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_bmi2, 9, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_bmi2, 10, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_bmi2, 11, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_bmi2, 12, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_bmi2, 13, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_bmi2, 14, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_bmi2, 15, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_bmi2, 16, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_bmi2, 17, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_bmi2, 18, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_bmi2, 19, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_bmi2, 20, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_bmi2, 21, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_bmi2, 22, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_bmi2, 23, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_bmi2, 24, true)
        default:
            throw std::invalid_argument("Unsupported bit width for BMI2 unpacking");
    }
}
}  // namespace pernix::bitpacking

#endif  // PERNIX_AVX2_ENABLED