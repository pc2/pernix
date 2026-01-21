#include <pernix/bitpacking/unpacking_avx2.h>

#include <stdexcept>

#ifdef PERNIX_AVX2_ENABLED
namespace pernix::bitpacking {
auto mm_unpack_aligned_epi32_avx2(const uint8_t bit_width, const uint8_t* __restrict__ input) -> __m128i {
    switch (bit_width) {
        BITUNPACKING_SWITCH_CASE(mm_unpack_aligned_epi32_avx2, 8, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_aligned_epi32_avx2, 16, true)
        default:
            throw std::invalid_argument("Unsupported bit width for aligned AVX2 unpacking");
    }
}

auto mm_unpack_epi32_avx2(const uint8_t bit_width, const uint8_t* __restrict__ input) -> __m128i {
    switch (bit_width) {
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx2, 1, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx2, 2, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx2, 3, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx2, 4, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx2, 5, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx2, 6, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx2, 7, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx2, 8, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx2, 9, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx2, 10, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx2, 11, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx2, 12, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx2, 13, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx2, 14, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx2, 15, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx2, 16, true)
        // BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx2, 17, true)
        // BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx2, 18, true)
        // BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx2, 19, true)
        // BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx2, 20, true)
        // BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx2, 21, true)
        // BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx2, 22, true)
        // BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx2, 23, true)
        // BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx2, 24, true)
        default:
            throw std::invalid_argument("Unsupported bit width for AVX2 unpacking");
    }
}

auto mm256_unpack_aligned_epi32_avx2(const uint8_t bit_width, const uint8_t* __restrict__ input) -> __m256i {
    switch (bit_width) {
        BITUNPACKING_SWITCH_CASE(mm256_unpack_aligned_epi32_avx2, 8, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_aligned_epi32_avx2, 16, true)
        default:
            throw std::invalid_argument("Unsupported bit width for aligned AVX2 unpacking");
    }
}

auto mm256_unpack_epi32_avx2(const uint8_t bit_width, const uint8_t* __restrict__ input) -> __m256i {
    switch (bit_width) {
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx2, 1, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx2, 2, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx2, 3, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx2, 4, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx2, 5, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx2, 6, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx2, 7, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx2, 8, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx2, 9, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx2, 10, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx2, 11, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx2, 12, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx2, 13, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx2, 14, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx2, 15, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx2, 16, true)
        // BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx2, 17, true)
        // BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx2, 18, true)
        // BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx2, 19, true)
        // BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx2, 20, true)
        // BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx2, 21, true)
        // BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx2, 22, true)
        // BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx2, 23, true)
        // BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx2, 24, true)
        default:
            throw std::invalid_argument("Unsupported bit width for AVX2 unpacking");
    }
}
}  // namespace pernix::bitpacking
#endif  // PERNIX_AVX2_ENABLED