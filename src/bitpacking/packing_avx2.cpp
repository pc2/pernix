#include <libcompression/bitpacking/packing_avx2.h>

#ifdef LIBCOMPRESSION_AVX2_ENABLED

#include <stdexcept>

namespace libcompression::bitpacking {
auto mm_pack_aligned_epi32_avx2(const uint8_t bit_width, __m128i& input) -> __m128i {
    switch (bit_width) {
        BITPACKING_SWITCH_CASE(mm_pack_aligned_epi32_avx2, 8)
        BITPACKING_SWITCH_CASE(mm_pack_aligned_epi32_avx2, 16)
        default:
            throw std::invalid_argument("Unsupported bit width for aligned AVX2 packing");
    }
}

auto mm_pack_epi32_avx2(const uint8_t bit_width, __m128i& input) -> __m128i {
    switch (bit_width) {
        BITPACKING_SWITCH_CASE(mm_pack_epi32_avx2, 1)
        BITPACKING_SWITCH_CASE(mm_pack_epi32_avx2, 2)
        BITPACKING_SWITCH_CASE(mm_pack_epi32_avx2, 3)
        BITPACKING_SWITCH_CASE(mm_pack_epi32_avx2, 4)
        BITPACKING_SWITCH_CASE(mm_pack_epi32_avx2, 5)
        BITPACKING_SWITCH_CASE(mm_pack_epi32_avx2, 6)
        BITPACKING_SWITCH_CASE(mm_pack_epi32_avx2, 7)
        BITPACKING_SWITCH_CASE(mm_pack_epi32_avx2, 8)
        BITPACKING_SWITCH_CASE(mm_pack_epi32_avx2, 9)
        BITPACKING_SWITCH_CASE(mm_pack_epi32_avx2, 10)
        BITPACKING_SWITCH_CASE(mm_pack_epi32_avx2, 11)
        BITPACKING_SWITCH_CASE(mm_pack_epi32_avx2, 12)
        BITPACKING_SWITCH_CASE(mm_pack_epi32_avx2, 13)
        BITPACKING_SWITCH_CASE(mm_pack_epi32_avx2, 14)
        BITPACKING_SWITCH_CASE(mm_pack_epi32_avx2, 15)
        BITPACKING_SWITCH_CASE(mm_pack_epi32_avx2, 16)
        BITPACKING_SWITCH_CASE(mm_pack_epi32_avx2, 17)
        BITPACKING_SWITCH_CASE(mm_pack_epi32_avx2, 18)
        BITPACKING_SWITCH_CASE(mm_pack_epi32_avx2, 19)
        BITPACKING_SWITCH_CASE(mm_pack_epi32_avx2, 20)
        BITPACKING_SWITCH_CASE(mm_pack_epi32_avx2, 21)
        BITPACKING_SWITCH_CASE(mm_pack_epi32_avx2, 22)
        BITPACKING_SWITCH_CASE(mm_pack_epi32_avx2, 23)
        BITPACKING_SWITCH_CASE(mm_pack_epi32_avx2, 24)
        default:
            throw std::invalid_argument("Unsupported bit width for AVX2 packing");
    }
}

auto mm256_pack_aligned_epi32_avx2(const uint8_t bit_width, __m256i& input) -> __m256i {
    switch (bit_width) {
        BITPACKING_SWITCH_CASE(mm256_pack_aligned_epi32_avx2, 8)
        BITPACKING_SWITCH_CASE(mm256_pack_aligned_epi32_avx2, 16)
        default:
            throw std::invalid_argument("Unsupported bit width for aligned AVX2 unpacking");
    }
}

auto mm256_pack_epi32_avx2(const uint8_t bit_width, __m256i& input) -> __m256i {
    switch (bit_width) {
        BITPACKING_SWITCH_CASE(mm256_pack_epi32_avx2, 1)
        BITPACKING_SWITCH_CASE(mm256_pack_epi32_avx2, 2)
        BITPACKING_SWITCH_CASE(mm256_pack_epi32_avx2, 3)
        BITPACKING_SWITCH_CASE(mm256_pack_epi32_avx2, 4)
        BITPACKING_SWITCH_CASE(mm256_pack_epi32_avx2, 5)
        BITPACKING_SWITCH_CASE(mm256_pack_epi32_avx2, 6)
        BITPACKING_SWITCH_CASE(mm256_pack_epi32_avx2, 7)
        BITPACKING_SWITCH_CASE(mm256_pack_epi32_avx2, 8)
        BITPACKING_SWITCH_CASE(mm256_pack_epi32_avx2, 9)
        BITPACKING_SWITCH_CASE(mm256_pack_epi32_avx2, 10)
        BITPACKING_SWITCH_CASE(mm256_pack_epi32_avx2, 11)
        BITPACKING_SWITCH_CASE(mm256_pack_epi32_avx2, 12)
        BITPACKING_SWITCH_CASE(mm256_pack_epi32_avx2, 13)
        BITPACKING_SWITCH_CASE(mm256_pack_epi32_avx2, 14)
        BITPACKING_SWITCH_CASE(mm256_pack_epi32_avx2, 15)
        BITPACKING_SWITCH_CASE(mm256_pack_epi32_avx2, 16)
        BITPACKING_SWITCH_CASE(mm256_pack_epi32_avx2, 17)
        BITPACKING_SWITCH_CASE(mm256_pack_epi32_avx2, 18)
        BITPACKING_SWITCH_CASE(mm256_pack_epi32_avx2, 19)
        BITPACKING_SWITCH_CASE(mm256_pack_epi32_avx2, 20)
        BITPACKING_SWITCH_CASE(mm256_pack_epi32_avx2, 21)
        BITPACKING_SWITCH_CASE(mm256_pack_epi32_avx2, 22)
        BITPACKING_SWITCH_CASE(mm256_pack_epi32_avx2, 23)
        BITPACKING_SWITCH_CASE(mm256_pack_epi32_avx2, 24)
        default:
            throw std::invalid_argument("Unsupported bit width for AVX2 packing");
    }
}
}

#endif // LIBCOMPRESSION_AVX2_ENABLED
