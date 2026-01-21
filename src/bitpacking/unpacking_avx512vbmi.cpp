#include <pernix/bitpacking/unpacking_avx512vbmi.h>

#ifdef PERNIX_AVX512_VBMI_ENABLED
namespace pernix::bitpacking {
auto mm_unpack_epi32_avx512vbmi(const uint8_t bit_width, const uint8_t* __restrict__ input) -> __m128i {
    switch (bit_width) {
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx512vbmi, 1, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx512vbmi, 2, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx512vbmi, 3, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx512vbmi, 4, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx512vbmi, 5, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx512vbmi, 6, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx512vbmi, 7, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx512vbmi, 8, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx512vbmi, 9, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx512vbmi, 10, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx512vbmi, 11, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx512vbmi, 12, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx512vbmi, 13, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx512vbmi, 14, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx512vbmi, 15, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx512vbmi, 16, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx512vbmi, 17, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx512vbmi, 18, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx512vbmi, 19, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx512vbmi, 20, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx512vbmi, 21, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx512vbmi, 22, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx512vbmi, 23, true)
        BITUNPACKING_SWITCH_CASE(mm_unpack_epi32_avx512vbmi, 24, true)
        default:
            throw std::invalid_argument("Unsupported bit width for AVX512VBMI unpacking");
    }
}

auto mm256_unpack_epi32_avx512vbmi(const uint8_t bit_width, const uint8_t* __restrict__ input) -> __m256i {
    switch (bit_width) {
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx512vbmi, 1, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx512vbmi, 2, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx512vbmi, 3, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx512vbmi, 4, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx512vbmi, 5, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx512vbmi, 6, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx512vbmi, 7, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx512vbmi, 8, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx512vbmi, 9, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx512vbmi, 10, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx512vbmi, 11, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx512vbmi, 12, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx512vbmi, 13, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx512vbmi, 14, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx512vbmi, 15, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx512vbmi, 16, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx512vbmi, 17, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx512vbmi, 18, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx512vbmi, 19, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx512vbmi, 20, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx512vbmi, 21, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx512vbmi, 22, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx512vbmi, 23, true)
        BITUNPACKING_SWITCH_CASE(mm256_unpack_epi32_avx512vbmi, 24, true)
        default:
            throw std::invalid_argument("Unsupported bit width for AVX512VBMI unpacking");
    }
}

auto mm512_unpack_aligned_epi32_avx512vbmi(const uint8_t bit_width, const uint8_t* __restrict__ input) -> __m512i {
    switch (bit_width) {
        // BITUNPACKING_SWITCH_CASE(mm512_unpack_aligned_epi32_avx512vbmi, 8, true)
        // BITUNPACKING_SWITCH_CASE(mm512_unpack_aligned_epi32_avx512vbmi, 16, true)
        default:
            throw std::invalid_argument("Unsupported bit width for aligned AVX512VBMI unpacking");
    }
}

auto mm512_unpack_epi32_avx512vbmi(const uint8_t bit_width, const uint8_t* __restrict__ input) -> __m512i {
    switch (bit_width) {
        BITUNPACKING_SWITCH_CASE(mm512_unpack_epi32_avx512vbmi, 1, true)
        BITUNPACKING_SWITCH_CASE(mm512_unpack_epi32_avx512vbmi, 2, true)
        BITUNPACKING_SWITCH_CASE(mm512_unpack_epi32_avx512vbmi, 3, true)
        BITUNPACKING_SWITCH_CASE(mm512_unpack_epi32_avx512vbmi, 4, true)
        BITUNPACKING_SWITCH_CASE(mm512_unpack_epi32_avx512vbmi, 5, true)
        BITUNPACKING_SWITCH_CASE(mm512_unpack_epi32_avx512vbmi, 6, true)
        BITUNPACKING_SWITCH_CASE(mm512_unpack_epi32_avx512vbmi, 7, true)
        BITUNPACKING_SWITCH_CASE(mm512_unpack_epi32_avx512vbmi, 8, true)
        BITUNPACKING_SWITCH_CASE(mm512_unpack_epi32_avx512vbmi, 9, true)
        BITUNPACKING_SWITCH_CASE(mm512_unpack_epi32_avx512vbmi, 10, true)
        BITUNPACKING_SWITCH_CASE(mm512_unpack_epi32_avx512vbmi, 11, true)
        BITUNPACKING_SWITCH_CASE(mm512_unpack_epi32_avx512vbmi, 12, true)
        BITUNPACKING_SWITCH_CASE(mm512_unpack_epi32_avx512vbmi, 13, true)
        BITUNPACKING_SWITCH_CASE(mm512_unpack_epi32_avx512vbmi, 14, true)
        BITUNPACKING_SWITCH_CASE(mm512_unpack_epi32_avx512vbmi, 15, true)
        BITUNPACKING_SWITCH_CASE(mm512_unpack_epi32_avx512vbmi, 16, true)
        BITUNPACKING_SWITCH_CASE(mm512_unpack_epi32_avx512vbmi, 17, true)
        BITUNPACKING_SWITCH_CASE(mm512_unpack_epi32_avx512vbmi, 18, true)
        BITUNPACKING_SWITCH_CASE(mm512_unpack_epi32_avx512vbmi, 19, true)
        BITUNPACKING_SWITCH_CASE(mm512_unpack_epi32_avx512vbmi, 20, true)
        BITUNPACKING_SWITCH_CASE(mm512_unpack_epi32_avx512vbmi, 21, true)
        BITUNPACKING_SWITCH_CASE(mm512_unpack_epi32_avx512vbmi, 22, true)
        BITUNPACKING_SWITCH_CASE(mm512_unpack_epi32_avx512vbmi, 23, true)
        BITUNPACKING_SWITCH_CASE(mm512_unpack_epi32_avx512vbmi, 24, true)
        default:
            throw std::invalid_argument("Unsupported bit width for AVX512VBMI unpacking");
    }
}
}  // namespace pernix::bitpacking
#endif  // PERNIX_AVX512_VBMI_ENABLED
