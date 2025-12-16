#ifndef LIBCOMPRESSION_PACK_AVX2_BMI2_H
#define LIBCOMPRESSION_PACK_AVX2_BMI2_H

#include <immintrin.h>
#include <libcompression/helper.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>

#ifdef LIBCOMPRESSION_AVX2_ENABLED
namespace libcompression::bitpacking {
namespace internal {
template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 32)
static constexpr std::tuple<uint16_t, uint64_t, uint16_t, uint16_t> pack_avx2_bmi2_constants() {
    uint32_t mask = BIT_WIDTH == 32 ? std::numeric_limits<uint32_t>::max() : (1ULL << BIT_WIDTH) - 1U;
    uint64_t pext_mask;
    uint16_t shift1 = BIT_WIDTH * 4;
    uint16_t shift2 = 64 - shift1;

    if constexpr (BIT_WIDTH > 0 && BIT_WIDTH <= 8) {
        pext_mask = 0x0101010101010101ULL * mask;
    } else if constexpr (BIT_WIDTH > 8 && BIT_WIDTH <= 16) {
        pext_mask = 0x0001000100010001ULL * mask;
    } else {
        pext_mask = 0x0000000100000001ULL * mask;
    }

    return {
        mask,
        pext_mask,
        shift1,
        shift2,
    };
}
}  // namespace internal

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 32)
static inline auto mm_pack_epi32_bmi2(const __m128i& input) -> __m128i {
    const auto [mask, pext_mask, shift1, shift2] = internal::pack_avx2_bmi2_constants<BIT_WIDTH>();

    if constexpr (BIT_WIDTH > 0 && BIT_WIDTH <= 16) {
        const __m128i packed = _mm_packs_epi32(input, _mm_setzero_si128());
        const uint64_t value = _pext_u64(_mm_extract_epi64(packed, 0), pext_mask);

        const __m128i result = _mm_set_epi64x(0, value);
        return result;
    } else {
        alignas(16) uint64_t values[2];
        values[0] = _pext_u64(_mm_extract_epi64(input, 0), pext_mask);

        const uint64_t temp_combined = _pext_u64(_mm_extract_epi64(input, 1), pext_mask);
        values[1]                    = temp_combined >> shift2;
        values[0] |= (temp_combined << shift1);

        const __m128i result = _mm_set_epi64x(static_cast<int64_t>(values[1]), static_cast<int64_t>(values[0]));
        return result;
    }
}

template <uint8_t BIT_WIDTH>
    requires(BIT_WIDTH > 0 && BIT_WIDTH <= 16)
static inline auto mm256_pack_epi32_bmi2(const __m256i& input) -> __m256i {
    const auto [mask, pext_mask, shift1, shift2] = internal::pack_avx2_bmi2_constants<BIT_WIDTH>();

    const __m256i packed16 = _mm256_packs_epi32(input, _mm256_setzero_si256());

    if constexpr (BIT_WIDTH > 0 && BIT_WIDTH <= 8) {
        const __m256i permuted = _mm256_permute4x64_epi64(packed16, _MM_SHUFFLE(3, 1, 2, 0));
        const __m256i packed8  = _mm256_packs_epi16(permuted, _mm256_setzero_si256());
        const uint64_t value   = _pext_u64(_mm256_extract_epi64(packed8, 0), pext_mask);

        const __m256i result = _mm256_setr_epi64x(value, 0, 0, 0);
        return result;
    } else {
        alignas(16) int64_t values[2] = {};
        values[0]                     = _pext_u64(_mm256_extract_epi64(packed16, 0), pext_mask);

        const uint64_t temp_combined = _pext_u64(_mm256_extract_epi64(packed16, 2), pext_mask);
        values[1]                    = temp_combined >> shift2;
        if (shift1 < 64) {
            values[0] |= (temp_combined << shift1);
        }

        const __m256i result = _mm256_setr_epi64x(values[0], values[1], 0, 0);
        return result;
    }
}
}  // namespace libcompression::bitpacking
#endif  // LIBCOMPRESSION_AVX2_ENABLED
#endif  // LIBCOMPRESSION_PACK_AVX2_BMI2_H