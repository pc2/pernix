#ifndef LIBCOMPRESSION_PACKING_AVX2_H
#define LIBCOMPRESSION_PACKING_AVX2_H

#include <libcompression/helper.h>

#ifdef LIBCOMPRESSION_AVX2_ENABLED

#include <immintrin.h>
#include <libcompression/bitpacking/packing_tables.h>

#include <cstdint>

namespace libcompression::bitpacking {
    namespace internal {
        __always_inline static __m128i _mm_sllv_epi16(const __m128i a, const __m128i count) {
            const __m128i mask = _mm_set1_epi32(0xffff0000);
            const __m128i low_half = _mm_sllv_epi32(a, _mm_andnot_si128(mask, count));
            const __m128i high_half = _mm_sllv_epi32(_mm_and_si128(mask, a), _mm_srli_epi32(count, 16));
            return _mm_blend_epi16(low_half, high_half, 0xaa);
        }

        __always_inline static __m128i _mm_srlv_epi16(const __m128i a, const __m128i count) {
            const __m128i mask = _mm_set1_epi32(0x0000ffff);
            const __m128i low_half = _mm_srlv_epi32(_mm_and_si128(mask, a), _mm_and_si128(mask, count));
            const __m128i high_half = _mm_srlv_epi32(a, _mm_srli_epi32(count, 16));
            return _mm_blend_epi16(low_half, high_half, 0xaa);
        }

        __always_inline static __m256i _mm256_sllv_epi16(const __m256i a, const __m256i count) {
            const __m256i mask = _mm256_set1_epi32(0xffff0000);
            const __m256i low_half = _mm256_sllv_epi32(a, _mm256_andnot_si256(mask, count));
            const __m256i high_half = _mm256_sllv_epi32(_mm256_and_si256(mask, a), _mm256_srli_epi32(count, 16));
            return _mm256_blend_epi16(low_half, high_half, 0xaa);
        }

        __always_inline static __m256i _mm256_srlv_epi16(const __m256i a, const __m256i count) {
            const __m256i mask = _mm256_set1_epi32(0x0000ffff);
            const __m256i low_half = _mm256_srlv_epi32(_mm256_and_si256(mask, a), _mm256_and_si256(mask, count));
            const __m256i high_half = _mm256_srlv_epi32(a, _mm256_srli_epi32(count, 16));
            return _mm256_blend_epi16(low_half, high_half, 0xaa);
        }

        __always_inline static __m128i mm_blend_epi8(const __m128i X, const __m128i Y, const int8_t M) {
            return _mm_blendv_epi8(X, Y, _mm_set1_epi8(M));
        }

        __always_inline static __m256i mm256_blend_epi8(const __m256i X, const __m256i Y, const int8_t M) {
            return _mm256_blendv_epi8(X, Y, _mm256_set1_epi8(M));
        }

        __always_inline static __m128i _mm_sllv_epi8(const __m128i a, const __m128i count) {
            const __m128i mask = _mm_set1_epi16(0xff00);
            const __m128i low_half = _mm_sllv_epi16(a, _mm_andnot_si128(mask, count));
            const __m128i high_half = _mm_sllv_epi16(_mm_and_si128(mask, a), _mm_srli_epi16(count, 8));
            return mm_blend_epi8(low_half, high_half, 0xaa);
        }

        __always_inline static __m128i _mm_srlv_epi8(const __m128i a, const __m128i count) {
            const __m128i mask = _mm_set1_epi16(0x00ff);
            const __m128i low_half = _mm_srlv_epi16(_mm_and_si128(mask, a), _mm_and_si128(mask, count));
            const __m128i high_half = _mm_srlv_epi16(a, _mm_srli_epi16(count, 8));
            return mm_blend_epi8(low_half, high_half, 0xaa);
        }

        __always_inline static __m256i _mm256_sllv_epi8(const __m256i a, const __m256i count) {
            const __m256i mask = _mm256_set1_epi16(0xff00);
            const __m256i low_half = _mm256_sllv_epi16(a, _mm256_andnot_si256(mask, count));
            const __m256i high_half = _mm256_sllv_epi16(_mm256_and_si256(mask, a), _mm256_srli_epi16(count, 8));
            return mm256_blend_epi8(low_half, high_half, 0xaa);
        }

        __always_inline static __m256i _mm256_srlv_epi8(const __m256i a, const __m256i count) {
            const __m256i mask = _mm256_set1_epi16(0x00ff);
            const __m256i low_half = _mm256_srlv_epi16(_mm256_and_si256(mask, a), _mm256_and_si256(mask, count));
            const __m256i high_half = _mm256_srlv_epi16(a, _mm256_srli_epi16(count, 8));
            return mm256_blend_epi8(low_half, high_half, 0xaa);
        }

        template<uint8_t BIT_WIDTH>
            requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
        __always_inline auto mm_pack_epi32_avx2_9to16(__m128i &input) -> __m128i {
            using tables = pack_tables_avx2_16<BIT_WIDTH, __m128i>;
            constexpr uint16_t bitmask = (1 << BIT_WIDTH) - 1;
            const __m128i masked = _mm_and_si128(input, _mm_set1_epi16(bitmask));
            __m128i combined;

            if constexpr (BIT_WIDTH == 12 || BIT_WIDTH == 14 || BIT_WIDTH == 15) {
                const __m128i shuffled1 = _mm_shuffle_epi8(masked, tables::get_permute1());
                const __m128i shuffled2 = _mm_shuffle_epi8(masked, tables::get_permute2());

                const __m128i shifted1 = _mm_sllv_epi16(shuffled1, tables::get_shift1());
                const __m128i shifted2 = _mm_srlv_epi16(shuffled2, tables::get_shift2());

                combined = _mm_or_si128(shifted1, shifted2);
            } else {
                const __m128i shuffled1 = _mm_shuffle_epi8(masked, tables::get_permute1());
                const __m128i shuffled2 = _mm_shuffle_epi8(masked, tables::get_permute2());
                const __m128i shuffled3 = _mm_shuffle_epi8(masked, tables::get_permute3());

                const __m128i shifted1 = _mm_sllv_epi16(shuffled1, tables::get_shift1());
                const __m128i shifted2 = _mm_sllv_epi16(shuffled2, tables::get_shift2());
                const __m128i shifted3 = _mm_srlv_epi16(shuffled3, tables::get_shift3());

                combined = _mm_or_si128(_mm_or_si128(shifted1, shifted2), shifted3);
            }
            return combined;
        }

        template<uint8_t BIT_WIDTH>
            requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16)
        __always_inline auto mm256_pack_epi32_avx2_9to16(const __m256i &input) -> __m256i {
            using tables = pack_tables_avx2_16<BIT_WIDTH, __m128i>;
            constexpr uint16_t bitmask = (1 << BIT_WIDTH) - 1;
            const __m128i packed = _mm_packs_epi32(_mm256_castsi256_si128(input), _mm256_extracti128_si256(input, 1));
            const __m128i masked = _mm_and_si128(packed, _mm_set1_epi16(bitmask));
            __m128i combined;

            if constexpr (BIT_WIDTH == 12 || BIT_WIDTH == 14 || BIT_WIDTH == 15 || BIT_WIDTH == 16) {
                const __m128i shuffled1 = _mm_shuffle_epi8(masked, tables::get_permute1());
                const __m128i shuffled2 = _mm_shuffle_epi8(masked, tables::get_permute2());

                const __m128i shifted1 = _mm_sllv_epi16(shuffled1, tables::get_shift1());
                const __m128i shifted2 = _mm_srlv_epi16(shuffled2, tables::get_shift2());

                combined = _mm_or_si128(shifted1, shifted2);
            } else {
                const __m128i shuffled1 = _mm_shuffle_epi8(masked, tables::get_permute1());
                const __m128i shuffled2 = _mm_shuffle_epi8(masked, tables::get_permute2());
                const __m128i shuffled3 = _mm_shuffle_epi8(masked, tables::get_permute3());

                const __m128i shifted1 = _mm_sllv_epi16(shuffled1, tables::get_shift1());
                const __m128i shifted2 = _mm_sllv_epi16(shuffled2, tables::get_shift2());
                const __m128i shifted3 = _mm_srlv_epi16(shuffled3, tables::get_shift3());

                combined = _mm_or_si128(_mm_or_si128(shifted1, shifted2), shifted3);
            }
            return _mm256_castsi128_si256(combined);
        }

        template<uint8_t BIT_WIDTH>
            requires(BIT_WIDTH >= 4 && BIT_WIDTH <= 8)
        __always_inline auto mm256_pack_epi32_avx2_4to8(const __m256i &input) -> __m256i {
            // using tables              = pack_tables_avx2_16<BIT_WIDTH, __m128i>;
            constexpr uint8_t bitmask = (1 << BIT_WIDTH) - 1;

            const __m128i packed = _mm_packs_epi32(_mm256_castsi256_si128(input), _mm256_extracti128_si256(input, 1));
            const __m128i masked = _mm_and_si128(packed, _mm_set1_epi16(bitmask));
            const __m128i half1 = _mm_and_si128(masked, _mm_set_epi64x(0x5555555555555555, 0x5555555555555555));
            __m128i half2 = _mm_and_si128(masked, _mm_set_epi64x(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA));

            half2 = _mm_srli_epi16(half2, 2);
            const __m128i combined = _mm_or_si128(half1, half2);
            const __m256i result = _mm256_castsi128_si256(_mm_cvtepi16_epi32(combined));

            return mm256_pack_epi32_avx2_9to16<12>(result);
        }

        template<uint8_t BIT_WIDTH>
            requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24)
        __always_inline auto mm256_pack_epi32_avx2_17to24(const __m256i &input) -> __m256i {
            using tables = pack_tables_avx2_32<BIT_WIDTH, __m256i>;
            constexpr uint32_t bitmask = (1 << BIT_WIDTH) - 1;
            const __m256i masked = _mm256_and_si256(input, _mm256_set1_epi32(bitmask));
            __m256i combined;

            if constexpr (BIT_WIDTH == 24) {
                const __m256i shuffled1 = _mm256_permutevar8x32_epi32(masked, tables::get_permute1());
                const __m256i shuffled2 = _mm256_permutevar8x32_epi32(masked, tables::get_permute2());

                const __m256i shifted1 = _mm256_sllv_epi32(shuffled1, tables::get_shift1());
                const __m256i shifted2 = _mm256_srlv_epi32(shuffled2, tables::get_shift2());

                combined = _mm256_or_si256(shifted1, shifted2);
            } else {
                const __m256i shuffled1 = _mm256_permutevar8x32_epi32(masked, tables::get_permute1());
                const __m256i shuffled2 = _mm256_permutevar8x32_epi32(masked, tables::get_permute2());
                const __m256i shuffled3 = _mm256_permutevar8x32_epi32(masked, tables::get_permute3());

                const __m256i shifted1 = _mm256_sllv_epi32(shuffled1, tables::get_shift1());
                const __m256i shifted2 = _mm256_sllv_epi32(shuffled2, tables::get_shift2());
                const __m256i shifted3 = _mm256_srlv_epi32(shuffled3, tables::get_shift3());

                combined = _mm256_or_si256(_mm256_or_si256(shifted1, shifted2), shifted3);
            }

            return combined;
        }
    } // namespace internal

    template<uint8_t BIT_WIDTH>
        requires(BIT_WIDTH == 8 || BIT_WIDTH == 16)
    auto mm_pack_aligned_epi32_avx2(__m128i &input) -> __m128i {
        if constexpr (BIT_WIDTH == 8) {
            return _mm_packus_epi16(_mm_packs_epi32(input, _mm_setzero_si128()), _mm_setzero_si128());
        } else {
            return _mm_packs_epi32(input, _mm_setzero_si128());
        }
    }

    template<uint8_t BIT_WIDTH>
        requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
    auto mm_pack_epi32_avx2(__m128i &input) -> __m128i {
        if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 3) {
            // TODO: implementation for 1-3 bits
            return _mm_setzero_si128();
        } else if constexpr (BIT_WIDTH >= 4 && BIT_WIDTH <= 8) {
        } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
            return internal::mm_pack_epi32_avx2_9to16<BIT_WIDTH>(input);
        } else {
            return _mm_setzero_si128();
        }
    }

    template<uint8_t BIT_WIDTH>
        requires(BIT_WIDTH == 8 || BIT_WIDTH == 16)
    auto mm256_pack_aligned_epi32_avx2(const __m256i &input) -> __m256i {
        if constexpr (BIT_WIDTH == 8) {
            const __m128i packed16 = _mm_packs_epi32(_mm256_castsi256_si128(input), _mm256_extracti128_si256(input, 1));
            const __m128i packed8 = _mm_unpacklo_epi32(packed16, _mm_setzero_si128());
            return _mm256_castsi128_si256(packed8);
        } else {
            return _mm256_castsi128_si256(_mm_packs_epi32(_mm256_castsi256_si128(input),
                                                          _mm256_extracti128_si256(input, 1)));
        }
    }

    template<uint8_t BIT_WIDTH>
        requires(BIT_WIDTH > 0 && BIT_WIDTH <= 24)
    auto mm256_pack_epi32_avx2(const __m256i &input) -> __m256i {
        if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 3) {
            // TODO: implementation for 1-3 bits
            return _mm256_setzero_si256();
        } else if constexpr (BIT_WIDTH >= 4 && BIT_WIDTH < 8) {
            return internal::mm256_pack_epi32_avx2_4to8<BIT_WIDTH>(input);
        } else if constexpr (BIT_WIDTH == 8) {
            return mm256_pack_aligned_epi32_avx2<BIT_WIDTH>(input);
        } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH < 16) {
            return internal::mm256_pack_epi32_avx2_9to16<BIT_WIDTH>(input);
        } else if constexpr (BIT_WIDTH == 16) {
            return mm256_pack_aligned_epi32_avx2<BIT_WIDTH>(input);
        } else {
            return internal::mm256_pack_epi32_avx2_17to24<BIT_WIDTH>(input);
        }
    }

    auto mm_pack_aligned_epi32_avx2(uint8_t bit_width, __m128i &input) -> __m128i;

    auto mm_pack_epi32_avx2(uint8_t bit_width, __m128i &input) -> __m128i;

    auto mm256_pack_aligned_epi32_avx2(uint8_t bit_width, __m256i &input) -> __m256i;

    auto mm256_pack_epi32_avx2(uint8_t bit_width, __m256i &input) -> __m256i;
} // namespace libcompression::bitpacking
#endif  // LIBCOMPRESSION_AVX2_ENABLED
#endif  // LIBCOMPRESSION_PACKING_AVX2_H
