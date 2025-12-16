#ifndef LIBCOMPRESSION_HELPER_H
#define LIBCOMPRESSION_HELPER_H

#include <immintrin.h>
#include <cstdint>

#define LIBCOMPRESSION_MACHINE_ID_GENERIC 0
#define LIBCOMPRESSION_MACHINE_ID_V2 1
#define LIBCOMPRESSION_MACHINE_ID_V3 2
#define LIBCOMPRESSION_MACHINE_ID_V4 3
#define LIBCOMPRESSION_MACHINE_ID_V4_VBMI 4

#if (__SSE3__ && __SSE4_1__ && __SSE4_2__)
#if (__AVX__ && __AVX2__ && __FMA__ && __BMI__ && __BMI2__)
#if (__AVX512BW__ && __AVX512CD__ && __AVX512DQ__ && __AVX512F__ && __AVX512VL__)
#if (__AVX512VBMI__)
#define LIBCOMPRESSION_MACHINE_ID LIBCOMPRESSION_MACHINE_ID_V4_VBMI
#else
#define LIBCOMPRESSION_MACHINE_ID LIBCOMPRESSION_MACHINE_ID_V4
#endif

#else
#define LIBCOMPRESSION_MACHINE_ID LIBCOMPRESSION_MACHINE_ID_V3
#endif

#else
#define LIBCOMPRESSION_MACHINE_ID LIBCOMPRESSION_MACHINE_ID_V2
#endif

#else
#define LIBCOMPRESSION_MACHINE_ID LIBCOMPRESSION_MACHINE_ID_GENERIC
#endif

#if (LIBCOMPRESSION_MACHINE_ID >= LIBCOMPRESSION_MACHINE_ID_V2)
#define LIBCOMPRESSION_SSE_ENABLED
#endif
#if (LIBCOMPRESSION_MACHINE_ID >= LIBCOMPRESSION_MACHINE_ID_V3)
#define LIBCOMPRESSION_AVX2_ENABLED
#define LIBCOMPRESSION_BMI2_ENABLED
#endif
#if (LIBCOMPRESSION_MACHINE_ID >= LIBCOMPRESSION_MACHINE_ID_V4)
#define LIBCOMPRESSION_AVX512_ENABLED
#endif
#if (LIBCOMPRESSION_MACHINE_ID >= LIBCOMPRESSION_MACHINE_ID_V4_VBMI)
#define LIBCOMPRESSION_AVX512_VBMI_ENABLED
#endif

#ifdef LIBCOMPRESSION_DISABLE_BMI2
#undef LIBCOMPRESSION_BMI2_ENABLED
#endif
#ifdef LIBCOMPRESSION_DISABLE_AVX2
#undef LIBCOMPRESSION_AVX2_ENABLED
#endif
#ifdef LIBCOMPRESSION_DISABLE_AVX512
#undef LIBCOMPRESSION_AVX512_ENABLED
#endif
#ifdef LIBCOMPRESSION_DISABLE_AVX512_VBMI
#undef LIBCOMPRESSION_AVX512_VBMI_ENABLED
#endif

namespace libcompression::internal {
#ifdef LIBCOMPRESSION_SSE_ENABLED
static inline __m128i mm_convert_vmask_epi32(__mmask8 mask8) {
    mask8 = static_cast<__mmask8>((mask8 << 4) >> 4); // Ensure we only consider the lower 4 bits
    return _mm_setr_epi32((mask8 & 0x1) ? -1 : 0,
                          (mask8 & 0x2) ? -1 : 0,
                          (mask8 & 0x4) ? -1 : 0,
                          (mask8 & 0x8) ? -1 : 0);
}
#endif  // LIBCOMPRESSION_SSE_ENABLED
#ifdef LIBCOMPRESSION_AVX2_ENABLED
static inline __m256i mm256_convert_vmask_epi32(const __mmask8 mask8) {
    return _mm256_setr_epi32((mask8 & 0x1) ? -1 : 0,
                             (mask8 & 0x2) ? -1 : 0,
                             (mask8 & 0x4) ? -1 : 0,
                             (mask8 & 0x8) ? -1 : 0,
                             (mask8 & 0x10) ? -1 : 0,
                             (mask8 & 0x20) ? -1 : 0,
                             (mask8 & 0x40) ? -1 : 0,
                             (mask8 & 0x80) ? -1 : 0);
}
#endif  // LIBCOMPRESSION_AVX2_ENABLED
} // namespace libcompression::internal

namespace libcompression {
enum class AvailableImplementations : uint8_t {
    FALLBACK,
#ifdef LIBCOMPRESSION_AVX2_ENABLED
    AVX2,
    AVX2_BMI2,
#endif
#ifdef LIBCOMPRESSION_AVX512_ENABLED
    AVX512,
#endif
#ifdef LIBCOMPRESSION_AVX512_VBMI_ENABLED
    AVX512_VBMI
#endif
};

#define BITPACKING_SWITCH_CASE(METHOD, BIT_WIDTH)       \
    case BIT_WIDTH:                                             \
        return METHOD<BIT_WIDTH>(input);

#define BITUNPACKING_SWITCH_CASE(METHOD, BIT_WIDTH, SIGNED)       \
    case BIT_WIDTH:                                             \
        return METHOD<BIT_WIDTH, SIGNED>(input);

#define COMPRESSION_BLOCK_SWITCH_CASE(METHOD, BIT_WIDTH, SIGNED)       \
    case BIT_WIDTH:                                                     \
        return METHOD<BIT_WIDTH, SIGNED>(input, scale, output);

#define COMPRESSION_BLOCKS_SWITCH_CASE(METHOD, BIT_WIDTH, SIGNED)       \
    case BIT_WIDTH:                                                     \
        return METHOD<BIT_WIDTH, SIGNED>(input, scale, output, blocks);
} // namespace libcompression

#endif  // LIBCOMPRESSION_HELPER_H
