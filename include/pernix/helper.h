#ifndef PERNIX_HELPER_H
#define PERNIX_HELPER_H

#include <immintrin.h>

#include <cstdint>

#define PERNIX_MACHINE_ID_GENERIC 0
#define PERNIX_MACHINE_ID_V2 1
#define PERNIX_MACHINE_ID_V3 2
#define PERNIX_MACHINE_ID_V4 3
#define PERNIX_MACHINE_ID_V4_VBMI 4

#if (__SSE3__ && __SSE4_1__ && __SSE4_2__)
#if (__AVX__ && __AVX2__ && __FMA__ && __BMI__ && __BMI2__)
#if (__AVX512BW__ && __AVX512CD__ && __AVX512DQ__ && __AVX512F__ && __AVX512VL__)
#if (__AVX512VBMI__)
#define PERNIX_MACHINE_ID PERNIX_MACHINE_ID_V4_VBMI
#else
#define PERNIX_MACHINE_ID PERNIX_MACHINE_ID_V4
#endif

#else
#define PERNIX_MACHINE_ID PERNIX_MACHINE_ID_V3
#endif

#else
#define PERNIX_MACHINE_ID PERNIX_MACHINE_ID_V2
#endif

#else
#define PERNIX_MACHINE_ID PERNIX_MACHINE_ID_GENERIC
#endif

#if (PERNIX_MACHINE_ID >= PERNIX_MACHINE_ID_V2)
#define PERNIX_SSE_ENABLED
#endif
#if (PERNIX_MACHINE_ID >= PERNIX_MACHINE_ID_V3)
#define PERNIX_AVX2_ENABLED
#define PERNIX_BMI2_ENABLED
#endif
#if (PERNIX_MACHINE_ID >= PERNIX_MACHINE_ID_V4)
#define PERNIX_AVX512_ENABLED
#endif
#if (PERNIX_MACHINE_ID >= PERNIX_MACHINE_ID_V4_VBMI)
#define PERNIX_AVX512_VBMI_ENABLED
#endif

#ifdef PERNIX_DISABLE_BMI2
#undef PERNIX_BMI2_ENABLED
#endif
#ifdef PERNIX_DISABLE_AVX2
#undef PERNIX_AVX2_ENABLED
#endif
#ifdef PERNIX_DISABLE_AVX512
#undef PERNIX_AVX512_ENABLED
#endif
#ifdef PERNIX_DISABLE_AVX512_VBMI
#undef PERNIX_AVX512_VBMI_ENABLED
#endif

namespace pernix::internal {
#ifdef PERNIX_SSE_ENABLED
static inline __m128i mm_convert_vmask_epi32(__mmask8 mask8) {
    mask8 = static_cast<__mmask8>((mask8 << 4) >> 4);  // Ensure we only consider the lower 4 bits
    return _mm_setr_epi32((mask8 & 0x1) ? -1 : 0, (mask8 & 0x2) ? -1 : 0, (mask8 & 0x4) ? -1 : 0, (mask8 & 0x8) ? -1 : 0);
}
#endif  // PERNIX_SSE_ENABLED
#ifdef PERNIX_AVX2_ENABLED
static inline __m256i mm256_convert_vmask_epi32(const __mmask8 mask8) {
    return _mm256_setr_epi32((mask8 & 0x1) ? -1 : 0, (mask8 & 0x2) ? -1 : 0, (mask8 & 0x4) ? -1 : 0, (mask8 & 0x8) ? -1 : 0,
                             (mask8 & 0x10) ? -1 : 0, (mask8 & 0x20) ? -1 : 0, (mask8 & 0x40) ? -1 : 0, (mask8 & 0x80) ? -1 : 0);
}
#endif  // PERNIX_AVX2_ENABLED
}  // namespace pernix::internal

namespace pernix {
enum class AvailableImplementations : uint8_t {
    FALLBACK,
#ifdef PERNIX_AVX2_ENABLED
    AVX2,
    AVX2_BMI2,
#endif
#ifdef PERNIX_AVX512_ENABLED
    AVX512,
#endif
#ifdef PERNIX_AVX512_VBMI_ENABLED
    AVX512_VBMI
#endif
};

#define BITPACKING_SWITCH_CASE(METHOD, BIT_WIDTH) \
    case BIT_WIDTH:                               \
        return METHOD<BIT_WIDTH>(input);

#define BITUNPACKING_SWITCH_CASE(METHOD, BIT_WIDTH, SIGNED) \
    case BIT_WIDTH:                                         \
        return METHOD<BIT_WIDTH, SIGNED>(input);

#define DECOMPRESSION_BLOCK_SWITCH_CASE(METHOD, BIT_WIDTH, SIGNED) \
    case BIT_WIDTH:                                                \
        return METHOD<BIT_WIDTH, SIGNED>(input, scale, output);

#define DECOMPRESSION_BLOCKS_SWITCH_CASE(METHOD, BIT_WIDTH, SIGNED) \
    case BIT_WIDTH:                                                 \
        return METHOD<BIT_WIDTH, SIGNED>(input, scale, output, blocks);

#define COMPRESSION_BLOCK_SWITCH_CASE(METHOD, BIT_WIDTH) \
    case BIT_WIDTH:                                      \
        return METHOD<BIT_WIDTH>(input, scale, output);

#define COMPRESSION_BLOCKS_SWITCH_CASE(METHOD, BIT_WIDTH) \
    case BIT_WIDTH:                                       \
        return METHOD<BIT_WIDTH>(input, scale, output, blocks);
}  // namespace pernix

#endif  // PERNIX_HELPER_H
