#ifndef LIBCOMPRESSION_DEQUANTIZATION_H
#define LIBCOMPRESSION_DEQUANTIZATION_H

#include <immintrin.h>
#include <libcompression/helper.h>

namespace libcompression::quantization {
// fallback implementation
__always_inline float dequantize_epi32(const int32_t input, const float scale) {
    return static_cast<float>(input) * scale;
}

#ifdef LIBCOMPRESSION_SSE_ENABLED
__always_inline __m128 mm_dequantize_epi32(const __m128i& input, const __m128& scale) {
    const __m128 converted = _mm_cvtepi32_ps(input);
    return _mm_mul_ps(converted, scale);
}

#endif  // LIBCOMPRESSION_SSE_ENABLED

#ifdef LIBCOMPRESSION_AVX2_ENABLED
__always_inline __m256 mm256_dequantize_epi32(const __m256i& input, const __m256& scale) {
    const __m256 converted = _mm256_cvtepi32_ps(input);
    return _mm256_mul_ps(converted, scale);
}
#endif  // LIBCOMPRESSION_AVX2_ENABLED

#ifdef LIBCOMPRESSION_AVX512_ENABLED
__always_inline __m128 mm_maskz_dequantize_epi32(const __mmask8& mask, const __m128i& input, const __m128& scale) {
    const __m128 converted = _mm_maskz_cvtepi32_ps(mask, input);
    return _mm_maskz_mul_ps(mask, converted, scale);
}

__always_inline __m256 mm256_maskz_dequantize_epi32(const __mmask8& mask, const __m256i& input, const __m256& scale) {
    const __m256 converted = _mm256_maskz_cvtepi32_ps(mask, input);
    return _mm256_maskz_mul_ps(mask, converted, scale);
}

__always_inline __m512 mm512_dequantize_epi32(const __m512i& input, const __m512& scale) {
    const __m512 converted = _mm512_cvtepi32_ps(input);
    return _mm512_mul_ps(converted, scale);
}

__always_inline __m512 mm512_maskz_dequantize_epi32(const __mmask8& mask, const __m512i& input, const __m512& scale) {
    const __m512 converted = _mm512_maskz_cvtepi32_ps(mask, input);
    return _mm512_maskz_mul_ps(mask, converted, scale);
}
#endif  // LIBCOMPRESSION_AVX512_ENABLED
} // namespace libcompression::quantization

#endif  // LIBCOMPRESSION_DEQUANTIZATION_H
