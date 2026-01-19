#ifndef LIBCOMPRESSION_QUANTIZATION_H
#define LIBCOMPRESSION_QUANTIZATION_H

#include <immintrin.h>
#include <libcompression/helper.h>

#include <cmath>

namespace libcompression::quantization {
    // fallback implementation
    __always_inline int32_t quantize_ps_epi32(const float input, const float scale) {
        return static_cast<int32_t>(std::nearbyintf(input * scale));
    }

#ifdef LIBCOMPRESSION_SSE_ENABLED
    __always_inline __m128i mm_quantize_ps_epi32(const __m128 &input, const __m128 &scale) {
        const __m128 scaled = _mm_mul_ps(input, scale);
        const __m128 rounded = _mm_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        return _mm_cvtps_epi32(rounded);
    }
#endif  // LIBCOMPRESSION_SSE_ENABLED

#ifdef LIBCOMPRESSION_AVX2_ENABLED

    __always_inline __m256i mm256_quantize_ps_epi32(const __m256 &input, const __m256 &scale) {
        const __m256 scaled = _mm256_mul_ps(input, scale);
        const __m256 rounded = _mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        return _mm256_cvtps_epi32(rounded);
    }
#endif  // LIBCOMPRESSION_AVX2_ENABLED

#ifdef LIBCOMPRESSION_AVX512_ENABLED
__always_inline __m128i mm_maskz_quantize_ps_epi32(const __mmask8 &mask, const __m128 &input, const __m128 &scale) {
        const __m128 scaled = _mm_maskz_mul_ps(mask, input, scale);
        const __m128 rounded = _mm_maskz_roundscale_ps(mask, scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        return _mm_maskz_cvtps_epi32(mask, rounded);
    }

__always_inline __m256i mm256_maskz_quantize_ps_epi32(const __mmask8 &mask, const __m256 &input, const __m256 &scale) {
        const __m256 scaled = _mm256_maskz_mul_ps(mask, input, scale);
        const __m256 rounded = _mm256_maskz_roundscale_ps(mask, scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        return _mm256_maskz_cvtps_epi32(mask, rounded);
    }

__always_inline __m512i mm512_quantize_ps_epi32(const __m512 &input, const __m512 &scale) {
        const __m512 scaled = _mm512_mul_ps(input, scale);
        return _mm512_cvt_roundps_epi32(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }

__always_inline __m512i mm512_maskz_quantize_ps_epi32(const __mmask16 &mask, const __m512 &input, const __m512 &scale) {
        const __m512 scaled = _mm512_maskz_mul_ps(mask, input, scale);
        return _mm512_maskz_cvt_roundps_epi32(mask, scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
#endif // LIBCOMPRESSION_AVX512_ENABLED
} // namespace libcompression::quantization

#endif  // LIBCOMPRESSION_QUANTIZATION_H
