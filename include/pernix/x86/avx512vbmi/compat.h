#ifndef PERNIX_AVX512_COMPAT_H
#define PERNIX_AVX512_COMPAT_H

#include <pernix/simd_compat.h>

#include <cmath>
#include <cstdint>
#include <cstring>

namespace pernix::internal {
static __always_inline __mmask8 element_mask8(const u32 e) {
    return static_cast<__mmask8>(e >= 8 ? 0xFFu : ((1u << e) - 1u));
}

static __always_inline __mmask16 element_mask16(const u32 e) {
    return static_cast<__mmask16>(e >= 16 ? 0xFFFFu : ((1u << e) - 1u));
}

static __always_inline __mmask32 element_mask32(const u32 e) {
    return e >= 32 ? 0xFFFFFFFFu : (1u << e) - 1u;
}

static __always_inline __mmask64 element_mask64(const u32 e) {
    return e >= 64 ? 0xFFFFFFFFFFFFFFFFull : (1ull << e) - 1ull;
}

static __always_inline __m512i mm512_loadu_elements_epi64(const u32 e, const void* mem_addr) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    __m512i a = _mm512_setzero_si512();
    std::memcpy(&a, mem_addr, e * sizeof(i64));
    return a;
#else
    return _mm512_maskz_loadu_epi64(element_mask8(e), mem_addr);
#endif
}

static __always_inline __m256i mm256_loadu_elements_epi64(const u32 e, const void* mem_addr) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    __m256i a = _mm256_setzero_si256();
    std::memcpy(&a, mem_addr, e * sizeof(i64));
    return a;
#else
    return _mm256_maskz_loadu_epi64(element_mask8(e), mem_addr);
#endif
}

static __always_inline __m128i mm_loadu_elements_epi64(const u32 e, const void* mem_addr) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    __m128i a = _mm_setzero_si128();
    std::memcpy(&a, mem_addr, e * sizeof(i64));
    return a;
#else
    return _mm_maskz_loadu_epi64(element_mask8(e), mem_addr);
#endif
}

static __always_inline __m512i mm512_loadu_elements_epi32(const u32 e, const void* mem_addr) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    __m512i a = _mm512_setzero_si512();
    std::memcpy(&a, mem_addr, e * sizeof(i32));
    return a;
#else
    return _mm512_maskz_loadu_epi32(element_mask16(e), mem_addr);
#endif
}

static __always_inline __m256i mm256_loadu_elements_epi32(const u32 e, const void* mem_addr) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    __m256i a = _mm256_setzero_si256();
    std::memcpy(&a, mem_addr, e * sizeof(i32));
    return a;
#else
    return _mm256_maskz_loadu_epi32(element_mask8(e), mem_addr);
#endif
}

static __always_inline __m128i mm_loadu_elements_epi32(const u32 e, const void* mem_addr) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    __m128i a = _mm_setzero_si128();
    std::memcpy(&a, mem_addr, e * sizeof(i32));
    return a;
#else
    return _mm_maskz_loadu_epi32(element_mask8(e), mem_addr);
#endif
}

static __always_inline __m512i mm512_loadu_elements_epi16(const u32 e, const void* mem_addr) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    __m512i a = _mm512_setzero_si512();
    std::memcpy(&a, mem_addr, e * sizeof(i16));
    return a;
#else
    return _mm512_maskz_loadu_epi16(element_mask32(e), mem_addr);
#endif
}

static __always_inline __m256i mm256_loadu_elements_epi16(const u32 e, const void* mem_addr) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    __m256i a = _mm256_setzero_si256();
    std::memcpy(&a, mem_addr, e * sizeof(i16));
    return a;
#else
    return _mm256_maskz_loadu_epi16(element_mask16(e), mem_addr);
#endif
}

static __always_inline __m128i mm_loadu_elements_epi16(const u32 e, const void* mem_addr) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    __m128i a = _mm_setzero_si128();
    std::memcpy(&a, mem_addr, e * sizeof(i16));
    return a;
#else
    return _mm_maskz_loadu_epi16(element_mask8(e), mem_addr);
#endif
}

static __always_inline __m512i mm512_loadu_elements_epi8(const u32 e, const void* mem_addr) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    __m512i a = _mm512_setzero_si512();
    std::memcpy(&a, mem_addr, e * sizeof(i8));
    return a;
#else
    return _mm512_maskz_loadu_epi8(element_mask64(e), mem_addr);
#endif
}

static __always_inline __m256i mm256_loadu_elements_epi8(const u32 e, const void* mem_addr) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    __m256i a = _mm256_setzero_si256();
    std::memcpy(&a, mem_addr, e * sizeof(i8));
    return a;
#else
    return _mm256_maskz_loadu_epi8(element_mask32(e), mem_addr);
#endif
}

static __always_inline __m128i mm_loadu_elements_epi8(const u32 e, const void* mem_addr) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    __m128i a = _mm_setzero_si128();
    std::memcpy(&a, mem_addr, e * sizeof(i8));
    return a;
#else
    return _mm_maskz_loadu_epi8(element_mask16(e), mem_addr);
#endif
}

static __always_inline void mm512_storeu_elements_epi64(void* mem_addr, const u32 e, const __m512i a) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    alignas(64) u8 bytes[64];
    _mm512_storeu_si512(bytes, a);
    std::memcpy(mem_addr, bytes, e * sizeof(i64));
#else
    _mm512_mask_storeu_epi64(mem_addr, element_mask8(e), a);
#endif
}

static __always_inline void mm256_storeu_elements_epi64(void* mem_addr, const u32 e, const __m256i a) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    alignas(32) u8 bytes[32];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(bytes), a);
    std::memcpy(mem_addr, bytes, e * sizeof(i64));
#else
    _mm256_mask_storeu_epi64(mem_addr, element_mask8(e), a);
#endif
}

static __always_inline void mm_storeu_elements_epi64(void* mem_addr, const u32 e, const __m128i a) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    alignas(16) u8 bytes[16];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(bytes), a);
    std::memcpy(mem_addr, bytes, e * sizeof(i64));
#else
    _mm_mask_storeu_epi64(mem_addr, element_mask8(e), a);
#endif
}

static __always_inline void mm512_storeu_elements_epi32(void* mem_addr, const u32 e, const __m512i a) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    alignas(64) u8 bytes[64];
    _mm512_storeu_si512(bytes, a);
    std::memcpy(mem_addr, bytes, e * sizeof(i32));
#else
    _mm512_mask_storeu_epi32(mem_addr, element_mask16(e), a);
#endif
}

static __always_inline void mm256_storeu_elements_epi32(void* mem_addr, const u32 e, const __m256i a) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    alignas(32) u8 bytes[32];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(bytes), a);
    std::memcpy(mem_addr, bytes, e * sizeof(i32));
#else
    _mm256_mask_storeu_epi32(mem_addr, element_mask8(e), a);
#endif
}

static __always_inline void mm_storeu_elements_epi32(void* mem_addr, const u32 e, const __m128i a) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    alignas(16) u8 bytes[16];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(bytes), a);
    std::memcpy(mem_addr, bytes, e * sizeof(i32));
#else
    _mm_mask_storeu_epi32(mem_addr, element_mask8(e), a);
#endif
}

static __always_inline void mm512_storeu_elements_epi16(void* mem_addr, const u32 e, const __m512i a) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    alignas(64) u8 bytes[64];
    _mm512_storeu_si512(bytes, a);
    std::memcpy(mem_addr, bytes, e * sizeof(i16));
#else
    _mm512_mask_storeu_epi16(mem_addr, element_mask32(e), a);
#endif
}

static __always_inline void mm256_storeu_elements_epi16(void* mem_addr, const u32 e, const __m256i a) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    alignas(32) u8 bytes[32];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(bytes), a);
    std::memcpy(mem_addr, bytes, e * sizeof(i16));
#else
    _mm256_mask_storeu_epi16(mem_addr, element_mask16(e), a);
#endif
}

static __always_inline void mm_storeu_elements_epi16(void* mem_addr, const u32 e, const __m128i a) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    alignas(16) u8 bytes[16];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(bytes), a);
    std::memcpy(mem_addr, bytes, e * sizeof(i16));
#else
    _mm_mask_storeu_epi16(mem_addr, element_mask8(e), a);
#endif
}

static __always_inline void mm512_storeu_elements_epi8(void* mem_addr, const u32 e, const __m512i a) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    alignas(64) u8 bytes[64];
    _mm512_storeu_si512(bytes, a);
    std::memcpy(mem_addr, bytes, e * sizeof(i8));
#else
    _mm512_mask_storeu_epi8(mem_addr, element_mask64(e), a);
#endif
}

static __always_inline void mm256_storeu_elements_epi8(void* mem_addr, const u32 e, const __m256i a) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    alignas(32) u8 bytes[32];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(bytes), a);
    std::memcpy(mem_addr, bytes, e * sizeof(i8));
#else
    _mm256_mask_storeu_epi8(mem_addr, element_mask32(e), a);
#endif
}

static __always_inline void mm_storeu_elements_epi8(void* mem_addr, const u32 e, const __m128i a) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    alignas(16) u8 bytes[16];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(bytes), a);
    std::memcpy(mem_addr, bytes, e * sizeof(i8));
#else
    _mm_mask_storeu_epi8(mem_addr, element_mask16(e), a);
#endif
}

static __always_inline __m512 mm512_loadu_elements_ps(const u32 e, const void* mem_addr) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    __m512 a = _mm512_setzero_ps();
    std::memcpy(&a, mem_addr, e * sizeof(f32));
    return a;
#else
    return _mm512_maskz_loadu_ps(element_mask16(e), mem_addr);
#endif
}

static __always_inline __m256 mm256_loadu_elements_ps(const u32 e, const void* mem_addr) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    __m256 a = _mm256_setzero_ps();
    std::memcpy(&a, mem_addr, e * sizeof(f32));
    return a;
#else
    return _mm256_maskz_loadu_ps(element_mask8(e), mem_addr);
#endif
}

static __always_inline __m128 mm_loadu_elements_ps(const u32 e, const void* mem_addr) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    __m128 a = _mm_setzero_ps();
    std::memcpy(&a, mem_addr, e * sizeof(f32));
    return a;
#else
    return _mm_maskz_loadu_ps(element_mask8(e), mem_addr);
#endif
}

static __always_inline __m512d mm512_loadu_elements_pd(const u32 e, const void* mem_addr) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    __m512d a = _mm512_setzero_pd();
    std::memcpy(&a, mem_addr, e * sizeof(f64));
    return a;
#else
    return _mm512_maskz_loadu_pd(element_mask8(e), mem_addr);
#endif
}

static __always_inline __m256d mm256_loadu_elements_pd(const u32 e, const void* mem_addr) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    __m256d a = _mm256_setzero_pd();
    std::memcpy(&a, mem_addr, e * sizeof(f64));
    return a;
#else
    return _mm256_maskz_loadu_pd(element_mask8(e), mem_addr);
#endif
}

static __always_inline __m128d mm_loadu_elements_pd(const u32 e, const void* mem_addr) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    __m128d a = _mm_setzero_pd();
    std::memcpy(&a, mem_addr, e * sizeof(f64));
    return a;
#else
    return _mm_maskz_loadu_pd(element_mask8(e), mem_addr);
#endif
}

static __always_inline void mm512_storeu_elements_ps(void* mem_addr, const u32 e, const __m512 a) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    alignas(64) f32 values[16];
    _mm512_storeu_ps(values, a);
    std::memcpy(mem_addr, values, e * sizeof(f32));
#else
    _mm512_mask_storeu_ps(mem_addr, element_mask16(e), a);
#endif
}

static __always_inline void mm256_storeu_elements_ps(void* mem_addr, const u32 e, const __m256 a) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    alignas(32) f32 values[8];
    _mm256_storeu_ps(values, a);
    std::memcpy(mem_addr, values, e * sizeof(f32));
#else
    _mm256_mask_storeu_ps(mem_addr, element_mask8(e), a);
#endif
}

static __always_inline void mm_storeu_elements_ps(void* mem_addr, const u32 e, const __m128 a) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    alignas(16) f32 values[4];
    _mm_storeu_ps(values, a);
    std::memcpy(mem_addr, values, e * sizeof(f32));
#else
    _mm_mask_storeu_ps(mem_addr, element_mask8(e), a);
#endif
}

static __always_inline void mm512_storeu_elements_pd(void* mem_addr, const u32 e, const __m512d a) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    alignas(64) f64 values[8];
    _mm512_storeu_pd(values, a);
    std::memcpy(mem_addr, values, e * sizeof(f64));
#else
    _mm512_mask_storeu_pd(mem_addr, element_mask8(e), a);
#endif
}

static __always_inline void mm256_storeu_elements_pd(void* mem_addr, const u32 e, const __m256d a) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    alignas(32) f64 values[4];
    _mm256_storeu_pd(values, a);
    std::memcpy(mem_addr, values, e * sizeof(f64));
#else
    _mm256_mask_storeu_pd(mem_addr, element_mask8(e), a);
#endif
}

static __always_inline void mm_storeu_elements_pd(void* mem_addr, const u32 e, const __m128d a) {
#if defined(PERNIX_USE_SIMDE) && !defined(SIMDE_X86_AVX512F_NATIVE)
    alignas(16) f64 values[2];
    _mm_storeu_pd(values, a);
    std::memcpy(mem_addr, values, e * sizeof(f64));
#else
    _mm_mask_storeu_pd(mem_addr, element_mask8(e), a);
#endif
}
}  // namespace pernix::internal

#endif  // PERNIX_AVX512_COMPAT_H
