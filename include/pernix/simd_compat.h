#ifndef PERNIX_SIMD_COMPAT_H
#define PERNIX_SIMD_COMPAT_H

#include <cstdint>

#if defined(PERNIX_USE_SIMDE)
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx512.h>
#include <simde/x86/bmi.h>

// #ifndef __mmask8
// typedef uint8_t __mmask8;
// #endif
// #ifndef __mmask16
// typedef uint16_t __mmask16;
// #endifc
// #ifndef __mmask32
// typedef uint32_t __mmask32;
// #endif
// #ifndef __mmask64
// typedef uint64_t __mmask64;
// #endif

#elif defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif

#ifndef __always_inline
#if defined(__GNUC__) || defined(__clang__)
#define __always_inline inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define __always_inline __forceinline
#else
#define __always_inline inline
#endif
#endif

template <typename T>
    requires(std::is_integral_v<T> && sizeof(T) <= 8)
static constexpr T tail_mask(const uint8_t bit_width, const uint32_t remaining_elements) {
    const uint32_t tail_bits  = remaining_elements * bit_width;
    const uint32_t tail_bytes = (tail_bits + 7u) / 8u;
    return (static_cast<T>(1) << (tail_bytes * 8)) - 1u;
}

#endif  // PERNIX_SIMD_COMPAT_H
