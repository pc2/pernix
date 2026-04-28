#ifndef PERNIX_SIMD_COMPAT_H
#define PERNIX_SIMD_COMPAT_H

#include <cstdint>
#include <type_traits>

#if defined(PERNIX_USE_SIMDE)
#define SIMDE_ENABLE_NATIVE_ALIASES
#undef SIMDE_X86_AVX512FP16_NATIVE
// #define SIMDE_NO_NATIVE
#include <simde/x86/avx2.h>
#include <simde/x86/bmi.h>
#include <simde/x86/avx512.h>

// #ifndef __mmask8
// typedef uint8_t __mmask8;
// #endif
// #ifndef __mmask16
// typedef uint16_t __mmask16;
// #endif
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
    if (tail_bytes == 0u) {
        return static_cast<T>(0);
    }
    if (tail_bytes >= 64u) {
        return static_cast<T>(~uint64_t{0});
    }
    const uint64_t mask = (uint64_t{1} << tail_bytes) - 1u;
    return static_cast<T>(mask);
}

#endif  // PERNIX_SIMD_COMPAT_H
