#ifndef PERNIX_SIMD_COMPAT_H
#define PERNIX_SIMD_COMPAT_H

#include <pernix/compat.h>

#include <cstdint>
#include <type_traits>

#if defined(PERNIX_USE_SIMDE)
#define SIMDE_ENABLE_NATIVE_ALIASES
#undef SIMDE_X86_AVX512FP16_NATIVE
#if defined(__clang__)
#define SIMDE_X86_AVX512BF16_NATIVE
#endif
// #define SIMDE_NO_NATIVE
#if defined(PERNIX_BACKEND_X86)
#include <simde/x86/avx2.h>
#include <simde/x86/avx512.h>
#include <simde/x86/bmi.h>
#elif defined(PERNIX_BACKEND_ARM64_NEON)
#include <simde/arm/neon.h>
#elif defined(PERNIX_BACKEND_ARM64_SVE) || defined(PERNIX_BACKEND_ARM64_SVE2)
#include <simde/arm/sve.h>
#endif

#elif defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(__arm64ec__)
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#endif

#endif  // PERNIX_SIMD_COMPAT_H
