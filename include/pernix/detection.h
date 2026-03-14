#ifndef PERNIX_HELPER_H
#define PERNIX_HELPER_H

#include <immintrin.h>

#include <cstdint>

// Internal capability tiers used to derive the public PERNIX_* feature macros.
#define PERNIX_MACHINE_ID_GENERIC 0
#define PERNIX_MACHINE_ID_V2 1
#define PERNIX_MACHINE_ID_V3 2
#define PERNIX_MACHINE_ID_V4 3
#define PERNIX_MACHINE_ID_V4_VBMI 4

// Map the compiler's enabled ISA set to the highest supported Pernix target level.
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

// Feature-selection macros consumed by the public headers.
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

// Allow build systems or tests to force lower-tier implementations.
#ifdef PERNIX_DISABLE_AVX512
#undef PERNIX_AVX512_ENABLED
#undef PERNIX_AVX512_VBMI_ENABLED
#endif
#ifdef PERNIX_DISABLE_AVX2
#undef PERNIX_AVX2_ENABLED
#endif
#ifdef PERNIX_DISABLE_BMI2
#undef PERNIX_BMI2_ENABLED
#endif

#endif  // PERNIX_HELPER_H
