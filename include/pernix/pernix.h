#ifndef PERNIX_H
#define PERNIX_H

#include <pernix/detection.h>

// Include architecture-specific headers based on detected capabilities
// AVX2
#ifdef PERNIX_AVX2_ENABLED
#include <pernix/avx2/compression.h>
#include <pernix/avx2/decompression.h>

// BMI2: Needs AVX2 as well
#ifdef PERNIX_BMI2_ENABLED
#include <pernix/bmi2/compression.h>
#include <pernix/bmi2/decompression.h>
#endif  // PERNIX_BMI2_ENABLED

// AVX512 VBMI: Needs AVX2 as well
#ifdef PERNIX_AVX512_VBMI_ENABLED
#include <pernix/avx512vbmi/compression.h>
#include <pernix/avx512vbmi/decompression.h>
#endif  // PERNIX_AVX512_VBMI_ENABLED

#endif  // PERNIX_AVX2_ENABLED

// Fallback (non-SIMD) implementations
#include <pernix/fallback/compression.h>
#include <pernix/fallback/decompression.h>

#endif  // PERNIX_H