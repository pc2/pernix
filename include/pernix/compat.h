#ifndef PERNIX_COMPAT_H
#define PERNIX_COMPAT_H

#if defined(__cplusplus)
#include <cstddef>
#include <cstdint>
#else
#include <stddef.h>
#include <stdint.h>
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

#if defined(_WIN32) && defined(PERNIX_SHARED)
#if defined(PERNIX_BUILD_LIB)
#define PERNIX_API __declspec(dllexport)
#else
#define PERNIX_API __declspec(dllimport)
#endif
#else
#define PERNIX_API
#endif

// Convenient type declarations
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef uintptr_t uptr;

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef float f32;
typedef double f64;

typedef std::size_t usize;

#endif  // PERNIX_COMPAT_H
