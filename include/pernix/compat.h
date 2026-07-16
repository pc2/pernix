#ifndef PERNIX_COMPAT_H
#define PERNIX_COMPAT_H

#ifdef __cplusplus
#include <cmath>
#include <cstddef>
#include <cstdint>
#else
#include <math.h>
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
using u8   = std::uint8_t;
using u16  = std::uint16_t;
using u32  = std::uint32_t;
using u64  = std::uint64_t;
using uptr = std::uintptr_t;

using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using f32 = std::float_t;
using f64 = std::double_t;

using usize = std::size_t;

#endif  // PERNIX_COMPAT_H
