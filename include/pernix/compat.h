#ifndef PERNIX_COMPAT_H
#define PERNIX_COMPAT_H

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

#endif //PERNIX_COMPAT_H
