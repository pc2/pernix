#ifndef PERNIX_CPU_FEATURES_H
#define PERNIX_CPU_FEATURES_H

#include <pernix/compat.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#include <x86intrin.h>
#endif
#endif

#if defined(__linux__) && (defined(__aarch64__) || defined(_M_ARM64))
#include <sys/auxv.h>
#ifndef HWCAP_SVE
#define HWCAP_SVE (1 << 22)
#endif
#ifndef HWCAP2_SVE2
#define HWCAP2_SVE2 (1 << 1)
#endif
#endif

namespace pernix::internal {
struct CpuFeatures {
    bool avx2       = false;
    bool bmi2       = false;
    bool avx512f    = false;
    bool avx512dq   = false;
    bool avx512bw   = false;
    bool avx512vl   = false;
    bool avx512vbmi = false;
    bool neon       = false;
    bool sve        = false;
    bool sve2       = false;
};

#if defined(PERNIX_BUILD_LIB)
CpuFeatures detect_cpu_features();
CpuFeatures get_cached_cpu_features();
#else
inline CpuFeatures detect_cpu_features() {
    CpuFeatures features{};

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    int regs[4]{};

#if defined(_MSC_VER)
    __cpuidex(regs, 1, 0);
    const auto xgetbv = [](const unsigned int index) -> u64 { return _xgetbv(index); };
#else
    __cpuid_count(1, 0, regs[0], regs[1], regs[2], regs[3]);
    const auto xgetbv = [](const unsigned int index) -> u64 {
        u32 eax = 0;
        u32 edx = 0;
        __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
        return static_cast<u64>(eax) | (static_cast<u64>(edx) << 32U);
    };
#endif

    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    const bool avx     = (regs[2] & (1 << 28)) != 0;
    if (!osxsave || !avx) {
        return features;
    }

    const u64 xcr0         = xgetbv(0);
    const bool xmm_enabled = (xcr0 & 0x2) != 0;
    const bool ymm_enabled = (xcr0 & 0x4) != 0;
    const bool zmm_enabled = (xcr0 & 0x20) != 0 && (xcr0 & 0x40) != 0 && (xcr0 & 0x80) != 0;
    if (!xmm_enabled || !ymm_enabled) {
        return features;
    }

#if defined(_MSC_VER)
    __cpuidex(regs, 7, 0);
#else
    __cpuid_count(7, 0, regs[0], regs[1], regs[2], regs[3]);
#endif

    features.avx2 = (regs[1] & (1 << 5)) != 0;
    features.bmi2 = (regs[1] & (1 << 8)) != 0;
    if (zmm_enabled) {
        features.avx512f    = (regs[1] & (1 << 16)) != 0;
        features.avx512dq   = (regs[1] & (1 << 29)) != 0;
        features.avx512bw   = (regs[1] & (1 << 30)) != 0;
        features.avx512vl   = (regs[1] & (1 << 31)) != 0;
        features.avx512vbmi = (regs[2] & (1 << 1)) != 0;
    }
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
    features.neon = true;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    features.neon = true;
#endif

#if defined(__linux__) && (defined(__aarch64__) || defined(_M_ARM64))
    const unsigned long hwcap  = getauxval(AT_HWCAP);
    const unsigned long hwcap2 = getauxval(AT_HWCAP2);
    features.sve               = (hwcap & HWCAP_SVE) != 0;
    features.sve2              = (hwcap2 & HWCAP2_SVE2) != 0;
#endif

    return features;
}

inline CpuFeatures get_cached_cpu_features() {
    static const CpuFeatures features = detect_cpu_features();
    return features;
}
#endif
}  // namespace pernix::internal

#endif  // PERNIX_CPU_FEATURES_H
