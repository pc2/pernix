#include <pernix/dispatch/cpu_features.h>

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
CpuFeatures detect_cpu_features() {
    CpuFeatures features{};

    // neon
#if defined(__aarch64__) || defined(_M_ARM64)
    features.neon = true;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    features.neon = true;
#endif

    // sve / sve2 — runtime detection via getauxval on Linux
#if defined(__linux__) && (defined(__aarch64__) || defined(_M_ARM64))
    unsigned long hwcap  = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);
    features.sve  = (hwcap  & HWCAP_SVE)  != 0;
    features.sve2 = (hwcap2 & HWCAP2_SVE2) != 0;
#endif

    return features;
}

CpuFeatures get_cached_cpu_features() {
    static const CpuFeatures features = detect_cpu_features();
    return features;
}
}
