#include <pernix/dispatch/cpu_features.h>

namespace pernix::internal {
CpuFeatures detect_cpu_features() {
    CpuFeatures features{};

    // neon
#if defined(__aarch64__) || defined(_M_ARM64)
    features.neon = true;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    features.neon = true;
#endif

    // sve
#if defined(__aarch64__) || defined(_M_ARM64)
#ifdef __ARM_FEATURE_SVE
    features.sve = true;
#endif
#endif

    return features;
}
}