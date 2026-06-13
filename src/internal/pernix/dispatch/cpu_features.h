#ifndef PERNIX_CPU_FEATURES_H
#define PERNIX_CPU_FEATURES_H

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

CpuFeatures detect_cpu_features();

inline CpuFeatures get_cached_cpu_features() {
    static const CpuFeatures features = detect_cpu_features();
    return features;
}
}

#endif //PERNIX_CPU_FEATURES_H
