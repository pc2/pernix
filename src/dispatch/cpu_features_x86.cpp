#include <pernix/dispatch/cpu_features.h>

#include <pernix/compat.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#include <x86intrin.h>
#endif


namespace pernix::internal {
namespace {
#if defined(_MSC_VER)

void cpuid(int out[4], int leaf, int subleaf) {
    __cpuidex(out, leaf, subleaf);
}

u64 xgetbv(unsigned int index) {
    return _xgetbv(index);
}

#else

void cpuid(int out[4], int leaf, int subleaf) {
    __cpuid_count(leaf, subleaf, out[0], out[1], out[2], out[3]);
}

u64 xgetbv(unsigned int index) {
    u32 eax = 0;
    u32 edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
    return static_cast<u64>(eax) | (static_cast<u64>(edx) << 32U);
}

#endif

bool bit_set(int value, int bit) {
    return (value & (1 << bit)) != 0;
}
} // namespace

CpuFeatures detect_cpu_features() {
    CpuFeatures features{};

    int regs[4]{};

    cpuid(regs, 1, 0);

    const bool osxsave = bit_set(regs[2], 27);
    const bool avx     = bit_set(regs[2], 28);

    if (!osxsave || !avx) {
        return features;
    }

    const u64 xcr0 = xgetbv(0);

    const bool xmm_enabled = (xcr0 & 0x2) != 0;
    const bool ymm_enabled = (xcr0 & 0x4) != 0;
    const bool zmm_enabled =
        (xcr0 & 0x20) != 0 &&
        (xcr0 & 0x40) != 0 &&
        (xcr0 & 0x80) != 0;

    if (!xmm_enabled || !ymm_enabled) {
        return features;
    }

    cpuid(regs, 7, 0);

    features.avx2 = bit_set(regs[1], 5);
    features.bmi2 = bit_set(regs[1], 8);

    if (zmm_enabled) {
        features.avx512f    = bit_set(regs[1], 16);
        features.avx512dq   = bit_set(regs[1], 29);
        features.avx512bw   = bit_set(regs[1], 30);
        features.avx512vl   = bit_set(regs[1], 31);
        features.avx512vbmi = bit_set(regs[2], 1);
    }

    return features;
}

CpuFeatures get_cached_cpu_features() {
    static const CpuFeatures features = detect_cpu_features();
    return features;
}
} // namespace pernix::internal
#else

namespace pernix::internal {
CpuFeatures detect_cpu_features() {
    return {};
}

CpuFeatures get_cached_cpu_features() {
    static const CpuFeatures features = detect_cpu_features();
    return features;
}
} // namespace pernix::internal

#endif
