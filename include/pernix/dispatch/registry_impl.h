#ifndef PERNIX_REGISTRY_IMPL_H
#define PERNIX_REGISTRY_IMPL_H

#include <pernix/dispatch/cpu_features.h>
#include <pernix/dispatch/select.h>

#include <type_traits>

namespace pernix::internal {
namespace dispatch_registry {
using SelectBlockF32 = Kernel<KernelBlockF32Func> (*)(u8, u32);
using SelectBlocksF32 = Kernel<KernelBlocksF32Func> (*)(u8, u32);
using SelectBlockF64 = Kernel<KernelBlockF64Func> (*)(u8, u32);
using SelectBlocksF64 = Kernel<KernelBlocksF64Func> (*)(u8, u32);
using SelectDecompressBlockF32 = Kernel<KernelBlockF32Func> (*)(u8, u32, bool);
using SelectDecompressBlocksF32 = Kernel<KernelBlocksF32Func> (*)(u8, u32, bool);
using SelectDecompressBlockF64 = Kernel<KernelBlockF64Func> (*)(u8, u32, bool);
using SelectDecompressBlocksF64 = Kernel<KernelBlocksF64Func> (*)(u8, u32, bool);

enum CpuFeature : u32 {
    CpuAvx2 = 1U << 0U,
    CpuBmi2 = 1U << 1U,
    CpuAvx512F = 1U << 2U,
    CpuAvx512Dq = 1U << 3U,
    CpuAvx512Bw = 1U << 4U,
    CpuAvx512Vl = 1U << 5U,
    CpuAvx512Vbmi = 1U << 6U,
    CpuNeon = 1U << 7U,
    CpuSve2 = 1U << 8U,
};

struct BackendDescriptor {
    Backend backend;
    u32 required_cpu_features;
    SelectBlockF32 compress_block_f32;
    SelectBlocksF32 compress_blocks_f32;
    SelectBlockF64 compress_block_f64;
    SelectBlocksF64 compress_blocks_f64;
    SelectDecompressBlockF32 decompress_block_f32;
    SelectDecompressBlocksF32 decompress_blocks_f32;
    SelectDecompressBlockF64 decompress_block_f64;
    SelectDecompressBlocksF64 decompress_blocks_f64;
};

#define PERNIX_BACKEND_DESCRIPTOR(BACKEND, PREFIX, FEATURES)                                                                  \
    BackendDescriptor {                                                                                                       \
        Backend::BACKEND, FEATURES, &select_##PREFIX##_compress_block_f32, &select_##PREFIX##_compress_blocks_f32,             \
            &select_##PREFIX##_compress_block_f64, &select_##PREFIX##_compress_blocks_f64,                                     \
            &select_##PREFIX##_decompress_block_f32, &select_##PREFIX##_decompress_blocks_f32,                                 \
            &select_##PREFIX##_decompress_block_f64, &select_##PREFIX##_decompress_blocks_f64                                 \
    }

inline constexpr BackendDescriptor kBackendDescriptors[] = {
    PERNIX_BACKEND_DESCRIPTOR(Fallback, fallback, 0U),
#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
    PERNIX_BACKEND_DESCRIPTOR(FallbackStdpar, fallback_stdpar, 0U),
#endif
#if defined(PERNIX_BUILD_FALLBACK_SIMD)
    PERNIX_BACKEND_DESCRIPTOR(FallbackSimd, fallback_simd, 0U),
#endif
#if defined(PERNIX_BUILD_X86_AVX512_VBMI)
    PERNIX_BACKEND_DESCRIPTOR(X86Avx512Vbmi, avx512vbmi,
                              CpuAvx512F | CpuAvx512Dq | CpuAvx512Bw | CpuAvx512Vl | CpuAvx512Vbmi),
#endif
#if defined(PERNIX_BUILD_X86_BMI2)
    PERNIX_BACKEND_DESCRIPTOR(X86Bmi2, bmi2, CpuAvx2 | CpuBmi2),
#endif
#if defined(PERNIX_BUILD_X86_AVX2)
    PERNIX_BACKEND_DESCRIPTOR(X86Avx2, avx2, CpuAvx2),
#endif
#if defined(PERNIX_BUILD_ARM64_SVE2)
    PERNIX_BACKEND_DESCRIPTOR(Arm64Sve, sve2, CpuSve2),
#endif
#if defined(PERNIX_BUILD_ARM64_NEON)
    PERNIX_BACKEND_DESCRIPTOR(Arm64Neon, neon, CpuNeon),
#endif
};

#undef PERNIX_BACKEND_DESCRIPTOR

constexpr u32 cpu_feature_mask(const CpuFeatures& features) noexcept {
    u32 mask = 0;
    if (features.avx2) {
        mask |= CpuAvx2;
    }
    if (features.bmi2) {
        mask |= CpuBmi2;
    }
    if (features.avx512f) {
        mask |= CpuAvx512F;
    }
    if (features.avx512dq) {
        mask |= CpuAvx512Dq;
    }
    if (features.avx512bw) {
        mask |= CpuAvx512Bw;
    }
    if (features.avx512vl) {
        mask |= CpuAvx512Vl;
    }
    if (features.avx512vbmi) {
        mask |= CpuAvx512Vbmi;
    }
    if (features.neon) {
        mask |= CpuNeon;
    }
    if (features.sve2) {
        mask |= CpuSve2;
    }
    return mask;
}

constexpr bool is_known_backend(const Backend backend) noexcept {
    switch (backend) {
        case Backend::Auto:
        case Backend::Fallback:
        case Backend::X86Avx2:
        case Backend::X86Bmi2:
        case Backend::X86Avx512Vbmi:
        case Backend::Arm64Neon:
        case Backend::Arm64Sve:
        case Backend::FallbackStdpar:
        case Backend::FallbackSimd:
            return true;
        default:
            return false;
    }
}

constexpr const BackendDescriptor* find_backend(const Backend backend) noexcept {
    for (const auto& descriptor : kBackendDescriptors) {
        if (descriptor.backend == backend) {
            return &descriptor;
        }
    }
    return nullptr;
}

constexpr bool supports(const BackendDescriptor& descriptor, const CpuFeatures& features) noexcept {
    const u32 detected = cpu_feature_mask(features);
    return (detected & descriptor.required_cpu_features) == descriptor.required_cpu_features;
}

inline pernix_status resolve(const Backend requested, Backend* resolved) noexcept {
    if (resolved == nullptr) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }

    const CpuFeatures features = get_cached_cpu_features();
    if (requested == Backend::Auto) {
        constexpr Backend priority[] = {
            Backend::X86Avx512Vbmi,
            Backend::X86Bmi2,
            Backend::X86Avx2,
            Backend::Arm64Sve,
            Backend::Arm64Neon,
            Backend::Fallback,
        };
        for (const Backend candidate : priority) {
            const BackendDescriptor* descriptor = find_backend(candidate);
            if (descriptor != nullptr && supports(*descriptor, features)) {
                *resolved = candidate;
                return PERNIX_STATUS_OK;
            }
        }
        return PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION;
    }

    if (!is_known_backend(requested)) {
        return PERNIX_STATUS_UNSUPPORTED_BACKEND;
    }
    const BackendDescriptor* descriptor = find_backend(requested);
    if (descriptor == nullptr || !supports(*descriptor, features)) {
        return PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION;
    }

    *resolved = requested;
    return PERNIX_STATUS_OK;
}

template <typename Selector, typename... Args>
auto select(const Backend backend, const Selector selector, Args... args) noexcept {
    using KernelType = std::invoke_result_t<Selector, const BackendDescriptor&, Args...>;
    const BackendDescriptor* descriptor = find_backend(backend);
    if (descriptor == nullptr) {
        return KernelType::failure(is_known_backend(backend) ? PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION
                                                             : PERNIX_STATUS_UNSUPPORTED_BACKEND);
    }
    return selector(*descriptor, args...);
}

struct CompressBlockF32 {
    Kernel<KernelBlockF32Func> operator()(const BackendDescriptor& d, const u8 w, const u32 b) const { return d.compress_block_f32(w, b); }
};
struct CompressBlocksF32 {
    Kernel<KernelBlocksF32Func> operator()(const BackendDescriptor& d, const u8 w, const u32 b) const { return d.compress_blocks_f32(w, b); }
};
struct CompressBlockF64 {
    Kernel<KernelBlockF64Func> operator()(const BackendDescriptor& d, const u8 w, const u32 b) const { return d.compress_block_f64(w, b); }
};
struct CompressBlocksF64 {
    Kernel<KernelBlocksF64Func> operator()(const BackendDescriptor& d, const u8 w, const u32 b) const { return d.compress_blocks_f64(w, b); }
};
struct DecompressBlockF32 {
    Kernel<KernelBlockF32Func> operator()(const BackendDescriptor& d, const u8 w, const u32 b, const bool s) const {
        return d.decompress_block_f32(w, b, s);
    }
};
struct DecompressBlocksF32 {
    Kernel<KernelBlocksF32Func> operator()(const BackendDescriptor& d, const u8 w, const u32 b, const bool s) const {
        return d.decompress_blocks_f32(w, b, s);
    }
};
struct DecompressBlockF64 {
    Kernel<KernelBlockF64Func> operator()(const BackendDescriptor& d, const u8 w, const u32 b, const bool s) const {
        return d.decompress_block_f64(w, b, s);
    }
};
struct DecompressBlocksF64 {
    Kernel<KernelBlocksF64Func> operator()(const BackendDescriptor& d, const u8 w, const u32 b, const bool s) const {
        return d.decompress_blocks_f64(w, b, s);
    }
};
}  // namespace dispatch_registry

#if defined(PERNIX_BUILD_LIB)
#define PERNIX_DISPATCH_LINKAGE
#else
#define PERNIX_DISPATCH_LINKAGE inline
#endif

PERNIX_DISPATCH_LINKAGE pernix_status resolve_backend(const Backend requested, Backend* resolved) noexcept {
    return dispatch_registry::resolve(requested, resolved);
}

PERNIX_DISPATCH_LINKAGE Kernel<KernelBlockF32Func> select_compress_block_f32(const Backend backend, const u8 bit_width,
                                                                            const u32 block_size) {
    return dispatch_registry::select(backend, dispatch_registry::CompressBlockF32{}, bit_width, block_size);
}

PERNIX_DISPATCH_LINKAGE Kernel<KernelBlocksF32Func> select_compress_blocks_f32(const Backend backend, const u8 bit_width,
                                                                              const u32 block_size) {
    return dispatch_registry::select(backend, dispatch_registry::CompressBlocksF32{}, bit_width, block_size);
}

PERNIX_DISPATCH_LINKAGE Kernel<KernelBlockF64Func> select_compress_block_f64(const Backend backend, const u8 bit_width,
                                                                            const u32 block_size) {
    return dispatch_registry::select(backend, dispatch_registry::CompressBlockF64{}, bit_width, block_size);
}

PERNIX_DISPATCH_LINKAGE Kernel<KernelBlocksF64Func> select_compress_blocks_f64(const Backend backend, const u8 bit_width,
                                                                              const u32 block_size) {
    return dispatch_registry::select(backend, dispatch_registry::CompressBlocksF64{}, bit_width, block_size);
}

PERNIX_DISPATCH_LINKAGE Kernel<KernelBlockF32Func> select_decompress_block_f32(const Backend backend, const u8 bit_width,
                                                                               const u32 block_size, const bool sign_values) {
    return dispatch_registry::select(backend, dispatch_registry::DecompressBlockF32{}, bit_width, block_size, sign_values);
}

PERNIX_DISPATCH_LINKAGE Kernel<KernelBlocksF32Func> select_decompress_blocks_f32(const Backend backend, const u8 bit_width,
                                                                                 const u32 block_size, const bool sign_values) {
    return dispatch_registry::select(backend, dispatch_registry::DecompressBlocksF32{}, bit_width, block_size, sign_values);
}

PERNIX_DISPATCH_LINKAGE Kernel<KernelBlockF64Func> select_decompress_block_f64(const Backend backend, const u8 bit_width,
                                                                               const u32 block_size, const bool sign_values) {
    return dispatch_registry::select(backend, dispatch_registry::DecompressBlockF64{}, bit_width, block_size, sign_values);
}

PERNIX_DISPATCH_LINKAGE Kernel<KernelBlocksF64Func> select_decompress_blocks_f64(const Backend backend, const u8 bit_width,
                                                                                 const u32 block_size, const bool sign_values) {
    return dispatch_registry::select(backend, dispatch_registry::DecompressBlocksF64{}, bit_width, block_size, sign_values);
}

#undef PERNIX_DISPATCH_LINKAGE
}  // namespace pernix::internal

#endif  // PERNIX_REGISTRY_IMPL_H
