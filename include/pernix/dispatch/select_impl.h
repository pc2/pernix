#ifndef PERNIX_SELECT_IMPL_H
#define PERNIX_SELECT_IMPL_H

#include <pernix/dispatch/cpu_features.h>
#include <pernix/fallback/scalar_compression.h>
#include <pernix/fallback/scalar_decompression.h>
#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
#include <pernix/fallback/stdpar_compression.h>
#include <pernix/fallback/stdpar_decompression.h>
#endif
#if defined(PERNIX_BUILD_X86_AVX2)
#include <pernix/x86_64/avx2/avx2_compression.h>
#include <pernix/x86_64/avx2/avx2_decompression.h>
#endif
#if defined(PERNIX_BUILD_X86_BMI2)
#include <pernix/x86_64/bmi2/bmi2_compression.h>
#include <pernix/x86_64/bmi2/bmi2_decompression.h>
#endif
#if defined(PERNIX_BUILD_X86_AVX512_VBMI)
#include <pernix/x86_64/avx512vbmi/avx512vbmi_compression.h>
#include <pernix/x86_64/avx512vbmi/avx512vbmi_decompression.h>
#endif
#if defined(PERNIX_BUILD_ARM64_NEON)
#include <pernix/arm64/neon/compression.h>
#include <pernix/arm64/neon/decompression.h>
#endif
#if defined(PERNIX_BUILD_ARM64_SVE2)
#include <pernix/arm64/sve2/compression.h>
#include <pernix/arm64/sve2/decompression.h>
#endif

namespace pernix::internal {
#define PERNIX_CASE_COMPRESS_BLOCK_32(N, BS) \
    case N:                                  \
        return Kernel<KernelBlockF32Func>(&compress_block_fallback<N, BS>)

#define PERNIX_CASE_COMPRESS_BLOCKS_32(N, BS) \
    case N:                                   \
        return Kernel<KernelBlocksF32Func>(&compress_blocks_fallback<N, BS>)

#define PERNIX_CASE_COMPRESS_BLOCK_64(N, BS) \
    case N:                                  \
        return Kernel<KernelBlockF64Func>(&compress_block_fallback<N, BS>)

#define PERNIX_CASE_COMPRESS_BLOCKS_64(N, BS) \
    case N:                                   \
        return Kernel<KernelBlocksF64Func>(&compress_blocks_fallback<N, BS>)

#define PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32(BS)   \
    case BS:                                       \
        switch (bit_width) {                       \
            PERNIX_CASE_COMPRESS_BLOCK_32(1, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(2, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(3, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(4, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(5, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(6, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(7, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(8, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(9, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(10, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(11, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(12, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(13, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(14, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(15, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(16, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(17, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(18, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(19, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(20, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(21, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(22, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(23, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(24, BS); \
            default:                               \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};      \
        }

#define PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32(BS) \
    case BS:                                            \
        switch (bit_width) {                            \
            PERNIX_CASE_COMPRESS_BLOCKS_32(1, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(2, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(3, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(4, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(5, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(6, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(7, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(8, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(9, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(10, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(11, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(12, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(13, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(14, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(15, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(16, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(17, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(18, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(19, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(20, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(21, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(22, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(23, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(24, BS);     \
            default:                                    \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};           \
        }

#define PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64(BS)   \
    case BS:                                       \
        switch (bit_width) {                       \
            PERNIX_CASE_COMPRESS_BLOCK_64(1, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(2, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(3, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(4, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(5, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(6, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(7, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(8, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(9, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(10, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(11, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(12, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(13, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(14, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(15, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(16, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(17, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(18, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(19, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(20, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(21, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(22, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(23, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(24, BS); \
            default:                               \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};      \
        }

#define PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64(BS) \
    case BS:                                            \
        switch (bit_width) {                            \
            PERNIX_CASE_COMPRESS_BLOCKS_64(1, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(2, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(3, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(4, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(5, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(6, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(7, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(8, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(9, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(10, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(11, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(12, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(13, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(14, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(15, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(16, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(17, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(18, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(19, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(20, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(21, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(22, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(23, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(24, BS);     \
            default:                                    \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};           \
        }

inline Kernel<KernelBlockF32Func> select_fallback_compress_block_f32(const u8 bit_width, const u32 block_size) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32(64);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32(128);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32(256);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32(512);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlocksF32Func> select_fallback_compress_blocks_f32(const u8 bit_width, const u32 block_size) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32(64);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32(128);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32(256);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32(512);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlockF64Func> select_fallback_compress_block_f64(const u8 bit_width, const u32 block_size) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64(64);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64(128);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64(256);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64(512);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlocksF64Func> select_fallback_compress_blocks_f64(const u8 bit_width, const u32 block_size) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64(64);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64(128);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64(256);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64(512);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlockF32Func> select_fallback_scalar_compress_block_f32(const u8 bit_width, const u32 block_size) {
    return select_fallback_compress_block_f32(bit_width, block_size);
}

inline Kernel<KernelBlocksF32Func> select_fallback_scalar_compress_blocks_f32(const u8 bit_width, const u32 block_size) {
    return select_fallback_compress_blocks_f32(bit_width, block_size);
}

inline Kernel<KernelBlockF64Func> select_fallback_scalar_compress_block_f64(const u8 bit_width, const u32 block_size) {
    return select_fallback_compress_block_f64(bit_width, block_size);
}

inline Kernel<KernelBlocksF64Func> select_fallback_scalar_compress_blocks_f64(const u8 bit_width, const u32 block_size) {
    return select_fallback_compress_blocks_f64(bit_width, block_size);
}

#define PERNIX_CASE_STDPAR_COMPRESS_BLOCK_32(N, BS) \
    case N:                                         \
        return Kernel<KernelBlockF32Func>(&compress_block_fallback_stdpar<N, BS, f32>)

#define PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_32(N, BS) \
    case N:                                          \
        return Kernel<KernelBlocksF32Func>(&compress_blocks_fallback_stdpar<N, BS, f32>)

#define PERNIX_CASE_STDPAR_COMPRESS_BLOCK_64(N, BS) \
    case N:                                         \
        return Kernel<KernelBlockF64Func>(&compress_block_fallback_stdpar<N, BS, f64>)

#define PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_64(N, BS) \
    case N:                                          \
        return Kernel<KernelBlocksF64Func>(&compress_blocks_fallback_stdpar<N, BS, f64>)

#define PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_SWITCH_32(BS)   \
    case BS:                                              \
        switch (bit_width) {                              \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_32(1, BS);  \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_32(2, BS);  \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_32(3, BS);  \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_32(4, BS);  \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_32(5, BS);  \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_32(6, BS);  \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_32(7, BS);  \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_32(8, BS);  \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_32(9, BS);  \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_32(10, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_32(11, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_32(12, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_32(13, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_32(14, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_32(15, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_32(16, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_32(17, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_32(18, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_32(19, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_32(20, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_32(21, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_32(22, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_32(23, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_32(24, BS); \
            default:                                      \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};      \
        }

#define PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_BLOCKS_SWITCH_32(BS) \
    case BS:                                                   \
        switch (bit_width) {                                   \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_32(1, BS);      \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_32(2, BS);      \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_32(3, BS);      \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_32(4, BS);      \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_32(5, BS);      \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_32(6, BS);      \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_32(7, BS);      \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_32(8, BS);      \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_32(9, BS);      \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_32(10, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_32(11, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_32(12, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_32(13, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_32(14, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_32(15, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_32(16, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_32(17, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_32(18, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_32(19, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_32(20, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_32(21, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_32(22, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_32(23, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_32(24, BS);     \
            default:                                           \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};           \
        }

#define PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_SWITCH_64(BS)   \
    case BS:                                              \
        switch (bit_width) {                              \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_64(1, BS);  \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_64(2, BS);  \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_64(3, BS);  \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_64(4, BS);  \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_64(5, BS);  \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_64(6, BS);  \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_64(7, BS);  \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_64(8, BS);  \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_64(9, BS);  \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_64(10, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_64(11, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_64(12, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_64(13, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_64(14, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_64(15, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_64(16, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_64(17, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_64(18, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_64(19, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_64(20, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_64(21, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_64(22, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_64(23, BS); \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCK_64(24, BS); \
            default:                                      \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};      \
        }

#define PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_BLOCKS_SWITCH_64(BS) \
    case BS:                                                   \
        switch (bit_width) {                                   \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_64(1, BS);      \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_64(2, BS);      \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_64(3, BS);      \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_64(4, BS);      \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_64(5, BS);      \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_64(6, BS);      \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_64(7, BS);      \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_64(8, BS);      \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_64(9, BS);      \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_64(10, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_64(11, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_64(12, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_64(13, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_64(14, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_64(15, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_64(16, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_64(17, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_64(18, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_64(19, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_64(20, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_64(21, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_64(22, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_64(23, BS);     \
            PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_64(24, BS);     \
            default:                                           \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};           \
        }

inline Kernel<KernelBlockF32Func> select_fallback_stdpar_compress_block_f32(const u8 bit_width, const u32 block_size) {
#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
    switch (block_size) {
        PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_SWITCH_32(64);
        PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_SWITCH_32(128);
        PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_SWITCH_32(256);
        PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_SWITCH_32(512);
        PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_SWITCH_32(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
#else
    return {PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION, nullptr};
#endif
}

inline Kernel<KernelBlocksF32Func> select_fallback_stdpar_compress_blocks_f32(const u8 bit_width, const u32 block_size) {
#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
    switch (block_size) {
        PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_BLOCKS_SWITCH_32(64);
        PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_BLOCKS_SWITCH_32(128);
        PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_BLOCKS_SWITCH_32(256);
        PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_BLOCKS_SWITCH_32(512);
        PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_BLOCKS_SWITCH_32(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
#else
    return {PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION, nullptr};
#endif
}

inline Kernel<KernelBlockF64Func> select_fallback_stdpar_compress_block_f64(const u8 bit_width, const u32 block_size) {
#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
    switch (block_size) {
        PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_SWITCH_64(64);
        PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_SWITCH_64(128);
        PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_SWITCH_64(256);
        PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_SWITCH_64(512);
        PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_SWITCH_64(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
#else
    return {PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION, nullptr};
#endif
}

inline Kernel<KernelBlocksF64Func> select_fallback_stdpar_compress_blocks_f64(const u8 bit_width, const u32 block_size) {
#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
    switch (block_size) {
        PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_BLOCKS_SWITCH_64(64);
        PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_BLOCKS_SWITCH_64(128);
        PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_BLOCKS_SWITCH_64(256);
        PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_BLOCKS_SWITCH_64(512);
        PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_BLOCKS_SWITCH_64(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
#else
    return {PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION, nullptr};
#endif
}

#undef PERNIX_CASE_COMPRESS_BLOCK_32
#undef PERNIX_CASE_COMPRESS_BLOCKS_32
#undef PERNIX_CASE_COMPRESS_BLOCK_64
#undef PERNIX_CASE_COMPRESS_BLOCKS_64
#undef PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32
#undef PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32
#undef PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64
#undef PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64
}  // namespace pernix::internal

namespace pernix::internal {
#define PERNIX_CASE_DECOMPRESS_BLOCK_32(N, BS)                                                           \
    case N:                                                                                              \
        if (sign_values)                                                                                 \
            return Kernel<KernelBlockF32Func>(&decompress_block_fallback<N, true, BS, f32>); \
        return Kernel<KernelBlockF32Func>(&decompress_block_fallback<N, false, BS, f32>)

#define PERNIX_CASE_DECOMPRESS_BLOCKS_32(N, BS)                                                            \
    case N:                                                                                                \
        if (sign_values)                                                                                   \
            return Kernel<KernelBlocksF32Func>(&decompress_blocks_fallback<N, true, BS, f32>); \
        return Kernel<KernelBlocksF32Func>(&decompress_blocks_fallback<N, false, BS, f32>)

#define PERNIX_CASE_DECOMPRESS_BLOCK_64(N, BS)                                                           \
    case N:                                                                                              \
        if (sign_values)                                                                                 \
            return Kernel<KernelBlockF64Func>(&decompress_block_fallback<N, true, BS, f64>); \
        return Kernel<KernelBlockF64Func>(&decompress_block_fallback<N, false, BS, f64>)

#define PERNIX_CASE_DECOMPRESS_BLOCKS_64(N, BS)                                                            \
    case N:                                                                                                \
        if (sign_values)                                                                                   \
            return Kernel<KernelBlocksF64Func>(&decompress_blocks_fallback<N, true, BS, f64>); \
        return Kernel<KernelBlocksF64Func>(&decompress_blocks_fallback<N, false, BS, f64>)
#define PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(BS)   \
    case BS:                                         \
        switch (bit_width) {                         \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(1, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(2, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(3, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(4, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(5, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(6, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(7, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(8, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(9, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(10, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(11, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(12, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(13, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(14, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(15, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(16, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(17, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(18, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(19, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(20, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(21, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(22, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(23, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(24, BS); \
            default:                                 \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};        \
        }                                            \
        break

#define PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(BS) \
    case BS:                                              \
        switch (bit_width) {                              \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(1, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(2, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(3, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(4, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(5, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(6, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(7, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(8, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(9, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(10, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(11, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(12, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(13, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(14, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(15, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(16, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(17, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(18, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(19, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(20, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(21, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(22, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(23, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(24, BS);     \
            default:                                      \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};             \
        }                                                 \
        break

#define PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(BS)   \
    case BS:                                         \
        switch (bit_width) {                         \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(1, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(2, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(3, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(4, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(5, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(6, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(7, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(8, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(9, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(10, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(11, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(12, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(13, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(14, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(15, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(16, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(17, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(18, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(19, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(20, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(21, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(22, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(23, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(24, BS); \
            default:                                 \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};        \
        }                                            \
        break

#define PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(BS) \
    case BS:                                              \
        switch (bit_width) {                              \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(1, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(2, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(3, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(4, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(5, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(6, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(7, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(8, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(9, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(10, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(11, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(12, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(13, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(14, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(15, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(16, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(17, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(18, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(19, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(20, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(21, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(22, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(23, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(24, BS);     \
            default:                                      \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};             \
        }                                                 \
        break

inline Kernel<KernelBlockF32Func> select_fallback_decompress_block_f32(const u8 bit_width, const u32 block_size, bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(64);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(128);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(256);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(512);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlocksF32Func> select_fallback_decompress_blocks_f32(const u8 bit_width, const u32 block_size, bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(64);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(128);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(256);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(512);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlockF64Func> select_fallback_decompress_block_f64(const u8 bit_width, const u32 block_size, bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(64);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(128);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(256);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(512);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlocksF64Func> select_fallback_decompress_blocks_f64(const u8 bit_width, const u32 block_size, bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(64);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(128);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(256);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(512);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlockF32Func> select_fallback_scalar_decompress_block_f32(const u8 bit_width, const u32 block_size,
                                                                              const bool sign_values) {
    return select_fallback_decompress_block_f32(bit_width, block_size, sign_values);
}

inline Kernel<KernelBlocksF32Func> select_fallback_scalar_decompress_blocks_f32(const u8 bit_width, const u32 block_size,
                                                                                const bool sign_values) {
    return select_fallback_decompress_blocks_f32(bit_width, block_size, sign_values);
}

inline Kernel<KernelBlockF64Func> select_fallback_scalar_decompress_block_f64(const u8 bit_width, const u32 block_size,
                                                                              const bool sign_values) {
    return select_fallback_decompress_block_f64(bit_width, block_size, sign_values);
}

inline Kernel<KernelBlocksF64Func> select_fallback_scalar_decompress_blocks_f64(const u8 bit_width, const u32 block_size,
                                                                                const bool sign_values) {
    return select_fallback_decompress_blocks_f64(bit_width, block_size, sign_values);
}

#define PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32(N, BS)                                                                  \
    case N:                                                                                                            \
        if (sign_values)                                                                                               \
            return Kernel<KernelBlockF32Func>(&decompress_block_fallback_stdpar<N, true, BS, f32>); \
        return Kernel<KernelBlockF32Func>(&decompress_block_fallback_stdpar<N, false, BS, f32>)

#define PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32(N, BS)                                                                   \
    case N:                                                                                                              \
        if (sign_values)                                                                                                 \
            return Kernel<KernelBlocksF32Func>(&decompress_blocks_fallback_stdpar<N, true, BS, f32>); \
        return Kernel<KernelBlocksF32Func>(&decompress_blocks_fallback_stdpar<N, false, BS, f32>)

#define PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64(N, BS)                                                                  \
    case N:                                                                                                            \
        if (sign_values)                                                                                               \
            return Kernel<KernelBlockF64Func>(&decompress_block_fallback_stdpar<N, true, BS, f64>); \
        return Kernel<KernelBlockF64Func>(&decompress_block_fallback_stdpar<N, false, BS, f64>)

#define PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64(N, BS)                                                                   \
    case N:                                                                                                              \
        if (sign_values)                                                                                                 \
            return Kernel<KernelBlocksF64Func>(&decompress_blocks_fallback_stdpar<N, true, BS, f64>); \
        return Kernel<KernelBlocksF64Func>(&decompress_blocks_fallback_stdpar<N, false, BS, f64>)

#define PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_SWITCH_32(BS)   \
    case BS:                                                \
        switch (bit_width) {                                \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32(1, BS);  \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32(2, BS);  \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32(3, BS);  \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32(4, BS);  \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32(5, BS);  \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32(6, BS);  \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32(7, BS);  \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32(8, BS);  \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32(9, BS);  \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32(10, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32(11, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32(12, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32(13, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32(14, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32(15, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32(16, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32(17, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32(18, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32(19, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32(20, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32(21, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32(22, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32(23, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32(24, BS); \
            default:                                        \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};        \
        }                                                   \
        break

#define PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_BLOCKS_SWITCH_32(BS) \
    case BS:                                                     \
        switch (bit_width) {                                     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32(1, BS);      \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32(2, BS);      \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32(3, BS);      \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32(4, BS);      \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32(5, BS);      \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32(6, BS);      \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32(7, BS);      \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32(8, BS);      \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32(9, BS);      \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32(10, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32(11, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32(12, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32(13, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32(14, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32(15, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32(16, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32(17, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32(18, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32(19, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32(20, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32(21, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32(22, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32(23, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32(24, BS);     \
            default:                                             \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};             \
        }                                                        \
        break

#define PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_SWITCH_64(BS)   \
    case BS:                                                \
        switch (bit_width) {                                \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64(1, BS);  \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64(2, BS);  \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64(3, BS);  \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64(4, BS);  \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64(5, BS);  \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64(6, BS);  \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64(7, BS);  \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64(8, BS);  \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64(9, BS);  \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64(10, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64(11, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64(12, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64(13, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64(14, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64(15, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64(16, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64(17, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64(18, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64(19, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64(20, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64(21, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64(22, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64(23, BS); \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64(24, BS); \
            default:                                        \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};        \
        }                                                   \
        break

#define PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_BLOCKS_SWITCH_64(BS) \
    case BS:                                                     \
        switch (bit_width) {                                     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64(1, BS);      \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64(2, BS);      \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64(3, BS);      \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64(4, BS);      \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64(5, BS);      \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64(6, BS);      \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64(7, BS);      \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64(8, BS);      \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64(9, BS);      \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64(10, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64(11, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64(12, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64(13, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64(14, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64(15, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64(16, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64(17, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64(18, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64(19, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64(20, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64(21, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64(22, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64(23, BS);     \
            PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64(24, BS);     \
            default:                                             \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};             \
        }                                                        \
        break

inline Kernel<KernelBlockF32Func> select_fallback_stdpar_decompress_block_f32(const u8 bit_width, const u32 block_size,
                                                                              const bool sign_values) {
#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
    switch (block_size) {
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_SWITCH_32(64);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_SWITCH_32(128);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_SWITCH_32(256);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_SWITCH_32(512);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_SWITCH_32(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
#else
    return {PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION, nullptr};
#endif
}

inline Kernel<KernelBlocksF32Func> select_fallback_stdpar_decompress_blocks_f32(const u8 bit_width, const u32 block_size,
                                                                                const bool sign_values) {
#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
    switch (block_size) {
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_BLOCKS_SWITCH_32(64);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_BLOCKS_SWITCH_32(128);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_BLOCKS_SWITCH_32(256);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_BLOCKS_SWITCH_32(512);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_BLOCKS_SWITCH_32(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
#else
    return {PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION, nullptr};
#endif
}

inline Kernel<KernelBlockF64Func> select_fallback_stdpar_decompress_block_f64(const u8 bit_width, const u32 block_size,
                                                                              const bool sign_values) {
#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
    switch (block_size) {
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_SWITCH_64(64);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_SWITCH_64(128);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_SWITCH_64(256);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_SWITCH_64(512);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_SWITCH_64(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
#else
    return {PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION, nullptr};
#endif
}

inline Kernel<KernelBlocksF64Func> select_fallback_stdpar_decompress_blocks_f64(const u8 bit_width, const u32 block_size,
                                                                                const bool sign_values) {
#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
    switch (block_size) {
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_BLOCKS_SWITCH_64(64);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_BLOCKS_SWITCH_64(128);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_BLOCKS_SWITCH_64(256);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_BLOCKS_SWITCH_64(512);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_BLOCKS_SWITCH_64(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
#else
    return {PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION, nullptr};
#endif
}

#undef PERNIX_CASE_DECOMPRESS_BLOCK_32
#undef PERNIX_CASE_DECOMPRESS_BLOCKS_32
#undef PERNIX_CASE_DECOMPRESS_BLOCK_64
#undef PERNIX_CASE_DECOMPRESS_BLOCKS_64
#undef PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32
#undef PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32
#undef PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64
#undef PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64
}  // namespace pernix::internal
#if defined(PERNIX_BUILD_X86_AVX2)

namespace pernix::internal {
#define PERNIX_CASE_COMPRESS_BLOCK_32(N, BS) \
    case N:                                  \
        return Kernel<KernelBlockF32Func>(&mm256_compress_block_avx2<N, BS>)

#define PERNIX_CASE_COMPRESS_BLOCKS_32(N, BS) \
    case N:                                   \
        return Kernel<KernelBlocksF32Func>(&mm256_compress_blocks_avx2<N, BS>)

#define PERNIX_CASE_COMPRESS_BLOCK_64(N, BS) \
    case N:                                  \
        return Kernel<KernelBlockF64Func>(&mm256_compress_block_avx2<N, BS>)

#define PERNIX_CASE_COMPRESS_BLOCKS_64(N, BS) \
    case N:                                   \
        return Kernel<KernelBlocksF64Func>(&mm256_compress_blocks_avx2<N, BS>)

#define PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32(BS)   \
    case BS:                                       \
        switch (bit_width) {                       \
            PERNIX_CASE_COMPRESS_BLOCK_32(1, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(2, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(3, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(4, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(5, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(6, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(7, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(8, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(9, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(10, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(11, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(12, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(13, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(14, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(15, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(16, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(17, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(18, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(19, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(20, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(21, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(22, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(23, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(24, BS); \
            default:                               \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};          \
        }

#define PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32(BS) \
    case BS:                                            \
        switch (bit_width) {                            \
            PERNIX_CASE_COMPRESS_BLOCKS_32(1, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(2, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(3, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(4, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(5, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(6, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(7, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(8, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(9, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(10, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(11, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(12, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(13, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(14, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(15, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(16, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(17, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(18, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(19, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(20, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(21, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(22, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(23, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(24, BS);     \
            default:                                    \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};               \
        }

#define PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64(BS)   \
    case BS:                                       \
        switch (bit_width) {                       \
            PERNIX_CASE_COMPRESS_BLOCK_64(1, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(2, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(3, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(4, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(5, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(6, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(7, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(8, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(9, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(10, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(11, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(12, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(13, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(14, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(15, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(16, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(17, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(18, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(19, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(20, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(21, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(22, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(23, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(24, BS); \
            default:                               \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};          \
        }

#define PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64(BS) \
    case BS:                                            \
        switch (bit_width) {                            \
            PERNIX_CASE_COMPRESS_BLOCKS_64(1, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(2, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(3, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(4, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(5, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(6, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(7, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(8, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(9, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(10, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(11, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(12, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(13, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(14, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(15, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(16, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(17, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(18, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(19, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(20, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(21, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(22, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(23, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(24, BS);     \
            default:                                    \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};               \
        }

inline Kernel<KernelBlockF32Func> select_avx2_compress_block_f32(const u8 bit_width, const u32 block_size) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32(64);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32(128);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32(256);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32(512);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlocksF32Func> select_avx2_compress_blocks_f32(const u8 bit_width, const u32 block_size) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32(64);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32(128);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32(256);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32(512);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlockF64Func> select_avx2_compress_block_f64(const u8 bit_width, const u32 block_size) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64(64);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64(128);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64(256);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64(512);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlocksF64Func> select_avx2_compress_blocks_f64(const u8 bit_width, const u32 block_size) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64(64);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64(128);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64(256);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64(512);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

#undef PERNIX_CASE_COMPRESS_BLOCK_32
#undef PERNIX_CASE_COMPRESS_BLOCKS_32
#undef PERNIX_CASE_COMPRESS_BLOCK_64
#undef PERNIX_CASE_COMPRESS_BLOCKS_64
#undef PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32
#undef PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32
#undef PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64
#undef PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64
#undef PERNIX_CASE_STDPAR_COMPRESS_BLOCK_32
#undef PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_32
#undef PERNIX_CASE_STDPAR_COMPRESS_BLOCK_64
#undef PERNIX_CASE_STDPAR_COMPRESS_BLOCKS_64
#undef PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_SWITCH_32
#undef PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_BLOCKS_SWITCH_32
#undef PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_SWITCH_64
#undef PERNIX_BLOCK_SIZE_STDPAR_COMPRESS_BLOCKS_SWITCH_64
}  // namespace pernix::internal

namespace pernix::internal {
#define PERNIX_CASE_DECOMPRESS_BLOCK_32(N, BS)                                                    \
    case N:                                                                                       \
        if (sign_values)                                                                          \
            return Kernel<KernelBlockF32Func>(&mm256_decompress_block_avx2<N, true, BS>); \
        return Kernel<KernelBlockF32Func>(&mm256_decompress_block_avx2<N, false, BS>)

#define PERNIX_CASE_DECOMPRESS_BLOCKS_32(N, BS)                                                     \
    case N:                                                                                         \
        if (sign_values)                                                                            \
            return Kernel<KernelBlocksF32Func>(&mm256_decompress_blocks_avx2<N, true, BS>); \
        return Kernel<KernelBlocksF32Func>(&mm256_decompress_blocks_avx2<N, false, BS>)

#define PERNIX_CASE_DECOMPRESS_BLOCK_64(N, BS)                                                    \
    case N:                                                                                       \
        if (sign_values)                                                                          \
            return Kernel<KernelBlockF64Func>(&mm256_decompress_block_avx2<N, true, BS>); \
        return Kernel<KernelBlockF64Func>(&mm256_decompress_block_avx2<N, false, BS>)

#define PERNIX_CASE_DECOMPRESS_BLOCKS_64(N, BS)                                                     \
    case N:                                                                                         \
        if (sign_values)                                                                            \
            return Kernel<KernelBlocksF64Func>(&mm256_decompress_blocks_avx2<N, true, BS>); \
        return Kernel<KernelBlocksF64Func>(&mm256_decompress_blocks_avx2<N, false, BS>)
#define PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(BS)   \
    case BS:                                         \
        switch (bit_width) {                         \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(1, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(2, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(3, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(4, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(5, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(6, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(7, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(8, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(9, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(10, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(11, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(12, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(13, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(14, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(15, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(16, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(17, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(18, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(19, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(20, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(21, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(22, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(23, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(24, BS); \
            default:                                 \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};            \
        }                                            \
        break

#define PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(BS) \
    case BS:                                              \
        switch (bit_width) {                              \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(1, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(2, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(3, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(4, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(5, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(6, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(7, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(8, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(9, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(10, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(11, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(12, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(13, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(14, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(15, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(16, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(17, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(18, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(19, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(20, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(21, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(22, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(23, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(24, BS);     \
            default:                                      \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};                 \
        }                                                 \
        break

#define PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(BS)   \
    case BS:                                         \
        switch (bit_width) {                         \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(1, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(2, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(3, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(4, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(5, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(6, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(7, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(8, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(9, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(10, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(11, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(12, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(13, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(14, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(15, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(16, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(17, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(18, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(19, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(20, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(21, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(22, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(23, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(24, BS); \
            default:                                 \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};            \
        }                                            \
        break

#define PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(BS) \
    case BS:                                              \
        switch (bit_width) {                              \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(1, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(2, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(3, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(4, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(5, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(6, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(7, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(8, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(9, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(10, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(11, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(12, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(13, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(14, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(15, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(16, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(17, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(18, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(19, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(20, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(21, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(22, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(23, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(24, BS);     \
            default:                                      \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};                 \
        }                                                 \
        break

inline Kernel<KernelBlockF32Func> select_avx2_decompress_block_f32(const u8 bit_width, const u32 block_size, bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(64);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(128);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(256);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(512);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlocksF32Func> select_avx2_decompress_blocks_f32(const u8 bit_width, const u32 block_size, bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(64);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(128);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(256);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(512);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlockF64Func> select_avx2_decompress_block_f64(const u8 bit_width, const u32 block_size, bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(64);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(128);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(256);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(512);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlocksF64Func> select_avx2_decompress_blocks_f64(const u8 bit_width, const u32 block_size, bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(64);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(128);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(256);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(512);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

#undef PERNIX_CASE_DECOMPRESS_BLOCK_32
#undef PERNIX_CASE_DECOMPRESS_BLOCKS_32
#undef PERNIX_CASE_DECOMPRESS_BLOCK_64
#undef PERNIX_CASE_DECOMPRESS_BLOCKS_64
#undef PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32
#undef PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32
#undef PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64
#undef PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64
#undef PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32
#undef PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32
#undef PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64
#undef PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64
#undef PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_SWITCH_32
#undef PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_BLOCKS_SWITCH_32
#undef PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_SWITCH_64
#undef PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_BLOCKS_SWITCH_64
}  // namespace pernix::internal
#endif
#if defined(PERNIX_BUILD_X86_BMI2)

namespace pernix::internal {
#define PERNIX_CASE_COMPRESS_BLOCK_32(N, BS) \
    case N:                                  \
        return Kernel<KernelBlockF32Func>(&mm256_compress_block_bmi2<N, BS>)

#define PERNIX_CASE_COMPRESS_BLOCKS_32(N, BS) \
    case N:                                   \
        return Kernel<KernelBlocksF32Func>(&mm256_compress_blocks_bmi2<N, BS>)

#define PERNIX_CASE_COMPRESS_BLOCK_64(N, BS) \
    case N:                                  \
        return Kernel<KernelBlockF64Func>(&mm256_compress_block_bmi2<N, BS>)

#define PERNIX_CASE_COMPRESS_BLOCKS_64(N, BS) \
    case N:                                   \
        return Kernel<KernelBlocksF64Func>(&mm256_compress_blocks_bmi2<N, BS>)

#define PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32(BS)   \
    case BS:                                       \
        switch (bit_width) {                       \
            PERNIX_CASE_COMPRESS_BLOCK_32(1, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(2, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(3, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(4, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(5, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(6, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(7, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(8, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(9, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(10, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(11, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(12, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(13, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(14, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(15, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(16, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(17, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(18, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(19, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(20, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(21, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(22, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(23, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(24, BS); \
            default:                               \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};          \
        }

#define PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32(BS) \
    case BS:                                            \
        switch (bit_width) {                            \
            PERNIX_CASE_COMPRESS_BLOCKS_32(1, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(2, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(3, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(4, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(5, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(6, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(7, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(8, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(9, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(10, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(11, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(12, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(13, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(14, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(15, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(16, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(17, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(18, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(19, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(20, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(21, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(22, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(23, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(24, BS);     \
            default:                                    \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};               \
        }

#define PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64(BS)   \
    case BS:                                       \
        switch (bit_width) {                       \
            PERNIX_CASE_COMPRESS_BLOCK_64(1, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(2, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(3, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(4, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(5, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(6, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(7, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(8, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(9, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(10, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(11, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(12, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(13, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(14, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(15, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(16, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(17, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(18, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(19, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(20, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(21, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(22, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(23, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(24, BS); \
            default:                               \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};          \
        }

#define PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64(BS) \
    case BS:                                            \
        switch (bit_width) {                            \
            PERNIX_CASE_COMPRESS_BLOCKS_64(1, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(2, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(3, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(4, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(5, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(6, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(7, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(8, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(9, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(10, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(11, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(12, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(13, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(14, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(15, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(16, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(17, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(18, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(19, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(20, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(21, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(22, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(23, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(24, BS);     \
            default:                                    \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};               \
        }

inline Kernel<KernelBlockF32Func> select_bmi2_compress_block_f32(const u8 bit_width, const u32 block_size) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32(64);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32(128);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32(256);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32(512);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlocksF32Func> select_bmi2_compress_blocks_f32(const u8 bit_width, const u32 block_size) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32(64);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32(128);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32(256);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32(512);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlockF64Func> select_bmi2_compress_block_f64(const u8 bit_width, const u32 block_size) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64(64);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64(128);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64(256);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64(512);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlocksF64Func> select_bmi2_compress_blocks_f64(const u8 bit_width, const u32 block_size) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64(64);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64(128);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64(256);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64(512);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

#undef PERNIX_CASE_COMPRESS_BLOCK_32
#undef PERNIX_CASE_COMPRESS_BLOCKS_32
#undef PERNIX_CASE_COMPRESS_BLOCK_64
#undef PERNIX_CASE_COMPRESS_BLOCKS_64
#undef PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32
#undef PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32
#undef PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64
#undef PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64
}  // namespace pernix::internal

namespace pernix::internal {
#define PERNIX_CASE_DECOMPRESS_BLOCK_32(N, BS)                                                    \
    case N:                                                                                       \
        if (sign_values)                                                                          \
            return Kernel<KernelBlockF32Func>(&mm256_decompress_block_bmi2<N, true, BS>); \
        return Kernel<KernelBlockF32Func>(&mm256_decompress_block_bmi2<N, false, BS>)

#define PERNIX_CASE_DECOMPRESS_BLOCKS_32(N, BS)                                                     \
    case N:                                                                                         \
        if (sign_values)                                                                            \
            return Kernel<KernelBlocksF32Func>(&mm256_decompress_blocks_bmi2<N, true, BS>); \
        return Kernel<KernelBlocksF32Func>(&mm256_decompress_blocks_bmi2<N, false, BS>)

#define PERNIX_CASE_DECOMPRESS_BLOCK_64(N, BS)                                                    \
    case N:                                                                                       \
        if (sign_values)                                                                          \
            return Kernel<KernelBlockF64Func>(&mm256_decompress_block_bmi2<N, true, BS>); \
        return Kernel<KernelBlockF64Func>(&mm256_decompress_block_bmi2<N, false, BS>)

#define PERNIX_CASE_DECOMPRESS_BLOCKS_64(N, BS)                                                     \
    case N:                                                                                         \
        if (sign_values)                                                                            \
            return Kernel<KernelBlocksF64Func>(&mm256_decompress_blocks_bmi2<N, true, BS>); \
        return Kernel<KernelBlocksF64Func>(&mm256_decompress_blocks_bmi2<N, false, BS>)
#define PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(BS)   \
    case BS:                                         \
        switch (bit_width) {                         \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(1, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(2, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(3, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(4, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(5, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(6, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(7, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(8, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(9, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(10, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(11, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(12, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(13, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(14, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(15, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(16, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(17, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(18, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(19, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(20, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(21, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(22, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(23, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(24, BS); \
            default:                                 \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};            \
        }                                            \
        break

#define PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(BS) \
    case BS:                                              \
        switch (bit_width) {                              \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(1, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(2, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(3, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(4, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(5, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(6, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(7, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(8, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(9, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(10, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(11, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(12, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(13, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(14, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(15, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(16, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(17, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(18, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(19, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(20, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(21, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(22, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(23, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(24, BS);     \
            default:                                      \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};                 \
        }                                                 \
        break

#define PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(BS)   \
    case BS:                                         \
        switch (bit_width) {                         \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(1, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(2, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(3, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(4, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(5, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(6, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(7, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(8, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(9, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(10, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(11, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(12, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(13, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(14, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(15, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(16, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(17, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(18, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(19, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(20, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(21, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(22, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(23, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(24, BS); \
            default:                                 \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};            \
        }                                            \
        break

#define PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(BS) \
    case BS:                                              \
        switch (bit_width) {                              \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(1, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(2, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(3, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(4, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(5, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(6, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(7, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(8, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(9, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(10, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(11, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(12, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(13, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(14, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(15, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(16, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(17, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(18, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(19, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(20, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(21, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(22, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(23, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(24, BS);     \
            default:                                      \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};                 \
        }                                                 \
        break

inline Kernel<KernelBlockF32Func> select_bmi2_decompress_block_f32(const u8 bit_width, const u32 block_size, bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(64);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(128);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(256);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(512);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlocksF32Func> select_bmi2_decompress_blocks_f32(const u8 bit_width, const u32 block_size, bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(64);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(128);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(256);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(512);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlockF64Func> select_bmi2_decompress_block_f64(const u8 bit_width, const u32 block_size, bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(64);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(128);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(256);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(512);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlocksF64Func> select_bmi2_decompress_blocks_f64(const u8 bit_width, const u32 block_size, bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(64);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(128);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(256);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(512);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

#undef PERNIX_CASE_DECOMPRESS_BLOCK_32
#undef PERNIX_CASE_DECOMPRESS_BLOCKS_32
#undef PERNIX_CASE_DECOMPRESS_BLOCK_64
#undef PERNIX_CASE_DECOMPRESS_BLOCKS_64
#undef PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32
#undef PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32
#undef PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64
#undef PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64
}  // namespace pernix::internal
#endif
#if defined(PERNIX_BUILD_X86_AVX512_VBMI)

namespace pernix::internal {
#define PERNIX_CASE_COMPRESS_BLOCK_32(N, BS) \
    case N:                                  \
        return Kernel<KernelBlockF32Func>(&mm512_compress_block_avx512vbmi<N, BS>)

#define PERNIX_CASE_COMPRESS_BLOCKS_32(N, BS) \
    case N:                                   \
        return Kernel<KernelBlocksF32Func>(&mm512_compress_blocks_avx512vbmi<N, BS>)

#define PERNIX_CASE_COMPRESS_BLOCK_64(N, BS) \
    case N:                                  \
        return Kernel<KernelBlockF64Func>(&mm512_compress_block_avx512vbmi<N, BS>)

#define PERNIX_CASE_COMPRESS_BLOCKS_64(N, BS) \
    case N:                                   \
        return Kernel<KernelBlocksF64Func>(&mm512_compress_blocks_avx512vbmi<N, BS>)

#define PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32(BS)   \
    case BS:                                       \
        switch (bit_width) {                       \
            PERNIX_CASE_COMPRESS_BLOCK_32(1, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(2, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(3, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(4, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(5, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(6, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(7, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(8, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(9, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_32(10, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(11, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(12, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(13, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(14, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(15, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(16, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(17, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(18, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(19, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(20, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(21, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(22, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(23, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_32(24, BS); \
            default:                               \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};    \
        }

#define PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32(BS) \
    case BS:                                            \
        switch (bit_width) {                            \
            PERNIX_CASE_COMPRESS_BLOCKS_32(1, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(2, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(3, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(4, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(5, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(6, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(7, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(8, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(9, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_32(10, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(11, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(12, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(13, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(14, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(15, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(16, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(17, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(18, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(19, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(20, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(21, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(22, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(23, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_32(24, BS);     \
            default:                                    \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};         \
        }

#define PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64(BS)   \
    case BS:                                       \
        switch (bit_width) {                       \
            PERNIX_CASE_COMPRESS_BLOCK_64(1, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(2, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(3, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(4, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(5, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(6, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(7, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(8, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(9, BS);  \
            PERNIX_CASE_COMPRESS_BLOCK_64(10, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(11, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(12, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(13, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(14, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(15, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(16, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(17, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(18, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(19, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(20, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(21, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(22, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(23, BS); \
            PERNIX_CASE_COMPRESS_BLOCK_64(24, BS); \
            default:                               \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};    \
        }

#define PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64(BS) \
    case BS:                                            \
        switch (bit_width) {                            \
            PERNIX_CASE_COMPRESS_BLOCKS_64(1, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(2, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(3, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(4, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(5, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(6, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(7, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(8, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(9, BS);      \
            PERNIX_CASE_COMPRESS_BLOCKS_64(10, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(11, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(12, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(13, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(14, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(15, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(16, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(17, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(18, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(19, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(20, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(21, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(22, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(23, BS);     \
            PERNIX_CASE_COMPRESS_BLOCKS_64(24, BS);     \
            default:                                    \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};         \
        }

inline Kernel<KernelBlockF32Func> select_avx512vbmi_compress_block_f32(const u8 bit_width, const u32 block_size) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32(64);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32(128);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32(256);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32(512);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlocksF32Func> select_avx512vbmi_compress_blocks_f32(const u8 bit_width, const u32 block_size) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32(64);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32(128);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32(256);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32(512);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlockF64Func> select_avx512vbmi_compress_block_f64(const u8 bit_width, const u32 block_size) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64(64);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64(128);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64(256);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64(512);
        PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlocksF64Func> select_avx512vbmi_compress_blocks_f64(const u8 bit_width, const u32 block_size) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64(64);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64(128);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64(256);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64(512);
        PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

#undef PERNIX_CASE_COMPRESS_BLOCK_32
#undef PERNIX_CASE_COMPRESS_BLOCKS_32
#undef PERNIX_CASE_COMPRESS_BLOCK_64
#undef PERNIX_CASE_COMPRESS_BLOCKS_64
#undef PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_32
#undef PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_32
#undef PERNIX_BLOCK_SIZE_COMPRESS_SWITCH_64
#undef PERNIX_BLOCK_SIZE_COMPRESS_BLOCKS_SWITCH_64
}  // namespace pernix::internal

namespace pernix::internal {
#define PERNIX_CASE_DECOMPRESS_BLOCK_32(N, BS)                                                                \
    case N:                                                                                                   \
        if (sign_values)                                                                                      \
            return Kernel<KernelBlockF32Func>(&mm512_decompress_block_avx512vbmi<N, true, BS>); \
        return Kernel<KernelBlockF32Func>(&mm512_decompress_block_avx512vbmi<N, false, BS>)

#define PERNIX_CASE_DECOMPRESS_BLOCKS_32(N, BS)                                                                 \
    case N:                                                                                                     \
        if (sign_values)                                                                                        \
            return Kernel<KernelBlocksF32Func>(&mm512_decompress_blocks_avx512vbmi<N, true, BS>); \
        return Kernel<KernelBlocksF32Func>(&mm512_decompress_blocks_avx512vbmi<N, false, BS>)

#define PERNIX_CASE_DECOMPRESS_BLOCK_64(N, BS)                                                                \
    case N:                                                                                                   \
        if (sign_values)                                                                                      \
            return Kernel<KernelBlockF64Func>(&mm512_decompress_block_avx512vbmi<N, true, BS>); \
        return Kernel<KernelBlockF64Func>(&mm512_decompress_block_avx512vbmi<N, false, BS>)

#define PERNIX_CASE_DECOMPRESS_BLOCKS_64(N, BS)                                                                 \
    case N:                                                                                                     \
        if (sign_values)                                                                                        \
            return Kernel<KernelBlocksF64Func>(&mm512_decompress_blocks_avx512vbmi<N, true, BS>); \
        return Kernel<KernelBlocksF64Func>(&mm512_decompress_blocks_avx512vbmi<N, false, BS>)
#define PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(BS)   \
    case BS:                                         \
        switch (bit_width) {                         \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(1, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(2, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(3, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(4, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(5, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(6, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(7, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(8, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(9, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(10, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(11, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(12, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(13, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(14, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(15, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(16, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(17, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(18, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(19, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(20, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(21, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(22, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(23, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(24, BS); \
            default:                                 \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};      \
        }                                            \
        break

#define PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(BS) \
    case BS:                                              \
        switch (bit_width) {                              \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(1, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(2, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(3, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(4, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(5, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(6, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(7, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(8, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(9, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(10, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(11, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(12, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(13, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(14, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(15, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(16, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(17, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(18, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(19, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(20, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(21, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(22, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(23, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(24, BS);     \
            default:                                      \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};           \
        }                                                 \
        break

#define PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(BS)   \
    case BS:                                         \
        switch (bit_width) {                         \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(1, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(2, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(3, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(4, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(5, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(6, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(7, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(8, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(9, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(10, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(11, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(12, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(13, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(14, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(15, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(16, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(17, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(18, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(19, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(20, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(21, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(22, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(23, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(24, BS); \
            default:                                 \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};      \
        }                                            \
        break

#define PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(BS) \
    case BS:                                              \
        switch (bit_width) {                              \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(1, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(2, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(3, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(4, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(5, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(6, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(7, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(8, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(9, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(10, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(11, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(12, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(13, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(14, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(15, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(16, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(17, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(18, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(19, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(20, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(21, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(22, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(23, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(24, BS);     \
            default:                                      \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};           \
        }                                                 \
        break

inline Kernel<KernelBlockF32Func> select_avx512vbmi_decompress_block_f32(const u8 bit_width, const u32 block_size, bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(64);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(128);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(256);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(512);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlocksF32Func> select_avx512vbmi_decompress_blocks_f32(const u8 bit_width, const u32 block_size, bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(64);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(128);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(256);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(512);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlockF64Func> select_avx512vbmi_decompress_block_f64(const u8 bit_width, const u32 block_size, bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(64);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(128);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(256);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(512);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlocksF64Func> select_avx512vbmi_decompress_blocks_f64(const u8 bit_width, const u32 block_size, bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(64);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(128);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(256);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(512);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

#undef PERNIX_CASE_DECOMPRESS_BLOCK_32
#undef PERNIX_CASE_DECOMPRESS_BLOCKS_32
#undef PERNIX_CASE_DECOMPRESS_BLOCK_64
#undef PERNIX_CASE_DECOMPRESS_BLOCKS_64
#undef PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32
#undef PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32
#undef PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64
#undef PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64
}  // namespace pernix::internal
#endif
#if defined(PERNIX_BUILD_ARM64_NEON)

namespace pernix::internal {
inline Kernel<KernelBlockF32Func> select_neon_compress_block_f32(const u8 bit_width, const u32 block_size) {
    (void)bit_width;
    (void)block_size;
    return {PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION, nullptr};
}

inline Kernel<KernelBlocksF32Func> select_neon_compress_blocks_f32(const u8 bit_width, const u32 block_size) {
    (void)bit_width;
    (void)block_size;
    return {PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION, nullptr};
}

inline Kernel<KernelBlockF64Func> select_neon_compress_block_f64(const u8 bit_width, const u32 block_size) {
    (void)bit_width;
    (void)block_size;
    return {PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION, nullptr};
}

inline Kernel<KernelBlocksF64Func> select_neon_compress_blocks_f64(const u8 bit_width, const u32 block_size) {
    (void)bit_width;
    (void)block_size;
    return {PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION, nullptr};
}
}  // namespace pernix::internal

using pernix::arm64::neon::neon_decompress_block;
using pernix::arm64::neon::neon_decompress_blocks;

namespace pernix::internal {
#define PERNIX_CASE_DECOMPRESS_BLOCK_32(N, BS)                                              \
    case N:                                                                                 \
        if (sign_values)                                                                    \
            return Kernel<KernelBlockF32Func>(&neon_decompress_block<N, true, BS>); \
        return Kernel<KernelBlockF32Func>(&neon_decompress_block<N, false, BS>)

#define PERNIX_CASE_DECOMPRESS_BLOCKS_32(N, BS)                                               \
    case N:                                                                                   \
        if (sign_values)                                                                      \
            return Kernel<KernelBlocksF32Func>(&neon_decompress_blocks<N, true, BS>); \
        return Kernel<KernelBlocksF32Func>(&neon_decompress_blocks<N, false, BS>)

#define PERNIX_CASE_DECOMPRESS_BLOCK_64(N, BS)                                              \
    case N:                                                                                 \
        if (sign_values)                                                                    \
            return Kernel<KernelBlockF64Func>(&neon_decompress_block<N, true, BS>); \
        return Kernel<KernelBlockF64Func>(&neon_decompress_block<N, false, BS>)

#define PERNIX_CASE_DECOMPRESS_BLOCKS_64(N, BS)                                               \
    case N:                                                                                   \
        if (sign_values)                                                                      \
            return Kernel<KernelBlocksF64Func>(&neon_decompress_blocks<N, true, BS>); \
        return Kernel<KernelBlocksF64Func>(&neon_decompress_blocks<N, false, BS>)

#define PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(BS)   \
    case BS:                                         \
        switch (bit_width) {                         \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(1, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(2, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(3, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(4, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(5, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(6, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(7, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(8, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(9, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(10, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(11, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(12, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(13, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(14, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(15, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(16, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(17, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(18, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(19, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(20, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(21, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(22, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(23, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(24, BS); \
            default:                                 \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};            \
        }                                            \
        break

#define PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(BS) \
    case BS:                                              \
        switch (bit_width) {                              \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(1, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(2, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(3, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(4, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(5, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(6, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(7, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(8, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(9, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(10, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(11, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(12, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(13, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(14, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(15, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(16, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(17, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(18, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(19, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(20, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(21, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(22, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(23, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(24, BS);     \
            default:                                      \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};                 \
        }                                                 \
        break

#define PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(BS)   \
    case BS:                                         \
        switch (bit_width) {                         \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(1, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(2, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(3, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(4, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(5, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(6, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(7, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(8, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(9, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(10, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(11, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(12, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(13, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(14, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(15, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(16, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(17, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(18, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(19, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(20, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(21, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(22, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(23, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(24, BS); \
            default:                                 \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};            \
        }                                            \
        break

#define PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(BS) \
    case BS:                                              \
        switch (bit_width) {                              \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(1, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(2, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(3, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(4, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(5, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(6, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(7, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(8, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(9, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(10, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(11, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(12, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(13, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(14, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(15, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(16, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(17, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(18, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(19, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(20, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(21, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(22, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(23, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(24, BS);     \
            default:                                      \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};                 \
        }                                                 \
        break

inline Kernel<KernelBlockF32Func> select_neon_decompress_block_f32(const u8 bit_width, const u32 block_size, bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(64);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(128);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(256);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(512);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlocksF32Func> select_neon_decompress_blocks_f32(const u8 bit_width, const u32 block_size, bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(64);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(128);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(256);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(512);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlockF64Func> select_neon_decompress_block_f64(const u8 bit_width, const u32 block_size, bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(64);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(128);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(256);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(512);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlocksF64Func> select_neon_decompress_blocks_f64(const u8 bit_width, const u32 block_size, bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(64);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(128);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(256);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(512);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

#undef PERNIX_CASE_DECOMPRESS_BLOCK_32
#undef PERNIX_CASE_DECOMPRESS_BLOCKS_32
#undef PERNIX_CASE_DECOMPRESS_BLOCK_64
#undef PERNIX_CASE_DECOMPRESS_BLOCKS_64
#undef PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32
#undef PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32
#undef PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64
#undef PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64
}  // namespace pernix::internal
#endif
#if defined(PERNIX_BUILD_ARM64_SVE2)

namespace pernix::internal {
inline Kernel<KernelBlockF32Func> select_sve2_compress_block_f32(const u8 bit_width, const u32 block_size) {
    (void)bit_width;
    (void)block_size;
    return {PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION, nullptr};
}

inline Kernel<KernelBlocksF32Func> select_sve2_compress_blocks_f32(const u8 bit_width, const u32 block_size) {
    (void)bit_width;
    (void)block_size;
    return {PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION, nullptr};
}

inline Kernel<KernelBlockF64Func> select_sve2_compress_block_f64(const u8 bit_width, const u32 block_size) {
    (void)bit_width;
    (void)block_size;
    return {PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION, nullptr};
}

inline Kernel<KernelBlocksF64Func> select_sve2_compress_blocks_f64(const u8 bit_width, const u32 block_size) {
    (void)bit_width;
    (void)block_size;
    return {PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION, nullptr};
}
}  // namespace pernix::internal

using pernix::arm64::sve2::sve2_decompress_block;
using pernix::arm64::sve2::sve2_decompress_blocks;

namespace pernix::internal {
#define PERNIX_CASE_DECOMPRESS_BLOCK_32(N, BS)                                              \
    case N:                                                                                 \
        if (sign_values)                                                                    \
            return Kernel<KernelBlockF32Func>(&sve2_decompress_block<N, true, BS>); \
        return Kernel<KernelBlockF32Func>(&sve2_decompress_block<N, false, BS>)

#define PERNIX_CASE_DECOMPRESS_BLOCKS_32(N, BS)                                               \
    case N:                                                                                   \
        if (sign_values)                                                                      \
            return Kernel<KernelBlocksF32Func>(&sve2_decompress_blocks<N, true, BS>); \
        return Kernel<KernelBlocksF32Func>(&sve2_decompress_blocks<N, false, BS>)

#define PERNIX_CASE_DECOMPRESS_BLOCK_64(N, BS)                                              \
    case N:                                                                                 \
        if (sign_values)                                                                    \
            return Kernel<KernelBlockF64Func>(&sve2_decompress_block<N, true, BS>); \
        return Kernel<KernelBlockF64Func>(&sve2_decompress_block<N, false, BS>)

#define PERNIX_CASE_DECOMPRESS_BLOCKS_64(N, BS)                                               \
    case N:                                                                                   \
        if (sign_values)                                                                      \
            return Kernel<KernelBlocksF64Func>(&sve2_decompress_blocks<N, true, BS>); \
        return Kernel<KernelBlocksF64Func>(&sve2_decompress_blocks<N, false, BS>)

#define PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(BS)   \
    case BS:                                         \
        switch (bit_width) {                         \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(1, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(2, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(3, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(4, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(5, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(6, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(7, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(8, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(9, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(10, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(11, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(12, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(13, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(14, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(15, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(16, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(17, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(18, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(19, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(20, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(21, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(22, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(23, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_32(24, BS); \
            default:                                 \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};            \
        }                                            \
        break

#define PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(BS) \
    case BS:                                              \
        switch (bit_width) {                              \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(1, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(2, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(3, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(4, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(5, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(6, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(7, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(8, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(9, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(10, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(11, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(12, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(13, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(14, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(15, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(16, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(17, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(18, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(19, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(20, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(21, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(22, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(23, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_32(24, BS);     \
            default:                                      \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};                 \
        }                                                 \
        break

#define PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(BS)   \
    case BS:                                         \
        switch (bit_width) {                         \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(1, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(2, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(3, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(4, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(5, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(6, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(7, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(8, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(9, BS);  \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(10, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(11, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(12, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(13, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(14, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(15, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(16, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(17, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(18, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(19, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(20, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(21, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(22, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(23, BS); \
            PERNIX_CASE_DECOMPRESS_BLOCK_64(24, BS); \
            default:                                 \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};            \
        }                                            \
        break

#define PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(BS) \
    case BS:                                              \
        switch (bit_width) {                              \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(1, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(2, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(3, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(4, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(5, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(6, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(7, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(8, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(9, BS);      \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(10, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(11, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(12, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(13, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(14, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(15, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(16, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(17, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(18, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(19, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(20, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(21, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(22, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(23, BS);     \
            PERNIX_CASE_DECOMPRESS_BLOCKS_64(24, BS);     \
            default:                                      \
                return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};                 \
        }                                                 \
        break

inline Kernel<KernelBlockF32Func> select_sve2_decompress_block_f32(const u8 bit_width, const u32 block_size, bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(64);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(128);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(256);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(512);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlocksF32Func> select_sve2_decompress_blocks_f32(const u8 bit_width, const u32 block_size, bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(64);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(128);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(256);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(512);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlockF64Func> select_sve2_decompress_block_f64(const u8 bit_width, const u32 block_size, bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(64);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(128);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(256);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(512);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

inline Kernel<KernelBlocksF64Func> select_sve2_decompress_blocks_f64(const u8 bit_width, const u32 block_size, bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(64);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(128);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(256);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(512);
        PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64(1024);
        default:
            return {PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH, nullptr};
    }
}

#undef PERNIX_CASE_DECOMPRESS_BLOCK_32
#undef PERNIX_CASE_DECOMPRESS_BLOCKS_32
#undef PERNIX_CASE_DECOMPRESS_BLOCK_64
#undef PERNIX_CASE_DECOMPRESS_BLOCKS_64
#undef PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_32
#undef PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_32
#undef PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_64
#undef PERNIX_BLOCK_SIZE_DECOMPRESS_SWITCH_BLOCKS_64
}  // namespace pernix::internal
#endif

#include <pernix/dispatch/registry_impl.h>

#endif  // PERNIX_SELECT_IMPL_H
