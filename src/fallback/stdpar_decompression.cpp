#include <pernix/dispatch/select.h>
#include <pernix/fallback/stdpar_decompression.h>

namespace pernix::internal {
#define PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32(N, BS)                                                                  \
    case N:                                                                                                            \
        if (sign_values)                                                                                               \
            return Kernel<KernelBlockF32Func>("fallback_stdpar", &decompress_block_fallback_stdpar<N, true, BS, f32>); \
        return Kernel<KernelBlockF32Func>("fallback_stdpar", &decompress_block_fallback_stdpar<N, false, BS, f32>)

#define PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32(N, BS)                                                                   \
    case N:                                                                                                              \
        if (sign_values)                                                                                                 \
            return Kernel<KernelBlocksF32Func>("fallback_stdpar", &decompress_blocks_fallback_stdpar<N, true, BS, f32>); \
        return Kernel<KernelBlocksF32Func>("fallback_stdpar", &decompress_blocks_fallback_stdpar<N, false, BS, f32>)

#define PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64(N, BS)                                                                  \
    case N:                                                                                                            \
        if (sign_values)                                                                                               \
            return Kernel<KernelBlockF64Func>("fallback_stdpar", &decompress_block_fallback_stdpar<N, true, BS, f64>); \
        return Kernel<KernelBlockF64Func>("fallback_stdpar", &decompress_block_fallback_stdpar<N, false, BS, f64>)

#define PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64(N, BS)                                                                   \
    case N:                                                                                                              \
        if (sign_values)                                                                                                 \
            return Kernel<KernelBlocksF64Func>("fallback_stdpar", &decompress_blocks_fallback_stdpar<N, true, BS, f64>); \
        return Kernel<KernelBlocksF64Func>("fallback_stdpar", &decompress_blocks_fallback_stdpar<N, false, BS, f64>)

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
                return {"fallback_stdpar", nullptr};        \
        }

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
                return {"fallback_stdpar", nullptr};             \
        }

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
                return {"fallback_stdpar", nullptr};        \
        }

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
                return {"fallback_stdpar", nullptr};             \
        }

Kernel<KernelBlockF32Func> select_fallback_stdpar_decompress_block_f32(const u8 bit_width, const u32 block_size, const bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_SWITCH_32(64);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_SWITCH_32(128);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_SWITCH_32(256);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_SWITCH_32(512);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_SWITCH_32(1024);
        default:
            return {"fallback_stdpar", nullptr};
    }
}

Kernel<KernelBlocksF32Func> select_fallback_stdpar_decompress_blocks_f32(const u8 bit_width, const u32 block_size, const bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_BLOCKS_SWITCH_32(64);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_BLOCKS_SWITCH_32(128);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_BLOCKS_SWITCH_32(256);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_BLOCKS_SWITCH_32(512);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_BLOCKS_SWITCH_32(1024);
        default:
            return {"fallback_stdpar", nullptr};
    }
}

Kernel<KernelBlockF64Func> select_fallback_stdpar_decompress_block_f64(const u8 bit_width, const u32 block_size, const bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_SWITCH_64(64);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_SWITCH_64(128);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_SWITCH_64(256);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_SWITCH_64(512);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_SWITCH_64(1024);
        default:
            return {"fallback_stdpar", nullptr};
    }
}

Kernel<KernelBlocksF64Func> select_fallback_stdpar_decompress_blocks_f64(const u8 bit_width, const u32 block_size, const bool sign_values) {
    switch (block_size) {
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_BLOCKS_SWITCH_64(64);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_BLOCKS_SWITCH_64(128);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_BLOCKS_SWITCH_64(256);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_BLOCKS_SWITCH_64(512);
        PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_BLOCKS_SWITCH_64(1024);
        default:
            return {"fallback_stdpar", nullptr};
    }
}

#undef PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_32
#undef PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_32
#undef PERNIX_CASE_STDPAR_DECOMPRESS_BLOCK_64
#undef PERNIX_CASE_STDPAR_DECOMPRESS_BLOCKS_64
#undef PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_SWITCH_32
#undef PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_BLOCKS_SWITCH_32
#undef PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_SWITCH_64
#undef PERNIX_BLOCK_SIZE_STDPAR_DECOMPRESS_BLOCKS_SWITCH_64
}  // namespace pernix::internal
