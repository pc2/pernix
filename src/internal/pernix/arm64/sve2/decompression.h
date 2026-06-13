#ifndef PERNIX_ARM64_SVE2_DECOMPRESSION_H
#define PERNIX_ARM64_SVE2_DECOMPRESSION_H

#include <pernix/arm64/sve2/tables.h>
#include <pernix/arm64/sve2/unpacking.h>
#include <pernix/simd_compat.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace pernix::arm64::sve2 {
namespace internal {
template <uint8_t BIT_WIDTH>
[[nodiscard]] __always_inline constexpr uint32_t packed_bytes(const uint32_t elements) {
    return (elements * BIT_WIDTH + 7) / 8;
}

[[nodiscard]] __always_inline svuint8_t sve2_load_packed_bytes(const uint8_t* __restrict__ input,
                                                               const uint32_t bytes) {
    const svbool_t pg = svwhilelt_b8(uint64_t{0}, static_cast<uint64_t>(bytes));
    return svld1_u8(pg, input);
}

template <bool SIGN_VALUES>
__always_inline void sve2_store_dequantized_i8_f32(svint8_t values, const svfloat32_t scale_v,
                                                   float_t* __restrict__ output,
                                                   const uint32_t count) {
    alignas(64) std::vector<int8_t> temp(svcntb());

    svst1_s8(svptrue_b8(), temp.data(), values);

    uint32_t offset = 0;
    while (offset < count) {
        const svbool_t pg = svwhilelt_b32(static_cast<uint64_t>(offset), static_cast<uint64_t>(count));

        svfloat32_t dequantized;

        if constexpr (SIGN_VALUES) {
            const svint32_t widened = svld1sb_s32(pg, temp.data() + offset);
            dequantized             = svmul_f32_x(pg, svcvt_f32_s32_x(pg, widened), scale_v);
        } else {
            const svuint32_t widened = svld1ub_u32(pg, reinterpret_cast<const uint8_t*>(temp.data() + offset));
            dequantized              = svmul_f32_x(pg, svcvt_f32_u32_x(pg, widened), scale_v);
        }

        svst1_f32(pg, output + offset, dequantized);

        offset += static_cast<uint32_t>(svcntw());
    }
}

template <bool SIGN_VALUES>
__always_inline void sve2_store_dequantized_i8_f64(svint8_t values, const double_t scale,
                                                   double_t* __restrict__ output,
                                                   const uint32_t count) {
    std::vector<int8_t> temp(svcntb());

    svst1_s8(svptrue_b8(), temp.data(), values);

    for (uint32_t i = 0; i < count; ++i) {
        if constexpr (SIGN_VALUES) {
            output[i] = static_cast<double_t>(temp[i]) * scale;
        } else {
            output[i] = static_cast<double_t>(static_cast<uint8_t>(temp[i])) * scale;
        }
    }
}

template <bool SIGN_VALUES>
__always_inline void sve2_store_dequantized_i16_f32(svint16_t values, const svfloat32_t scale_v,
                                                    float_t* __restrict__ output,
                                                    const uint32_t count) {
    alignas(64) std::vector<int16_t> temp(svcnth());

    svst1_s16(svptrue_b16(), temp.data(), values);

    uint32_t offset = 0;
    while (offset < count) {
        const svbool_t pg = svwhilelt_b32(static_cast<uint64_t>(offset), static_cast<uint64_t>(count));

        svfloat32_t dequantized;
        if constexpr (SIGN_VALUES) {
            const svint32_t widened = svld1sh_s32(pg, temp.data() + offset);
            dequantized             = svmul_f32_x(pg, svcvt_f32_s32_x(pg, widened), scale_v);
        } else {
            const svuint32_t widened =
                svld1uh_u32(pg, reinterpret_cast<const uint16_t*>(temp.data() + offset));
            dequantized = svmul_f32_x(pg, svcvt_f32_u32_x(pg, widened), scale_v);
        }

        svst1_f32(pg, output + offset, dequantized);

        offset += static_cast<uint32_t>(svcntw());
    }
}

template <bool SIGN_VALUES>
__always_inline void sve2_store_dequantized_i16_f64(svint16_t values, const double_t scale,
                                                    double_t* __restrict__ output,
                                                    const uint32_t count) {
    std::vector<int16_t> temp(svcnth());

    svst1_s16(svptrue_b16(), temp.data(), values);

    for (uint32_t i = 0; i < count; ++i) {
        if constexpr (SIGN_VALUES) {
            output[i] = static_cast<double_t>(temp[i]) * scale;
        } else {
            output[i] = static_cast<double_t>(static_cast<uint16_t>(temp[i])) * scale;
        }
    }
}

template <bool SIGN_VALUES>
__always_inline void sve2_store_dequantized_i32_f32(svint32_t values, const svfloat32_t scale_v,
                                                    float_t* __restrict__ output,
                                                    const uint32_t count) {
    const svbool_t pg = svwhilelt_b32(uint64_t{0}, static_cast<uint64_t>(count));

    svfloat32_t dequantized;
    if constexpr (SIGN_VALUES) {
        dequantized = svmul_f32_x(pg, svcvt_f32_s32_x(pg, values), scale_v);
    } else {
        dequantized = svmul_f32_x(pg, svcvt_f32_u32_x(pg, svreinterpret_u32_s32(values)), scale_v);
    }

    svst1_f32(pg, output, dequantized);
}

template <bool SIGN_VALUES>
__always_inline void sve2_store_dequantized_i32_f64(svint32_t values, const double_t scale,
                                                    double_t* __restrict__ output,
                                                    const uint32_t count) {
    std::vector<int32_t> temp(svcntw());

    svst1_s32(svptrue_b32(), temp.data(), values);

    for (uint32_t i = 0; i < count; ++i) {
        if constexpr (SIGN_VALUES) {
            output[i] = static_cast<double_t>(temp[i]) * scale;
        } else {
            output[i] = static_cast<double_t>(static_cast<uint32_t>(temp[i])) * scale;
        }
    }
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8) && (BLOCK_SIZE % 32 == 0)
__always_inline int sve2_decompress_block_1to8(const uint8_t* __restrict__ input, const float_t scale,
                                               float_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    const auto lanes            = static_cast<uint32_t>(svcntb());
    uint32_t input_bit_offset   = 0;
    uint32_t processed_elements = 0;

    const svfloat32_t scale_v = svdup_n_f32(scale);

    const table_unpacking<BIT_WIDTH> table;
    const svuint8_t permute = table.permute();
    const svuint8_t shift   = table.shift();
    svuint8_t spill_permute = svdup_n_u8(0);
    svuint8_t spill_shift   = svdup_n_u8(0);
    if constexpr (BIT_WIDTH == 3 || BIT_WIDTH == 5 || BIT_WIDTH == 6 || BIT_WIDTH == 7) {
        spill_permute = table.spill_permute();
        spill_shift   = table.spill_shift();
    }

    while (processed_elements < elements_per_block) {
        const uint32_t count = std::min(elements_per_block - processed_elements, lanes);

        const uint32_t bytes       = packed_bytes<BIT_WIDTH>(count);
        const uint8_t* chunk_input = input + input_bit_offset / 8;

        const svuint8_t source  = sve2_load_packed_bytes(chunk_input, bytes);
        const svint8_t unpacked = sve2_unpack_epi8_1to8<BIT_WIDTH, SIGN_VALUES
            >
            (source, permute, shift, spill_permute, spill_shift);

        sve2_store_dequantized_i8_f32<SIGN_VALUES>(unpacked, scale_v, output + processed_elements, count);

        processed_elements += count;
        input_bit_offset   += count * BIT_WIDTH;
    }

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16) && (BLOCK_SIZE % 32 == 0)
__always_inline int sve2_decompress_block_9to16(const uint8_t* __restrict__ input, const float_t scale,
                                                float_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    const auto lanes            = static_cast<uint32_t>(svcnth());
    uint32_t input_bit_offset   = 0;
    uint32_t processed_elements = 0;

    const svfloat32_t scale_v = svdup_n_f32(scale);

    const table_unpacking<BIT_WIDTH> table;
    const svuint8_t permute = table.permute();
    const svuint16_t shift  = table.shift();
    svuint8_t spill_permute = svdup_n_u8(0);
    svuint16_t spill_shift  = svdup_n_u16(0);
    if constexpr (BIT_WIDTH == 11 || BIT_WIDTH == 13 || BIT_WIDTH == 14 || BIT_WIDTH == 15) {
        spill_permute = table.spill_permute();
        spill_shift   = table.spill_shift();
    }

    while (processed_elements < elements_per_block) {
        const uint32_t count = std::min(elements_per_block - processed_elements, lanes);

        const uint32_t bytes       = packed_bytes<BIT_WIDTH>(count);
        const uint8_t* chunk_input = input + input_bit_offset / 8;

        const svuint8_t source   = sve2_load_packed_bytes(chunk_input, bytes);
        const svint16_t unpacked =
            sve2_unpack_epi16_9to16<BIT_WIDTH, SIGN_VALUES
            >
            (svreinterpret_u16_u8(source), permute, shift, spill_permute, spill_shift);

        sve2_store_dequantized_i16_f32<SIGN_VALUES>(unpacked, scale_v, output + processed_elements, count);

        processed_elements += count;
        input_bit_offset   += count * BIT_WIDTH;
    }

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int sve2_decompress_block_17to24(const uint8_t* __restrict__ input, const float_t scale,
                                                 float_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    const auto lanes            = static_cast<uint32_t>(svcntw());
    uint32_t input_bit_offset   = 0;
    uint32_t processed_elements = 0;

    const svfloat32_t scale_v = svdup_n_f32(scale);

    while (processed_elements < elements_per_block) {
        const uint32_t count = std::min(elements_per_block - processed_elements, lanes);

        const uint8_t* chunk_input = input + input_bit_offset / 8;
        const uint32_t bit_offset  = input_bit_offset % 8;
        const uint32_t bytes       = (bit_offset + count * BIT_WIDTH + 7u) / 8u;

        const svuint8_t source = sve2_load_packed_bytes(chunk_input, bytes);
        svint32_t unpacked;
        if (bit_offset == 0) {
            unpacked = sve2_unpack_epi32_17to24<BIT_WIDTH, SIGN_VALUES, 0>(source);
        } else {
            unpacked = sve2_unpack_epi32_17to24<BIT_WIDTH, SIGN_VALUES, 4>(source);
        }

        sve2_store_dequantized_i32_f32<SIGN_VALUES>(unpacked, scale_v, output + processed_elements, count);

        processed_elements += count;
        input_bit_offset   += count * BIT_WIDTH;
    }

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 8) && (BLOCK_SIZE % 32 == 0)
__always_inline int sve2_decompress_block_1to8(const uint8_t* __restrict__ input, const double_t scale,
                                               double_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    const auto lanes            = static_cast<uint32_t>(svcntb());
    uint32_t input_bit_offset   = 0;
    uint32_t processed_elements = 0;

    const table_unpacking<BIT_WIDTH> table;
    const svuint8_t permute = table.permute();
    const svuint8_t shift   = table.shift();
    svuint8_t spill_permute = svdup_n_u8(0);
    svuint8_t spill_shift   = svdup_n_u8(0);
    if constexpr (BIT_WIDTH == 3 || BIT_WIDTH == 5 || BIT_WIDTH == 6 || BIT_WIDTH == 7) {
        spill_permute = table.spill_permute();
        spill_shift   = table.spill_shift();
    }

    while (processed_elements < elements_per_block) {
        const uint32_t count = std::min(elements_per_block - processed_elements, lanes);

        const uint32_t bytes       = packed_bytes<BIT_WIDTH>(count);
        const uint8_t* chunk_input = input + input_bit_offset / 8;

        const svuint8_t source  = sve2_load_packed_bytes(chunk_input, bytes);
        const svint8_t unpacked = sve2_unpack_epi8_1to8<BIT_WIDTH, SIGN_VALUES
            >
            (source, permute, shift, spill_permute, spill_shift);

        sve2_store_dequantized_i8_f64<SIGN_VALUES>(unpacked, scale, output + processed_elements, count);

        processed_elements += count;
        input_bit_offset   += count * BIT_WIDTH;
    }

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 9 && BIT_WIDTH <= 16) && (BLOCK_SIZE % 32 == 0)
__always_inline int sve2_decompress_block_9to16(const uint8_t* __restrict__ input, const double_t scale,
                                                double_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    const auto lanes            = static_cast<uint32_t>(svcnth());
    uint32_t input_bit_offset   = 0;
    uint32_t processed_elements = 0;

    const table_unpacking<BIT_WIDTH> table;
    const svuint8_t permute = table.permute();
    const svuint16_t shift  = table.shift();
    svuint8_t spill_permute = svdup_n_u8(0);
    svuint16_t spill_shift  = svdup_n_u16(0);
    if constexpr (BIT_WIDTH == 11 || BIT_WIDTH == 13 || BIT_WIDTH == 14 || BIT_WIDTH == 15) {
        spill_permute = table.spill_permute();
        spill_shift   = table.spill_shift();
    }

    while (processed_elements < elements_per_block) {
        const uint32_t count = std::min(elements_per_block - processed_elements, lanes);

        const uint32_t bytes       = packed_bytes<BIT_WIDTH>(count);
        const uint8_t* chunk_input = input + input_bit_offset / 8;

        const svuint8_t source   = sve2_load_packed_bytes(chunk_input, bytes);
        const svint16_t unpacked =
            sve2_unpack_epi16_9to16<BIT_WIDTH, SIGN_VALUES
            >
            (svreinterpret_u16_u8(source), permute, shift, spill_permute, spill_shift);

        sve2_store_dequantized_i16_f64<SIGN_VALUES>(unpacked, scale, output + processed_elements, count);

        processed_elements += count;
        input_bit_offset   += count * BIT_WIDTH;
    }

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 17 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
__always_inline int sve2_decompress_block_17to24(const uint8_t* __restrict__ input, const double_t scale,
                                                 double_t* __restrict__ output) {
    constexpr uint32_t elements_per_block = (BLOCK_SIZE * 8) / BIT_WIDTH;

    const auto lanes            = static_cast<uint32_t>(svcntw());
    uint32_t input_bit_offset   = 0;
    uint32_t processed_elements = 0;

    while (processed_elements < elements_per_block) {
        const uint32_t count = std::min(elements_per_block - processed_elements, lanes);

        const uint8_t* chunk_input = input + input_bit_offset / 8;
        const uint32_t bit_offset  = input_bit_offset % 8;
        const uint32_t bytes       = (bit_offset + count * BIT_WIDTH + 7u) / 8u;

        const svuint8_t source = sve2_load_packed_bytes(chunk_input, bytes);
        svint32_t unpacked;
        if (bit_offset == 0) {
            unpacked = sve2_unpack_epi32_17to24<BIT_WIDTH, SIGN_VALUES, 0>(source);
        } else {
            unpacked = sve2_unpack_epi32_17to24<BIT_WIDTH, SIGN_VALUES, 4>(source);
        }

        sve2_store_dequantized_i32_f64<SIGN_VALUES>(unpacked, scale, output + processed_elements, count);

        processed_elements += count;
        input_bit_offset   += count * BIT_WIDTH;
    }

    return 0;
}
} // namespace internal

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int sve2_decompress_block(const void* __restrict__ input, const float_t scale, void* __restrict__ output) {
    const auto* typed_input = static_cast<const uint8_t*>(input);
    auto* typed_output      = static_cast<float_t*>(output);
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
        return internal::sve2_decompress_block_1to8<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(typed_input, scale, typed_output);
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
        return internal::sve2_decompress_block_9to16<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(typed_input, scale, typed_output);
    } else if constexpr (BIT_WIDTH >= 17 && BIT_WIDTH <= 24) {
        return internal::sve2_decompress_block_17to24<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(typed_input, scale, typed_output);
    }
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int sve2_decompress_block(const void* __restrict__ input, const double_t scale,
                          void* __restrict__ output) {
    const auto* typed_input = static_cast<const uint8_t*>(input);
    auto* typed_output      = static_cast<double_t*>(output);
    if constexpr (BIT_WIDTH >= 1 && BIT_WIDTH <= 8) {
        return internal::sve2_decompress_block_1to8<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(typed_input, scale, typed_output);
    } else if constexpr (BIT_WIDTH >= 9 && BIT_WIDTH <= 16) {
        return internal::sve2_decompress_block_9to16<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(typed_input, scale, typed_output);
    } else if constexpr (BIT_WIDTH >= 17 && BIT_WIDTH <= 24) {
        return internal::sve2_decompress_block_17to24<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(typed_input, scale, typed_output);
    }
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int sve2_decompress_blocks(const void* __restrict__ input, const float_t scale, void* __restrict__ output,
                           const uint32_t blocks) {
    const auto* typed_input    = static_cast<const uint8_t*>(input);
    auto* typed_output         = static_cast<float_t*>(output);
    const uint8_t* block_input = typed_input;
    float_t* block_output      = typed_output;

    for (uint32_t block = 0; block < blocks; ++block) {
        sve2_decompress_block<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(block_input, scale, block_output);
        block_input  += BLOCK_SIZE;
        block_output += (BLOCK_SIZE * 8) / BIT_WIDTH;
    }

    return 0;
}

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, uint32_t BLOCK_SIZE>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && (BLOCK_SIZE % 32 == 0)
int sve2_decompress_blocks(const void* __restrict__ input, const double_t scale, void* __restrict__ output,
                           const uint32_t blocks) {
    const auto* typed_input    = static_cast<const uint8_t*>(input);
    auto* typed_output         = static_cast<double_t*>(output);
    const uint8_t* block_input = typed_input;
    double_t* block_output     = typed_output;

    for (uint32_t block = 0; block < blocks; ++block) {
        sve2_decompress_block<BIT_WIDTH, SIGN_VALUES, BLOCK_SIZE>(block_input, scale, block_output);
        block_input  += BLOCK_SIZE;
        block_output += (BLOCK_SIZE * 8) / BIT_WIDTH;
    }

    return 0;
}
} // namespace pernix::arm64::sve2

#endif  // PERNIX_ARM64_SVE2_DECOMPRESSION_H
