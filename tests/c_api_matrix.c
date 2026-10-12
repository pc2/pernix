#include <pernix/pernix.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    block_count = 3
};

static int32_t value_for_index(const u8 bit_width, const u32 index) {
    if (bit_width == 1) {
        return (index & 1U) == 0U ? 0 : 1;
    }

    const int32_t minimum = -(INT32_C(1) << (bit_width - 1U));
    const int32_t maximum = (INT32_C(1) << (bit_width - 1U)) - 1;
    switch (index % 5U) {
        case 0:
            return minimum;
        case 1:
            return maximum;
        case 2:
            return -1;
        case 3:
            return 0;
        default:
            return 1;
    }
}

#define DEFINE_FLOAT_MATRIX_CHECK(SUFFIX, TYPE, COMPRESS_BLOCK, COMPRESS_BLOCKS, DECOMPRESS_BLOCK, DECOMPRESS_BLOCKS)                     \
    static int check_##SUFFIX(const pernix_dispatcher* dispatcher, const u8 bit_width, const u32 block_size) {                            \
        const u32 elements_per_block = (block_size * 8U) / bit_width;                                                                     \
        const u32 elements           = elements_per_block * block_count;                                                                  \
        const size_t packed_size     = (size_t)block_size * block_count;                                                                  \
        TYPE* input                  = (TYPE*)calloc(elements, sizeof(TYPE));                                                             \
        TYPE* restored               = (TYPE*)calloc(elements, sizeof(TYPE));                                                             \
        TYPE* unsigned_restored      = (TYPE*)calloc(elements, sizeof(TYPE));                                                             \
        TYPE* one_block_restored     = (TYPE*)calloc(elements_per_block, sizeof(TYPE));                                                   \
        u8* packed                   = (u8*)calloc(packed_size, sizeof(u8));                                                              \
        u8* one_block                = (u8*)calloc(block_size, sizeof(u8));                                                               \
        int result                   = 1;                                                                                                 \
        if (input == NULL || restored == NULL || unsigned_restored == NULL || one_block_restored == NULL || packed == NULL ||             \
            one_block == NULL) {                                                                                                          \
            fprintf(stderr, "allocation failed for %s, bit_width=%u, block_size=%u\n", #SUFFIX, (unsigned)bit_width,                      \
                    (unsigned)block_size);                                                                                                \
            goto cleanup;                                                                                                                 \
        }                                                                                                                                 \
        for (u32 index = 0; index < elements; ++index) {                                                                                  \
            input[index] = (TYPE)value_for_index(bit_width, index);                                                                       \
        }                                                                                                                                 \
        if (COMPRESS_BLOCKS(dispatcher, bit_width, block_size, input, (TYPE)1, packed, block_count) != PERNIX_STATUS_OK) {                \
            fprintf(stderr, "compress blocks failed for %s, bit_width=%u, block_size=%u\n", #SUFFIX, (unsigned)bit_width,                 \
                    (unsigned)block_size);                                                                                                \
            goto cleanup;                                                                                                                 \
        }                                                                                                                                 \
        for (u32 block = 0; block < block_count; ++block) {                                                                               \
            const size_t input_offset  = (size_t)block * elements_per_block;                                                              \
            const size_t packed_offset = (size_t)block * block_size;                                                                      \
            if (COMPRESS_BLOCK(dispatcher, bit_width, block_size, input + input_offset, (TYPE)1, one_block) != PERNIX_STATUS_OK ||        \
                memcmp(one_block, packed + packed_offset, block_size) != 0) {                                                             \
                fprintf(stderr, "single/block compression mismatch for %s, bit_width=%u, block_size=%u, block=%u\n", #SUFFIX,             \
                        (unsigned)bit_width, (unsigned)block_size, (unsigned)block);                                                      \
                goto cleanup;                                                                                                             \
            }                                                                                                                             \
            if (DECOMPRESS_BLOCK(dispatcher, bit_width, block_size, packed + packed_offset, (TYPE)1, one_block_restored, true) !=         \
                PERNIX_STATUS_OK) {                                                                                                       \
                fprintf(stderr, "single-block decompression failed for %s, bit_width=%u, block_size=%u, block=%u\n", #SUFFIX,             \
                        (unsigned)bit_width, (unsigned)block_size, (unsigned)block);                                                      \
                goto cleanup;                                                                                                             \
            }                                                                                                                             \
            for (u32 element = 0; element < elements_per_block; ++element) {                                                              \
                if (one_block_restored[element] != input[input_offset + element]) {                                                       \
                    fprintf(stderr, "single-block decode mismatch for %s, bit_width=%u, block_size=%u, block=%u, element=%u\n", #SUFFIX,  \
                            (unsigned)bit_width, (unsigned)block_size, (unsigned)block, (unsigned)element);                               \
                    goto cleanup;                                                                                                         \
                }                                                                                                                         \
            }                                                                                                                             \
        }                                                                                                                                 \
        if (DECOMPRESS_BLOCKS(dispatcher, bit_width, block_size, packed, (TYPE)1, restored, block_count, true) != PERNIX_STATUS_OK ||     \
            DECOMPRESS_BLOCKS(dispatcher, bit_width, block_size, packed, (TYPE)1, unsigned_restored, block_count, false) !=               \
                PERNIX_STATUS_OK) {                                                                                                       \
            fprintf(stderr, "decompress blocks failed for %s, bit_width=%u, block_size=%u\n", #SUFFIX, (unsigned)bit_width,               \
                    (unsigned)block_size);                                                                                                \
            goto cleanup;                                                                                                                 \
        }                                                                                                                                 \
        for (u32 index = 0; index < elements; ++index) {                                                                                  \
            const int32_t signed_code = value_for_index(bit_width, index);                                                                \
            const u32 mask            = (1U << bit_width) - 1U;                                                                           \
            const u32 unsigned_code   = (u32)signed_code & mask;                                                                          \
            if (restored[index] != (TYPE)signed_code || unsigned_restored[index] != (TYPE)unsigned_code) {                                \
                fprintf(stderr, "decoded value mismatch for %s, bit_width=%u, block_size=%u, element=%u\n", #SUFFIX, (unsigned)bit_width, \
                        (unsigned)block_size, (unsigned)index);                                                                           \
                goto cleanup;                                                                                                             \
            }                                                                                                                             \
        }                                                                                                                                 \
        result = 0;                                                                                                                       \
    cleanup:                                                                                                                              \
        free(one_block);                                                                                                                  \
        free(packed);                                                                                                                     \
        free(one_block_restored);                                                                                                         \
        free(unsigned_restored);                                                                                                          \
        free(restored);                                                                                                                   \
        free(input);                                                                                                                      \
        return result;                                                                                                                    \
    }

DEFINE_FLOAT_MATRIX_CHECK(f32, float, pernix_compress_block_f32, pernix_compress_blocks_f32, pernix_decompress_block_f32,
                          pernix_decompress_blocks_f32)
DEFINE_FLOAT_MATRIX_CHECK(f64, double, pernix_compress_block_f64, pernix_compress_blocks_f64, pernix_decompress_block_f64,
                          pernix_decompress_blocks_f64)

#undef DEFINE_FLOAT_MATRIX_CHECK

int main(void) {
    static const u32 block_sizes[]    = {64, 128, 256, 512, 1024};
    pernix_dispatcher* dispatcher     = NULL;
    const pernix_status create_status = pernix_dispatcher_create(PERNIX_BACKEND_FALLBACK, &dispatcher);
    if (create_status != PERNIX_STATUS_OK) {
        fprintf(stderr, "fallback dispatcher creation failed: %s\n", pernix_status_string(create_status));
        return 1;
    }

    for (u32 size_index = 0; size_index < (u32)(sizeof(block_sizes) / sizeof(block_sizes[0])); ++size_index) {
        for (u8 bit_width = pernix_min_bit_width(); bit_width <= pernix_max_bit_width(); ++bit_width) {
            if (check_f32(dispatcher, bit_width, block_sizes[size_index]) != 0 ||
                check_f64(dispatcher, bit_width, block_sizes[size_index]) != 0) {
                pernix_dispatcher_destroy(dispatcher);
                return 2;
            }
        }
    }

    pernix_dispatcher_destroy(dispatcher);
    return 0;
}
