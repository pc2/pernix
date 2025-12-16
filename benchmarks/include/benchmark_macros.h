#ifndef LIBCOMPRESSION_BENCHMARK_MACROS_H
#define LIBCOMPRESSION_BENCHMARK_MACROS_H

#include <functional>

#include <libcompression.h>

#define BENCHMARK_DECOMPRESS_BLOCKS_REGISTER(name) \
    BENCHMARK(BM_##name)->RangeMultiplier(2)->Range(1 << 0, 1 << 22)

#define BENCHMARK_DECOMPRESS_BLOCKS_FUNCTION(name, func, N, MEM)                \
    static void BM_##name##_##MEM##_##N(benchmark::State& state) {              \
        BM_decompress_blocks<N, true, MEM>(state, func<N>);                     \
    }                                                                           \
    BENCHMARK_DECOMPRESS_BLOCKS_REGISTER(name##_##MEM##_##N);


template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true, bool DISABLE_MEM = false>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
__always_inline void BM_decompress_blocks(benchmark::State& state,
                                          const std::function<int(const uint8_t*, float_t, float_t*)>& decompress_function) {
    const size_t elements_per_block = 512 / BIT_WIDTH;
    const auto number_of_blocks     = static_cast<size_t>(state.range(0));
    const auto benchmark_set        = create_decompression_benchmark_set<BIT_WIDTH, SIGN_VALUES>(static_cast<int64_t>(number_of_blocks));

    const size_t bytes_read_per_block    = (elements_per_block * BIT_WIDTH + 7) / 8;
    const size_t bytes_written_per_block = elements_per_block * sizeof(float_t);

    if constexpr (DISABLE_MEM) {
        alignas(64) thread_local uint8_t dummy_input[32]                      = {0};
        alignas(64) thread_local float_t dummy_output[elements_per_block * 1] = {0};

        for (auto _ : state) {
            for (uint32_t block = 0; block < number_of_blocks; block++) {
                decompress_function(dummy_input, benchmark_set.scales[block], dummy_output);
                asm volatile("" :: "r"(dummy_input), "r"(dummy_output));
            }
        }
    } else {
        for (auto _ : state) {
            alignas(64) const uint8_t* block_input = benchmark_set.input_ptr;
            alignas(64) float_t* block_output      = benchmark_set.output_ptr;

            for (uint32_t block = 0; block < number_of_blocks; block++) {
                decompress_function(block_input, benchmark_set.scales[block], block_output);
                benchmark::DoNotOptimize(block_input);
                benchmark::DoNotOptimize(benchmark_set.scales.data());
                benchmark::DoNotOptimize(block_output);
                block_input += 32;
                block_output += elements_per_block;
                benchmark::ClobberMemory();
            }
        }
    }
    const auto iters  = static_cast<uint64_t>(state.iterations());
    const auto blocks = static_cast<uint64_t>(number_of_blocks);

    const uint64_t bytes_per_block = bytes_read_per_block + bytes_written_per_block;
    state.SetBytesProcessed(static_cast<int64_t>(iters * blocks * bytes_per_block));

    const auto items = static_cast<int64_t>(iters * blocks * elements_per_block);
    state.SetItemsProcessed(items);
}

#endif //LIBCOMPRESSION_BENCHMARK_MACROS_H