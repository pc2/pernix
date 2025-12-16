#ifndef LIBCOMPRESSION_FIXTURES_H
#define LIBCOMPRESSION_FIXTURES_H

#include <benchmark/benchmark.h>
#include <functional>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <random>


template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires (BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
struct DecompressionBenchmarkSet {
    int64_t number_of_blocks = 0;

    alignas(64) uint8_t* input_ptr  = nullptr;
    alignas(64) float_t* output_ptr = nullptr;
    std::vector<float_t> scales{};

    ~DecompressionBenchmarkSet() {
        if (input_ptr) {
            std::free(input_ptr);
            input_ptr = nullptr;
        }
        if (output_ptr) {
            std::free(output_ptr);
            output_ptr = nullptr;
        }
    }
};

template <uint8_t BIT_WIDTH, bool SIGN_VALUES = true>
    requires (BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
DecompressionBenchmarkSet<BIT_WIDTH, SIGN_VALUES> create_decompression_benchmark_set(int64_t number_of_blocks) {
    DecompressionBenchmarkSet<BIT_WIDTH, SIGN_VALUES> benchmark_set;
    benchmark_set.number_of_blocks = number_of_blocks;

    const int64_t elements_per_block = 512 / BIT_WIDTH;
    const int64_t total_bits         = number_of_blocks * elements_per_block * BIT_WIDTH;
    const int64_t num_bytes          = (total_bits + 7) / 8;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int8_t> dis{};
    std::uniform_real_distribution scale_dis(0.001f, 1.0f);

    benchmark_set.input_ptr  = static_cast<uint8_t*>(std::aligned_alloc(64, num_bytes * sizeof(uint8_t)));
    benchmark_set.output_ptr = static_cast<float_t*>(std::aligned_alloc(64, number_of_blocks * elements_per_block * sizeof(float_t)));
    benchmark_set.scales.resize(number_of_blocks);

    for (auto& scale : benchmark_set.scales) {
        scale = scale_dis(gen);
    }

    for (int64_t i = 0; i < num_bytes; ++i) {
        benchmark_set.input_ptr[i] = static_cast<uint8_t>(dis(gen));
    }

    return benchmark_set;
}

#endif //LIBCOMPRESSION_FIXTURES_H