#include <pernix/pernix.hpp>
#include <pernix/cuda/codec.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define PERNIX_CHECK(call)                                    \
    do {                                                      \
        const auto status = (call);                           \
        if (status != PERNIX_STATUS_OK) {                     \
            throw std::runtime_error(                         \
                std::string(#call) + ": " +                   \
                pernix::status_string(status));               \
        }                                                     \
    } while (false)



namespace {
constexpr usize block_bytes = 64;
constexpr usize blocks = 1 << 20;

void run_test(const pernix::Dispatcher& cpu_dispatcher, const pernix::Dispatcher& cuda_dispatcher,
              const std::uint8_t bit_width, const bool signed_values) {
    const usize elements_per_block = pernix::elements_per_block(bit_width);
    const usize element_count = blocks * elements_per_block;
    constexpr usize compressed_bytes = blocks * block_bytes;

    constexpr f32 scale = 0.125F;
    constexpr f32 inverse_scale = 8.0F;

    std::vector<float> input(element_count);
    std::vector<float> cpu_output(element_count);
    std::vector<float> gpu_output(element_count);
    std::vector<std::uint8_t> compressed(compressed_bytes);

    const i32 max_quantized = (1U << (bit_width - 1U)) - 1;
    const i32 magnitude = std::min(max_quantized, 100);

    for (std::size_t i = 0; i < element_count; ++i) {
        const int q = magnitude == 0 ? 0 : static_cast<int>( i % static_cast<std::size_t>(2 * magnitude + 1)) - magnitude;

        input[i] = static_cast<float>(q) * scale;
    }

    PERNIX_CHECK(cpu_dispatcher.compress_blocks(
        bit_width,
        std::span<const float>(input),
        inverse_scale,
        std::span(compressed), blocks));

    PERNIX_CHECK(cpu_dispatcher.decompress_blocks(
        bit_width,
        std::span<const u8>(compressed),
        scale,
        std::span(cpu_output), blocks,
        signed_values));

    PERNIX_CHECK(cuda_dispatcher.decompress_blocks(
        bit_width,
        std::span<const u8>(compressed),
        scale,
        std::span(gpu_output),
        blocks,
        signed_values));

    for (std::size_t i = 0; i < element_count; ++i) {
        if (cpu_output[i] != gpu_output[i]) {
            throw std::runtime_error(
                "Mismatch: bit_width=" +
                std::to_string(bit_width) +
                ", signed=" +
                std::to_string(signed_values) +
                ", index=" +
                std::to_string(i) +
                ", CPU=" +
                std::to_string(cpu_output[i]) +
                ", GPU=" +
                std::to_string(gpu_output[i]));
        }
    }

    std::cout << "PASS CUDA dispatcher N="
              << static_cast<int>(bit_width)
              << " signed=" << signed_values
              << " elements=" << element_count
              << '\n';

}

}

int main() {
    try {
        const int device_count = pernix::cuda::device_count();
        if (device_count == 0) {
            std::cerr << "CUDA decompression test skipped: no CUDA device or driver is available\n";
            return 77;
        }

        std::cout << "CUDA decompression using " << device_count << " device(s)\n";

        pernix::Dispatcher cpu_dispatcher;
        pernix::Dispatcher cuda_dispatcher;
        PERNIX_CHECK(pernix::Dispatcher::create(pernix::Backend::Fallback, cpu_dispatcher));
        PERNIX_CHECK(pernix::Dispatcher::create(pernix::Backend::Cuda, cuda_dispatcher));

        for (std::uint8_t bits = 1; bits <= 24; ++bits) {
            run_test(cpu_dispatcher, cuda_dispatcher, bits, true);
            run_test(cpu_dispatcher, cuda_dispatcher, bits, false);
        }

        std::cout
            << "All CUDA decompression tests PASSED\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
