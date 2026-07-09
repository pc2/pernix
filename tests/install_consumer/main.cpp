#include <array>
#include <cmath>
#include <cstddef>
#include <pernix/pernix.hpp>

int main() {
    constexpr u8 bit_width         = 8;
    constexpr u32 block_size       = 64;
    constexpr std::size_t elements = (block_size * 8U) / bit_width;

    std::array<float, elements> input{};
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(static_cast<int>(i % 17U) - 8) * 0.125f;
    }

    std::array<u8, block_size> compressed{};
    std::array<float, elements> restored{};

    if (pernix::compress_block(pernix::Backend::Fallback, bit_width, block_size, input, 8.0f, compressed) != PERNIX_STATUS_OK) {
        return 1;
    }
    if (pernix::decompress_block(pernix::Backend::Fallback, bit_width, block_size, compressed, 0.125f, restored) != PERNIX_STATUS_OK) {
        return 2;
    }

    for (std::size_t i = 0; i < input.size(); ++i) {
        if (std::abs(input[i] - restored[i]) > 0.0f) {
            return 3;
        }
    }
    return 0;
}
