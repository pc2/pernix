#include <pernix/pernix.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <span>

int main() {
    constexpr u8 bit_width   = 8;
    constexpr u32 block_size = 64;
    constexpr usize elements = (block_size * 8U) / bit_width;

    std::array<float, elements> input{};
    for (usize i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(static_cast<int>(i % 17U) - 8) * 0.125f;
    }

    std::array<u8, block_size> compressed{};
    std::array<float, elements> restored{};
    pernix::Dispatcher dispatcher;
    if (pernix::Dispatcher::create(pernix::Backend::Fallback, dispatcher) != PERNIX_STATUS_OK) {
        return 1;
    }

    if (dispatcher.compress_block(bit_width, block_size, std::span<const float>(input), 8.0f, std::span<u8>(compressed)) !=
        PERNIX_STATUS_OK) {
        return 2;
    }
    if (dispatcher.decompress_block(bit_width, block_size, std::span<const u8>(compressed), 0.125f, std::span<float>(restored)) !=
        PERNIX_STATUS_OK) {
        return 3;
    }

    for (usize i = 0; i < input.size(); ++i) {
        if (std::abs(input[i] - restored[i]) > 0.0f) {
            return 4;
        }
    }
    return 0;
}
