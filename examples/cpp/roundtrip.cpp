#include <pernix/pernix.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

int main() {
    constexpr u8 bit_width = 16;
    constexpr u32 block_size = pernix::compressed_block_size();
    constexpr std::size_t elements = pernix::elements_per_block(bit_width);

    std::array<float, elements> input{};
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.25f);
    }

    float bmax = 0.0f;
    for (const float value : input) {
        bmax = std::max(bmax, std::abs(value));
    }
    float scale = 0.0f;
    float inverse_scale = 0.0f;
    if (pernix::decompression_scale_from_bmax(bmax, bit_width, scale) != PERNIX_STATUS_OK ||
        pernix::inverse_scale(scale, inverse_scale) != PERNIX_STATUS_OK) {
        return 1;
    }

    std::array<u8, block_size> compressed{};
    std::array<float, elements> restored{};

    if (pernix::compress_block(pernix::Backend::Fallback, bit_width, block_size, input, inverse_scale, compressed) !=
        PERNIX_STATUS_OK) {
        return 1;
    }
    if (pernix::decompress_block(pernix::Backend::Fallback, bit_width, block_size, compressed, scale, restored) !=
        PERNIX_STATUS_OK) {
        return 2;
    }

    for (std::size_t i = 0; i < input.size(); ++i) {
        if (std::abs(restored[i] - input[i]) > scale) {
            return 3;
        }
    }
    return 0;
}
