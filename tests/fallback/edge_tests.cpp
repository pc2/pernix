#include <pernix/fallback/compression.h>
#include <pernix/fallback/decompression.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

TEST(FallbackDecompressionEdgeTest, SignExtensionIsWellDefinedForNegativeValues) {
    const std::array<uint8_t, 64> input{0x08};
    std::array<float_t, 128> output{};

    ASSERT_EQ(pernix::decompress_block_fallback<4>(input.data(), 1.0F, output.data()), 0);

    EXPECT_EQ(output[0], -8.0F);
}

TEST(FallbackCompressionEdgeTest, ClearsUnusedPaddingBytes) {
    std::array<float_t, (64 * 8) / 24> input{};
    std::array<uint8_t, 64> output{};
    output.fill(0xA5);

    ASSERT_EQ(pernix::compress_block_fallback<24>(input.data(), 1.0F, output.data()), 0);

    EXPECT_EQ(output[63], 0);
}

TEST(FallbackCompressionEdgeTest, ClampsNonFiniteAndOutOfRangeBeforeNarrowing) {
    std::array<float_t, (64 * 8) / 4> input{};
    input[0] = std::numeric_limits<float_t>::infinity();
    input[1] = -std::numeric_limits<float_t>::infinity();
    input[2] = std::numeric_limits<float_t>::quiet_NaN();
    std::array<uint8_t, 64> compressed{};
    std::array<float_t, (64 * 8) / 4> restored{};

    ASSERT_EQ(pernix::compress_block_fallback<4>(input.data(), 1.0F, compressed.data()), 0);
    ASSERT_EQ(pernix::decompress_block_fallback<4>(compressed.data(), 1.0F, restored.data()), 0);

    EXPECT_EQ(restored[0], 7.0F);
    EXPECT_EQ(restored[1], -8.0F);
    EXPECT_EQ(restored[2], 0.0F);
}
