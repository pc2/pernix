#include <testset.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

// ---------------------------------------------------------------------------
// Fallback compress: verify byte-exact match against the reference
// ---------------------------------------------------------------------------

TYPED_TEST(CompressionTest, FallbackCompressBlock) {
    std::vector<std::vector<uint8_t>> compressed(this->testSet.numberOfBlocks);

    for (uint32_t b = 0; b < this->testSet.numberOfBlocks; b++) {
        compressed[b].resize(TestFixture::BlockSize);

        const auto status = pernix_compress_block_f32(
            PERNIX_BACKEND_FALLBACK, TestFixture::BitWidth, TestFixture::BlockSize,
            this->testSet.getDecompressedData()[b].data(), 1.0f / this->testSet.getScales()[b],
            compressed[b].data());
        ASSERT_EQ(status, PERNIX_STATUS_OK);
    }

    for (uint32_t b = 0; b < this->testSet.numberOfBlocks; b++) {
        expectCompressedBlockEqualsReference(*this, compressed[b], b);
    }
}

TYPED_TEST(CompressionTest64, FallbackCompressBlock) {
    std::vector<std::vector<uint8_t>> compressed(this->testSet.numberOfBlocks);

    for (uint32_t b = 0; b < this->testSet.numberOfBlocks; b++) {
        compressed[b].resize(TestFixture::BlockSize);

        const auto status = pernix_compress_block_f64(
            PERNIX_BACKEND_FALLBACK, TestFixture::BitWidth, TestFixture::BlockSize,
            this->testSet.getDecompressedData()[b].data(), 1.0 / this->testSet.getScales()[b],
            compressed[b].data());
        ASSERT_EQ(status, PERNIX_STATUS_OK);
    }

    for (uint32_t b = 0; b < this->testSet.numberOfBlocks; b++) {
        expectCompressedBlockEqualsReference(*this, compressed[b], b);
    }
}

// ---------------------------------------------------------------------------
// Fallback decompress: verify near-source match
// ---------------------------------------------------------------------------

TYPED_TEST(DecompressionTest, FallbackDecompressBlock) {
    std::vector<std::vector<float_t>> decompressed(this->testSet.numberOfBlocks);

    for (uint32_t b = 0; b < this->testSet.numberOfBlocks; b++) {
        decompressed[b].resize(this->testSet.elementsPerBlock);

        const auto status = pernix_decompress_block_f32(
            PERNIX_BACKEND_FALLBACK, TestFixture::BitWidth, TestFixture::BlockSize,
            this->testSet.getCompressedData()[b].data(), this->testSet.getScales()[b],
            decompressed[b].data(), true);
        ASSERT_EQ(status, PERNIX_STATUS_OK);
    }

    for (uint32_t b = 0; b < this->testSet.numberOfBlocks; b++) {
        expectDecompressedBlockNearSource(*this, decompressed[b], b);
    }
}

TYPED_TEST(DecompressionTest64, FallbackDecompressBlock) {
    std::vector<std::vector<double_t>> decompressed(this->testSet.numberOfBlocks);

    for (uint32_t b = 0; b < this->testSet.numberOfBlocks; b++) {
        decompressed[b].resize(this->testSet.elementsPerBlock);

        const auto status = pernix_decompress_block_f64(
            PERNIX_BACKEND_FALLBACK, TestFixture::BitWidth, TestFixture::BlockSize,
            this->testSet.getCompressedData()[b].data(), this->testSet.getScales()[b],
            decompressed[b].data(), true);
        ASSERT_EQ(status, PERNIX_STATUS_OK);
    }

    for (uint32_t b = 0; b < this->testSet.numberOfBlocks; b++) {
        expectDecompressedBlockNearSource(*this, decompressed[b], b);
    }
}

// ---------------------------------------------------------------------------
// Multi-block roundtrip via compress_blocks / decompress_blocks (fallback)
// ---------------------------------------------------------------------------

TYPED_TEST(CompressionTest, FallbackCompressBlocksRoundtrip) {
    const uint32_t nb = this->testSet.numberOfBlocks;
    const uint32_t epb = this->testSet.elementsPerBlock;
    const uint32_t total = nb * epb;

    std::vector<float_t> flat(total);
    for (uint32_t b = 0; b < nb; b++) {
        std::copy_n(this->testSet.getDecompressedData()[b].data(), epb,
                    flat.data() + b * epb);
    }

    // Compute a single scale that covers all blocks
    float max_abs = 0.0f;
    for (uint32_t i = 0; i < total; i++) {
        max_abs = std::max(max_abs, std::abs(flat[i]));
    }
    const float q = static_cast<float>(decltype(this->testSet)::quantization_levels);
    const float scale = (max_abs > 0.0f && q > 0.0f) ? (max_abs / q) : std::numeric_limits<float_t>::epsilon();
    const float scale_inv = 1.0f / scale;

    std::vector<uint8_t> compressed(nb * TestFixture::BlockSize);
    auto status = pernix_compress_blocks_f32(
        PERNIX_BACKEND_FALLBACK, TestFixture::BitWidth, TestFixture::BlockSize,
        flat.data(), scale_inv, compressed.data(), nb);
    ASSERT_EQ(status, PERNIX_STATUS_OK);

    std::vector<float_t> restored(total);
    status = pernix_decompress_blocks_f32(
        PERNIX_BACKEND_FALLBACK, TestFixture::BitWidth, TestFixture::BlockSize,
        compressed.data(), scale, restored.data(), nb, true);
    ASSERT_EQ(status, PERNIX_STATUS_OK);

    const float tol = (std::abs(scale) * 0.5f) + (std::numeric_limits<float_t>::epsilon() * 16.0f);
    for (uint32_t i = 0; i < total; i++) {
        EXPECT_NEAR(restored[i], flat[i], tol);
    }
}

TYPED_TEST(CompressionTest64, FallbackCompressBlocksRoundtrip) {
    const uint32_t nb = this->testSet.numberOfBlocks;
    const uint32_t epb = this->testSet.elementsPerBlock;
    const uint32_t total = nb * epb;

    std::vector<double_t> flat(total);
    for (uint32_t b = 0; b < nb; b++) {
        std::copy_n(this->testSet.getDecompressedData()[b].data(), epb,
                    flat.data() + b * epb);
    }

    // Compute a single scale that covers all blocks
    double max_abs = 0.0;
    for (uint32_t i = 0; i < total; i++) {
        max_abs = std::max(max_abs, std::abs(flat[i]));
    }
    const double q = static_cast<double>(decltype(this->testSet)::quantization_levels);
    const double scale = (max_abs > 0.0 && q > 0.0) ? (max_abs / q) : std::numeric_limits<double_t>::epsilon();
    const double scale_inv = 1.0 / scale;

    std::vector<uint8_t> compressed(nb * TestFixture::BlockSize);
    auto status = pernix_compress_blocks_f64(
        PERNIX_BACKEND_FALLBACK, TestFixture::BitWidth, TestFixture::BlockSize,
        flat.data(), scale_inv, compressed.data(), nb);
    ASSERT_EQ(status, PERNIX_STATUS_OK);

    std::vector<double_t> restored(total);
    status = pernix_decompress_blocks_f64(
        PERNIX_BACKEND_FALLBACK, TestFixture::BitWidth, TestFixture::BlockSize,
        compressed.data(), scale, restored.data(), nb, true);
    ASSERT_EQ(status, PERNIX_STATUS_OK);

    const double tol = (std::abs(scale) * 0.5) + (std::numeric_limits<double_t>::epsilon() * 16.0);
    for (uint32_t i = 0; i < total; i++) {
        EXPECT_NEAR(restored[i], flat[i], tol);
    }
}

// ---------------------------------------------------------------------------
// blocks API with a single block should match the block API exactly
// ---------------------------------------------------------------------------

TYPED_TEST(CompressionTest, SingleBlockCompressBlocksMatchesBlock) {
    const auto& src = this->testSet.getDecompressedData()[0];
    const float scale_inv = 1.0f / this->testSet.getScales()[0];

    std::vector<uint8_t> blockOut(TestFixture::BlockSize);
    std::vector<uint8_t> blocksOut(TestFixture::BlockSize);

    auto s1 = pernix_compress_block_f32(
        PERNIX_BACKEND_FALLBACK, TestFixture::BitWidth, TestFixture::BlockSize,
        src.data(), scale_inv, blockOut.data());
    ASSERT_EQ(s1, PERNIX_STATUS_OK);

    auto s2 = pernix_compress_blocks_f32(
        PERNIX_BACKEND_FALLBACK, TestFixture::BitWidth, TestFixture::BlockSize,
        src.data(), scale_inv, blocksOut.data(), 1);
    ASSERT_EQ(s2, PERNIX_STATUS_OK);

    for (uint32_t i = 0; i < TestFixture::BlockSize; i++) {
        EXPECT_EQ(blockOut[i], blocksOut[i]) << "byte " << i;
    }
}

TYPED_TEST(DecompressionTest, SingleBlockDecompressBlocksMatchesBlock) {
    const auto& compressed = this->testSet.getCompressedData()[0];
    const float scale = this->testSet.getScales()[0];
    const uint32_t epb = this->testSet.elementsPerBlock;

    std::vector<float_t> blockOut(epb);
    std::vector<float_t> blocksOut(epb);

    auto s1 = pernix_decompress_block_f32(
        PERNIX_BACKEND_FALLBACK, TestFixture::BitWidth, TestFixture::BlockSize,
        compressed.data(), scale, blockOut.data(), true);
    ASSERT_EQ(s1, PERNIX_STATUS_OK);

    auto s2 = pernix_decompress_blocks_f32(
        PERNIX_BACKEND_FALLBACK, TestFixture::BitWidth, TestFixture::BlockSize,
        compressed.data(), scale, blocksOut.data(), 1, true);
    ASSERT_EQ(s2, PERNIX_STATUS_OK);

    for (uint32_t i = 0; i < epb; i++) {
        EXPECT_FLOAT_EQ(blockOut[i], blocksOut[i]) << "element " << i;
    }
}

// ---------------------------------------------------------------------------
// Edge-case behavioural tests (fallback, block_size=64)
// ---------------------------------------------------------------------------

TEST(FallbackEdgeTest, SignExtensionIsWellDefinedForNegativeValues) {
    constexpr uint32_t BS = 64;
    const std::array<uint8_t, BS> input{0x08};

    pernix_status st;
    std::array<float_t, (BS * 8) / 4> output{};

    st = pernix_decompress_block_f32(PERNIX_BACKEND_FALLBACK, 4, BS, input.data(), 1.0f, output.data(), true);
    ASSERT_EQ(st, PERNIX_STATUS_OK);
    EXPECT_EQ(output[0], -8.0f);
}

TEST(FallbackEdgeTest, ClearsUnusedPaddingBytes) {
    constexpr uint32_t BS = 64;
    constexpr uint32_t BW = 24;
    constexpr uint32_t EPB = (BS * 8) / BW;

    std::array<float_t, EPB> input{};
    std::array<uint8_t, BS> output{};
    output.fill(0xA5);

    auto st = pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, BW, BS, input.data(), 1.0f, output.data());
    ASSERT_EQ(st, PERNIX_STATUS_OK);
    EXPECT_EQ(output[BS - 1], 0);
}

TEST(FallbackEdgeTest, ClampsNonFiniteAndOutOfRangeBeforeNarrowing) {
    constexpr uint32_t BS = 64;
    constexpr uint32_t BW = 4;
    constexpr uint32_t EPB = (BS * 8) / BW;

    std::array<float_t, EPB> input{};
    input[0] = std::numeric_limits<float_t>::infinity();
    input[1] = -std::numeric_limits<float_t>::infinity();
    input[2] = std::numeric_limits<float_t>::quiet_NaN();

    std::array<uint8_t, BS> compressed{};
    std::array<float_t, EPB> restored{};

    auto st = pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, BW, BS, input.data(), 1.0f, compressed.data());
    ASSERT_EQ(st, PERNIX_STATUS_OK);

    st = pernix_decompress_block_f32(PERNIX_BACKEND_FALLBACK, BW, BS, compressed.data(), 1.0f, restored.data(), true);
    ASSERT_EQ(st, PERNIX_STATUS_OK);

    EXPECT_EQ(restored[0], 7.0f);
    EXPECT_EQ(restored[1], -8.0f);
    EXPECT_EQ(restored[2], 0.0f);
}

// ---------------------------------------------------------------------------
// Error-code contract tests
// ---------------------------------------------------------------------------

TEST(ErrorCodeTest, UnsupportedBlockSizeReturnsError) {
    constexpr uint32_t BS = 32;
    float_t src[32] = {};
    uint8_t dst[32] = {};

    auto st = pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, 8, BS, src, 1.0f, dst);
    EXPECT_EQ(st, PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE);

    st = pernix_decompress_block_f32(PERNIX_BACKEND_FALLBACK, 8, BS, dst, 1.0f, src, true);
    EXPECT_EQ(st, PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE);

    st = pernix_compress_blocks_f32(PERNIX_BACKEND_FALLBACK, 8, BS, src, 1.0f, dst, 1);
    EXPECT_EQ(st, PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE);

    st = pernix_decompress_blocks_f32(PERNIX_BACKEND_FALLBACK, 8, BS, dst, 1.0f, src, 1, true);
    EXPECT_EQ(st, PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE);
}

TEST(ErrorCodeTest, UnsupportedBitWidthReturnsError) {
    constexpr uint32_t BS = 64;
    float_t src[256] = {};
    uint8_t dst[64] = {};

    auto st = pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, 0, BS, src, 1.0f, dst);
    EXPECT_EQ(st, PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH);

    st = pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, 25, BS, src, 1.0f, dst);
    EXPECT_EQ(st, PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH);
}

TEST(ErrorCodeTest, NullPointerReturnsError) {
    float_t src[64] = {};
    uint8_t dst[64] = {};

    auto st = pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, 8, 64, nullptr, 1.0f, dst);
    EXPECT_EQ(st, PERNIX_STATUS_INVALID_ARGUMENT);

    st = pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, 8, 64, src, 1.0f, nullptr);
    EXPECT_EQ(st, PERNIX_STATUS_INVALID_ARGUMENT);
}
