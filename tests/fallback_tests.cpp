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
    std::vector<std::vector<u8> > compressed(this->testSet.numberOfBlocks);

    for (u32 b = 0; b < this->testSet.numberOfBlocks; b++) {
        compressed[b].resize(TestFixture::BlockSize);

        const auto status = pernix_compress_block_f32(
            PERNIX_BACKEND_FALLBACK, TestFixture::BitWidth, TestFixture::BlockSize,
            this->testSet.getDecompressedData()[b].data(), 1.0f / this->testSet.getScales()[b],
            compressed[b].data());
        ASSERT_EQ(status, PERNIX_STATUS_OK);
    }

    for (u32 b = 0; b < this->testSet.numberOfBlocks; b++) {
        expectCompressedBlockEqualsReference(*this, compressed[b], b);
    }
}

TYPED_TEST(CompressionTest64, FallbackCompressBlock) {
    std::vector<std::vector<u8> > compressed(this->testSet.numberOfBlocks);

    for (u32 b = 0; b < this->testSet.numberOfBlocks; b++) {
        compressed[b].resize(TestFixture::BlockSize);

        const auto status = pernix_compress_block_f64(
            PERNIX_BACKEND_FALLBACK, TestFixture::BitWidth, TestFixture::BlockSize,
            this->testSet.getDecompressedData()[b].data(), 1.0 / this->testSet.getScales()[b],
            compressed[b].data());
        ASSERT_EQ(status, PERNIX_STATUS_OK);
    }

    for (u32 b = 0; b < this->testSet.numberOfBlocks; b++) {
        expectCompressedBlockEqualsReference(*this, compressed[b], b);
    }
}

// ---------------------------------------------------------------------------
// Fallback decompress: verify near-source match
// ---------------------------------------------------------------------------

TYPED_TEST(DecompressionTest, FallbackDecompressBlock) {
    std::vector<std::vector<f32> > decompressed(this->testSet.numberOfBlocks);

    for (u32 b = 0; b < this->testSet.numberOfBlocks; b++) {
        decompressed[b].resize(this->testSet.elementsPerBlock);

        const auto status = pernix_decompress_block_f32(
            PERNIX_BACKEND_FALLBACK, TestFixture::BitWidth, TestFixture::BlockSize,
            this->testSet.getCompressedData()[b].data(), this->testSet.getScales()[b],
            decompressed[b].data(), true);
        ASSERT_EQ(status, PERNIX_STATUS_OK);
    }

    for (u32 b = 0; b < this->testSet.numberOfBlocks; b++) {
        expectDecompressedBlockNearSource(*this, decompressed[b], b);
    }
}

TYPED_TEST(DecompressionTest64, FallbackDecompressBlock) {
    std::vector<std::vector<f64> > decompressed(this->testSet.numberOfBlocks);

    for (u32 b = 0; b < this->testSet.numberOfBlocks; b++) {
        decompressed[b].resize(this->testSet.elementsPerBlock);

        const auto status = pernix_decompress_block_f64(
            PERNIX_BACKEND_FALLBACK, TestFixture::BitWidth, TestFixture::BlockSize,
            this->testSet.getCompressedData()[b].data(), this->testSet.getScales()[b],
            decompressed[b].data(), true);
        ASSERT_EQ(status, PERNIX_STATUS_OK);
    }

    for (u32 b = 0; b < this->testSet.numberOfBlocks; b++) {
        expectDecompressedBlockNearSource(*this, decompressed[b], b);
    }
}

// ---------------------------------------------------------------------------
// Multi-block roundtrip via compress_blocks / decompress_blocks (fallback)
// ---------------------------------------------------------------------------

TYPED_TEST(CompressionTest, FallbackCompressBlocksRoundtrip) {
    const u32 nb = this->testSet.numberOfBlocks;
    const u32 epb = this->testSet.elementsPerBlock;
    const u32 total = nb * epb;

    std::vector<f32> flat(total);
    for (u32 b = 0; b < nb; b++) {
        std::copy_n(this->testSet.getDecompressedData()[b].data(), epb,
                    flat.data() + b * epb);
    }

    // Compute a single scale that covers all blocks
    float max_abs = 0.0f;
    for (u32 i = 0; i < total; i++) {
        max_abs = std::max(max_abs, std::abs(flat[i]));
    }
    const float q = static_cast<float>(decltype(this->testSet)::quantization_levels);
    const float scale = (max_abs > 0.0f && q > 0.0f) ? (max_abs / q) : std::numeric_limits<f32>::epsilon();
    const float scale_inv = 1.0f / scale;

    std::vector<u8> compressed(nb * TestFixture::BlockSize);
    auto status = pernix_compress_blocks_f32(
        PERNIX_BACKEND_FALLBACK, TestFixture::BitWidth, TestFixture::BlockSize,
        flat.data(), scale_inv, compressed.data(), nb);
    ASSERT_EQ(status, PERNIX_STATUS_OK);

    std::vector<f32> restored(total);
    status = pernix_decompress_blocks_f32(
        PERNIX_BACKEND_FALLBACK, TestFixture::BitWidth, TestFixture::BlockSize,
        compressed.data(), scale, restored.data(), nb, true);
    ASSERT_EQ(status, PERNIX_STATUS_OK);

    const float tol = (std::abs(scale) * 0.5f) + (std::numeric_limits<f32>::epsilon() * 16.0f);
    for (u32 i = 0; i < total; i++) {
        EXPECT_NEAR(restored[i], flat[i], tol);
    }
}

TYPED_TEST(CompressionTest64, FallbackCompressBlocksRoundtrip) {
    const u32 nb = this->testSet.numberOfBlocks;
    const u32 epb = this->testSet.elementsPerBlock;
    const u32 total = nb * epb;

    std::vector<f64> flat(total);
    for (u32 b = 0; b < nb; b++) {
        std::copy_n(this->testSet.getDecompressedData()[b].data(), epb,
                    flat.data() + b * epb);
    }

    // Compute a single scale that covers all blocks
    double max_abs = 0.0;
    for (u32 i = 0; i < total; i++) {
        max_abs = std::max(max_abs, std::abs(flat[i]));
    }
    const double q = static_cast<double>(decltype(this->testSet)::quantization_levels);
    const double scale = (max_abs > 0.0 && q > 0.0) ? (max_abs / q) : std::numeric_limits<f64>::epsilon();
    const double scale_inv = 1.0 / scale;

    std::vector<u8> compressed(nb * TestFixture::BlockSize);
    auto status = pernix_compress_blocks_f64(
        PERNIX_BACKEND_FALLBACK, TestFixture::BitWidth, TestFixture::BlockSize,
        flat.data(), scale_inv, compressed.data(), nb);
    ASSERT_EQ(status, PERNIX_STATUS_OK);

    std::vector<f64> restored(total);
    status = pernix_decompress_blocks_f64(
        PERNIX_BACKEND_FALLBACK, TestFixture::BitWidth, TestFixture::BlockSize,
        compressed.data(), scale, restored.data(), nb, true);
    ASSERT_EQ(status, PERNIX_STATUS_OK);

    const double tol = (std::abs(scale) * 0.5) + (std::numeric_limits<f64>::epsilon() * 16.0);
    for (u32 i = 0; i < total; i++) {
        EXPECT_NEAR(restored[i], flat[i], tol);
    }
}

// ---------------------------------------------------------------------------
// blocks API with a single block should match the block API exactly
// ---------------------------------------------------------------------------

TYPED_TEST(CompressionTest, SingleBlockCompressBlocksMatchesBlock) {
    const auto &src = this->testSet.getDecompressedData()[0];
    const float scale_inv = 1.0f / this->testSet.getScales()[0];

    std::vector<u8> blockOut(TestFixture::BlockSize);
    std::vector<u8> blocksOut(TestFixture::BlockSize);

    auto s1 = pernix_compress_block_f32(
        PERNIX_BACKEND_FALLBACK, TestFixture::BitWidth, TestFixture::BlockSize,
        src.data(), scale_inv, blockOut.data());
    ASSERT_EQ(s1, PERNIX_STATUS_OK);

    auto s2 = pernix_compress_blocks_f32(
        PERNIX_BACKEND_FALLBACK, TestFixture::BitWidth, TestFixture::BlockSize,
        src.data(), scale_inv, blocksOut.data(), 1);
    ASSERT_EQ(s2, PERNIX_STATUS_OK);

    for (u32 i = 0; i < TestFixture::BlockSize; i++) {
        EXPECT_EQ(blockOut[i], blocksOut[i]) << "byte " << i;
    }
}

TYPED_TEST(DecompressionTest, SingleBlockDecompressBlocksMatchesBlock) {
    const auto &compressed = this->testSet.getCompressedData()[0];
    const float scale = this->testSet.getScales()[0];
    const u32 epb = this->testSet.elementsPerBlock;

    std::vector<f32> blockOut(epb);
    std::vector<f32> blocksOut(epb);

    auto s1 = pernix_decompress_block_f32(
        PERNIX_BACKEND_FALLBACK, TestFixture::BitWidth, TestFixture::BlockSize,
        compressed.data(), scale, blockOut.data(), true);
    ASSERT_EQ(s1, PERNIX_STATUS_OK);

    auto s2 = pernix_decompress_blocks_f32(
        PERNIX_BACKEND_FALLBACK, TestFixture::BitWidth, TestFixture::BlockSize,
        compressed.data(), scale, blocksOut.data(), 1, true);
    ASSERT_EQ(s2, PERNIX_STATUS_OK);

    for (u32 i = 0; i < epb; i++) {
        EXPECT_FLOAT_EQ(blockOut[i], blocksOut[i]) << "element " << i;
    }
}

// ---------------------------------------------------------------------------
// Edge-case behavioural tests (fallback, block_size=64)
// ---------------------------------------------------------------------------

TEST(FallbackEdgeTest, SignExtensionIsWellDefinedForNegativeValues) {
    constexpr u32 BS = 64;
    const std::array<u8, BS> input{0x08};

    pernix_status st;
    std::array<f32, (BS * 8) / 4> output{};

    st = pernix_decompress_block_f32(PERNIX_BACKEND_FALLBACK, 4, BS, input.data(), 1.0f, output.data(), true);
    ASSERT_EQ(st, PERNIX_STATUS_OK);
    EXPECT_EQ(output[0], -8.0f);
}

TEST(FallbackEdgeTest, ClearsUnusedPaddingBytes) {
    constexpr u32 BS = 64;
    constexpr u32 BW = 24;
    constexpr u32 EPB = (BS * 8) / BW;

    std::array<f32, EPB> input{};
    std::array<u8, BS> output{};
    output.fill(0xA5);

    auto st = pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, BW, BS, input.data(), 1.0f, output.data());
    ASSERT_EQ(st, PERNIX_STATUS_OK);
    EXPECT_EQ(output[BS - 1], 0);
}

TEST(FallbackEdgeTest, ClampsNonFiniteAndOutOfRangeBeforeNarrowing) {
    constexpr u32 BS = 64;
    constexpr u32 BW = 4;
    constexpr u32 EPB = (BS * 8) / BW;

    std::array<f32, EPB> input{};
    input[0] = std::numeric_limits<f32>::infinity();
    input[1] = -std::numeric_limits<f32>::infinity();
    input[2] = std::numeric_limits<f32>::quiet_NaN();

    std::array<u8, BS> compressed{};
    std::array<f32, EPB> restored{};

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
    constexpr u32 BS = 32;
    f32 src[32] = {};
    u8 dst[32] = {};

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
    constexpr u32 BS = 64;
    f32 src[256] = {};
    u8 dst[64] = {};

    auto st = pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, 0, BS, src, 1.0f, dst);
    EXPECT_EQ(st, PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH);

    st = pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, 25, BS, src, 1.0f, dst);
    EXPECT_EQ(st, PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH);
}

TEST(ErrorCodeTest, NullPointerReturnsError) {
    f32 src[64] = {};
    u8 dst[64] = {};

    auto st = pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, 8, 64, nullptr, 1.0f, dst);
    EXPECT_EQ(st, PERNIX_STATUS_INVALID_ARGUMENT);

    st = pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, 8, 64, src, 1.0f, nullptr);
    EXPECT_EQ(st, PERNIX_STATUS_INVALID_ARGUMENT);
}
