#include <testset.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__linux__) && defined(__aarch64__)
#include <sys/auxv.h>
#ifndef HWCAP_SVE
#define HWCAP_SVE (1 << 22)
#endif
#ifndef HWCAP2_SVE2
#define HWCAP2_SVE2 (1 << 1)
#endif
#endif

namespace {
template <typename FixtureT, typename CompressFn>
void expect_backend_compressed_block_matches_reference(FixtureT& fixture, const pernix_backend backend,
                                                       CompressFn compress_fn) {
    using ScaleType = std::remove_cvref_t<decltype(fixture.testSet.getScales()[0])>;
    std::vector<std::vector<u8> > compressed(fixture.testSet.numberOfBlocks);

    for (u32 b = 0; b < fixture.testSet.numberOfBlocks; ++b) {
        compressed[b].resize(FixtureT::BlockSize);
        const auto status = compress_fn(backend, FixtureT::BitWidth, FixtureT::BlockSize,
                                        fixture.testSet.getDecompressedData()[b].data(),
                                        ScaleType{1} / fixture.testSet.getScales()[b],
                                        compressed[b].data());
        ASSERT_EQ(status, PERNIX_STATUS_OK);
    }

    for (u32 b = 0; b < fixture.testSet.numberOfBlocks; ++b) {
        expectCompressedBlockEqualsReference(fixture, compressed[b], b);
    }
}

template <typename FixtureT, typename DecompressFn>
void expect_backend_decompressed_block_matches_source(FixtureT& fixture, const pernix_backend backend,
                                                      DecompressFn decompress_fn) {
    using ValueType = std::remove_cvref_t<decltype(fixture.testSet.getDecompressedData()[0][0])>;
    std::vector<std::vector<ValueType> > decompressed(fixture.testSet.numberOfBlocks);

    for (u32 b = 0; b < fixture.testSet.numberOfBlocks; ++b) {
        decompressed[b].resize(fixture.testSet.elementsPerBlock);
        const auto status = decompress_fn(backend, FixtureT::BitWidth, FixtureT::BlockSize,
                                          fixture.testSet.getCompressedData()[b].data(),
                                          fixture.testSet.getScales()[b], decompressed[b].data(), true);
        ASSERT_EQ(status, PERNIX_STATUS_OK);
    }

    for (u32 b = 0; b < fixture.testSet.numberOfBlocks; ++b) {
        expectDecompressedBlockNearSource(fixture, decompressed[b], b);
    }
}
}

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

#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
TYPED_TEST(CompressionTest, FallbackStdparCompressBlock) {
    expect_backend_compressed_block_matches_reference(*this, PERNIX_BACKEND_FALLBACK_STDPAR,
                                                      &pernix_compress_block_f32);
}

TYPED_TEST(CompressionTest64, FallbackStdparCompressBlock) {
    expect_backend_compressed_block_matches_reference(*this, PERNIX_BACKEND_FALLBACK_STDPAR,
                                                      &pernix_compress_block_f64);
}

TYPED_TEST(DecompressionTest, FallbackStdparDecompressBlock) {
    expect_backend_decompressed_block_matches_source(*this, PERNIX_BACKEND_FALLBACK_STDPAR,
                                                     &pernix_decompress_block_f32);
}

TYPED_TEST(DecompressionTest64, FallbackStdparDecompressBlock) {
    expect_backend_decompressed_block_matches_source(*this, PERNIX_BACKEND_FALLBACK_STDPAR,
                                                     &pernix_decompress_block_f64);
}
#endif

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

#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
TYPED_TEST(CompressionTest, FallbackStdparCompressBlocksRoundtrip) {
    const u32 nb = this->testSet.numberOfBlocks;
    const u32 epb = this->testSet.elementsPerBlock;
    const u32 total = nb * epb;

    std::vector<f32> flat(total);
    for (u32 b = 0; b < nb; ++b) {
        std::copy_n(this->testSet.getDecompressedData()[b].data(), epb, flat.data() + b * epb);
    }

    float max_abs = 0.0f;
    for (u32 i = 0; i < total; ++i) {
        max_abs = std::max(max_abs, std::abs(flat[i]));
    }
    const float q = static_cast<float>(decltype(this->testSet)::quantization_levels);
    const float scale = (max_abs > 0.0f && q > 0.0f) ? (max_abs / q) : std::numeric_limits<f32>::epsilon();
    const float scale_inv = 1.0f / scale;

    std::vector<u8> compressed(nb * TestFixture::BlockSize);
    auto status = pernix_compress_blocks_f32(PERNIX_BACKEND_FALLBACK_STDPAR, TestFixture::BitWidth,
                                             TestFixture::BlockSize, flat.data(), scale_inv, compressed.data(), nb);
    ASSERT_EQ(status, PERNIX_STATUS_OK);

    std::vector<f32> restored(total);
    status = pernix_decompress_blocks_f32(PERNIX_BACKEND_FALLBACK_STDPAR, TestFixture::BitWidth,
                                          TestFixture::BlockSize, compressed.data(), scale, restored.data(), nb, true);
    ASSERT_EQ(status, PERNIX_STATUS_OK);

    const float tol = (std::abs(scale) * 0.5f) + (std::numeric_limits<f32>::epsilon() * 16.0f);
    for (u32 i = 0; i < total; ++i) {
        EXPECT_NEAR(restored[i], flat[i], tol);
    }
}

TYPED_TEST(CompressionTest64, FallbackStdparCompressBlocksRoundtrip) {
    const u32 nb = this->testSet.numberOfBlocks;
    const u32 epb = this->testSet.elementsPerBlock;
    const u32 total = nb * epb;

    std::vector<f64> flat(total);
    for (u32 b = 0; b < nb; ++b) {
        std::copy_n(this->testSet.getDecompressedData()[b].data(), epb, flat.data() + b * epb);
    }

    double max_abs = 0.0;
    for (u32 i = 0; i < total; ++i) {
        max_abs = std::max(max_abs, std::abs(flat[i]));
    }
    const double q = static_cast<double>(decltype(this->testSet)::quantization_levels);
    const double scale = (max_abs > 0.0 && q > 0.0) ? (max_abs / q) : std::numeric_limits<f64>::epsilon();
    const double scale_inv = 1.0 / scale;

    std::vector<u8> compressed(nb * TestFixture::BlockSize);
    auto status = pernix_compress_blocks_f64(PERNIX_BACKEND_FALLBACK_STDPAR, TestFixture::BitWidth,
                                             TestFixture::BlockSize, flat.data(), scale_inv, compressed.data(), nb);
    ASSERT_EQ(status, PERNIX_STATUS_OK);

    std::vector<f64> restored(total);
    status = pernix_decompress_blocks_f64(PERNIX_BACKEND_FALLBACK_STDPAR, TestFixture::BitWidth,
                                          TestFixture::BlockSize, compressed.data(), scale, restored.data(), nb, true);
    ASSERT_EQ(status, PERNIX_STATUS_OK);

    const double tol = (std::abs(scale) * 0.5) + (std::numeric_limits<f64>::epsilon() * 16.0);
    for (u32 i = 0; i < total; ++i) {
        EXPECT_NEAR(restored[i], flat[i], tol);
    }
}
#endif

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

#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
TYPED_TEST(CompressionTest, SingleBlockStdparCompressBlocksMatchesBlock) {
    const auto& src = this->testSet.getDecompressedData()[0];
    const float scale_inv = 1.0f / this->testSet.getScales()[0];

    std::vector<u8> block_out(TestFixture::BlockSize);
    std::vector<u8> blocks_out(TestFixture::BlockSize);

    auto s1 = pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK_STDPAR, TestFixture::BitWidth,
                                        TestFixture::BlockSize, src.data(), scale_inv, block_out.data());
    ASSERT_EQ(s1, PERNIX_STATUS_OK);

    auto s2 = pernix_compress_blocks_f32(PERNIX_BACKEND_FALLBACK_STDPAR, TestFixture::BitWidth,
                                         TestFixture::BlockSize, src.data(), scale_inv, blocks_out.data(), 1);
    ASSERT_EQ(s2, PERNIX_STATUS_OK);

    for (u32 i = 0; i < TestFixture::BlockSize; ++i) {
        EXPECT_EQ(block_out[i], blocks_out[i]) << "byte " << i;
    }
}

TYPED_TEST(DecompressionTest, SingleBlockStdparDecompressBlocksMatchesBlock) {
    const auto& compressed = this->testSet.getCompressedData()[0];
    const float scale = this->testSet.getScales()[0];
    const u32 epb = this->testSet.elementsPerBlock;

    std::vector<f32> block_out(epb);
    std::vector<f32> blocks_out(epb);

    auto s1 = pernix_decompress_block_f32(PERNIX_BACKEND_FALLBACK_STDPAR, TestFixture::BitWidth,
                                          TestFixture::BlockSize, compressed.data(), scale, block_out.data(), true);
    ASSERT_EQ(s1, PERNIX_STATUS_OK);

    auto s2 = pernix_decompress_blocks_f32(PERNIX_BACKEND_FALLBACK_STDPAR, TestFixture::BitWidth,
                                           TestFixture::BlockSize, compressed.data(), scale, blocks_out.data(), 1,
                                           true);
    ASSERT_EQ(s2, PERNIX_STATUS_OK);

    for (u32 i = 0; i < epb; ++i) {
        EXPECT_FLOAT_EQ(block_out[i], blocks_out[i]) << "element " << i;
    }
}
#endif

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

TEST(FallbackEdgeTest, SignValuesFalseTreatsPackedValuesAsUnsigned) {
    constexpr u32 BS = 64;
    const std::array<u8, BS> input{0x0F};

    std::array<f32, (BS * 8) / 4> signed_output{};
    std::array<f32, (BS * 8) / 4> unsigned_output{};

    auto st = pernix_decompress_block_f32(PERNIX_BACKEND_FALLBACK, 4, BS, input.data(), 1.0f, signed_output.data(),
                                          true);
    ASSERT_EQ(st, PERNIX_STATUS_OK);
    st = pernix_decompress_block_f32(PERNIX_BACKEND_FALLBACK, 4, BS, input.data(), 1.0f, unsigned_output.data(),
                                     false);
    ASSERT_EQ(st, PERNIX_STATUS_OK);

    EXPECT_EQ(signed_output[0], -1.0f);
    EXPECT_EQ(unsigned_output[0], 15.0f);
}

TEST(FallbackEdgeTest, FixedBlockInvariantAndMixedSignMultiBlockRoundTrip) {
    constexpr u32 BS = 64;
    constexpr u32 BW = 12;
    constexpr u32 blocks = 3;
    constexpr u32 EPB = (BS * 8) / BW;
    constexpr u32 total = EPB * blocks;

    static_assert(BS == 64);
    static_assert(EPB == 42);
    EXPECT_EQ(pernix_compressed_block_size(), BS);
    EXPECT_EQ(pernix_elements_per_block(BW), EPB);

    std::array<f32, total> input_f32{};
    std::array<f64, total> input_f64{};
    for (u32 i = 0; i < total; ++i) {
        const auto centered = static_cast<i32>(i % 31U) - 15;
        input_f32[i] = static_cast<f32>(centered) * 0.25f;
        input_f64[i] = static_cast<f64>(centered) * 0.25;
    }

    std::array<u8, BS * blocks> compressed_f32{};
    std::array<u8, BS * blocks> compressed_f64{};
    std::array<f32, total> restored_f32{};
    std::array<f64, total> restored_f64{};

    auto status = pernix_compress_blocks_f32(PERNIX_BACKEND_FALLBACK, BW, BS, input_f32.data(), 4.0f,
                                             compressed_f32.data(), blocks);
    ASSERT_EQ(status, PERNIX_STATUS_OK);
    status = pernix_decompress_blocks_f32(PERNIX_BACKEND_FALLBACK, BW, BS, compressed_f32.data(), 0.25f,
                                          restored_f32.data(), blocks, true);
    ASSERT_EQ(status, PERNIX_STATUS_OK);

    status = pernix_compress_blocks_f64(PERNIX_BACKEND_FALLBACK, BW, BS, input_f64.data(), 4.0,
                                        compressed_f64.data(), blocks);
    ASSERT_EQ(status, PERNIX_STATUS_OK);
    status = pernix_decompress_blocks_f64(PERNIX_BACKEND_FALLBACK, BW, BS, compressed_f64.data(), 0.25,
                                          restored_f64.data(), blocks, true);
    ASSERT_EQ(status, PERNIX_STATUS_OK);

    for (u32 i = 0; i < total; ++i) {
        EXPECT_FLOAT_EQ(restored_f32[i], input_f32[i]) << "f32 element " << i;
        EXPECT_DOUBLE_EQ(restored_f64[i], input_f64[i]) << "f64 element " << i;
    }

    std::array<f32, (BS * 8) / 4> unsigned_output{};
    status = pernix_decompress_block_f32(PERNIX_BACKEND_FALLBACK, 4, BS, compressed_f32.data(), 1.0f,
                                         unsigned_output.data(), false);
    ASSERT_EQ(status, PERNIX_STATUS_OK);
}

TEST(FallbackEdgeTest, CApiRejectsInvalidScale) {
    constexpr u32 BS = 64;
    constexpr u32 BW = 8;
    constexpr u32 EPB = (BS * 8) / BW;

    std::array<f32, EPB> input{};
    std::array<u8, BS> compressed{};
    std::array<f32, EPB> output{};

    EXPECT_EQ(pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, BW, BS, input.data(), 0.0f, compressed.data()),
              PERNIX_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(pernix_decompress_block_f32(PERNIX_BACKEND_FALLBACK, BW, BS, compressed.data(), -1.0f, output.data(),
                                          true),
              PERNIX_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, BW, BS, input.data(),
                                        std::numeric_limits<f32>::infinity(), compressed.data()),
              PERNIX_STATUS_INVALID_ARGUMENT);
}

TEST(FallbackEdgeTest, CApiRejectsInvalidScaleF64) {
    constexpr u32 BS = 64;
    constexpr u32 BW = 8;
    constexpr u32 EPB = (BS * 8) / BW;

    std::array<f64, EPB> input{};
    std::array<u8, BS> compressed{};
    std::array<f64, EPB> output{};

    EXPECT_EQ(pernix_compress_block_f64(PERNIX_BACKEND_FALLBACK, BW, BS, input.data(), 0.0, compressed.data()),
              PERNIX_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(pernix_decompress_block_f64(PERNIX_BACKEND_FALLBACK, BW, BS, compressed.data(), -1.0, output.data(),
                                          true),
              PERNIX_STATUS_INVALID_ARGUMENT);
}

TEST(FallbackEdgeTest, ScaleHelpersValidateInputs) {
    float scale_f32 = 0.0f;
    double scale_f64 = 0.0;

    EXPECT_EQ(pernix_scale_f32(32767.0f, 16, &scale_f32), PERNIX_STATUS_OK);
    EXPECT_FLOAT_EQ(scale_f32, 1.0f);
    EXPECT_EQ(pernix_scale_f64(32767.0, 16, &scale_f64), PERNIX_STATUS_OK);
    EXPECT_DOUBLE_EQ(scale_f64, 1.0);
    EXPECT_EQ(pernix_scale_f32(1.0f, 0, &scale_f32), PERNIX_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(pernix_scale_f32(-1.0f, 16, &scale_f32), PERNIX_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(pernix_scale_f64(std::numeric_limits<double>::infinity(), 16, &scale_f64),
              PERNIX_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(pernix_scale_f64(1.0, 16, nullptr), PERNIX_STATUS_INVALID_ARGUMENT);
}

#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
TEST(FallbackStdparEdgeTest, SignExtensionIsWellDefinedForNegativeValues) {
    constexpr u32 BS = 64;
    const std::array<u8, BS> input{0x08};

    std::array<f32, (BS * 8) / 4> output{};
    const auto st = pernix_decompress_block_f32(PERNIX_BACKEND_FALLBACK_STDPAR, 4, BS, input.data(), 1.0f,
                                                output.data(), true);
    ASSERT_EQ(st, PERNIX_STATUS_OK);
    EXPECT_EQ(output[0], -8.0f);
}

TEST(FallbackStdparEdgeTest, ClearsUnusedPaddingBytes) {
    constexpr u32 BS = 64;
    constexpr u32 BW = 24;
    constexpr u32 EPB = (BS * 8) / BW;

    std::array<f32, EPB> input{};
    std::array<u8, BS> output{};
    output.fill(0xA5);

    const auto st = pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK_STDPAR, BW, BS, input.data(), 1.0f,
                                              output.data());
    ASSERT_EQ(st, PERNIX_STATUS_OK);
    EXPECT_EQ(output[BS - 1], 0);
}

TEST(FallbackStdparEdgeTest, ClampsNonFiniteAndOutOfRangeBeforeNarrowing) {
    constexpr u32 BS = 64;
    constexpr u32 BW = 4;
    constexpr u32 EPB = (BS * 8) / BW;

    std::array<f32, EPB> input{};
    input[0] = std::numeric_limits<f32>::infinity();
    input[1] = -std::numeric_limits<f32>::infinity();
    input[2] = std::numeric_limits<f32>::quiet_NaN();

    std::array<u8, BS> compressed{};
    std::array<f32, EPB> restored{};

    auto st = pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK_STDPAR, BW, BS, input.data(), 1.0f,
                                        compressed.data());
    ASSERT_EQ(st, PERNIX_STATUS_OK);

    st = pernix_decompress_block_f32(PERNIX_BACKEND_FALLBACK_STDPAR, BW, BS, compressed.data(), 1.0f,
                                     restored.data(), true);
    ASSERT_EQ(st, PERNIX_STATUS_OK);

    EXPECT_EQ(restored[0], 7.0f);
    EXPECT_EQ(restored[1], -8.0f);
    EXPECT_EQ(restored[2], 0.0f);
}

TEST(FallbackStdparEdgeTest, TailWidthCompressMatchesScalarF32) {
    constexpr u32 BS  = 64;
    constexpr u32 BW  = 24;
    constexpr u32 EPB = (BS * 8) / BW;

    std::array<f32, EPB> input{};
    for (u32 i = 0; i < EPB; ++i) {
        input[i] = static_cast<f32>(static_cast<i32>(i) - 10);
    }

    std::array<u8, BS> scalar_block{};
    std::array<u8, BS> stdpar_block{};
    std::array<u8, BS> stdpar_blocks{};

    auto st = pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, BW, BS, input.data(), 1.0f, scalar_block.data());
    ASSERT_EQ(st, PERNIX_STATUS_OK);

    st = pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK_STDPAR, BW, BS, input.data(), 1.0f, stdpar_block.data());
    ASSERT_EQ(st, PERNIX_STATUS_OK);

    st = pernix_compress_blocks_f32(PERNIX_BACKEND_FALLBACK_STDPAR, BW, BS, input.data(), 1.0f,
                                    stdpar_blocks.data(), 1);
    ASSERT_EQ(st, PERNIX_STATUS_OK);

    for (u32 i = 0; i < BS; ++i) {
        EXPECT_EQ(stdpar_block[i], scalar_block[i]) << "block byte " << i;
        EXPECT_EQ(stdpar_blocks[i], scalar_block[i]) << "blocks byte " << i;
    }
}

TEST(FallbackStdparEdgeTest, TailWidthCompressMatchesScalarF64) {
    constexpr u32 BS  = 64;
    constexpr u32 BW  = 24;
    constexpr u32 EPB = (BS * 8) / BW;

    std::array<f64, EPB> input{};
    for (u32 i = 0; i < EPB; ++i) {
        input[i] = static_cast<f64>((static_cast<i32>(i) - 10) * 2);
    }

    std::array<u8, BS> scalar_block{};
    std::array<u8, BS> stdpar_block{};
    std::array<u8, BS> stdpar_blocks{};

    auto st = pernix_compress_block_f64(PERNIX_BACKEND_FALLBACK, BW, BS, input.data(), 1.0, scalar_block.data());
    ASSERT_EQ(st, PERNIX_STATUS_OK);

    st = pernix_compress_block_f64(PERNIX_BACKEND_FALLBACK_STDPAR, BW, BS, input.data(), 1.0, stdpar_block.data());
    ASSERT_EQ(st, PERNIX_STATUS_OK);

    st = pernix_compress_blocks_f64(PERNIX_BACKEND_FALLBACK_STDPAR, BW, BS, input.data(), 1.0,
                                    stdpar_blocks.data(), 1);
    ASSERT_EQ(st, PERNIX_STATUS_OK);

    for (u32 i = 0; i < BS; ++i) {
        EXPECT_EQ(stdpar_block[i], scalar_block[i]) << "block byte " << i;
        EXPECT_EQ(stdpar_blocks[i], scalar_block[i]) << "blocks byte " << i;
    }
}

TEST(FallbackStdparEdgeTest, TailWidthDecompressMatchesScalarF32) {
    constexpr u32 BS  = 64;
    constexpr u32 BW  = 24;
    constexpr u32 EPB = (BS * 8) / BW;

    std::array<f32, EPB> input{};
    for (u32 i = 0; i < EPB; ++i) {
        input[i] = static_cast<f32>(static_cast<i32>(i) - 10);
    }

    std::array<u8, BS> compressed{};
    std::array<f32, EPB> scalar_block{};
    std::array<f32, EPB> stdpar_block{};
    std::array<f32, EPB> stdpar_blocks{};

    auto st = pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, BW, BS, input.data(), 1.0f, compressed.data());
    ASSERT_EQ(st, PERNIX_STATUS_OK);

    st = pernix_decompress_block_f32(PERNIX_BACKEND_FALLBACK, BW, BS, compressed.data(), 1.0f, scalar_block.data(), true);
    ASSERT_EQ(st, PERNIX_STATUS_OK);

    st = pernix_decompress_block_f32(PERNIX_BACKEND_FALLBACK_STDPAR, BW, BS, compressed.data(), 1.0f,
                                     stdpar_block.data(), true);
    ASSERT_EQ(st, PERNIX_STATUS_OK);

    st = pernix_decompress_blocks_f32(PERNIX_BACKEND_FALLBACK_STDPAR, BW, BS, compressed.data(), 1.0f,
                                      stdpar_blocks.data(), 1, true);
    ASSERT_EQ(st, PERNIX_STATUS_OK);

    for (u32 i = 0; i < EPB; ++i) {
        EXPECT_FLOAT_EQ(stdpar_block[i], scalar_block[i]) << "block element " << i;
        EXPECT_FLOAT_EQ(stdpar_blocks[i], scalar_block[i]) << "blocks element " << i;
    }
}

TEST(FallbackStdparEdgeTest, TailWidthDecompressMatchesScalarF64) {
    constexpr u32 BS  = 64;
    constexpr u32 BW  = 24;
    constexpr u32 EPB = (BS * 8) / BW;

    std::array<f64, EPB> input{};
    for (u32 i = 0; i < EPB; ++i) {
        input[i] = static_cast<f64>((static_cast<i32>(i) - 10) * 2);
    }

    std::array<u8, BS> compressed{};
    std::array<f64, EPB> scalar_block{};
    std::array<f64, EPB> stdpar_block{};
    std::array<f64, EPB> stdpar_blocks{};

    auto st = pernix_compress_block_f64(PERNIX_BACKEND_FALLBACK, BW, BS, input.data(), 1.0, compressed.data());
    ASSERT_EQ(st, PERNIX_STATUS_OK);

    st = pernix_decompress_block_f64(PERNIX_BACKEND_FALLBACK, BW, BS, compressed.data(), 1.0, scalar_block.data(), true);
    ASSERT_EQ(st, PERNIX_STATUS_OK);

    st = pernix_decompress_block_f64(PERNIX_BACKEND_FALLBACK_STDPAR, BW, BS, compressed.data(), 1.0,
                                     stdpar_block.data(), true);
    ASSERT_EQ(st, PERNIX_STATUS_OK);

    st = pernix_decompress_blocks_f64(PERNIX_BACKEND_FALLBACK_STDPAR, BW, BS, compressed.data(), 1.0,
                                      stdpar_blocks.data(), 1, true);
    ASSERT_EQ(st, PERNIX_STATUS_OK);

    for (u32 i = 0; i < EPB; ++i) {
        EXPECT_DOUBLE_EQ(stdpar_block[i], scalar_block[i]) << "block element " << i;
        EXPECT_DOUBLE_EQ(stdpar_blocks[i], scalar_block[i]) << "blocks element " << i;
    }
}
#endif

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

TEST(ErrorCodeTest, FallbackAliasMatchesScalar) {
    EXPECT_EQ(PERNIX_BACKEND_FALLBACK, PERNIX_BACKEND_FALLBACK_SCALAR);
}

TEST(ErrorCodeTest, FallbackStdparReturnsUnsupportedImplementationByDefault) {
    constexpr u32 BS = 64;
    constexpr u8 BW = 8;

    f32 src[(BS * 8U) / BW] = {};
    u8 dst[BS] = {};

#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
    const auto st = pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK_STDPAR, BW, BS, src, 1.0f, dst);
    EXPECT_EQ(st, PERNIX_STATUS_OK);
#else
    const auto st = pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK_STDPAR, BW, BS, src, 1.0f, dst);
    EXPECT_EQ(st, PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION);
#endif
}

TEST(ErrorCodeTest, FallbackSimdReturnsUnsupportedImplementationByDefault) {
    constexpr u32 BS = 64;
    constexpr u8 BW = 8;

    f32 src[(BS * 8U) / BW] = {};
    u8 dst[BS] = {};

    const auto st = pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK_SIMD, BW, BS, src, 1.0f, dst);
    EXPECT_EQ(st, PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION);
}

TEST(ErrorCodeTest, ExplicitUnavailableBackendReturnsError) {
    constexpr u32 BS = 64;
    constexpr u8 BW = 8;

    f32 src[(BS * 8U) / BW] = {};
    u8 dst[BS] = {};

    pernix_backend backend = PERNIX_BACKEND_FALLBACK;
    bool found = false;

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    if (!__builtin_cpu_supports("avx512vbmi")) {
        backend = PERNIX_BACKEND_X86_AVX512_VBMI;
        found = true;
    } else if (!(__builtin_cpu_supports("avx2") && __builtin_cpu_supports("bmi2"))) {
        backend = PERNIX_BACKEND_X86_BMI2;
        found = true;
    } else if (!__builtin_cpu_supports("avx2")) {
        backend = PERNIX_BACKEND_X86_AVX2;
        found = true;
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    const bool neon = true;
#if defined(__linux__) && defined(__aarch64__)
    const bool sve2 = (getauxval(AT_HWCAP2) & HWCAP2_SVE2) != 0;
#else
    const bool sve2 = false;
#endif
    if (!sve2) {
        backend = PERNIX_BACKEND_ARM64_SVE;
        found = true;
    } else if (!neon) {
        backend = PERNIX_BACKEND_ARM64_NEON;
        found = true;
    }
#endif

    if (!found) {
        GTEST_SKIP() << "No compiled explicit backend is unsupported on this machine";
    }

    const auto st = pernix_compress_block_f32(backend, BW, BS, src, 1.0f, dst);
    // A backend can be absent from the build, or compiled but unsupported by this CPU.
    EXPECT_TRUE(st == PERNIX_STATUS_UNSUPPORTED_BACKEND ||
                st == PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION);
}
