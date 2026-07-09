#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <pernix/pernix.hpp>
#include <vector>

namespace {
constexpr u8 kBitWidth       = 8;
constexpr u32 kBlockSize     = PERNIX_TEST_BLOCK_SIZE;
constexpr u32 kBlockElements = (kBlockSize * 8U) / kBitWidth;

template <typename FloatT>
std::vector<FloatT> make_input() {
    std::vector<FloatT> values(kBlockElements);
    for (u32 i = 0; i < kBlockElements; ++i) {
        values[static_cast<usize>(i)] = static_cast<FloatT>((static_cast<i32>(i % 13U) - 6) * 0.25);
    }
    return values;
}

template <typename FloatT>
void expect_round_trip(const pernix::Backend backend) {
    auto input = make_input<FloatT>();
    std::vector<u8> compressed(kBlockSize);
    std::vector<FloatT> restored(kBlockElements, static_cast<FloatT>(0));

    ASSERT_EQ(pernix::compress_block(backend, kBitWidth, kBlockSize, std::span<const FloatT>(input), static_cast<FloatT>(1),
                                     std::span<u8>(compressed)),
              PERNIX_STATUS_OK);
    ASSERT_EQ(pernix::decompress_block(backend, kBitWidth, kBlockSize, std::span<const u8>(compressed), static_cast<FloatT>(1),
                                       std::span<FloatT>(restored)),
              PERNIX_STATUS_OK);

    for (usize i = 0; i < restored.size(); ++i) {
        EXPECT_NEAR(restored[i], input[i], static_cast<double>(1.0));
    }
}
}  // namespace

TEST(HeaderOnlyPernix, FallbackFloatRoundTrip) {
    expect_round_trip<float>(pernix::Backend::Fallback);
}

TEST(HeaderOnlyPernix, PublicHelpersDescribeFixedBlockFormat) {
    EXPECT_EQ(pernix::min_bit_width(), 1);
    EXPECT_EQ(pernix::max_bit_width(), 24);
    EXPECT_TRUE(pernix::is_valid_bit_width(1));
    EXPECT_TRUE(pernix::is_valid_bit_width(24));
    EXPECT_FALSE(pernix::is_valid_bit_width(0));
    EXPECT_FALSE(pernix::is_valid_bit_width(25));
    EXPECT_TRUE(pernix::is_valid_block_size(64));
    EXPECT_FALSE(pernix::is_valid_block_size(96));
    EXPECT_EQ(pernix::compressed_block_size(), 64);
    EXPECT_EQ(pernix::elements_per_block(16), 32);
    EXPECT_EQ(pernix::elements_per_block(0), 0);

    float scale_f32 = 0.0f;
    EXPECT_EQ(pernix::scale_from_bmax(32767.0f, 16, scale_f32), PERNIX_STATUS_OK);
    EXPECT_FLOAT_EQ(scale_f32, 1.0f);

    double scale_f64 = 0.0;
    EXPECT_EQ(pernix::scale_from_bmax(0.0, 16, scale_f64), PERNIX_STATUS_OK);
    EXPECT_GT(scale_f64, 0.0);
}

TEST(HeaderOnlyPernix, ScaleHelpersNameCompressionAndDecompressionConventions) {
    float decompression_scale = 0.0f;
    float compression_scale   = 0.0f;
    float inverse_scale       = 0.0f;

    EXPECT_EQ(pernix::decompression_scale_from_bmax(127.0f, 8, decompression_scale), PERNIX_STATUS_OK);
    EXPECT_FLOAT_EQ(decompression_scale, 1.0f);
    EXPECT_EQ(pernix::compression_scale_from_bmax(127.0f, 8, compression_scale), PERNIX_STATUS_OK);
    EXPECT_FLOAT_EQ(compression_scale, 1.0f);
    EXPECT_EQ(pernix::inverse_scale(decompression_scale, inverse_scale), PERNIX_STATUS_OK);
    EXPECT_FLOAT_EQ(inverse_scale, compression_scale);

    EXPECT_EQ(pernix::inverse_scale(0.0f, inverse_scale), PERNIX_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(pernix::compression_scale_from_bmax(1.0f, 0, compression_scale), PERNIX_STATUS_INVALID_ARGUMENT);
}

TEST(HeaderOnlyPernix, StatusStringsArePublic) {
    pernix::Status status = PERNIX_STATUS_OK;
    EXPECT_EQ(status, PERNIX_STATUS_OK);
    EXPECT_STREQ(pernix::status_string(status), "PERNIX_STATUS_OK");
    EXPECT_STREQ(pernix::status_string(PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION), "PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION");
    EXPECT_STREQ(pernix::status_string(static_cast<pernix_status>(123)), "PERNIX_STATUS_UNKNOWN");
}

TEST(HeaderOnlyPernix, RejectsUndersizedCompressInputSpan) {
    std::vector<float> input(kBlockElements - 1U, 0.0f);
    std::vector<u8> output(kBlockSize);

    EXPECT_EQ(pernix::compress_block(pernix::Backend::Fallback, kBitWidth, kBlockSize, std::span<const float>(input), 1.0f,
                                     std::span<u8>(output)),
              PERNIX_STATUS_INVALID_ARGUMENT);
}

TEST(HeaderOnlyPernix, RejectsUndersizedCompressOutputSpan) {
    std::vector<float> input(kBlockElements, 0.0f);
    std::vector<u8> output(kBlockSize - 1U);

    EXPECT_EQ(pernix::compress_block(pernix::Backend::Fallback, kBitWidth, kBlockSize, std::span<const float>(input), 1.0f,
                                     std::span<u8>(output)),
              PERNIX_STATUS_INVALID_ARGUMENT);
}

TEST(HeaderOnlyPernix, RejectsUndersizedDecompressSpans) {
    std::vector<u8> short_input(kBlockSize - 1U);
    std::vector<float> output(kBlockElements);
    EXPECT_EQ(pernix::decompress_block(pernix::Backend::Fallback, kBitWidth, kBlockSize, std::span<const u8>(short_input), 1.0f,
                                       std::span<float>(output)),
              PERNIX_STATUS_INVALID_ARGUMENT);

    std::vector<u8> input(kBlockSize);
    std::vector<float> short_output(kBlockElements - 1U);
    EXPECT_EQ(pernix::decompress_block(pernix::Backend::Fallback, kBitWidth, kBlockSize, std::span<const u8>(input), 1.0f,
                                       std::span<float>(short_output)),
              PERNIX_STATUS_INVALID_ARGUMENT);
}

TEST(HeaderOnlyPernix, RejectsInvalidSpanParametersBeforeDispatch) {
    std::vector<float> input(kBlockElements, 0.0f);
    std::vector<u8> output(kBlockSize);

    EXPECT_EQ(pernix::compress_block(pernix::Backend::Fallback, 0, kBlockSize, std::span<const float>(input), 1.0f, std::span<u8>(output)),
              PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH);
    EXPECT_EQ(pernix::compress_block(pernix::Backend::Fallback, kBitWidth, 96, std::span<const float>(input), 1.0f, std::span<u8>(output)),
              PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE);
    EXPECT_EQ(pernix::compress_blocks(pernix::Backend::Fallback, kBitWidth, kBlockSize, std::span<const float>(input), 1.0f,
                                      std::span<u8>(output), 0),
              PERNIX_STATUS_INVALID_ARGUMENT);
}

TEST(HeaderOnlyPernix, RejectsInvalidScale) {
    std::vector<float> input(kBlockElements, 0.0f);
    std::vector<u8> compressed(kBlockSize);
    std::vector<float> output(input.size());

    EXPECT_EQ(pernix::compress_block(pernix::Backend::Fallback, kBitWidth, kBlockSize, std::span<const float>(input), 0.0f,
                                     std::span<u8>(compressed)),
              PERNIX_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(pernix::decompress_block(pernix::Backend::Fallback, kBitWidth, kBlockSize, std::span<const u8>(compressed), -1.0f,
                                       std::span<float>(output)),
              PERNIX_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(pernix::compress_block(pernix::Backend::Fallback, kBitWidth, kBlockSize, std::span<const float>(input),
                                     std::numeric_limits<float>::infinity(), std::span<u8>(compressed)),
              PERNIX_STATUS_INVALID_ARGUMENT);
}

TEST(HeaderOnlyPernix, FallbackAliasMatchesScalar) {
    EXPECT_EQ(static_cast<int>(pernix::Backend::Fallback), static_cast<int>(pernix::Backend::FallbackScalar));
}

TEST(HeaderOnlyPernix, FallbackStdparReturnsUnsupportedImplementation) {
    auto input = make_input<float>();
    std::vector<u8> compressed(kBlockSize);

#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
    EXPECT_EQ(pernix::compress_block(pernix::Backend::FallbackStdpar, kBitWidth, kBlockSize, std::span<const float>(input), 1.0f,
                                     std::span<u8>(compressed)),
              PERNIX_STATUS_OK);
#else
    EXPECT_EQ(pernix::compress_block(pernix::Backend::FallbackStdpar, kBitWidth, kBlockSize, std::span<const float>(input), 1.0f,
                                     std::span<u8>(compressed)),
              PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION);
#endif
}

TEST(HeaderOnlyPernix, FallbackSimdReturnsUnsupportedImplementation) {
    auto input = make_input<float>();
    std::vector<u8> compressed(kBlockSize);

    EXPECT_EQ(pernix::compress_block(pernix::Backend::FallbackSimd, kBitWidth, kBlockSize, std::span<const float>(input), 1.0f,
                                     std::span<u8>(compressed)),
              PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION);
}

TEST(HeaderOnlyPernix, AutoDoubleRoundTrip) {
    expect_round_trip<double>(pernix::Backend::Auto);
}
