#include <pernix/pernix.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

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

template <typename FloatT>
std::vector<FloatT> make_block_pattern(const u8 bit_width, const u32 blocks) {
    const u32 elements_per_block = pernix::elements_per_block(bit_width);
    std::vector<FloatT> values(static_cast<usize>(elements_per_block) * blocks);

    for (usize i = 0; i < values.size(); ++i) {
        values[i] = static_cast<FloatT>((static_cast<i32>(i % 29U) - 14) * 0.25);
    }

    return values;
}

template <typename FloatT>
void expect_stdpar_matches_scalar_for_all_bit_widths() {
    constexpr u32 blocks = 3;

    for (u8 bit_width = pernix::min_bit_width(); bit_width <= pernix::max_bit_width(); ++bit_width) {
        const u32 elements_per_block = pernix::elements_per_block(bit_width);
        auto input                   = make_block_pattern<FloatT>(bit_width, blocks);

        std::vector<u8> scalar_block(kBlockSize);
        std::vector<u8> stdpar_block(kBlockSize);
        std::vector<u8> scalar_blocks(static_cast<usize>(kBlockSize) * blocks);
        std::vector<u8> stdpar_blocks(static_cast<usize>(kBlockSize) * blocks);
        std::vector<FloatT> scalar_restored(elements_per_block * blocks);
        std::vector<FloatT> stdpar_restored(elements_per_block * blocks);

        ASSERT_EQ(pernix::compress_block(pernix::Backend::Fallback, bit_width, kBlockSize,
                                         std::span<const FloatT>(input.data(), elements_per_block), static_cast<FloatT>(4),
                                         std::span<u8>(scalar_block)),
                  PERNIX_STATUS_OK)
            << "bit_width=" << static_cast<u32>(bit_width);
        ASSERT_EQ(pernix::compress_block(pernix::Backend::FallbackStdpar, bit_width, kBlockSize,
                                         std::span<const FloatT>(input.data(), elements_per_block), static_cast<FloatT>(4),
                                         std::span<u8>(stdpar_block)),
                  PERNIX_STATUS_OK)
            << "bit_width=" << static_cast<u32>(bit_width);
        EXPECT_EQ(stdpar_block, scalar_block) << "bit_width=" << static_cast<u32>(bit_width);

        ASSERT_EQ(pernix::compress_blocks(pernix::Backend::Fallback, bit_width, kBlockSize, std::span<const FloatT>(input),
                                          static_cast<FloatT>(4), std::span<u8>(scalar_blocks), blocks),
                  PERNIX_STATUS_OK)
            << "bit_width=" << static_cast<u32>(bit_width);
        ASSERT_EQ(pernix::compress_blocks(pernix::Backend::FallbackStdpar, bit_width, kBlockSize, std::span<const FloatT>(input),
                                          static_cast<FloatT>(4), std::span<u8>(stdpar_blocks), blocks),
                  PERNIX_STATUS_OK)
            << "bit_width=" << static_cast<u32>(bit_width);
        EXPECT_EQ(stdpar_blocks, scalar_blocks) << "bit_width=" << static_cast<u32>(bit_width);

        ASSERT_EQ(pernix::decompress_blocks(pernix::Backend::Fallback, bit_width, kBlockSize, std::span<const u8>(scalar_blocks),
                                            static_cast<FloatT>(0.25), std::span<FloatT>(scalar_restored), blocks, true),
                  PERNIX_STATUS_OK)
            << "bit_width=" << static_cast<u32>(bit_width);
        ASSERT_EQ(pernix::decompress_blocks(pernix::Backend::FallbackStdpar, bit_width, kBlockSize, std::span<const u8>(stdpar_blocks),
                                            static_cast<FloatT>(0.25), std::span<FloatT>(stdpar_restored), blocks, true),
                  PERNIX_STATUS_OK)
            << "bit_width=" << static_cast<u32>(bit_width);
        EXPECT_EQ(stdpar_restored, scalar_restored) << "bit_width=" << static_cast<u32>(bit_width);
    }
}

template <typename FloatT>
void expect_cross_compatibility(const u8 bit_width, const u32 blocks, const FloatT compression_scale, const FloatT decompression_scale,
                                const bool sign_values = true) {
    const u32 elements_per_block = pernix::elements_per_block(bit_width);
    auto input                   = make_block_pattern<FloatT>(bit_width, blocks);

    std::vector<u8> scalar_compressed(static_cast<usize>(kBlockSize) * blocks);
    std::vector<u8> stdpar_compressed(static_cast<usize>(kBlockSize) * blocks);
    std::vector<FloatT> scalar_to_stdpar(static_cast<usize>(elements_per_block) * blocks, static_cast<FloatT>(0));
    std::vector<FloatT> stdpar_to_scalar(static_cast<usize>(elements_per_block) * blocks, static_cast<FloatT>(0));

    ASSERT_EQ(pernix::compress_blocks(pernix::Backend::Fallback, bit_width, kBlockSize, std::span<const FloatT>(input), compression_scale,
                                      std::span<u8>(scalar_compressed), blocks),
              PERNIX_STATUS_OK);
    ASSERT_EQ(pernix::compress_blocks(pernix::Backend::FallbackStdpar, bit_width, kBlockSize, std::span<const FloatT>(input),
                                      compression_scale, std::span<u8>(stdpar_compressed), blocks),
              PERNIX_STATUS_OK);

    ASSERT_EQ(pernix::decompress_blocks(pernix::Backend::FallbackStdpar, bit_width, kBlockSize, std::span<const u8>(scalar_compressed),
                                        decompression_scale, std::span<FloatT>(scalar_to_stdpar), blocks, sign_values),
              PERNIX_STATUS_OK);
    ASSERT_EQ(pernix::decompress_blocks(pernix::Backend::Fallback, bit_width, kBlockSize, std::span<const u8>(stdpar_compressed),
                                        decompression_scale, std::span<FloatT>(stdpar_to_scalar), blocks, sign_values),
              PERNIX_STATUS_OK);

    EXPECT_EQ(scalar_to_stdpar, stdpar_to_scalar) << "bit_width=" << static_cast<u32>(bit_width);
}

template <typename FloatT>
void expect_single_block_partial_group_matches_scalar(const u8 bit_width) {
    const u32 elements_per_block = pernix::elements_per_block(bit_width);
    auto input                   = make_block_pattern<FloatT>(bit_width, 1);

    ASSERT_GT(elements_per_block, 8U) << "bit_width=" << static_cast<u32>(bit_width);
    ASSERT_NE(elements_per_block % 8U, 0U) << "bit_width=" << static_cast<u32>(bit_width);

    std::vector<u8> scalar_compressed(kBlockSize);
    std::vector<u8> stdpar_compressed(kBlockSize);
    std::vector<FloatT> scalar_restored(elements_per_block, static_cast<FloatT>(0));
    std::vector<FloatT> stdpar_restored(elements_per_block, static_cast<FloatT>(0));

    ASSERT_EQ(pernix::compress_block(pernix::Backend::Fallback, bit_width, kBlockSize, std::span<const FloatT>(input),
                                     static_cast<FloatT>(4), std::span<u8>(scalar_compressed)),
              PERNIX_STATUS_OK);
    ASSERT_EQ(pernix::compress_block(pernix::Backend::FallbackStdpar, bit_width, kBlockSize, std::span<const FloatT>(input),
                                     static_cast<FloatT>(4), std::span<u8>(stdpar_compressed)),
              PERNIX_STATUS_OK);
    EXPECT_EQ(stdpar_compressed, scalar_compressed) << "bit_width=" << static_cast<u32>(bit_width);

    ASSERT_EQ(pernix::decompress_block(pernix::Backend::Fallback, bit_width, kBlockSize, std::span<const u8>(scalar_compressed),
                                       static_cast<FloatT>(0.25), std::span<FloatT>(scalar_restored), true),
              PERNIX_STATUS_OK);
    ASSERT_EQ(pernix::decompress_block(pernix::Backend::FallbackStdpar, bit_width, kBlockSize, std::span<const u8>(stdpar_compressed),
                                       static_cast<FloatT>(0.25), std::span<FloatT>(stdpar_restored), true),
              PERNIX_STATUS_OK);
    EXPECT_EQ(stdpar_restored, scalar_restored) << "bit_width=" << static_cast<u32>(bit_width);
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

#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
TEST(HeaderOnlyPernix, FallbackStdparMatchesScalarForAllBitWidthsF32) {
    expect_stdpar_matches_scalar_for_all_bit_widths<float>();
}

TEST(HeaderOnlyPernix, FallbackStdparMatchesScalarForAllBitWidthsF64) {
    expect_stdpar_matches_scalar_for_all_bit_widths<double>();
}

TEST(HeaderOnlyPernix, FallbackStdparMatchesScalarForEdgeValues) {
    constexpr u8 bit_width = 4;
    constexpr u32 blocks   = 2;

    const u32 elements_per_block = pernix::elements_per_block(bit_width);
    std::vector<float> input(static_cast<usize>(elements_per_block) * blocks, 0.0f);
    input[0] = 0.49f;
    input[1] = 0.50f;
    input[2] = 1.49f;
    input[3] = 1.50f;
    input[4] = std::numeric_limits<float>::infinity();
    input[5] = -std::numeric_limits<float>::infinity();
    input[6] = std::numeric_limits<float>::quiet_NaN();
    input[7] = 99.0f;
    input[8] = -99.0f;

    std::vector<u8> scalar_compressed(static_cast<usize>(kBlockSize) * blocks);
    std::vector<u8> stdpar_compressed(static_cast<usize>(kBlockSize) * blocks);
    std::vector<float> scalar_signed(input.size());
    std::vector<float> stdpar_signed(input.size());

    ASSERT_EQ(pernix::compress_blocks(pernix::Backend::Fallback, bit_width, kBlockSize, std::span<const float>(input), 1.0f,
                                      std::span<u8>(scalar_compressed), blocks),
              PERNIX_STATUS_OK);
    ASSERT_EQ(pernix::compress_blocks(pernix::Backend::FallbackStdpar, bit_width, kBlockSize, std::span<const float>(input), 1.0f,
                                      std::span<u8>(stdpar_compressed), blocks),
              PERNIX_STATUS_OK);
    EXPECT_EQ(stdpar_compressed, scalar_compressed);

    ASSERT_EQ(pernix::decompress_blocks(pernix::Backend::Fallback, bit_width, kBlockSize, std::span<const u8>(scalar_compressed), 1.0f,
                                        std::span<float>(scalar_signed), blocks, true),
              PERNIX_STATUS_OK);
    ASSERT_EQ(pernix::decompress_blocks(pernix::Backend::FallbackStdpar, bit_width, kBlockSize, std::span<const u8>(stdpar_compressed),
                                        1.0f, std::span<float>(stdpar_signed), blocks, true),
              PERNIX_STATUS_OK);
    EXPECT_EQ(stdpar_signed, scalar_signed);
}

TEST(HeaderOnlyPernix, FallbackStdparSingleBlockPartialGroupsMatchScalar) {
    for (const u8 bit_width : std::array<u8, 8>{3, 5, 6, 7, 10, 11, 12, 24}) {
        expect_single_block_partial_group_matches_scalar<float>(bit_width);
    }
}

TEST(HeaderOnlyPernix, FallbackStdparCrossCompatibilityMatchesScalarForSignedValues) {
    for (const u8 bit_width : std::array<u8, 8>{1, 3, 5, 8, 9, 12, 17, 24}) {
        expect_cross_compatibility<float>(bit_width, 2, 4.0f, 0.25f, true);
    }
}

TEST(HeaderOnlyPernix, FallbackStdparCrossCompatibilityMatchesScalarForUnsignedDecode) {
    expect_cross_compatibility<float>(4, 2, 1.0f, 1.0f, false);
}

TEST(HeaderOnlyPernix, FallbackStdparSignValuesFalseMatchesScalar) {
    constexpr u8 bit_width       = 4;
    const u32 elements_per_block = pernix::elements_per_block(bit_width);

    std::vector<u8> compressed(kBlockSize, 0);
    compressed[0] = 0x0F;

    std::vector<float> scalar_output(elements_per_block, 0.0f);
    std::vector<float> stdpar_output(elements_per_block, 0.0f);

    ASSERT_EQ(pernix::decompress_block(pernix::Backend::Fallback, bit_width, kBlockSize, std::span<const u8>(compressed), 1.0f,
                                       std::span<float>(scalar_output), false),
              PERNIX_STATUS_OK);
    ASSERT_EQ(pernix::decompress_block(pernix::Backend::FallbackStdpar, bit_width, kBlockSize, std::span<const u8>(compressed), 1.0f,
                                       std::span<float>(stdpar_output), false),
              PERNIX_STATUS_OK);
    EXPECT_EQ(stdpar_output, scalar_output);
}

TEST(HeaderOnlyPernix, FallbackStdparRepeatedExecutionIsDeterministic) {
    constexpr u8 bit_width = 12;
    constexpr u32 blocks   = 3;

    auto input = make_block_pattern<float>(bit_width, blocks);
    std::vector<u8> first_compressed(static_cast<usize>(kBlockSize) * blocks);
    std::vector<u8> second_compressed(static_cast<usize>(kBlockSize) * blocks);
    std::vector<float> first_restored(input.size(), 0.0f);
    std::vector<float> second_restored(input.size(), 0.0f);

    ASSERT_EQ(pernix::compress_blocks(pernix::Backend::FallbackStdpar, bit_width, kBlockSize, std::span<const float>(input), 4.0f,
                                      std::span<u8>(first_compressed), blocks),
              PERNIX_STATUS_OK);
    ASSERT_EQ(pernix::compress_blocks(pernix::Backend::FallbackStdpar, bit_width, kBlockSize, std::span<const float>(input), 4.0f,
                                      std::span<u8>(second_compressed), blocks),
              PERNIX_STATUS_OK);
    EXPECT_EQ(second_compressed, first_compressed);

    ASSERT_EQ(pernix::decompress_blocks(pernix::Backend::FallbackStdpar, bit_width, kBlockSize, std::span<const u8>(first_compressed),
                                        0.25f, std::span<float>(first_restored), blocks, true),
              PERNIX_STATUS_OK);
    ASSERT_EQ(pernix::decompress_blocks(pernix::Backend::FallbackStdpar, bit_width, kBlockSize, std::span<const u8>(second_compressed),
                                        0.25f, std::span<float>(second_restored), blocks, true),
              PERNIX_STATUS_OK);
    EXPECT_EQ(second_restored, first_restored);
}

TEST(HeaderOnlyPernix, FallbackStdparRepeatedSingleBlockCompressionIsDeterministic) {
    constexpr u8 bit_width = 17;

    auto input = make_block_pattern<double>(bit_width, 1);
    std::vector<u8> first_compressed(kBlockSize);
    std::vector<u8> second_compressed(kBlockSize);

    ASSERT_EQ(pernix::compress_block(pernix::Backend::FallbackStdpar, bit_width, kBlockSize, std::span<const double>(input), 2.0,
                                     std::span<u8>(first_compressed)),
              PERNIX_STATUS_OK);
    ASSERT_EQ(pernix::compress_block(pernix::Backend::FallbackStdpar, bit_width, kBlockSize, std::span<const double>(input), 2.0,
                                     std::span<u8>(second_compressed)),
              PERNIX_STATUS_OK);
    EXPECT_EQ(second_compressed, first_compressed);
}
#endif

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
