#include <gtest/gtest.h>

#include <pernix/pernix.hpp>

#include <array>
#include <vector>

namespace {
constexpr u8 kBitWidth = 8;
constexpr u32 kBlockSize = PERNIX_TEST_BLOCK_SIZE;
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

    ASSERT_EQ(pernix::compress_block(backend, kBitWidth, kBlockSize, std::span<const FloatT>(input),
                                     static_cast<FloatT>(1), std::span<u8>(compressed)),
              PERNIX_STATUS_OK);
    ASSERT_EQ(pernix::decompress_block(backend, kBitWidth, kBlockSize, std::span<const u8>(compressed),
                                       static_cast<FloatT>(1), std::span<FloatT>(restored)),
              PERNIX_STATUS_OK);

    for (usize i = 0; i < restored.size(); ++i) {
        EXPECT_NEAR(restored[i], input[i], static_cast<double>(1.0));
    }
}
}

TEST(HeaderOnlyPernix, FallbackFloatRoundTrip) {
    expect_round_trip<float>(pernix::Backend::Fallback);
}

TEST(HeaderOnlyPernix, FallbackAliasMatchesScalar) {
    EXPECT_EQ(static_cast<int>(pernix::Backend::Fallback), static_cast<int>(pernix::Backend::FallbackScalar));
}

TEST(HeaderOnlyPernix, FallbackStdparReturnsUnsupportedImplementation) {
    auto input = make_input<float>();
    std::vector<u8> compressed(kBlockSize);

#if defined(PERNIX_BUILD_FALLBACK_STDPAR)
    EXPECT_EQ(pernix::compress_block(pernix::Backend::FallbackStdpar, kBitWidth, kBlockSize,
                                     std::span<const float>(input), 1.0f, std::span<u8>(compressed)),
              PERNIX_STATUS_OK);
#else
    EXPECT_EQ(pernix::compress_block(pernix::Backend::FallbackStdpar, kBitWidth, kBlockSize,
                                     std::span<const float>(input), 1.0f, std::span<u8>(compressed)),
              PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION);
#endif
}

TEST(HeaderOnlyPernix, FallbackSimdReturnsUnsupportedImplementation) {
    auto input = make_input<float>();
    std::vector<u8> compressed(kBlockSize);

    EXPECT_EQ(pernix::compress_block(pernix::Backend::FallbackSimd, kBitWidth, kBlockSize,
                                     std::span<const float>(input), 1.0f, std::span<u8>(compressed)),
              PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION);
}

TEST(HeaderOnlyPernix, AutoDoubleRoundTrip) {
    expect_round_trip<double>(pernix::Backend::Auto);
}
