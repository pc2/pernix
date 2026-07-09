#include <testset.h>

#include <pernix/pernix.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#if PERNIX_TEST_BLOCK_SIZE == 64
namespace {
constexpr u32 kBlockSize = 64;
constexpr u32 kBlocks = 3;

enum class Pattern {
    Zero,
    Constant,
    Increasing,
    Decreasing,
    SmallMagnitude,
    LargeMagnitude,
    QuantizationBoundary,
    PseudoRandom
};

struct PatternCase {
    Pattern pattern;
    std::string_view name;
};

constexpr std::array<PatternCase, 8> kPatternCases{{
    {Pattern::Zero, "zero"},
    {Pattern::Constant, "constant"},
    {Pattern::Increasing, "increasing"},
    {Pattern::Decreasing, "decreasing"},
    {Pattern::SmallMagnitude, "small_magnitude"},
    {Pattern::LargeMagnitude, "large_magnitude"},
    {Pattern::QuantizationBoundary, "quantization_boundary"},
    {Pattern::PseudoRandom, "pseudo_random"},
}};

constexpr std::array<pernix_backend, 5> kComparisonBackends{{
    PERNIX_BACKEND_X86_AVX2,
    PERNIX_BACKEND_X86_BMI2,
    PERNIX_BACKEND_X86_AVX512_VBMI,
    PERNIX_BACKEND_ARM64_NEON,
    PERNIX_BACKEND_ARM64_SVE,
}};

template <typename FloatT>
FloatT abs_value(const FloatT value) {
    return value < FloatT{0} ? -value : value;
}

template <typename FloatT>
FloatT max_abs(const std::vector<FloatT>& values) {
    FloatT bmax = FloatT{0};
    for (const FloatT value : values) {
        bmax = std::max(bmax, abs_value(value));
    }
    return bmax;
}

template <typename FloatT>
pernix_status compute_scale(const FloatT bmax, const u8 bit_width, FloatT& scale) {
    if constexpr (std::is_same_v<FloatT, float>) {
        return pernix_scale_f32(bmax, bit_width, &scale);
    } else {
        return pernix_scale_f64(bmax, bit_width, &scale);
    }
}

template <typename FloatT>
FloatT tolerance_for_scale(const FloatT scale) {
    return (abs_value(scale) * static_cast<FloatT>(0.55)) +
           (std::numeric_limits<FloatT>::epsilon() * static_cast<FloatT>(64));
}

template <typename FloatT>
std::vector<FloatT> make_dataset(const Pattern pattern, const u8 bit_width, const u32 blocks = kBlocks) {
    const u32 elements_per_block = pernix_elements_per_block(bit_width);
    std::vector<FloatT> values(static_cast<usize>(elements_per_block) * blocks);
    const bool one_bit = bit_width == 1;
    const FloatT amplitude = [&] {
        switch (pattern) {
        case Pattern::SmallMagnitude:
            return static_cast<FloatT>(1.0e-5);
        case Pattern::LargeMagnitude:
            return static_cast<FloatT>(1.0e6);
        default:
            return static_cast<FloatT>(16);
        }
    }();

    std::mt19937 rng(0x5065726eU + bit_width);
    std::uniform_real_distribution<double> uniform(one_bit ? 0.0 : -1.0, 1.0);

    for (usize i = 0; i < values.size(); ++i) {
        const FloatT unit = values.size() > 1
                                ? static_cast<FloatT>((static_cast<double>(i) / static_cast<double>(values.size() - 1)) *
                                                      2.0 - 1.0)
                                : FloatT{0};
        switch (pattern) {
        case Pattern::Zero:
            values[i] = FloatT{0};
            break;
        case Pattern::Constant:
            values[i] = one_bit ? amplitude : amplitude / FloatT{2};
            break;
        case Pattern::Increasing:
            values[i] = one_bit ? static_cast<FloatT>(i % 2U) * amplitude : unit * amplitude;
            break;
        case Pattern::Decreasing:
            values[i] = one_bit ? static_cast<FloatT>((i + 1U) % 2U) * amplitude : -unit * amplitude;
            break;
        case Pattern::SmallMagnitude:
        case Pattern::LargeMagnitude:
            values[i] = one_bit ? static_cast<FloatT>(i % 2U) * amplitude
                                : static_cast<FloatT>(((static_cast<int>(i % 9U) - 4) / 4.0)) * amplitude;
            break;
        case Pattern::QuantizationBoundary: {
            const FloatT step = amplitude / static_cast<FloatT>(17);
            if (one_bit) {
                values[i] = static_cast<FloatT>(i % 2U) * step;
            } else {
                const int bucket = static_cast<int>(i % 9U) - 4;
                const FloatT nudge = (i % 2U == 0U) ? step * static_cast<FloatT>(0.49)
                                                    : step * static_cast<FloatT>(-0.49);
                values[i] = (static_cast<FloatT>(bucket) * step) + nudge;
            }
            break;
        }
        case Pattern::PseudoRandom:
            values[i] = static_cast<FloatT>(uniform(rng)) * amplitude;
            break;
        }
    }

    return values;
}

template <typename FloatT>
void expect_near_input(const std::vector<FloatT>& input, const std::vector<FloatT>& restored, const FloatT scale) {
    ASSERT_EQ(restored.size(), input.size());
    const FloatT tolerance = tolerance_for_scale(scale);
    for (usize i = 0; i < input.size(); ++i) {
        ASSERT_NEAR(restored[i], input[i], tolerance) << "element=" << i << ", scale=" << scale;
    }
}

template <typename FloatT>
pernix_status compress_blocks(pernix_backend backend, const u8 bit_width, const std::vector<FloatT>& input,
                              const FloatT inverse_scale, std::vector<u8>& compressed, const u32 blocks) {
    if constexpr (std::is_same_v<FloatT, float>) {
        return pernix_compress_blocks_f32(backend, bit_width, kBlockSize, input.data(), inverse_scale,
                                          compressed.data(), blocks);
    } else {
        return pernix_compress_blocks_f64(backend, bit_width, kBlockSize, input.data(), inverse_scale,
                                          compressed.data(), blocks);
    }
}

template <typename FloatT>
pernix_status decompress_blocks(pernix_backend backend, const u8 bit_width, const std::vector<u8>& compressed,
                                const FloatT scale, std::vector<FloatT>& output, const u32 blocks,
                                const bool sign_values = true) {
    if constexpr (std::is_same_v<FloatT, float>) {
        return pernix_decompress_blocks_f32(backend, bit_width, kBlockSize, compressed.data(), scale, output.data(),
                                            blocks, sign_values);
    } else {
        return pernix_decompress_blocks_f64(backend, bit_width, kBlockSize, compressed.data(), scale, output.data(),
                                            blocks, sign_values);
    }
}

template <typename FloatT>
void expect_fallback_round_trip(const u8 bit_width, const PatternCase& pattern_case) {
    SCOPED_TRACE(::testing::Message() << "bit_width=" << static_cast<unsigned>(bit_width)
                                      << ", pattern=" << pattern_case.name);

    const auto input = make_dataset<FloatT>(pattern_case.pattern, bit_width);
    FloatT scale = FloatT{0};
    ASSERT_EQ(compute_scale(max_abs(input), bit_width, scale), PERNIX_STATUS_OK);

    std::vector<u8> compressed(kBlocks * kBlockSize);
    std::vector<FloatT> restored(input.size(), FloatT{});
    ASSERT_EQ(compress_blocks(PERNIX_BACKEND_FALLBACK, bit_width, input, FloatT{1} / scale, compressed, kBlocks),
              PERNIX_STATUS_OK);
    ASSERT_EQ(decompress_blocks(PERNIX_BACKEND_FALLBACK, bit_width, compressed, scale, restored, kBlocks),
              PERNIX_STATUS_OK);
    expect_near_input(input, restored, scale);
}

template <typename FloatT>
void expect_backend_matches_fallback(const pernix_backend backend, const u8 bit_width) {
    SCOPED_TRACE(::testing::Message() << "backend=" << static_cast<int>(backend)
                                      << ", bit_width=" << static_cast<unsigned>(bit_width));

    const auto input = make_dataset<FloatT>(Pattern::PseudoRandom, bit_width, 2);
    FloatT scale = FloatT{0};
    ASSERT_EQ(compute_scale(max_abs(input), bit_width, scale), PERNIX_STATUS_OK);

    std::vector<u8> fallback_compressed(2 * kBlockSize);
    std::vector<u8> backend_compressed(2 * kBlockSize);
    ASSERT_EQ(compress_blocks(PERNIX_BACKEND_FALLBACK, bit_width, input, FloatT{1} / scale, fallback_compressed, 2),
              PERNIX_STATUS_OK);

    const auto compress_status = compress_blocks(backend, bit_width, input, FloatT{1} / scale, backend_compressed, 2);
    if (compress_status != PERNIX_STATUS_OK) {
        EXPECT_TRUE(compress_status == PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION ||
                    compress_status == PERNIX_STATUS_UNSUPPORTED_BACKEND ||
                    compress_status == PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH);
        return;
    }

    std::vector<FloatT> fallback_restored(input.size(), FloatT{});
    std::vector<FloatT> backend_restored(input.size(), FloatT{});
    ASSERT_EQ(decompress_blocks(PERNIX_BACKEND_FALLBACK, bit_width, fallback_compressed, scale, fallback_restored, 2),
              PERNIX_STATUS_OK);
    ASSERT_EQ(decompress_blocks(PERNIX_BACKEND_FALLBACK, bit_width, backend_compressed, scale, backend_restored, 2),
              PERNIX_STATUS_OK);

    const FloatT tolerance = (abs_value(scale) * static_cast<FloatT>(1.1)) +
                             (std::numeric_limits<FloatT>::epsilon() * static_cast<FloatT>(64));
    for (usize i = 0; i < input.size(); ++i) {
        ASSERT_NEAR(backend_restored[i], fallback_restored[i], tolerance)
            << "element=" << i << ", scale=" << scale;
    }

    std::vector<FloatT> backend_decompressed(input.size(), FloatT{});
    const auto decompress_status =
        decompress_blocks(backend, bit_width, fallback_compressed, scale, backend_decompressed, 2);
    if (decompress_status != PERNIX_STATUS_OK) {
        EXPECT_TRUE(decompress_status == PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION ||
                    decompress_status == PERNIX_STATUS_UNSUPPORTED_BACKEND ||
                    decompress_status == PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH);
        return;
    }
    for (usize i = 0; i < input.size(); ++i) {
        ASSERT_NEAR(backend_decompressed[i], fallback_restored[i], tolerance)
            << "element=" << i << ", scale=" << scale;
    }
}
} // namespace

TEST(DeterministicCorrectnessTest, F32FallbackRoundTripsAllPatternsAndBitWidths) {
    for (u8 bit_width = pernix_min_bit_width(); bit_width <= pernix_max_bit_width(); ++bit_width) {
        for (const PatternCase& pattern_case : kPatternCases) {
            expect_fallback_round_trip<float>(bit_width, pattern_case);
        }
    }
}

TEST(DeterministicCorrectnessTest, F64FallbackRoundTripsAllPatternsAndBitWidths) {
    for (u8 bit_width = pernix_min_bit_width(); bit_width <= pernix_max_bit_width(); ++bit_width) {
        for (const PatternCase& pattern_case : kPatternCases) {
            expect_fallback_round_trip<double>(bit_width, pattern_case);
        }
    }
}

TEST(DeterministicCorrectnessTest, SimdBackendsMatchFallbackWhereAvailable) {
    for (const pernix_backend backend : kComparisonBackends) {
        for (u8 bit_width = pernix_min_bit_width(); bit_width <= pernix_max_bit_width(); ++bit_width) {
            expect_backend_matches_fallback<float>(backend, bit_width);
            expect_backend_matches_fallback<double>(backend, bit_width);
        }
    }
}

TEST(DeterministicCorrectnessTest, F32AndF64HaveConsistentFallbackSemantics) {
    constexpr u8 bit_width = 12;
    const u32 elements = pernix_elements_per_block(bit_width) * 2U;
    std::vector<float> input_f32(elements);
    for (usize i = 0; i < input_f32.size(); ++i) {
        input_f32[i] = static_cast<float>(static_cast<int>(i % 9U) - 4);
    }
    std::vector<double> input_f64(input_f32.begin(), input_f32.end());

    std::vector<u8> compressed_f32(2 * kBlockSize);
    std::vector<u8> compressed_f64(2 * kBlockSize);
    ASSERT_EQ(compress_blocks(PERNIX_BACKEND_FALLBACK, bit_width, input_f32, 1.0f, compressed_f32, 2),
              PERNIX_STATUS_OK);
    ASSERT_EQ(compress_blocks(PERNIX_BACKEND_FALLBACK, bit_width, input_f64, 1.0, compressed_f64, 2),
              PERNIX_STATUS_OK);
    EXPECT_EQ(compressed_f64, compressed_f32);

    std::vector<float> restored_f32(input_f32.size());
    std::vector<double> restored_f64(input_f64.size());
    ASSERT_EQ(decompress_blocks(PERNIX_BACKEND_FALLBACK, bit_width, compressed_f32, 1.0f, restored_f32, 2),
              PERNIX_STATUS_OK);
    ASSERT_EQ(decompress_blocks(PERNIX_BACKEND_FALLBACK, bit_width, compressed_f64, 1.0, restored_f64, 2),
              PERNIX_STATUS_OK);

    for (usize i = 0; i < restored_f32.size(); ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(restored_f32[i]), restored_f64[i]);
    }
}

TEST(DeterministicCorrectnessTest, SignValuesFalseUsesUnsignedPackedValuesForAllBitWidths) {
    for (u8 bit_width = pernix_min_bit_width(); bit_width <= pernix_max_bit_width(); ++bit_width) {
        SCOPED_TRACE(::testing::Message() << "bit_width=" << static_cast<unsigned>(bit_width));
        std::vector<u8> packed(kBlockSize, 0xFFU);
        const u32 elements = pernix_elements_per_block(bit_width);
        std::vector<float> signed_values(elements);
        std::vector<float> unsigned_values(elements);

        ASSERT_EQ(pernix_decompress_block_f32(PERNIX_BACKEND_FALLBACK, bit_width, kBlockSize, packed.data(), 1.0f,
                                              signed_values.data(), true),
                  PERNIX_STATUS_OK);
        ASSERT_EQ(pernix_decompress_block_f32(PERNIX_BACKEND_FALLBACK, bit_width, kBlockSize, packed.data(), 1.0f,
                                              unsigned_values.data(), false),
                  PERNIX_STATUS_OK);

        const float expected_unsigned = static_cast<float>((1U << bit_width) - 1U);
        EXPECT_EQ(unsigned_values[0], expected_unsigned);
        if (bit_width == 1) {
            EXPECT_EQ(signed_values[0], 1.0f);
        } else {
            EXPECT_EQ(signed_values[0], -1.0f);
        }
    }
}

TEST(DeterministicCorrectnessTest, CppSpanWrappersMatchCApiOutputAndStatus) {
    constexpr u8 bit_width = 10;
    const auto input = make_dataset<float>(Pattern::Increasing, bit_width, 1);
    float scale = 0.0f;
    ASSERT_EQ(compute_scale(max_abs(input), bit_width, scale), PERNIX_STATUS_OK);

    std::vector<u8> c_compressed(kBlockSize);
    std::vector<u8> cpp_compressed(kBlockSize);
    EXPECT_EQ(pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, bit_width, kBlockSize, input.data(), 1.0f / scale,
                                        c_compressed.data()),
              PERNIX_STATUS_OK);
    EXPECT_EQ(pernix::compress_block(pernix::Backend::Fallback, bit_width, kBlockSize, std::span<const float>(input),
                                     1.0f / scale, std::span<u8>(cpp_compressed)),
              PERNIX_STATUS_OK);
    EXPECT_EQ(cpp_compressed, c_compressed);

    std::vector<float> c_restored(input.size());
    std::vector<float> cpp_restored(input.size());
    EXPECT_EQ(pernix_decompress_block_f32(PERNIX_BACKEND_FALLBACK, bit_width, kBlockSize, c_compressed.data(), scale,
                                          c_restored.data(), true),
              PERNIX_STATUS_OK);
    EXPECT_EQ(pernix::decompress_block(pernix::Backend::Fallback, bit_width, kBlockSize,
                                       std::span<const u8>(cpp_compressed), scale, std::span<float>(cpp_restored)),
              PERNIX_STATUS_OK);
    EXPECT_EQ(cpp_restored, c_restored);

    EXPECT_EQ(pernix::compress_block(pernix::Backend::Fallback, 0, kBlockSize, std::span<const float>(input),
                                     1.0f / scale, std::span<u8>(cpp_compressed)),
              pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, 0, kBlockSize, input.data(), 1.0f / scale,
                                        c_compressed.data()));
}

TEST(DeterministicCorrectnessTest, SpanValidationRejectsUndersizedMultiBlockBuffers) {
    constexpr u8 bit_width = 8;
    const u32 elements = pernix_elements_per_block(bit_width) * 2U;
    std::vector<float> input(elements, 1.0f);
    std::vector<u8> compressed(2U * kBlockSize);
    std::vector<float> restored(elements);

    EXPECT_EQ(pernix::compress_blocks(pernix::Backend::Fallback, bit_width, kBlockSize,
                                      std::span<const float>(input).first(input.size() - 1U), 1.0f,
                                      std::span<u8>(compressed), 2),
              PERNIX_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(pernix::compress_blocks(pernix::Backend::Fallback, bit_width, kBlockSize, std::span<const float>(input),
                                      1.0f, std::span<u8>(compressed).first(compressed.size() - 1U), 2),
              PERNIX_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(pernix::decompress_blocks(pernix::Backend::Fallback, bit_width, kBlockSize,
                                        std::span<const u8>(compressed).first(compressed.size() - 1U), 1.0f,
                                        std::span<float>(restored), 2),
              PERNIX_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(pernix::decompress_blocks(pernix::Backend::Fallback, bit_width, kBlockSize,
                                        std::span<const u8>(compressed), 1.0f,
                                        std::span<float>(restored).first(restored.size() - 1U), 2),
              PERNIX_STATUS_INVALID_ARGUMENT);
}
#endif
