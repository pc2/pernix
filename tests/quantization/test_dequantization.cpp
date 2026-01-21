#include <gtest/gtest.h>
#include <pernix/helper.h>
#include <pernix/quantization/dequantization.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <type_traits>

using namespace pernix::quantization;

template <std::size_t size, typename T, typename F>
    requires(std::is_integral_v<T> && std::is_floating_point_v<F>)
struct DequantizationTestSet {
    alignas(64) std::array<T, size> input;
    alignas(64) std::array<F, size> dequantized;
    alignas(64) std::array<F, size> result;
    float scale;
};

template <std::size_t size, typename T, typename F>
    requires(std::is_integral_v<T> && std::is_floating_point_v<F>)
auto operator<<(std::ostream& os, const DequantizationTestSet<size, T, F>& test_set) -> std::ostream& {
    os << "Input: ";
    for (const auto& val : test_set.input) {
        os << val << " ";
    }
    os << "\nQuantized: ";
    for (const auto& val : test_set.quantized) {
        os << val << " ";
    }
    os << "\nResult: ";
    for (const auto& val : test_set.result) {
        os << val << " ";
    }
    os << "\nScale: " << test_set.scale;
    return os;
}

template <std::size_t size>
static auto create_signed_dequantization_test_set(const uint8_t bit_width, const float_t min_value, const float_t max_value)
    -> DequantizationTestSet<size, int32_t, float_t> {
    DequantizationTestSet<size, int32_t, float_t> test_set{};
    std::random_device random_device;
    std::mt19937 gen(random_device());
    std::uniform_real_distribution dis(min_value, max_value);

    for (auto& val : test_set.dequantized) {
        val = dis(gen);
    }
    const float_t b_max = *std::max_element(test_set.dequantized.begin(), test_set.dequantized.end());

    test_set.scale = b_max / static_cast<float_t>((1 << (bit_width - 1)) - 1);
    if (bit_width == 1) {
        test_set.scale = b_max;
    }
    const float_t inv_scale = 1.0F / test_set.scale;
    for (uint32_t i = 0; i < size; ++i) {
        test_set.input[i] = static_cast<int32_t>(std::roundf(test_set.dequantized[i] * inv_scale));
    }

    return test_set;
}

template <std::size_t size>
static void assert_dequantization_result(const DequantizationTestSet<size, int32_t, float_t>& test_set, const uint8_t bit_width) {
    // const float_t max_error = test_set.scale / 2.0f;
    const float_t max_error = test_set.scale;
    for (std::size_t i = 0; i < size; ++i) {
        ASSERT_LE(std::fabs(test_set.result[i] - test_set.dequantized[i]), max_error)
            << "Dequantization failed for bit width " << static_cast<int>(bit_width) << " at index " << i << ": expected "
            << test_set.dequantized[i] << ", got " << test_set.result[i] << ", max error allowed " << max_error;
    }
}

using DequantizationBitWidthTest = testing::TestWithParam<int>;

void test_dequantization_fallback(const uint8_t bit_width) {
    auto test_set = create_signed_dequantization_test_set<32>(bit_width, -1.0f, 1.0f);

    for (std::size_t i = 0; i < test_set.input.size(); ++i) {
        test_set.result[i] = dequantize_epi32(test_set.input[i], test_set.scale);
    }

    assert_dequantization_result(test_set, bit_width);
}

TEST_P(DequantizationBitWidthTest, DequantizeFallback) {
    test_dequantization_fallback(static_cast<uint8_t>(GetParam()));
}

#ifdef PERNIX_SSE_ENABLED
void test_dequantization_dequantize4(const uint8_t bit_width) {
    auto test_set = create_signed_dequantization_test_set<32>(bit_width, -1.0f, 1.0f);

    const __m128 scale_v = _mm_set1_ps(test_set.scale);
    for (int i = 0; i < 8; ++i) {
        const __m128i input_v      = _mm_loadu_si128(reinterpret_cast<const __m128i*>(test_set.input.data() + i * 4));
        const __m128 dequantized_v = mm_dequantize_epi32(input_v, scale_v);
        _mm_storeu_ps(test_set.result.data() + i * 4, dequantized_v);
    }

    assert_dequantization_result(test_set, bit_width);
}

TEST_P(DequantizationBitWidthTest, DequantizeSSE) {
    test_dequantization_dequantize4(static_cast<uint8_t>(GetParam()));
}
#endif  // PERNIX_SSE_ENABLED

#ifdef PERNIX_AVX2_ENABLED
void test_dequantization_dequantize8(const uint8_t bit_width) {
    auto test_set = create_signed_dequantization_test_set<32>(bit_width, -1.0f, 1.0f);

    const __m256 scale_v = _mm256_set1_ps(test_set.scale);
    for (int i = 0; i < 4; ++i) {
        const __m256i input_v      = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(test_set.input.data() + i * 8));
        const __m256 dequantized_v = mm256_dequantize_epi32(input_v, scale_v);
        _mm256_storeu_ps(test_set.result.data() + i * 8, dequantized_v);
    }

    assert_dequantization_result(test_set, bit_width);
}

TEST_P(DequantizationBitWidthTest, DequantizeAVX2) {
    test_dequantization_dequantize8(static_cast<uint8_t>(GetParam()));
}
#endif  // PERNIX_AVX2_ENABLED

#ifdef PERNIX_AVX512_ENABLED
void test_dequantization_dequantize16(const uint8_t bit_width) {
    auto test_set = create_signed_dequantization_test_set<32>(bit_width, -1.0f, 1.0f);

    const __m512 scale_v = _mm512_set1_ps(test_set.scale);
    for (int i = 0; i < 2; ++i) {
        const __m512i input_v      = _mm512_loadu_si512(test_set.input.data() + i * 16);
        const __m512 dequantized_v = mm512_dequantize_epi32(input_v, scale_v);
        _mm512_storeu_ps(test_set.result.data() + i * 16, dequantized_v);
    }

    assert_dequantization_result(test_set, bit_width);
}

TEST_P(DequantizationBitWidthTest, DequantizeAVX512) {
    test_dequantization_dequantize16(static_cast<uint8_t>(GetParam()));
}
#endif  // PERNIX_AVX512_ENABLED

INSTANTIATE_TEST_SUITE_P(AllBitWidths, DequantizationBitWidthTest, testing::Range(1, 25));
