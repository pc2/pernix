#include <gtest/gtest.h>
#include <pernix/helper.h>
#include <pernix/quantization/quantization.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <type_traits>

using namespace pernix::quantization;

template <std::size_t size, typename T>
    requires(std::is_floating_point_v<T>)
struct QuantizationTestSet {
    alignas(64) std::array<T, size> input;
    alignas(64) std::array<int32_t, size> quantized;
    alignas(64) std::array<int32_t, size> result;
    float scale;
};

template <std::size_t size, typename T>
    requires(std::is_floating_point_v<T>)
auto operator<<(std::ostream& os, const QuantizationTestSet<size, T>& test_set) -> std::ostream& {
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
static auto create_quantization_test_set(const uint8_t bit_width, const float_t min_value, const float_t max_value)
    -> QuantizationTestSet<size, float_t> {
    QuantizationTestSet<size, float_t> test_set{};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dis(min_value, max_value);

    for (auto& val : test_set.input) {
        val = dis(gen);
    }
    const float_t b_max = *std::max_element(test_set.input.begin(), test_set.input.end());

    test_set.scale = 1.0f / (b_max / static_cast<float_t>((1 << (bit_width - 1)) - 1));

    for (std::size_t i = 0; i < size; ++i) {
        test_set.quantized[i] = static_cast<int32_t>(std::nearbyintf(test_set.input[i] * test_set.scale));
    }

    return test_set;
}

using QuantizationBitWidthTest = testing::TestWithParam<int>;

void test_quantization_fallback(const uint8_t bit_width) {
    auto test_set = create_quantization_test_set<32>(bit_width, -1.0f, 1.0f);

    for (std::size_t i = 0; i < test_set.input.size(); ++i) {
        test_set.result[i] = quantize_ps_epi32(test_set.input[i], test_set.scale);
    }

    for (std::size_t i = 0; i < test_set.input.size(); ++i) {
        ASSERT_EQ(test_set.quantized[i], test_set.result[i])
            << "Mismatch at index " << i << ": expected " << test_set.quantized[i] << ", got " << test_set.result[i];
    }
}

TEST_P(QuantizationBitWidthTest, QuantizeFallback) {
    test_quantization_fallback(static_cast<uint8_t>(GetParam()));
}

#ifdef PERNIX_SSE_ENABLED
void test_quantization_quantize4(const uint8_t bit_width) {
    auto test_set = create_quantization_test_set<32>(bit_width, -1.0f, 1.0f);

    const __m128 scale_v = _mm_set1_ps(test_set.scale);
    for (int i = 0; i < 8; ++i) {
        const __m128 input_v      = _mm_loadu_ps(test_set.input.data() + i * 4);
        const __m128i quantized_v = mm_quantize_ps_epi32(input_v, scale_v);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(test_set.result.data() + i * 4), quantized_v);
    }

    for (std::size_t i = 0; i < test_set.input.size(); ++i) {
        ASSERT_EQ(test_set.quantized[i], test_set.result[i])
            << "Mismatch at index " << i << ": expected " << test_set.quantized[i] << ", got " << test_set.result[i];
    }
}
#endif  // PERNIX_SSE_ENABLED

#ifdef PERNIX_AVX2_ENABLED
void test_quantization_quantize8(const uint8_t bit_width) {
    auto test_set = create_quantization_test_set<32>(bit_width, -1.0f, 1.0f);

    const __m256 scale_v = _mm256_set1_ps(test_set.scale);
    for (int i = 0; i < 4; ++i) {
        const __m256 input_v      = _mm256_loadu_ps(test_set.input.data() + i * 8);
        const __m256i quantized_v = mm256_quantize_ps_epi32(input_v, scale_v);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(test_set.result.data() + i * 8), quantized_v);
    }

    for (std::size_t i = 0; i < test_set.input.size(); ++i) {
        ASSERT_EQ(test_set.quantized[i], test_set.result[i])
            << "Mismatch at index " << i << ": expected " << test_set.quantized[i] << ", got " << test_set.result[i];
    }
}

TEST_P(QuantizationBitWidthTest, Quantize8) {
    test_quantization_quantize8(static_cast<uint8_t>(GetParam()));
}
#endif  // PERNIX_AVX2_ENABLED

#ifdef PERNIX_AVX512_ENABLED
void test_quantization_quantize16(const uint8_t bit_width) {
    auto test_set = create_quantization_test_set<32>(bit_width, -1.0f, 1.0f);

    const __m512 scale_v = _mm512_set1_ps(test_set.scale);
    for (int i = 0; i < 2; ++i) {
        const __m512 input_v      = _mm512_loadu_ps(test_set.input.data() + i * 16);
        const __m512i quantized_v = mm512_quantize_ps_epi32(input_v, scale_v);
        _mm512_storeu_si512(test_set.result.data() + i * 16, quantized_v);
    }

    for (std::size_t i = 0; i < test_set.input.size(); ++i) {
        ASSERT_EQ(test_set.quantized[i], test_set.result[i])
            << "Mismatch at index " << i << ": expected " << test_set.quantized[i] << ", got " << test_set.result[i];
    }
}

TEST_P(QuantizationBitWidthTest, Quantize16) {
    test_quantization_quantize16(static_cast<uint8_t>(GetParam()));
}
#endif  // PERNIX_AVX512_ENABLED

INSTANTIATE_TEST_SUITE_P(AllBitWidths, QuantizationBitWidthTest, testing::Range(1, 25));