#ifndef PERNIX_TESTSET_H
#define PERNIX_TESTSET_H

#include <../../include/pernix/pernix.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

/**
 * A test set for compression and decompression tests.
 * It generates random float data, compresses it, and verifies the decompression using the fallback implementation.
 *
 * @tparam BitWidth The bit width used for compression (1 to 32).
 * @tparam Signed Indicates whether the values are signed or unsigned.
 */
template <uint8_t BitWidth, typename T = float_t, uint32_t BLOCK_SIZE = 64, bool Signed = true>
    requires(BitWidth >= 1 && BitWidth <= 32) && std::is_floating_point_v<T>
class TestSet {
    // using ValueType = std::conditional_t<Signed, int8_t, uint8_t>;
    using ValueType = uint8_t;

    alignas(64) std::vector<std::vector<ValueType>> compressedData;
    alignas(64) std::vector<std::vector<T>> decompressedData;
    alignas(64) std::vector<T> scalesData;

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_real_distribution<T> dis{};

public:
    static constexpr uint32_t elementsPerBlock = (BLOCK_SIZE * 8) / BitWidth;

    static constexpr T quantization_levels = Signed ? static_cast<T>((1u << (BitWidth - 1u)) - 1u) : static_cast<T>((1u << BitWidth) - 1u);

    uint32_t numberOfBlocks;

    [[nodiscard]] constexpr uint32_t totalElements() const { return numberOfBlocks * elementsPerBlock; }

    [[nodiscard]] T blockTolerance(const uint32_t block) const {
        // Half-step quantization bound + tiny FP slack for rounding edge cases.
        return (std::abs(scalesData[block]) * static_cast<T>(0.5)) + (std::numeric_limits<T>::epsilon() * static_cast<T>(16));
    }

    explicit TestSet(const uint32_t number_of_blocks) : numberOfBlocks(number_of_blocks) {
        compressedData.resize(numberOfBlocks);  // 64 bytes per block
        decompressedData.resize(number_of_blocks);
        scalesData.resize(numberOfBlocks);

        generateData();
    }

    [[nodiscard]] const std::vector<T>& getScales() const { return scalesData; }

    [[nodiscard]] const std::vector<std::vector<ValueType>>& getCompressedData() const { return compressedData; }

    [[nodiscard]] const std::vector<std::vector<T>>& getDecompressedData() const { return decompressedData; }

private:
    // Generate random data, compress it, and verify decompression
    void generateData() {
        for (uint32_t i = 0; i < numberOfBlocks; i++) {
            compressedData[i].resize(BLOCK_SIZE);  // 64 bytes per block
            decompressedData[i].resize(elementsPerBlock);

            for (uint32_t j = 0; j < elementsPerBlock; j++) {
                decompressedData[i][j] = dis(gen);
            }

            const T b_max = *std::ranges::max_element(decompressedData[i]);
            const T b_min = *std::ranges::min_element(decompressedData[i]);
            const T b_abs = std::max(std::abs(b_max), std::abs(b_min));
            scalesData[i] = (b_abs > static_cast<T>(0) && quantization_levels > static_cast<T>(0)) ? (b_abs / quantization_levels)
                                                                                                   : std::numeric_limits<T>::epsilon();

            // Compress the data using the fallback implementation
            pernix::compress_block_fallback<BitWidth>(decompressedData[i].data(), 1 / scalesData[i],
                                                      reinterpret_cast<uint8_t*>(compressedData[i].data()));

            // Decompress and verify using the fallback implementation
            std::vector<T> decompressed_verify(elementsPerBlock);
            pernix::decompress_block_fallback<BitWidth>(reinterpret_cast<uint8_t*>(compressedData[i].data()), scalesData[i],
                                                        decompressed_verify.data());

            for (uint32_t j = 0; j < elementsPerBlock; j++) {
                ASSERT_NEAR(decompressed_verify[j], decompressedData[i][j], blockTolerance(i));
            }
        }
    }
};

#define BitWithType(N)                          \
    struct BitWidth##N {                        \
        static constexpr uint8_t bit_width = N; \
    }

BitWithType(8);
BitWithType(9);
BitWithType(10);
BitWithType(11);
BitWithType(12);
BitWithType(13);
BitWithType(14);
BitWithType(15);
BitWithType(16);

using testing::Types;
using BitWidthTypes = Types<BitWidth8, BitWidth9, BitWidth10, BitWidth11, BitWidth12, BitWidth13, BitWidth14, BitWidth15, BitWidth16>;

template <typename BitWidthT>
class CompressionTest : public ::testing::Test {
public:
    static constexpr uint8_t BitWidth = BitWidthT::bit_width;

    TestSet<BitWidth> testSet;

    CompressionTest() : testSet(1u << 10) {}
};

template <typename BitWidthT>
class DecompressionTest : public ::testing::Test {
public:
    static constexpr uint8_t BitWidth = BitWidthT::bit_width;

    TestSet<BitWidth> testSet;

    DecompressionTest() : testSet(1u << 10) {}
};

template <typename BitWidthT>
class CompressionTest64 : public ::testing::Test {
public:
    static constexpr uint8_t BitWidth = BitWidthT::bit_width;

    TestSet<BitWidth, double_t> testSet;

    CompressionTest64() : testSet(1u << 10) {}
};

template <typename BitWidthT>
class DecompressionTest64 : public ::testing::Test {
public:
    static constexpr uint8_t BitWidth = BitWidthT::bit_width;

    TestSet<BitWidth, double_t> testSet;

    DecompressionTest64() : testSet(1u << 10) {}
};

TYPED_TEST_SUITE(CompressionTest, BitWidthTypes);
TYPED_TEST_SUITE(DecompressionTest, BitWidthTypes);
TYPED_TEST_SUITE(CompressionTest64, BitWidthTypes);
TYPED_TEST_SUITE(DecompressionTest64, BitWidthTypes);

#endif  // PERNIX_TESTSET_H
