#ifndef PERNIX_TESTSET_H
#define PERNIX_TESTSET_H

#include <../../include/pernix/pernix.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>

#ifndef PERNIX_TEST_BLOCK_SIZE
#define PERNIX_TEST_BLOCK_SIZE 64
#endif

static_assert(PERNIX_TEST_BLOCK_SIZE % 32 == 0,
              "PERNIX_TEST_BLOCK_SIZE must be dividable by 32 bytes");

/**
 * A test set for compression and decompression tests.
 * It generates random float data, compresses it, and verifies the decompression using the fallback implementation.
 *
 * @tparam BIT_WIDTH The bit width used for compression (1 to 24).
 * @tparam SIGN_VALUES Indicates whether the values are signed or unsigned.
 */
template <uint8_t BIT_WIDTH, typename T = float_t, uint32_t BLOCK_SIZE = 64, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && std::is_floating_point_v<T>
class TestSet {
    // using ValueType = std::conditional_t<Signed, int8_t, uint8_t>;
    using ValueType = uint8_t;
    using SeedType  = std::mt19937::result_type;

    alignas(64) std::vector<std::vector<ValueType> > compressedData;
    alignas(64) std::vector<std::vector<T> > decompressedData;
    alignas(64) std::vector<T> scalesData;

    SeedType seed;
    std::mt19937 gen;
    std::uniform_real_distribution<T> dis{};

public:
    static constexpr uint32_t elementsPerBlock = (BLOCK_SIZE * 8) / BIT_WIDTH;
    static constexpr SeedType defaultSeed      = 0x5eed1234u;

    static constexpr T quantization_levels =
        SIGN_VALUES ? static_cast<T>(BIT_WIDTH == 1 ? 1u : ((1u << (BIT_WIDTH - 1u)) - 1u)) : static_cast<T>((1u << BIT_WIDTH) - 1u);

    uint32_t numberOfBlocks;

    [[nodiscard]] constexpr uint32_t totalElements() const { return numberOfBlocks * elementsPerBlock; }

    [[nodiscard]] T blockTolerance(const uint32_t block) const {
        // Half-step quantization bound + tiny FP slack for rounding edge cases.
        return (std::abs(scalesData[block]) * static_cast<T>(0.5)) + (std::numeric_limits<T>::epsilon() * static_cast<T>(16));
    }

    explicit TestSet(const uint32_t number_of_blocks, const SeedType initial_seed = testSeed())
        : seed(initial_seed), gen(seed), numberOfBlocks(number_of_blocks) {
        compressedData.resize(numberOfBlocks);
        decompressedData.resize(number_of_blocks);
        scalesData.resize(numberOfBlocks);

        generateData();
    }

    [[nodiscard]] const std::vector<T>& getScales() const { return scalesData; }

    [[nodiscard]] const std::vector<std::vector<ValueType> >& getCompressedData() const { return compressedData; }

    [[nodiscard]] const std::vector<std::vector<T> >& getDecompressedData() const { return decompressedData; }

    [[nodiscard]] SeedType getSeed() const { return seed; }

    [[nodiscard]] static SeedType testSeed() {
        const char* env_seed = std::getenv("PERNIX_TEST_SEED");
        if (env_seed == nullptr || *env_seed == '\0') {
            return defaultSeed;
        }

        char* end                 = nullptr;
        const unsigned long value = std::strtoul(env_seed, &end, 0);
        return (end != env_seed && *end == '\0') ? static_cast<SeedType>(value) : defaultSeed;
    }

private:
    // Generate deterministic source data and its fallback-compressed reference.
    void generateData() {
        for (uint32_t i = 0; i < numberOfBlocks; i++) {
            compressedData[i].resize(BLOCK_SIZE);
            decompressedData[i].resize(elementsPerBlock);

            for (uint32_t j = 0; j < elementsPerBlock; j++) {
                decompressedData[i][j] = dis(gen);
            }

            const T b_max = *std::ranges::max_element(decompressedData[i]);
            const T b_min = *std::ranges::min_element(decompressedData[i]);
            const T b_abs = std::max(std::abs(b_max), std::abs(b_min));
            scalesData[i] = (b_abs > static_cast<T>(0) && quantization_levels > static_cast<T>(0))
                                ? (b_abs / quantization_levels)
                                : std::numeric_limits<T>::epsilon();

            // Compress the data using the fallback implementation
            pernix::compress_block_fallback<BIT_WIDTH, BLOCK_SIZE>(decompressedData[i].data(), 1 / scalesData[i],
                                                                   reinterpret_cast<uint8_t*>(compressedData[i].data()));
        }
    }
};

template <uint8_t BIT_WIDTH, uint32_t BLOCK_SIZE>
struct BitWidthBlockSize {
    static constexpr uint8_t bit_width   = BIT_WIDTH;
    static constexpr uint32_t block_size = BLOCK_SIZE;
};

using testing::Types;
using BitWidthBlockSizeTypes =
Types<BitWidthBlockSize<1, PERNIX_TEST_BLOCK_SIZE>, BitWidthBlockSize<2, PERNIX_TEST_BLOCK_SIZE>,
      BitWidthBlockSize<3, PERNIX_TEST_BLOCK_SIZE>, BitWidthBlockSize<4, PERNIX_TEST_BLOCK_SIZE>,
      BitWidthBlockSize<5, PERNIX_TEST_BLOCK_SIZE>, BitWidthBlockSize<6, PERNIX_TEST_BLOCK_SIZE>,
      BitWidthBlockSize<7, PERNIX_TEST_BLOCK_SIZE>, BitWidthBlockSize<8, PERNIX_TEST_BLOCK_SIZE>,
      BitWidthBlockSize<9, PERNIX_TEST_BLOCK_SIZE>, BitWidthBlockSize<10, PERNIX_TEST_BLOCK_SIZE>,
      BitWidthBlockSize<11, PERNIX_TEST_BLOCK_SIZE>, BitWidthBlockSize<12, PERNIX_TEST_BLOCK_SIZE>,
      BitWidthBlockSize<13, PERNIX_TEST_BLOCK_SIZE>, BitWidthBlockSize<14, PERNIX_TEST_BLOCK_SIZE>,
      BitWidthBlockSize<15, PERNIX_TEST_BLOCK_SIZE>, BitWidthBlockSize<16, PERNIX_TEST_BLOCK_SIZE>,
      BitWidthBlockSize<17, PERNIX_TEST_BLOCK_SIZE>, BitWidthBlockSize<18, PERNIX_TEST_BLOCK_SIZE>,
      BitWidthBlockSize<19, PERNIX_TEST_BLOCK_SIZE>, BitWidthBlockSize<20, PERNIX_TEST_BLOCK_SIZE>,
      BitWidthBlockSize<21, PERNIX_TEST_BLOCK_SIZE>, BitWidthBlockSize<22, PERNIX_TEST_BLOCK_SIZE>,
      BitWidthBlockSize<23, PERNIX_TEST_BLOCK_SIZE>, BitWidthBlockSize<24, PERNIX_TEST_BLOCK_SIZE> >;

struct BitWidthBlockSizeName {
    template <typename TestConfigT>
    static std::string GetName(int) {
        std::ostringstream name;
        name << "BitWidth" << static_cast<uint32_t>(TestConfigT::bit_width) << "BlockSize" << TestConfigT::block_size;
        return name.str();
    }
};

template <typename TestConfigT>
class CompressionTest : public ::testing::Test {
public:
    static constexpr uint8_t BitWidth   = TestConfigT::bit_width;
    static constexpr uint32_t BlockSize = TestConfigT::block_size;

    TestSet<BitWidth, float_t, BlockSize> testSet;

    CompressionTest() : testSet(1u << 10) {
    }
};

template <typename TestConfigT>
class DecompressionTest : public ::testing::Test {
public:
    static constexpr uint8_t BitWidth   = TestConfigT::bit_width;
    static constexpr uint32_t BlockSize = TestConfigT::block_size;

    TestSet<BitWidth, float_t, BlockSize> testSet;

    DecompressionTest() : testSet(1u << 10) {
    }
};

template <typename TestConfigT>
class CompressionTest64 : public ::testing::Test {
public:
    static constexpr uint8_t BitWidth   = TestConfigT::bit_width;
    static constexpr uint32_t BlockSize = TestConfigT::block_size;

    TestSet<BitWidth, double_t, BlockSize> testSet;

    CompressionTest64() : testSet(1u << 10) {
    }
};

template <typename TestConfigT>
class DecompressionTest64 : public ::testing::Test {
public:
    static constexpr uint8_t BitWidth   = TestConfigT::bit_width;
    static constexpr uint32_t BlockSize = TestConfigT::block_size;

    TestSet<BitWidth, double_t, BlockSize> testSet;

    DecompressionTest64() : testSet(1u << 10) {
    }
};

template <typename FixtureT>
[[nodiscard]] std::string testContext(const FixtureT& fixture, const uint32_t block) {
    std::ostringstream message;
    message << "bit_width=" << static_cast<uint32_t>(FixtureT::BitWidth) << ", block_size=" << FixtureT::BlockSize
        << ", block=" << block << ", scale=" << fixture.testSet.getScales()[block]
        << ", tolerance=" << fixture.testSet.blockTolerance(block) << ", seed=" << fixture.testSet.getSeed();
    return message.str();
}

template <typename FixtureT>
void expectCompressedBlockEqualsReference(const FixtureT& fixture, const std::vector<uint8_t>& actual, const uint32_t block) {
    SCOPED_TRACE(testContext(fixture, block));

    const auto& expected = fixture.testSet.getCompressedData()[block];
    ASSERT_EQ(actual.size(), expected.size()) << "Compressed block byte count differs from reference";

    for (uint32_t byte = 0; byte < actual.size(); byte++) {
        ASSERT_EQ(actual[byte], expected[byte])
            << "Compressed byte mismatch at byte=" << byte << ", actual=" << static_cast<uint32_t>(actual[byte])
            << ", expected=" << static_cast<uint32_t>(expected[byte]);
    }
}

template <typename FixtureT, typename T>
void expectDecompressedBlockNearSource(const FixtureT& fixture, const std::vector<T>& actual, const uint32_t block) {
    SCOPED_TRACE(testContext(fixture, block));

    const auto& expected = fixture.testSet.getDecompressedData()[block];
    ASSERT_EQ(actual.size(), expected.size()) << "Decompressed block element count differs from source";

    for (uint32_t element = 0; element < actual.size(); element++) {
        ASSERT_NEAR(actual[element], expected[element], fixture.testSet.blockTolerance(block))
            << "Decompressed element mismatch at element=" << element << ", actual=" << actual[element]
            << ", expected=" << expected[element] << ", absolute_error=" << std::abs(actual[element] - expected[element]);
    }
}

TYPED_TEST_SUITE(CompressionTest, BitWidthBlockSizeTypes, BitWidthBlockSizeName);
TYPED_TEST_SUITE(DecompressionTest, BitWidthBlockSizeTypes, BitWidthBlockSizeName);
TYPED_TEST_SUITE(CompressionTest64, BitWidthBlockSizeTypes, BitWidthBlockSizeName);
TYPED_TEST_SUITE(DecompressionTest64, BitWidthBlockSizeTypes, BitWidthBlockSizeName);

#endif  // PERNIX_TESTSET_H
