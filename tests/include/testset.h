#ifndef PERNIX_TESTSET_H
#define PERNIX_TESTSET_H

#include <gtest/gtest.h>
#include <pernix/pernix.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>

#ifndef PERNIX_TEST_BLOCK_SIZE
#define PERNIX_TEST_BLOCK_SIZE 64
#endif

static_assert(PERNIX_TEST_BLOCK_SIZE % 32 == 0, "PERNIX_TEST_BLOCK_SIZE must be dividable by 32 bytes");

template <u8 BIT_WIDTH, typename T = f32, u32 BLOCK_SIZE = 64, bool SIGN_VALUES = true>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24) && std::is_floating_point_v<T>
class TestSet {
    using ValueType = u8;
    using SeedType  = std::mt19937::result_type;

    alignas(64) std::vector<std::vector<ValueType> > compressedData;
    alignas(64) std::vector<std::vector<T> > sourceData;
    alignas(64) std::vector<T> scalesData;

    SeedType seed;
    std::mt19937 gen;
    std::uniform_real_distribution<T> dis{};

public:
    static constexpr u32 elementsPerBlock = (BLOCK_SIZE * 8) / BIT_WIDTH;
    static constexpr SeedType defaultSeed = 0x5eed1234u;

    static constexpr T quantization_levels =
        SIGN_VALUES ? static_cast<T>(BIT_WIDTH == 1 ? 1u : ((1u << (BIT_WIDTH - 1u)) - 1u)) : static_cast<T>((1u << BIT_WIDTH) - 1u);

    u32 numberOfBlocks;

    [[nodiscard]] constexpr u32 totalElements() const { return numberOfBlocks * elementsPerBlock; }

    [[nodiscard]] T blockTolerance(const u32 block) const {
        return (std::abs(scalesData[block]) * static_cast<T>(0.5)) + (std::numeric_limits<T>::epsilon() * static_cast<T>(16));
    }

    explicit TestSet(const u32 number_of_blocks, const SeedType initial_seed = testSeed())
        : seed(initial_seed), gen(seed), numberOfBlocks(number_of_blocks) {
        compressedData.resize(numberOfBlocks);
        sourceData.resize(number_of_blocks);
        scalesData.resize(numberOfBlocks);

        generateData();
    }

    [[nodiscard]] const std::vector<T>& getScales() const { return scalesData; }

    [[nodiscard]] const std::vector<std::vector<ValueType> >& getCompressedData() const { return compressedData; }

    [[nodiscard]] const std::vector<std::vector<T> >& getDecompressedData() const { return sourceData; }

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
    void generateData() {
        for (u32 i = 0; i < numberOfBlocks; i++) {
            compressedData[i].resize(BLOCK_SIZE);
            sourceData[i].resize(elementsPerBlock);

            for (u32 j = 0; j < elementsPerBlock; j++) {
                sourceData[i][j] = dis(gen);
            }

            const T b_max = *std::ranges::max_element(sourceData[i]);
            const T b_min = *std::ranges::min_element(sourceData[i]);
            const T b_abs = std::max(std::abs(b_max), std::abs(b_min));
            scalesData[i] = (b_abs > static_cast<T>(0) && quantization_levels > static_cast<T>(0)) ? (b_abs / quantization_levels)
                                                                                                   : std::numeric_limits<T>::epsilon();

            if constexpr (std::is_same_v<T, float>) {
                pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, BIT_WIDTH, BLOCK_SIZE, sourceData[i].data(), 1.0f / scalesData[i],
                                          compressedData[i].data());
            } else {
                pernix_compress_block_f64(PERNIX_BACKEND_FALLBACK, BIT_WIDTH, BLOCK_SIZE, sourceData[i].data(), 1.0 / scalesData[i],
                                          compressedData[i].data());
            }
        }
    }
};

template <u8 BIT_WIDTH, u32 BLOCK_SIZE>
struct BitWidthBlockSize {
    static constexpr u8 bit_width   = BIT_WIDTH;
    static constexpr u32 block_size = BLOCK_SIZE;
};

using testing::Types;
using BitWidthBlockSizeTypes = Types<BitWidthBlockSize<1, PERNIX_TEST_BLOCK_SIZE>, BitWidthBlockSize<2, PERNIX_TEST_BLOCK_SIZE>,
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
        name << "BitWidth" << static_cast<u32>(TestConfigT::bit_width) << "BlockSize" << TestConfigT::block_size;
        return name.str();
    }
};

template <typename TestConfigT>
class CompressionTest : public ::testing::Test {
public:
    static constexpr u8 BitWidth   = TestConfigT::bit_width;
    static constexpr u32 BlockSize = TestConfigT::block_size;

    TestSet<BitWidth, f32, BlockSize> testSet;

    CompressionTest() : testSet(1u << 10) {}
};

template <typename TestConfigT>
class DecompressionTest : public ::testing::Test {
public:
    static constexpr u8 BitWidth   = TestConfigT::bit_width;
    static constexpr u32 BlockSize = TestConfigT::block_size;

    TestSet<BitWidth, f32, BlockSize> testSet;

    DecompressionTest() : testSet(1u << 10) {}
};

template <typename TestConfigT>
class CompressionTest64 : public ::testing::Test {
public:
    static constexpr u8 BitWidth   = TestConfigT::bit_width;
    static constexpr u32 BlockSize = TestConfigT::block_size;

    TestSet<BitWidth, f64, BlockSize> testSet;

    CompressionTest64() : testSet(1u << 10) {}
};

template <typename TestConfigT>
class DecompressionTest64 : public ::testing::Test {
public:
    static constexpr u8 BitWidth   = TestConfigT::bit_width;
    static constexpr u32 BlockSize = TestConfigT::block_size;

    TestSet<BitWidth, f64, BlockSize> testSet;

    DecompressionTest64() : testSet(1u << 10) {}
};

template <typename FixtureT>
[[nodiscard]] std::string testContext(const FixtureT& fixture, const u32 block) {
    std::ostringstream message;
    message << "bit_width=" << static_cast<u32>(FixtureT::BitWidth) << ", block_size=" << FixtureT::BlockSize << ", block=" << block
            << ", scale=" << fixture.testSet.getScales()[block] << ", tolerance=" << fixture.testSet.blockTolerance(block)
            << ", seed=" << fixture.testSet.getSeed();
    return message.str();
}

template <typename FixtureT>
void expectCompressedBlockEqualsReference(const FixtureT& fixture, const std::vector<u8>& actual, const u32 block) {
    SCOPED_TRACE(testContext(fixture, block));

    const auto& expected = fixture.testSet.getCompressedData()[block];
    ASSERT_EQ(actual.size(), expected.size()) << "Compressed block byte count differs from reference";

    for (u32 byte = 0; byte < actual.size(); byte++) {
        ASSERT_EQ(actual[byte], expected[byte])
            << "Compressed byte mismatch at byte=" << byte << ", actual=" << static_cast<u32>(actual[byte])
            << ", expected=" << static_cast<u32>(expected[byte]);
    }
}

template <typename FixtureT, typename T>
void expectDecompressedBlockNearSource(const FixtureT& fixture, const std::vector<T>& actual, const u32 block) {
    SCOPED_TRACE(testContext(fixture, block));

    const auto& expected = fixture.testSet.getDecompressedData()[block];
    ASSERT_EQ(actual.size(), expected.size()) << "Decompressed block element count differs from source";

    for (u32 element = 0; element < actual.size(); element++) {
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
