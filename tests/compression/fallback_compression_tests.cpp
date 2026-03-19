#include <../../include/pernix/pernix.h>
#include <testset.h>

TYPED_TEST(CompressionTest, FallbackCompressBlock) {
    std::vector<std::vector<uint8_t>> compressedData(this->testSet.numberOfBlocks);

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        compressedData[block].resize(64u);

        pernix::compress_block_fallback<TestFixture::BitWidth>(this->testSet.getDecompressedData()[block].data(),
                                                               1 / this->testSet.getScales()[block],
                                                               reinterpret_cast<uint8_t*>(compressedData[block].data()));
    }

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        std::vector<uint8_t>& data = compressedData[block];

        for (uint32_t i = 0; i < data.size(); i++) {
            ASSERT_EQ(data[i], this->testSet.getCompressedData()[block][i])
                << "Mismatch at block " << block << ", byte " << i << ", element " << static_cast<int>(data[i]) << " vs "
                << static_cast<int>(this->testSet.getCompressedData()[block][i]);
        }
    }
}

TYPED_TEST(CompressionTest64, FallbackCompressBlock) {
    std::vector<std::vector<uint8_t>> compressedData(this->testSet.numberOfBlocks);

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        compressedData[block].resize(64u);

        pernix::compress_block_fallback<TestFixture::BitWidth>(this->testSet.getDecompressedData()[block].data(),
                                                               1 / this->testSet.getScales()[block],
                                                               reinterpret_cast<uint8_t*>(compressedData[block].data()));
    }

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        std::vector<uint8_t>& data = compressedData[block];

        for (uint32_t i = 0; i < data.size(); i++) {
            ASSERT_EQ(data[i], this->testSet.getCompressedData()[block][i])
                << "Mismatch at block " << block << ", byte " << i << ", element " << static_cast<int>(data[i]) << " vs "
                << static_cast<int>(this->testSet.getCompressedData()[block][i]);
        }
    }
}