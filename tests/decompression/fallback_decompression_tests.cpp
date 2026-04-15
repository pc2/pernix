#include <../../include/pernix/pernix.h>
#include <testset.h>

TYPED_TEST(DecompressionTest, FallbackDecompressBlock) {
    std::vector<std::vector<float_t>> decompressedData(this->testSet.numberOfBlocks);

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        decompressedData[block].resize(this->testSet.elementsPerBlock);

        pernix::decompress_block_fallback<TestFixture::BitWidth>(this->testSet.getCompressedData()[block].data(),
                                                                 this->testSet.getScales()[block], decompressedData[block].data());
    }

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        std::vector<float_t>& data = decompressedData[block];

        for (uint32_t i = 0; i < data.size(); i++) {
            ASSERT_NEAR(data[i], this->testSet.getDecompressedData()[block][i], this->testSet.blockTolerance(block))
                << "Mismatch at block " << block << ", element " << i << ": " << data[i] << " vs "
                << this->testSet.getDecompressedData()[block][i];
        }
    }
}

TYPED_TEST(DecompressionTest64, FallbackDecompressBlock) {
    std::vector<std::vector<double_t>> decompressedData(this->testSet.numberOfBlocks);

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        decompressedData[block].resize(this->testSet.elementsPerBlock);

        pernix::decompress_block_fallback<TestFixture::BitWidth>(this->testSet.getCompressedData()[block].data(),
                                                                 this->testSet.getScales()[block], decompressedData[block].data());
    }

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        std::vector<double_t>& data = decompressedData[block];

        for (uint32_t i = 0; i < data.size(); i++) {
            ASSERT_NEAR(data[i], this->testSet.getDecompressedData()[block][i], this->testSet.blockTolerance(block))
                << "Mismatch at block " << block << ", element " << i << ": " << data[i] << " vs "
                << this->testSet.getDecompressedData()[block][i];
        }
    }
}