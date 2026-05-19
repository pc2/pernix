#include <pernix/pernix.h>
#include <testset.h>

TYPED_TEST(CompressionTest, FallbackCompressBlock) {
    std::vector<std::vector<uint8_t>> compressedData(this->testSet.numberOfBlocks);

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        compressedData[block].resize(TestFixture::BlockSize);

        pernix::compress_block_fallback<TestFixture::BitWidth, TestFixture::BlockSize>(
            this->testSet.getDecompressedData()[block].data(), 1 / this->testSet.getScales()[block],
            reinterpret_cast<uint8_t*>(compressedData[block].data()));
    }

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        expectCompressedBlockEqualsReference(*this, compressedData[block], block);
    }
}

TYPED_TEST(CompressionTest64, FallbackCompressBlock) {
    std::vector<std::vector<uint8_t>> compressedData(this->testSet.numberOfBlocks);

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        compressedData[block].resize(TestFixture::BlockSize);

        pernix::compress_block_fallback<TestFixture::BitWidth, TestFixture::BlockSize>(
            this->testSet.getDecompressedData()[block].data(), 1 / this->testSet.getScales()[block],
            reinterpret_cast<uint8_t*>(compressedData[block].data()));
    }

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        expectCompressedBlockEqualsReference(*this, compressedData[block], block);
    }
}
