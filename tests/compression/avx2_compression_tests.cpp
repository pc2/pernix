#include <../../include/pernix/pernix.h>
#include <testset.h>

#ifdef PERNIX_AVX2_ENABLED

TYPED_TEST(CompressionTest, AVX2CompressBlock) {
    std::vector<std::vector<uint8_t>> compressedData(this->testSet.numberOfBlocks);

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        compressedData[block].resize(TestFixture::BlockSize);

        pernix::mm256_compress_block_avx2<TestFixture::BitWidth, TestFixture::BlockSize>(
            this->testSet.getDecompressedData()[block].data(), 1 / this->testSet.getScales()[block],
            reinterpret_cast<uint8_t*>(compressedData[block].data()));
    }

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        std::vector<float_t> restored(this->testSet.elementsPerBlock);
        pernix::decompress_block_fallback<TestFixture::BitWidth, true, TestFixture::BlockSize>(
            compressedData[block].data(), this->testSet.getScales()[block], restored.data());

        expectDecompressedBlockNearSource(*this, restored, block);
    }
}

TYPED_TEST(CompressionTest64, AVX2CompressBlock) {
    std::vector<std::vector<uint8_t>> compressedData(this->testSet.numberOfBlocks);

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        compressedData[block].resize(TestFixture::BlockSize);

        pernix::mm256_compress_block_avx2<TestFixture::BitWidth, TestFixture::BlockSize>(
            this->testSet.getDecompressedData()[block].data(), 1 / this->testSet.getScales()[block],
            reinterpret_cast<uint8_t*>(compressedData[block].data()));
    }

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        std::vector<double_t> restored(this->testSet.elementsPerBlock);
        pernix::decompress_block_fallback<TestFixture::BitWidth, true, TestFixture::BlockSize>(
            compressedData[block].data(), this->testSet.getScales()[block], restored.data());

        expectDecompressedBlockNearSource(*this, restored, block);
    }
}

#endif  // PERNIX_AVX2_ENABLED
