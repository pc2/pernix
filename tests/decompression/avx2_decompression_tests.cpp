#include <../../include/pernix/pernix.h>
#include <testset.h>

#ifdef PERNIX_AVX2_ENABLED

TYPED_TEST(DecompressionTest, AVX2DecompressBlock) {
    std::vector<std::vector<float_t>> decompressedData(this->testSet.numberOfBlocks);

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        decompressedData[block].resize(this->testSet.elementsPerBlock);

        pernix::mm256_decompress_block_avx2<TestFixture::BitWidth>(this->testSet.getCompressedData()[block].data(),
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

TYPED_TEST(DecompressionTest64, AVX2DecompressBlock) {
    std::vector<std::vector<double_t>> decompressedData(this->testSet.numberOfBlocks);

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        decompressedData[block].resize(this->testSet.elementsPerBlock);

        pernix::mm256_decompress_block_avx2<TestFixture::BitWidth>(this->testSet.getCompressedData()[block].data(),
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

#endif  // PERNIX_AVX2_ENABLED