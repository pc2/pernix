#include <../../include/pernix/pernix.h>
#include <testset.h>

#ifdef PERNIX_AVX512_VBMI_ENABLED

TYPED_TEST(DecompressionTest, AVX512VBMIDecompressBlock) {
    std::vector<std::vector<float_t>> decompressedData(this->testSet.numberOfBlocks);

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        decompressedData[block].resize(this->testSet.elementsPerBlock);

        pernix::mm512_decompress_block_avx512vbmi<TestFixture::BitWidth, true, TestFixture::BlockSize>(
            this->testSet.getCompressedData()[block].data(), this->testSet.getScales()[block], decompressedData[block].data());
    }

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        expectDecompressedBlockNearSource(*this, decompressedData[block], block);
    }
}

TYPED_TEST(DecompressionTest64, AVX512VBMIDecompressBlock) {
    std::vector<std::vector<double_t>> decompressedData(this->testSet.numberOfBlocks);

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        decompressedData[block].resize(this->testSet.elementsPerBlock);

        pernix::mm512_decompress_block_avx512vbmi<TestFixture::BitWidth, true, TestFixture::BlockSize>(
            this->testSet.getCompressedData()[block].data(), this->testSet.getScales()[block], decompressedData[block].data());
    }

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        expectDecompressedBlockNearSource(*this, decompressedData[block], block);
    }
}

#endif  // PERNIX_AVX512_VBMI_ENABLED
