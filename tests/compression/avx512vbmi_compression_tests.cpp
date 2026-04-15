#include <pernix/pernix.h>
#include <testset.h>

#ifdef PERNIX_AVX512_VBMI_ENABLED

TYPED_TEST(CompressionTest, AVX512VBMICompressBlock) {
    std::vector<std::vector<uint8_t>> compressedData(this->testSet.numberOfBlocks);

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        compressedData[block].resize(TestFixture::BlockSize);

        pernix::mm512_compress_block_avx512vbmi<TestFixture::BitWidth, TestFixture::BlockSize>(
            this->testSet.getDecompressedData()[block].data(), 1 / this->testSet.getScales()[block],
            reinterpret_cast<uint8_t*>(compressedData[block].data()));
    }

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        std::vector<float_t> restored(this->testSet.elementsPerBlock);
        pernix::decompress_block_fallback<TestFixture::BitWidth, true, TestFixture::BlockSize>(
            compressedData[block].data(), this->testSet.getScales()[block], restored.data());

        for (uint32_t i = 0; i < restored.size(); i++) {
            ASSERT_NEAR(restored[i], this->testSet.getDecompressedData()[block][i], this->testSet.blockTolerance(block))
                << "Mismatch at block " << block << ", element " << i << ": " << restored[i] << " vs "
                << this->testSet.getDecompressedData()[block][i];
        }
    }
}

TYPED_TEST(CompressionTest64, AVX512VBMICompressBlock) {
    std::vector<std::vector<uint8_t>> compressedData(this->testSet.numberOfBlocks);

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        compressedData[block].resize(TestFixture::BlockSize);

        pernix::mm512_compress_block_avx512vbmi<TestFixture::BitWidth, TestFixture::BlockSize>(
            this->testSet.getDecompressedData()[block].data(), 1 / this->testSet.getScales()[block],
            reinterpret_cast<uint8_t*>(compressedData[block].data()));
    }

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        std::vector<double_t> restored(this->testSet.elementsPerBlock);
        pernix::decompress_block_fallback<TestFixture::BitWidth, true, TestFixture::BlockSize>(
            compressedData[block].data(), this->testSet.getScales()[block], restored.data());

        for (uint32_t i = 0; i < restored.size(); i++) {
            ASSERT_NEAR(restored[i], this->testSet.getDecompressedData()[block][i], this->testSet.blockTolerance(block))
                << "Mismatch at block " << block << ", element " << i << ": " << restored[i] << " vs "
                << this->testSet.getDecompressedData()[block][i];
        }
    }
}

#endif  // PERNIX_AVX512_VBMI_ENABLED