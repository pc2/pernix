#include <pernix/pernix.h>
#include <testset.h>

#ifdef PERNIX_BACKEND_ARM64_NEON

using namespace pernix::arm64::neon;

TYPED_TEST(DecompressionTest, NeonDecompressBlock) {
    std::vector<std::vector<float_t> > decompressedData(this->testSet.numberOfBlocks);

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        decompressedData[block].resize(this->testSet.elementsPerBlock);

        neon_decompress_block<TestFixture::BitWidth, true, TestFixture::BlockSize>(
            this->testSet.getCompressedData()[block].data(), this->testSet.getScales()[block], decompressedData[block].data());
    }

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        expectDecompressedBlockNearSource(*this, decompressedData[block], block);
    }
}

TYPED_TEST(DecompressionTest64, NeonDecompressBlock) {
    std::vector<std::vector<double_t> > decompressedData(this->testSet.numberOfBlocks);

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        decompressedData[block].resize(this->testSet.elementsPerBlock);

        neon_decompress_block<TestFixture::BitWidth, true, TestFixture::BlockSize>(
            this->testSet.getCompressedData()[block].data(), this->testSet.getScales()[block], decompressedData[block].data());
    }

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        expectDecompressedBlockNearSource(*this, decompressedData[block], block);
    }
}

#endif