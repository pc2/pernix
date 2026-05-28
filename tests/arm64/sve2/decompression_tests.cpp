#include <pernix/pernix.h>
#include <testset.h>

#ifdef PERNIX_BACKEND_ARM64_SVE2

using namespace pernix::arm64::sve2;

TYPED_TEST(DecompressionTest, SVE2DecompressBlock) {
    std::vector<std::vector<float_t> > decompressedData(this->testSet.numberOfBlocks);

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        decompressedData[block].resize(this->testSet.elementsPerBlock);

        sve2_decompress_block<TestFixture::BitWidth, true, TestFixture::BlockSize>(
            this->testSet.getCompressedData()[block].data(), this->testSet.getScales()[block], decompressedData[block].data());
    }

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        expectDecompressedBlockNearSource(*this, decompressedData[block], block);
    }
}

TYPED_TEST(DecompressionTest64, SVE2DecompressBlock) {
    std::vector<std::vector<double_t> > decompressedData(this->testSet.numberOfBlocks);

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        decompressedData[block].resize(this->testSet.elementsPerBlock);

        sve2_decompress_block<TestFixture::BitWidth, true, TestFixture::BlockSize>(
            this->testSet.getCompressedData()[block].data(), this->testSet.getScales()[block], decompressedData[block].data());
    }

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        expectDecompressedBlockNearSource(*this, decompressedData[block], block);
    }
}

#endif