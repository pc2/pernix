#include <../../include/pernix/pernix.h>
#include <testset.h>

#ifdef PERNIX_AVX512_VBMI_ENABLED

TYPED_TEST(DecompressionTest, AVX512VBMIDecompressBlock) {
    std::vector<std::vector<float_t>> decompressedData(this->testSet.numberOfBlocks);

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        decompressedData[block].resize(this->testSet.elementsPerBlock);

        pernix::mm512_decompress_block_avx512vbmi<this->BitWidth>(this->testSet.getCompressedData()[block].data(),
                                                                  this->testSet.getScales()[block], decompressedData[block].data());
    }

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        std::vector<float_t>& data = decompressedData[block];

        for (uint32_t i = 0; i < data.size(); i++) {
            ASSERT_NEAR(decompressedData[block][i], decompressedData[block][i], this->testSet.getScales()[block] / 2);
        }
    }
}

#endif  // PERNIX_AVX512_VBMI_ENABLED