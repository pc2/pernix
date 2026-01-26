#include <pernix/pernix.h>
#include <testset.h>

#ifdef PERNIX_AVX512_VBMI_ENABLED

TYPED_TEST(CompressionTest, AVX512VBMICompressBlock) {
    std::vector<std::vector<uint8_t>> compressedData(this->testSet.numberOfBlocks);

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        compressedData[block].resize(64u);

        pernix::mm512_compress_block_avx512vbmi<this->BitWidth>(this->testSet.getDecompressedData()[block].data(),
                                                                1 / this->testSet.getScales()[block],
                                                                reinterpret_cast<uint8_t*>(compressedData[block].data()));
    }

    for (uint32_t block = 0; block < this->testSet.numberOfBlocks; block++) {
        std::vector<uint8_t>& data = compressedData[block];
        std::vector<float_t> decompressedData(this->testSet.elementsPerBlock);
        pernix::decompress_block_fallback<this->BitWidth>(data.data(), this->testSet.getScales()[block], decompressedData.data());

        // for (uint32_t i = 0; i < data.size(); i++) {
        //     ASSERT_EQ(data[i], this->testSet.getCompressedData()[block][i]) << "Mismatch at block " << block << ", byte " << i;
        // }

        const size_t compareCount = std::min(decompressedData.size(), decompressedData.size());
        for (uint32_t i = 0; i < compareCount; i++) {
            ASSERT_NEAR(decompressedData[i], this->testSet.getDecompressedData()[block][i], this->testSet.getScales()[block] / 2)
                << "Mismatch at block " << block << ", element " << i;
        }
    }
}

#endif  // PERNIX_AVX512_VBMI_ENABLED