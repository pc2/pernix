#include <cstdint>
#include <testset.h>
#include <vector>

// ---------------------------------------------------------------------------
// SIMD compress: compress via backend, decompress via fallback, compare source
// ---------------------------------------------------------------------------

template <typename FixtureT>
void testBackendCompressBlock(FixtureT& fixture, pernix_backend backend) {
    using T = std::remove_cvref_t<decltype(fixture.testSet.getScales()[0])>;

    {
        std::vector<u8> probe(FixtureT::BlockSize);
        pernix_status st;
        if constexpr (std::is_same_v<T, float>) {
            st =
                pernix_compress_block_f32(backend, FixtureT::BitWidth, FixtureT::BlockSize, fixture.testSet.getDecompressedData()[0].data(),
                                          1.0f / fixture.testSet.getScales()[0], probe.data());
        } else {
            st =
                pernix_compress_block_f64(backend, FixtureT::BitWidth, FixtureT::BlockSize, fixture.testSet.getDecompressedData()[0].data(),
                                          1.0 / fixture.testSet.getScales()[0], probe.data());
        }
        if (st != PERNIX_STATUS_OK) {
            GTEST_SKIP();
            return;
        }
    }

    for (u32 b = 0; b < fixture.testSet.numberOfBlocks; b++) {
        std::vector<u8> compressed(FixtureT::BlockSize);

        pernix_status st;
        if constexpr (std::is_same_v<T, float>) {
            st =
                pernix_compress_block_f32(backend, FixtureT::BitWidth, FixtureT::BlockSize, fixture.testSet.getDecompressedData()[b].data(),
                                          1.0f / fixture.testSet.getScales()[b], compressed.data());
        } else {
            st =
                pernix_compress_block_f64(backend, FixtureT::BitWidth, FixtureT::BlockSize, fixture.testSet.getDecompressedData()[b].data(),
                                          1.0 / fixture.testSet.getScales()[b], compressed.data());
        }
        ASSERT_EQ(st, PERNIX_STATUS_OK);

        std::vector<T> restored(fixture.testSet.elementsPerBlock);
        if constexpr (std::is_same_v<T, float>) {
            st = pernix_decompress_block_f32(PERNIX_BACKEND_FALLBACK, FixtureT::BitWidth, FixtureT::BlockSize, compressed.data(),
                                             fixture.testSet.getScales()[b], restored.data(), true);
        } else {
            st = pernix_decompress_block_f64(PERNIX_BACKEND_FALLBACK, FixtureT::BitWidth, FixtureT::BlockSize, compressed.data(),
                                             fixture.testSet.getScales()[b], restored.data(), true);
        }
        ASSERT_EQ(st, PERNIX_STATUS_OK);

        expectDecompressedBlockNearSource(fixture, restored, b);
    }
}

// ---------------------------------------------------------------------------
// SIMD decompress: decompress fallback-compressed data via backend, compare source
// ---------------------------------------------------------------------------

template <typename FixtureT>
void testBackendDecompressBlock(FixtureT& fixture, pernix_backend backend) {
    using T = std::remove_cvref_t<decltype(fixture.testSet.getScales()[0])>;

    {
        std::vector<T> probe(fixture.testSet.elementsPerBlock);
        pernix_status st;
        if constexpr (std::is_same_v<T, float>) {
            st =
                pernix_decompress_block_f32(backend, FixtureT::BitWidth, FixtureT::BlockSize, fixture.testSet.getCompressedData()[0].data(),
                                            fixture.testSet.getScales()[0], probe.data(), true);
        } else {
            st =
                pernix_decompress_block_f64(backend, FixtureT::BitWidth, FixtureT::BlockSize, fixture.testSet.getCompressedData()[0].data(),
                                            fixture.testSet.getScales()[0], probe.data(), true);
        }
        if (st != PERNIX_STATUS_OK) {
            GTEST_SKIP();
            return;
        }
    }

    for (u32 b = 0; b < fixture.testSet.numberOfBlocks; b++) {
        std::vector<T> decompressed(fixture.testSet.elementsPerBlock);

        pernix_status st;
        if constexpr (std::is_same_v<T, float>) {
            st =
                pernix_decompress_block_f32(backend, FixtureT::BitWidth, FixtureT::BlockSize, fixture.testSet.getCompressedData()[b].data(),
                                            fixture.testSet.getScales()[b], decompressed.data(), true);
        } else {
            st =
                pernix_decompress_block_f64(backend, FixtureT::BitWidth, FixtureT::BlockSize, fixture.testSet.getCompressedData()[b].data(),
                                            fixture.testSet.getScales()[b], decompressed.data(), true);
        }
        ASSERT_EQ(st, PERNIX_STATUS_OK);

        expectDecompressedBlockNearSource(fixture, decompressed, b);
    }
}

// ---------------------------------------------------------------------------
// x86: AVX2
// ---------------------------------------------------------------------------

TYPED_TEST(CompressionTest, AVX2CompressBlock) {
    testBackendCompressBlock(*this, PERNIX_BACKEND_X86_AVX2);
}

TYPED_TEST(DecompressionTest, AVX2DecompressBlock) {
    testBackendDecompressBlock(*this, PERNIX_BACKEND_X86_AVX2);
}

TYPED_TEST(CompressionTest64, AVX2CompressBlock) {
    testBackendCompressBlock(*this, PERNIX_BACKEND_X86_AVX2);
}

TYPED_TEST(DecompressionTest64, AVX2DecompressBlock) {
    testBackendDecompressBlock(*this, PERNIX_BACKEND_X86_AVX2);
}

// ---------------------------------------------------------------------------
// x86: BMI2
// ---------------------------------------------------------------------------

TYPED_TEST(CompressionTest, BMI2CompressBlock) {
    testBackendCompressBlock(*this, PERNIX_BACKEND_X86_BMI2);
}

TYPED_TEST(DecompressionTest, BMI2DecompressBlock) {
    testBackendDecompressBlock(*this, PERNIX_BACKEND_X86_BMI2);
}

TYPED_TEST(CompressionTest64, BMI2CompressBlock) {
    testBackendCompressBlock(*this, PERNIX_BACKEND_X86_BMI2);
}

TYPED_TEST(DecompressionTest64, BMI2DecompressBlock) {
    testBackendDecompressBlock(*this, PERNIX_BACKEND_X86_BMI2);
}

// ---------------------------------------------------------------------------
// x86: AVX512-VBMI
// ---------------------------------------------------------------------------

TYPED_TEST(CompressionTest, AVX512VBMICompressBlock) {
    testBackendCompressBlock(*this, PERNIX_BACKEND_X86_AVX512_VBMI);
}

TYPED_TEST(DecompressionTest, AVX512VBMIDecompressBlock) {
    testBackendDecompressBlock(*this, PERNIX_BACKEND_X86_AVX512_VBMI);
}

TYPED_TEST(CompressionTest64, AVX512VBMICompressBlock) {
    testBackendCompressBlock(*this, PERNIX_BACKEND_X86_AVX512_VBMI);
}

TYPED_TEST(DecompressionTest64, AVX512VBMIDecompressBlock) {
    testBackendDecompressBlock(*this, PERNIX_BACKEND_X86_AVX512_VBMI);
}

// ---------------------------------------------------------------------------
// ARM64: NEON (decompress only — no compress implementation)
// ---------------------------------------------------------------------------

TYPED_TEST(DecompressionTest, NeonDecompressBlock) {
    testBackendDecompressBlock(*this, PERNIX_BACKEND_ARM64_NEON);
}

TYPED_TEST(DecompressionTest64, NeonDecompressBlock) {
    testBackendDecompressBlock(*this, PERNIX_BACKEND_ARM64_NEON);
}

// ---------------------------------------------------------------------------
// ARM64: SVE2 (decompress only — no compress implementation)
// ---------------------------------------------------------------------------

TYPED_TEST(DecompressionTest, SVE2DecompressBlock) {
    testBackendDecompressBlock(*this, PERNIX_BACKEND_ARM64_SVE);
}

TYPED_TEST(DecompressionTest64, SVE2DecompressBlock) {
    testBackendDecompressBlock(*this, PERNIX_BACKEND_ARM64_SVE);
}
