#include <pernix/pernix.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace {
constexpr std::array<u32, 5> kBlockSizes{64, 128, 256, 512, 1024};
constexpr u32 kBlockCount = 3;

struct CodecCase {
    u8 bit_width;
    u32 block_size;
    bool use_double;
};

std::vector<CodecCase> make_codec_cases() {
    std::vector<CodecCase> cases;
    cases.reserve(24 * kBlockSizes.size() * 2);
    for (u8 bit_width = pernix::min_bit_width(); bit_width <= pernix::max_bit_width(); ++bit_width) {
        for (const u32 block_size : kBlockSizes) {
            cases.push_back({bit_width, block_size, false});
            cases.push_back({bit_width, block_size, true});
        }
    }
    return cases;
}

const std::vector<CodecCase> kCodecCases = make_codec_cases();

std::string codec_case_name(const testing::TestParamInfo<CodecCase>& info) {
    std::ostringstream name;
    name << "BitWidth_" << static_cast<u32>(info.param.bit_width) << "_BlockSize_" << info.param.block_size << '_'
         << (info.param.use_double ? "F64" : "F32");
    return name.str();
}

u32 test_seed() {
    constexpr u32 default_seed = 0x5eed1234U;
    const char* seed_text      = std::getenv("PERNIX_TEST_SEED");
    if (seed_text == nullptr || *seed_text == '\0') {
        return default_seed;
    }

    char* end                  = nullptr;
    const unsigned long parsed = std::strtoul(seed_text, &end, 0);
    if (end == seed_text || *end != '\0' || parsed > std::numeric_limits<u32>::max()) {
        return default_seed;
    }
    return static_cast<u32>(parsed);
}

i32 minimum_code(const u8 bit_width) {
    return bit_width == 1 ? 0 : -(1 << (bit_width - 1));
}

i32 maximum_code(const u8 bit_width) {
    return bit_width == 1 ? 1 : (1 << (bit_width - 1)) - 1;
}

std::vector<i32> make_codes(const CodecCase& test_case, const u32 seed) {
    const u32 elements_per_block = (test_case.block_size * 8U) / test_case.bit_width;
    const u32 elements           = elements_per_block * kBlockCount;
    const i32 minimum            = minimum_code(test_case.bit_width);
    const i32 maximum            = maximum_code(test_case.bit_width);
    std::mt19937 generator(seed);
    const u32 code_range = static_cast<u32>(maximum - minimum) + 1U;

    std::vector<i32> codes(elements);
    for (u32 index = 0; index < elements; ++index) {
        switch (index % 8U) {
            case 0:
                codes[index] = minimum;
                break;
            case 1:
                codes[index] = maximum;
                break;
            case 2:
                codes[index] = 0;
                break;
            case 3:
                codes[index] = test_case.bit_width == 1 ? 1 : -1;
                break;
            default:
                codes[index] = minimum + static_cast<i32>(generator() % code_range);
                break;
        }
    }
    return codes;
}

// Deliberately slow bit-at-a-time reference. It does not use the production packer.
std::vector<u8> reference_pack(const std::vector<i32>& codes, const u8 bit_width, const u32 block_size) {
    const u32 elements_per_block = (block_size * 8U) / bit_width;
    const u32 blocks             = static_cast<u32>(codes.size()) / elements_per_block;
    std::vector<u8> packed(static_cast<usize>(blocks) * block_size, 0);
    const u32 mask = (1U << bit_width) - 1U;

    for (u32 block = 0; block < blocks; ++block) {
        const usize output_offset = static_cast<usize>(block) * block_size;
        const usize input_offset  = static_cast<usize>(block) * elements_per_block;
        for (u32 element = 0; element < elements_per_block; ++element) {
            const u32 raw         = static_cast<u32>(codes[input_offset + element]) & mask;
            const usize first_bit = static_cast<usize>(element) * bit_width;
            for (u8 bit = 0; bit < bit_width; ++bit) {
                if (((raw >> bit) & 1U) != 0U) {
                    const usize output_bit = first_bit + bit;
                    packed[output_offset + (output_bit / 8U)] |= static_cast<u8>(1U << (output_bit % 8U));
                }
            }
        }
    }
    return packed;
}

std::vector<i32> reference_unpack(const std::vector<u8>& packed, const u8 bit_width, const u32 block_size, const u32 blocks,
                                  const bool sign_values) {
    const u32 elements_per_block = (block_size * 8U) / bit_width;
    const u32 mask               = (1U << bit_width) - 1U;
    std::vector<i32> codes(static_cast<usize>(elements_per_block) * blocks);

    for (u32 block = 0; block < blocks; ++block) {
        const usize input_offset  = static_cast<usize>(block) * block_size;
        const usize output_offset = static_cast<usize>(block) * elements_per_block;
        for (u32 element = 0; element < elements_per_block; ++element) {
            const usize first_bit = static_cast<usize>(element) * bit_width;
            u32 raw               = 0;
            for (u8 bit = 0; bit < bit_width; ++bit) {
                const usize input_bit = first_bit + bit;
                const u32 value       = (packed[input_offset + (input_bit / 8U)] >> (input_bit % 8U)) & 1U;
                raw |= value << bit;
            }

            if (sign_values && bit_width > 1 && (raw & (1U << (bit_width - 1U))) != 0U) {
                codes[output_offset + element] = static_cast<i32>(raw) - static_cast<i32>(1U << bit_width);
            } else {
                codes[output_offset + element] = static_cast<i32>(raw & mask);
            }
        }
    }
    return codes;
}

std::string backend_name(const pernix::Backend backend) {
    switch (backend) {
        case pernix::Backend::X86Avx2:
            return "X86Avx2";
        case pernix::Backend::X86Bmi2:
            return "X86Bmi2";
        case pernix::Backend::X86Avx512Vbmi:
            return "X86Avx512Vbmi";
        case pernix::Backend::Arm64Neon:
            return "Arm64Neon";
        case pernix::Backend::Arm64Sve:
            return "Arm64Sve";
        case pernix::Backend::FallbackStdpar:
            return "FallbackStdpar";
        default:
            return "Fallback";
    }
}

pernix::Backend backend_from_name(const std::string_view name) {
    if (name == "X86Avx2") {
        return pernix::Backend::X86Avx2;
    }
    if (name == "X86Bmi2") {
        return pernix::Backend::X86Bmi2;
    }
    if (name == "X86Avx512Vbmi") {
        return pernix::Backend::X86Avx512Vbmi;
    }
    if (name == "Arm64Neon") {
        return pernix::Backend::Arm64Neon;
    }
    if (name == "Arm64Sve") {
        return pernix::Backend::Arm64Sve;
    }
    return pernix::Backend::Auto;
}

template <typename FloatT>
void check_backend_matrix(const CodecCase& test_case, const u32 seed, const std::vector<i32>& source_codes,
                          const std::vector<u8>& reference_bytes) {
    constexpr std::array<pernix::Backend, 6> backends{
        pernix::Backend::X86Avx2,   pernix::Backend::X86Bmi2,  pernix::Backend::X86Avx512Vbmi,
        pernix::Backend::Arm64Neon, pernix::Backend::Arm64Sve, pernix::Backend::FallbackStdpar,
    };

    std::vector<FloatT> input(source_codes.size());
    std::transform(source_codes.begin(), source_codes.end(), input.begin(), [](const i32 value) { return static_cast<FloatT>(value); });

    for (const pernix::Backend backend : backends) {
        pernix::Dispatcher dispatcher;
        const pernix::Status create_status = pernix::Dispatcher::create(backend, dispatcher);
        if (create_status == PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION) {
            continue;
        }
        ASSERT_EQ(create_status, PERNIX_STATUS_OK) << "backend=" << backend_name(backend);

        SCOPED_TRACE("backend=" + backend_name(backend));
        std::vector<FloatT> restored(source_codes.size());
        ASSERT_EQ(dispatcher.decompress_blocks(test_case.bit_width, test_case.block_size, std::span<const u8>(reference_bytes), FloatT{1},
                                               std::span<FloatT>(restored), kBlockCount, true),
                  PERNIX_STATUS_OK);
        for (usize index = 0; index < source_codes.size(); ++index) {
            EXPECT_EQ(restored[index], static_cast<FloatT>(source_codes[index])) << "element=" << index << ", seed=" << seed;
        }

        std::vector<FloatT> unsigned_restored(source_codes.size());
        ASSERT_EQ(dispatcher.decompress_blocks(test_case.bit_width, test_case.block_size, std::span<const u8>(reference_bytes), FloatT{1},
                                               std::span<FloatT>(unsigned_restored), kBlockCount, false),
                  PERNIX_STATUS_OK);
        const auto expected_unsigned = reference_unpack(reference_bytes, test_case.bit_width, test_case.block_size, kBlockCount, false);
        for (usize index = 0; index < expected_unsigned.size(); ++index) {
            EXPECT_EQ(unsigned_restored[index], static_cast<FloatT>(expected_unsigned[index])) << "element=" << index << ", seed=" << seed;
        }

        const u32 elements_per_block = (test_case.block_size * 8U) / test_case.bit_width;
        std::vector<FloatT> single_restored(elements_per_block);
        ASSERT_EQ(dispatcher.decompress_block(test_case.bit_width, test_case.block_size,
                                              std::span<const u8>(reference_bytes).first(test_case.block_size), FloatT{1},
                                              std::span<FloatT>(single_restored), true),
                  PERNIX_STATUS_OK);
        for (u32 index = 0; index < elements_per_block; ++index) {
            EXPECT_EQ(single_restored[index], static_cast<FloatT>(source_codes[index])) << "element=" << index << ", seed=" << seed;
        }

        std::vector<u8> compressed(reference_bytes.size());
        const pernix::Status compress_status = dispatcher.compress_blocks(
            test_case.bit_width, test_case.block_size, std::span<const FloatT>(input), FloatT{1}, std::span<u8>(compressed), kBlockCount);
        std::vector<u8> single_compressed(test_case.block_size);
        const pernix::Status single_compress_status =
            dispatcher.compress_block(test_case.bit_width, test_case.block_size, std::span<const FloatT>(input).first(elements_per_block),
                                      FloatT{1}, std::span<u8>(single_compressed));
        if (backend == pernix::Backend::Arm64Neon || backend == pernix::Backend::Arm64Sve) {
            EXPECT_EQ(compress_status, PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION);
            EXPECT_EQ(single_compress_status, PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION);
        } else {
            ASSERT_EQ(compress_status, PERNIX_STATUS_OK)
                << "bit_width=" << static_cast<u32>(test_case.bit_width) << ", block_size=" << test_case.block_size;
            ASSERT_EQ(single_compress_status, PERNIX_STATUS_OK)
                << "bit_width=" << static_cast<u32>(test_case.bit_width) << ", block_size=" << test_case.block_size;
            EXPECT_EQ(compressed, reference_bytes);
            EXPECT_TRUE(std::equal(single_compressed.begin(), single_compressed.end(), reference_bytes.begin()));
        }
    }
}

template <typename FloatT>
void check_codec_case(const CodecCase& test_case) {
    const u32 seed =
        test_seed() ^ (static_cast<u32>(test_case.bit_width) << 24U) ^ test_case.block_size ^ (test_case.use_double ? 0xd0b1eU : 0xf032U);
    std::ostringstream context;
    context << "bit_width=" << static_cast<u32>(test_case.bit_width) << ", block_size=" << test_case.block_size << ", float_type=f"
            << (test_case.use_double ? 64 : 32) << ", seed=" << seed;
    SCOPED_TRACE(context.str());

    const std::vector<i32> source_codes  = make_codes(test_case, seed);
    const std::vector<u8> expected_bytes = reference_pack(source_codes, test_case.bit_width, test_case.block_size);
    std::vector<FloatT> input(source_codes.size());
    std::transform(source_codes.begin(), source_codes.end(), input.begin(), [](const i32 value) { return static_cast<FloatT>(value); });

    pernix::Dispatcher fallback;
    ASSERT_EQ(pernix::Dispatcher::create(pernix::Backend::Fallback, fallback), PERNIX_STATUS_OK);

    std::vector<u8> compressed(expected_bytes.size(), 0);
    ASSERT_EQ(fallback.compress_blocks(test_case.bit_width, test_case.block_size, std::span<const FloatT>(input), FloatT{1},
                                       std::span<u8>(compressed), kBlockCount),
              PERNIX_STATUS_OK);
    EXPECT_EQ(compressed, expected_bytes);

    std::vector<FloatT> restored(source_codes.size());
    ASSERT_EQ(fallback.decompress_blocks(test_case.bit_width, test_case.block_size, std::span<const u8>(expected_bytes), FloatT{1},
                                         std::span<FloatT>(restored), kBlockCount, true),
              PERNIX_STATUS_OK);
    for (usize index = 0; index < source_codes.size(); ++index) {
        EXPECT_EQ(restored[index], static_cast<FloatT>(source_codes[index])) << "element=" << index;
    }

    const std::vector<i32> expected_unsigned =
        reference_unpack(expected_bytes, test_case.bit_width, test_case.block_size, kBlockCount, false);
    std::vector<FloatT> unsigned_restored(source_codes.size());
    ASSERT_EQ(fallback.decompress_blocks(test_case.bit_width, test_case.block_size, std::span<const u8>(expected_bytes), FloatT{1},
                                         std::span<FloatT>(unsigned_restored), kBlockCount, false),
              PERNIX_STATUS_OK);
    for (usize index = 0; index < expected_unsigned.size(); ++index) {
        EXPECT_EQ(unsigned_restored[index], static_cast<FloatT>(expected_unsigned[index])) << "element=" << index;
    }

    const u32 elements_per_block = (test_case.block_size * 8U) / test_case.bit_width;
    std::vector<u8> single_block(test_case.block_size);
    ASSERT_EQ(fallback.compress_block(test_case.bit_width, test_case.block_size, std::span<const FloatT>(input).first(elements_per_block),
                                      FloatT{1}, std::span<u8>(single_block)),
              PERNIX_STATUS_OK);
    EXPECT_TRUE(std::equal(single_block.begin(), single_block.end(), expected_bytes.begin()));

    std::vector<FloatT> single_restored(elements_per_block);
    ASSERT_EQ(fallback.decompress_block(test_case.bit_width, test_case.block_size, std::span<const u8>(single_block), FloatT{1},
                                        std::span<FloatT>(single_restored), true),
              PERNIX_STATUS_OK);
    for (u32 index = 0; index < elements_per_block; ++index) {
        EXPECT_EQ(single_restored[index], static_cast<FloatT>(source_codes[index])) << "element=" << index;
    }

    const FloatT minimum = static_cast<FloatT>(minimum_code(test_case.bit_width));
    const FloatT maximum = static_cast<FloatT>(maximum_code(test_case.bit_width));
    const std::array<FloatT, 10> boundary_values{
        minimum - FloatT{1},      minimum - FloatT{0.5},    minimum + FloatT{0.5}, static_cast<FloatT>(-1.5), static_cast<FloatT>(-0.5),
        static_cast<FloatT>(0.5), static_cast<FloatT>(1.5), maximum - FloatT{0.5}, maximum + FloatT{0.5},     maximum + FloatT{1},
    };
    std::vector<FloatT> boundary_input(elements_per_block);
    std::vector<i32> boundary_codes(elements_per_block);
    for (u32 index = 0; index < elements_per_block; ++index) {
        const FloatT value    = boundary_values[index % boundary_values.size()];
        boundary_input[index] = value;
        const FloatT rounded  = std::clamp(std::nearbyint(value), minimum, maximum);
        boundary_codes[index] = static_cast<i32>(rounded);
    }
    const std::vector<u8> boundary_reference = reference_pack(boundary_codes, test_case.bit_width, test_case.block_size);
    std::vector<u8> boundary_compressed(test_case.block_size);
    ASSERT_EQ(fallback.compress_block(test_case.bit_width, test_case.block_size, std::span<const FloatT>(boundary_input), FloatT{1},
                                      std::span<u8>(boundary_compressed)),
              PERNIX_STATUS_OK);
    EXPECT_EQ(boundary_compressed, boundary_reference) << "quantization boundary rounding/clamping";

    check_backend_matrix<FloatT>(test_case, seed, source_codes, expected_bytes);
}

class CodecMatrixTest : public testing::TestWithParam<CodecCase> {};

void check_golden_vector(const u8 bit_width, const std::vector<i32>& prefix, const std::vector<u8>& expected_prefix) {
    constexpr u32 block_size = 64;
    const u32 elements       = (block_size * 8U) / bit_width;
    ASSERT_LE(prefix.size(), elements);

    std::vector<float> input(elements, 0.0f);
    std::transform(prefix.begin(), prefix.end(), input.begin(), [](const i32 value) { return static_cast<float>(value); });
    std::vector<u8> compressed(block_size, 0);
    std::vector<u8> expected(block_size, 0);
    std::copy(expected_prefix.begin(), expected_prefix.end(), expected.begin());

    pernix::Dispatcher fallback;
    ASSERT_EQ(pernix::Dispatcher::create(pernix::Backend::Fallback, fallback), PERNIX_STATUS_OK);
    ASSERT_EQ(fallback.compress_block(bit_width, block_size, std::span<const float>(input), 1.0f, std::span<u8>(compressed)),
              PERNIX_STATUS_OK);
    EXPECT_EQ(compressed, expected);
}

TEST_P(CodecMatrixTest, MatchesIndependentBitModelAcrossAllBackends) {
    const CodecCase test_case = GetParam();
    if (test_case.use_double) {
        check_codec_case<double>(test_case);
    } else {
        check_codec_case<float>(test_case);
    }
}

INSTANTIATE_TEST_SUITE_P(AllWidthsAndBlockSizes, CodecMatrixTest, testing::ValuesIn(kCodecCases), codec_case_name);

TEST(CodecGoldenVectorTest, PacksKnownValuesLeastSignificantBitsFirst) {
    check_golden_vector(1, {0, 1, 1, 0, 1, 0, 0, 1}, {0x96});
    check_golden_vector(3, {1, -1, 2, -2, 3, -3, 0, 0}, {0xb9, 0xbc, 0x02});
    check_golden_vector(4, {1, -2, 7, -8}, {0xe1, 0x87});
    check_golden_vector(24, {0x123456, -1}, {0x56, 0x34, 0x12, 0xff, 0xff, 0xff});
}

TEST(DispatcherContractTest, AutoChoosesTheFirstSupportedBackendInPriorityOrder) {
    constexpr std::array<pernix::Backend, 6> priority{
        pernix::Backend::X86Avx512Vbmi, pernix::Backend::X86Bmi2,   pernix::Backend::X86Avx2,
        pernix::Backend::Arm64Sve,      pernix::Backend::Arm64Neon, pernix::Backend::Fallback,
    };

    pernix::Backend expected = pernix::Backend::Auto;
    for (const pernix::Backend backend : priority) {
        pernix::Dispatcher candidate;
        const pernix::Status status = pernix::Dispatcher::create(backend, candidate);
        if (status == PERNIX_STATUS_OK) {
            expected = backend;
            break;
        }
        EXPECT_EQ(status, PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION) << "backend=" << backend_name(backend);
    }
    ASSERT_NE(expected, pernix::Backend::Auto);

    pernix::Dispatcher automatic;
    ASSERT_EQ(pernix::Dispatcher::create(pernix::Backend::Auto, automatic), PERNIX_STATUS_OK);
    EXPECT_EQ(automatic.backend(), expected);
}

TEST(DispatcherContractTest, RejectsInvalidBackendAndPreservesMoveOwnership) {
    pernix::Dispatcher output;
    EXPECT_EQ(pernix::Dispatcher::create(static_cast<pernix::Backend>(9), output), PERNIX_STATUS_UNSUPPORTED_BACKEND);
    EXPECT_EQ(pernix::Dispatcher::create(static_cast<pernix::Backend>(255), output), PERNIX_STATUS_UNSUPPORTED_BACKEND);
    EXPECT_FALSE(output.configured());
    std::array<float, 32> input{};
    std::array<u8, 64> compressed{};
    EXPECT_EQ(output.compress_block(16, 64, std::span<const float>(input), 1.0f, std::span<u8>(compressed)),
              PERNIX_STATUS_INVALID_ARGUMENT);

    ASSERT_EQ(pernix::Dispatcher::create(pernix::Backend::Fallback, output), PERNIX_STATUS_OK);
    pernix::Dispatcher moved(std::move(output));
    EXPECT_TRUE(moved.configured());
    EXPECT_FALSE(output.configured());
}

TEST(BackendAvailabilityTest, RequiredBackendsAreAvailableWhenConfiguredByCtest) {
    const char* required_text = std::getenv("PERNIX_TEST_REQUIRED_BACKENDS");
    if (required_text == nullptr || *required_text == '\0') {
        SUCCEED();
        return;
    }

    std::istringstream names(required_text);
    std::string name;
    while (std::getline(names, name, ',')) {
        const pernix::Backend backend = backend_from_name(name);
        ASSERT_NE(backend, pernix::Backend::Auto) << "unknown required backend: " << name;
        pernix::Dispatcher dispatcher;
        ASSERT_EQ(pernix::Dispatcher::create(backend, dispatcher), PERNIX_STATUS_OK) << "required backend=" << name;
    }
}
}  // namespace
