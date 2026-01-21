#ifndef PERNIX_PACKING_TESTSET_H
#define PERNIX_PACKING_TESTSET_H

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <random>
#include <type_traits>
#include <vector>

template <bool SIGNED, uint32_t SIZE>
class TestSet {
    using IntType = std::conditional_t<SIGNED, int32_t, uint32_t>;

    alignas(64) std::array<IntType, SIZE> unpacked_data;
    alignas(64) std::vector<uint8_t> packed_data;

public:
    explicit TestSet(const uint8_t bit_width) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<IntType> dis(SIGNED ? -(1 << (bit_width - 1)) : 0,
                                                   SIGNED ? (1 << (bit_width - 1)) - 1 : (1 << bit_width) - 1);

        for (auto& value : unpacked_data) {
            value = dis(gen);
        }

        const auto total_bits = SIZE * bit_width;
        const auto num_bytes  = (total_bits + 7) / 8;
        packed_data.resize(num_bytes, 0);

        for (uint32_t i = 0; i < SIZE; ++i) {
            const uint32_t bit_offset = i * bit_width;
            uint32_t byte_index       = bit_offset / 8;
            uint32_t bit_in_byte      = bit_offset % 8;
            uint32_t bits_to_write    = bit_width;
            uint32_t v_pos            = 0;

            while (bits_to_write > 0) {
                uint8_t& b            = packed_data[byte_index];
                const auto free_bits  = 8 - bit_in_byte;
                const auto write_bits = bits_to_write < free_bits ? bits_to_write : free_bits;
                const auto mask       = (1u << write_bits) - 1u;
                b |= static_cast<uint8_t>(((unpacked_data[i] >> v_pos) & mask) << bit_in_byte);

                bits_to_write -= write_bits;
                v_pos += write_bits;
                ++byte_index;
                bit_in_byte = 0;
            }
        }
    }

    void validate_unpacked(std::array<IntType, SIZE> test_data) const {
        for (uint32_t i = 0; i < SIZE; ++i) {
            ASSERT_EQ(test_data[i], unpacked_data[i])
                << "Mismatch at index " << i << ": expected " << unpacked_data[i] << ", got " << test_data[i];
        }
    }

    void validate_packed(const std::vector<uint8_t>& test_data) const {
        for (size_t i = 0; i < packed_data.size(); ++i) {
            ASSERT_EQ(test_data[i], packed_data[i]) << "Mismatch at byte index " << i << ": expected " << static_cast<int>(packed_data[i])
                                                    << ", got " << static_cast<int>(test_data[i]);
        }
    }

    [[nodiscard]] auto get_unpacked_data() const -> const std::array<IntType, SIZE>& { return unpacked_data; }

    [[nodiscard]] auto get_packed_data() const -> const std::vector<uint8_t>& { return packed_data; }

    [[nodiscard]] static constexpr auto get_size() noexcept -> uint32_t { return SIZE; }
};

#endif  // PERNIX_PACKING_TESTSET_H
