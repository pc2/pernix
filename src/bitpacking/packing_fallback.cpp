#include <pernix/bitpacking/packing_fallback.h>

#include <stdexcept>

#define FALLBACK_SWITCH_CASE(BIT_WIDTH) \
    case BIT_WIDTH:                     \
        return pack_epi32_fallback<BIT_WIDTH>(input, destination);

namespace pernix::bitpacking {
void pack_epi32_fallback(const uint8_t bit_width, const std::vector<uint32_t>& input, uint8_t* __restrict__ destination) {
    switch (bit_width) {
        FALLBACK_SWITCH_CASE(1)
        FALLBACK_SWITCH_CASE(2)
        FALLBACK_SWITCH_CASE(3)
        FALLBACK_SWITCH_CASE(4)
        FALLBACK_SWITCH_CASE(5)
        FALLBACK_SWITCH_CASE(6)
        FALLBACK_SWITCH_CASE(7)
        FALLBACK_SWITCH_CASE(8)
        FALLBACK_SWITCH_CASE(9)
        FALLBACK_SWITCH_CASE(10)
        FALLBACK_SWITCH_CASE(11)
        FALLBACK_SWITCH_CASE(12)
        FALLBACK_SWITCH_CASE(13)
        FALLBACK_SWITCH_CASE(14)
        FALLBACK_SWITCH_CASE(15)
        FALLBACK_SWITCH_CASE(16)
        FALLBACK_SWITCH_CASE(17)
        FALLBACK_SWITCH_CASE(18)
        FALLBACK_SWITCH_CASE(19)
        FALLBACK_SWITCH_CASE(20)
        FALLBACK_SWITCH_CASE(21)
        FALLBACK_SWITCH_CASE(22)
        FALLBACK_SWITCH_CASE(23)
        FALLBACK_SWITCH_CASE(24)
        default:
            throw std::invalid_argument("Unsupported bit width for fallback unpacking");
    }
}
}  // namespace pernix::bitpacking