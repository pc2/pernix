#include <pernix/bitpacking/unpacking_fallback.h>

#define FALLBACK_SWITCH_CASE(BIT_WIDTH, SIGNED) \
    case BIT_WIDTH:                             \
        return unpack_epi32_fallback<BIT_WIDTH, SIGNED>(input, elements);

namespace pernix::bitpacking {
auto unpack_epi32_fallback(const uint8_t bit_width, const uint8_t* __restrict__ input, const std::size_t elements) -> std::vector<int32_t> {
    switch (bit_width) {
        FALLBACK_SWITCH_CASE(1, true)
        FALLBACK_SWITCH_CASE(2, true)
        FALLBACK_SWITCH_CASE(3, true)
        FALLBACK_SWITCH_CASE(4, true)
        FALLBACK_SWITCH_CASE(5, true)
        FALLBACK_SWITCH_CASE(6, true)
        FALLBACK_SWITCH_CASE(7, true)
        FALLBACK_SWITCH_CASE(8, true)
        FALLBACK_SWITCH_CASE(9, true)
        FALLBACK_SWITCH_CASE(10, true)
        FALLBACK_SWITCH_CASE(11, true)
        FALLBACK_SWITCH_CASE(12, true)
        FALLBACK_SWITCH_CASE(13, true)
        FALLBACK_SWITCH_CASE(14, true)
        FALLBACK_SWITCH_CASE(15, true)
        FALLBACK_SWITCH_CASE(16, true)
        FALLBACK_SWITCH_CASE(17, true)
        FALLBACK_SWITCH_CASE(18, true)
        FALLBACK_SWITCH_CASE(19, true)
        FALLBACK_SWITCH_CASE(20, true)
        FALLBACK_SWITCH_CASE(21, true)
        FALLBACK_SWITCH_CASE(22, true)
        FALLBACK_SWITCH_CASE(23, true)
        FALLBACK_SWITCH_CASE(24, true)
        default:
            throw std::invalid_argument("Unsupported bit width for fallback unpacking");
    }
}
}  // namespace pernix::bitpacking