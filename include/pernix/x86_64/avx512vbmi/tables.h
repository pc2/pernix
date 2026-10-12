#ifndef PERNIX_AVX512VBMI_TABLES_H
#define PERNIX_AVX512VBMI_TABLES_H

#include <pernix/simd_compat.h>

#include <array>
#include <type_traits>

namespace pernix::internal {
template <typename Vec, typename U, usize N>
static __always_inline Vec load_table(const std::array<U, N>& table) {
    static_assert(sizeof(table) >= sizeof(Vec), "table is smaller than requested SIMD vector");
    if constexpr (std::is_same_v<Vec, __m512i>) {
        return _mm512_load_si512(static_cast<const void*>(table.data()));
    } else if constexpr (std::is_same_v<Vec, __m256i>) {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(table.data()));
    } else {
        return _mm_load_si128(reinterpret_cast<const __m128i*>(table.data()));
    }
}
}  // namespace pernix::internal

#endif  // PERNIX_AVX512VBMI_TABLES_H
