# PERNIX: Floating-Point Number De/Compression on CPUs
PERNIX is a high-throughput floating-point compression library for CPU-based scientific workloads. It quantizes floating-point values to a configurable bit width and packs them into fixed-size blocks, reducing memory and communication bandwidth while keeping decompression fast.

The library provides:

* C++ template API (`pernix::compress_block`, `pernix::decompress_block`)
* C ABI wrappers (`compress_block`, `decompress_block`, and `_f64` variants)
* Fortran bindings in `bindings/fortran`
* SIMD-optimized backends (AVX2, AVX-512 VBMI, BMI2) with fallback implementations

Compression of floating-point numbers $x_i$ to $N$-bit quantized numbers with scale $\varepsilon$ (M. Guidon, F. Schiffmann, J. Hutter and J. VandeVondele, The Journal of Chemical Physics, 2008, 128, 214104):

$$b_{\max} = \max\left(\lvert x_i \rvert\right) \quad \quad  \varepsilon = \frac{b_{\max}}{2^{N-1}-1}$$

$$x_{i,N} = \mathrm{ANINT}\left(x_i\cdot\varepsilon^{-1}\right)$$

PERNIX is block-based and uses 64-byte (512-bit) compressed blocks by default. For a bit width `N`, each block stores `(64 * 8) / N` values.

## Compiling
1. clone repository `git clone https://github.com/pc2/pernix`
2. build with CMake:
    * `cmake -E make_directory "build"`
    * `cmake -E chdir "build" cmake -DCMAKE_BUILD_TYPE=Release -DPERNIX_ENABLE_TESTS=off ../`
    * `cmake --build "build" --config Release`
3. `libpernix.so` will be in `build/src`

To enable Fortran bindings, configure with `-DPERNIX_ENABLE_FORTRAN_BINDINGS=ON`.

## Usage Examples

### C++ API example (single block)

```cpp
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <pernix/pernix.h>

int main() {
    constexpr uint8_t BIT_WIDTH = 16;
    constexpr uint32_t BLOCK_SIZE = 64;
    constexpr size_t ELEMENTS = (BLOCK_SIZE * 8) / BIT_WIDTH; // 32 values for 16-bit

    std::array<float, ELEMENTS> input{};
    for (size_t i = 0; i < ELEMENTS; ++i) {
        input[i] = std::sin(static_cast<float>(i));
    }

    float bmax = 0.0f;
    for (float x : input) {
        bmax = std::max(bmax, std::abs(x));
    }
    const float scale = bmax / ((1u << (BIT_WIDTH - 1)) - 1u);

    std::array<uint8_t, BLOCK_SIZE> compressed{};
    std::array<float, ELEMENTS> restored{};

    if (pernix::compress_block<BIT_WIDTH, BLOCK_SIZE>(input.data(), scale, compressed.data()) != 0) {
        return 1;
    }
    if (pernix::decompress_block<BIT_WIDTH, true, BLOCK_SIZE>(compressed.data(), scale, restored.data()) != 0) {
        return 1;
    }
    return 0;
}
```

### C ABI example (single block)

```c
#include <math.h>
#include <stdint.h>
#include <pernix/pernix.h>

int main(void) {
    const uint8_t bit_width = 16;
    float input[32];
    uint8_t compressed[64];
    float restored[32];
    float scale = 1.0f;

    for (int i = 0; i < 32; ++i) {
        input[i] = sinf((float)i);
    }

    if (compress_block(bit_width, input, scale, compressed) != 0) {
        return 1;
    }
    if (decompress_block(bit_width, compressed, scale, restored) != 0) {
        return 1;
    }
    return 0;
}
```

### Fortran example (using bindings)

```fortran
program pernix_example
    use iso_c_binding, only : c_int8_t, c_float, c_loc
    use pernix_compression
    use pernix_decompression
    implicit none

    integer(c_int8_t), parameter :: bit_width = 16_c_int8_t
    real(c_float), parameter :: scale = 1.5_c_float
    real(c_float), target :: input_data(32), output_data(32)
    integer(c_int8_t), target :: compressed_data(64)
    integer :: i

    do i = 1, size(input_data)
        input_data(i) = real(i, c_float)
    end do

    call compress_block(bit_width, c_loc(input_data), scale, c_loc(compressed_data))
    call decompress_block(bit_width, c_loc(compressed_data), scale, c_loc(output_data))
end program pernix_example
```

For a complete Fortran binding setup, see `bindings/README.md` and `bindings/fortran/main.f90`.

## Benchmarking
A benchmark framework for PERNIX can be found at https://github.com/pc2/pernix-benchmark.

