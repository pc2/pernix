# PERNIX: Floating-Point Number De/Compression on CPUs

PERNIX is a high-throughput floating-point compression library for CPU-based scientific workloads. It quantizes
floating-point values to a configurable bit width and packs them into fixed-size compressed blocks, reducing memory and
communication bandwidth while keeping decompression fast.

The library provides:

* C++ API functions such as `pernix::compress_block` and `pernix::decompress_block`
* C ABI wrappers named `pernix_compress_block_f32`, `pernix_decompress_block_f32`, and matching `_f64` variants
* Optional Fortran bindings in `bindings/fortran`
* SIMD-optimized backends where available, with a portable fallback backend

## Quantization Convention

For a block of floating-point numbers `x_i` and a bit width `N`, PERNIX uses the block maximum magnitude:

```text
bmax = max(abs(x_i))
scale = bmax / (2^(N - 1) - 1)
```

The current compression API expects the inverse scale, because compression computes:

```text
quantized = round(input * scale_inverse)
```

Decompression expects the forward scale, because decompression computes:

```text
output = quantized * scale
```

In practice, compute the decompression scale from `bmax`, pass its inverse to `compress_block`, and pass the forward
scale to `decompress_block`. The clearer helper names are:

* `pernix_decompression_scale_f32` / `pernix_decompression_scale_f64`
* `pernix_compression_scale_f32` / `pernix_compression_scale_f64`
* `pernix_inverse_scale_f32` / `pernix_inverse_scale_f64`
* `pernix::decompression_scale_from_bmax`, `pernix::compression_scale_from_bmax`, and `pernix::inverse_scale`

The older `pernix_scale_f32`, `pernix_scale_f64`, and `pernix::scale_from_bmax` names are kept as compatibility aliases
for the forward decompression scale. Scale-from-`bmax` helpers return a small positive scale for `bmax == 0`.

Scale arguments passed to compression and decompression must be finite and greater than zero. Passing zero, negative,
NaN, or infinity returns `PERNIX_STATUS_INVALID_ARGUMENT`.

## Block Format and Limits

PERNIX uses 64-byte (512-bit) compressed blocks by default. For bit width `N`, one 64-byte block stores
`(64 * 8) / N` values. The fallback implementation currently supports bit widths `1..24`. Public APIs also accept an
explicit `block_size`; supported values are `64`, `128`, `256`, `512`, and `1024`, but the documented default and normal
format is a fixed 64-byte block.

Packed values are signed by default during decompression. A `sign_values` argument is available in the C ABI and in the
C++ API to request unsigned interpretation.

The public helper APIs expose the fixed-format constants:

* `pernix_min_bit_width()` / `pernix::min_bit_width()` return `1`
* `pernix_max_bit_width()` / `pernix::max_bit_width()` return `24`
* `pernix_compressed_block_size()` / `pernix::compressed_block_size()` return `64`
* `pernix_elements_per_block(bit_width)` / `pernix::elements_per_block(bit_width)` return `(64 * 8) / bit_width`, or
  `0` for an invalid bit width
* `pernix_is_valid_bit_width(bit_width)` / `pernix::is_valid_bit_width(bit_width)` validate the public `1..24` range
* `pernix_is_valid_block_size(block_size)` / `pernix::is_valid_block_size(block_size)` validate the currently accepted
  block sizes

Input values should be finite for portable behavior. The scalar fallback clamps non-finite or out-of-range values before
narrowing, but that behavior is not a cross-backend contract.

## Status Codes

The C ABI returns `pernix_status`:

* `PERNIX_STATUS_OK`: success
* `PERNIX_STATUS_INVALID_ARGUMENT`: null pointer or invalid parameter
* `PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH`: unsupported bit width
* `PERNIX_STATUS_UNSUPPORTED_BACKEND`: unknown backend value
* `PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE`: unsupported block size
* `PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION`: backend was requested but is not available for this build or CPU

Use `pernix_status_string(status)` or `pernix::status_string(status)` for diagnostic text.

## f32 and f64 APIs

Use the `_f32` functions for `float` input/output and `_f64` functions for `double` input/output. Both variants pack to
the same integer bit stream for the selected bit width. The scale type matches the floating-point type:
`float` for `_f32`, `double` for `_f64`.

The C++ wrapper overloads return `pernix::Status`, an alias for the C ABI `pernix_status`. `std::span` arguments are
borrowed only for the duration of the call. For one 64-byte block, compression reads at least
`pernix::elements_per_block(bit_width)` input values and writes `pernix::compressed_block_size()` bytes. Multi-block
calls multiply those counts by `blocks`.

## Compiling

```bash
cmake -E make_directory build
cmake -E chdir build cmake -DCMAKE_BUILD_TYPE=Release -DPERNIX_ENABLE_TESTS=off ../
cmake --build build --config Release
```

The library is written under `build/src`. Shared builds are the default; pass `-DBUILD_SHARED_LIBS=OFF` for a static
library.

Optional build flags:

* `-DPERNIX_ENABLE_TESTS=ON` builds the test suite.
* `-DPERNIX_ENABLE_EXAMPLES=ON` builds the C and C++ examples.
* `-DPERNIX_ENABLE_FORTRAN_BINDINGS=ON` builds the Fortran bindings and Fortran round-trip program.
* `-DPERNIX_ENABLE_INSTALL=ON` installs the library, headers, pkg-config file, and CMake package files.
* `-DPERNIX_ENABLE_INSTALL_CONSUMER_TESTS=ON` adds a CTest that installs PERNIX into a temporary prefix and verifies
  C and C++ consumers with `find_package(pernix CONFIG REQUIRED)`.
* `-DPERNIX_ENABLE_FALLBACK_STDPAR=OFF` is the default and avoids the TBB dependency. Use `AUTO` to enable the
  header-only stdpar fallback backend only when TBB is found, or `ON` to require TBB and fail if missing. The compiled
  library target does not currently export stdpar dispatch.
* `-DPERNIX_ENABLE_X86_AVX2=OFF`, `-DPERNIX_ENABLE_X86_BMI2=OFF`, and
  `-DPERNIX_ENABLE_X86_AVX512VBMI=OFF` disable individual x86 SIMD backends.

## Usage Examples

### C++ API

```cpp
#include <pernix/pernix.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

int main() {
    constexpr u8 bit_width = 16;
    constexpr u32 block_size = 64;
    constexpr std::size_t elements = (block_size * 8U) / bit_width;

    std::array<float, elements> input{};
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.25f);
    }

    float bmax = 0.0f;
    for (float value : input) {
        bmax = std::max(bmax, std::abs(value));
    }
    float scale = 0.0f;
    float inverse_scale = 0.0f;
    if (pernix::decompression_scale_from_bmax(bmax, bit_width, scale) != PERNIX_STATUS_OK ||
        pernix::inverse_scale(scale, inverse_scale) != PERNIX_STATUS_OK) {
        return 1;
    }

    std::array<u8, block_size> compressed{};
    std::array<float, elements> restored{};

    if (pernix::compress_block(pernix::Backend::Fallback, bit_width, block_size, input, inverse_scale,
                               compressed) != PERNIX_STATUS_OK) {
        return 1;
    }
    if (pernix::decompress_block(pernix::Backend::Fallback, bit_width, block_size, compressed, scale,
                                 restored) != PERNIX_STATUS_OK) {
        return 1;
    }
    return 0;
}
```

### C ABI

```c
#include <pernix/pernix.h>

int main(void) {
    enum { bit_width = 16, block_size = 64, elements = (block_size * 8) / bit_width };
    float input[elements];
    float restored[elements];
    u8 compressed[block_size];
    float bmax = 0.0f;

    for (int i = 0; i < elements; ++i) {
        input[i] = ((float)i - 16.0f) * 0.125f;
        const float magnitude = input[i] < 0.0f ? -input[i] : input[i];
        bmax = bmax < magnitude ? magnitude : bmax;
    }

    float scale = 0.0f;
    float inverse_scale = 0.0f;
    if (pernix_decompression_scale_f32(bmax, bit_width, &scale) != PERNIX_STATUS_OK ||
        pernix_inverse_scale_f32(scale, &inverse_scale) != PERNIX_STATUS_OK) {
        return 1;
    }

    if (pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, bit_width, block_size, input, inverse_scale,
                                  compressed) != PERNIX_STATUS_OK) {
        return 1;
    }
    if (pernix_decompress_block_f32(PERNIX_BACKEND_FALLBACK, bit_width, block_size, compressed, scale, restored,
                                    true) != PERNIX_STATUS_OK) {
        return 1;
    }
    return 0;
}
```

### Fortran

Fortran bindings call the same C ABI symbols. Enable them with `-DPERNIX_ENABLE_FORTRAN_BINDINGS=ON`. A minimal
round-trip program is provided in `bindings/fortran/main.f90`, and a standalone example is in
`examples/fortran/roundtrip.f90`.

## Benchmarking

A benchmark framework for PERNIX can be found at https://github.com/pc2/pernix-benchmark.
