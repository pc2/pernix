# PERNIX

PERNIX is a small C/C++ library for CPU-based scientific data compression. It quantizes `float` or `double` values to a
user-chosen bit width and packs the quantized integers into fixed 64-byte blocks. The main goal is fast block
decompression with a portable fallback path and optional SIMD backends.

The public API includes:

* C++ functions in `<pernix/pernix.hpp>`, including `pernix::compress_block` and `pernix::decompress_block`
* a plain C ABI in `<pernix/pernix.h>`, including `pernix_compress_block_f32` and `pernix_decompress_block_f32`
* `_f64` variants for double-precision input/output
* optional Fortran bindings that call the C ABI

## Block Format

The normal PERNIX block is exactly 64 bytes, or 512 bits. For bit width `N`, one block stores:

```text
elements_per_block = (64 * 8) / N
```

Valid public bit widths are `1..24`. Public helpers expose these constants and calculations:

* `pernix::compressed_block_size()` / `pernix_compressed_block_size()` returns `64`
* `pernix::elements_per_block(bit_width)` / `pernix_elements_per_block(bit_width)` returns the 64-byte element count
* `pernix::is_valid_bit_width(bit_width)` / `pernix_is_valid_bit_width(bit_width)` checks `1..24`
* `pernix::min_bit_width()` and `pernix::max_bit_width()` return `1` and `24`

The implementation also accepts explicit block sizes `128`, `256`, `512`, and `1024` for internal/test coverage. The
documented interchange format is the 64-byte block.

Packed values are written as a byte stream, least-significant bits first within each value and byte. Compression
zero-fills the output block before packing, so unused tail bits in a block are zero. When `sign_values=true`, decompression
sign-extends each `N`-bit value before multiplying by scale. When `sign_values=false`, decompression treats the packed
value as unsigned. For `bit_width == 1`, signed fallback compression clamps to binary `0/1`.

The scalar fallback packs and unpacks byte-by-byte, so the serialized 64-byte block is not a native integer dump. Backends
are expected to produce compatible blocks for the same inputs, bit width, scale, and sign mode.

## Quantization And Scale

For a block with maximum magnitude `bmax` and bit width `N`, PERNIX uses this forward decompression scale:

```text
scale = bmax / (2^(N - 1) - 1)
```

For `N == 1`, the denominator is treated as `1`. For `bmax == 0`, helper functions return a small positive scale rather
than zero.

The current compression API expects the inverse scale:

```text
quantized = round(input * inverse_scale)
```

The decompression API expects the forward scale:

```text
output = quantized * scale
```

Use these helpers to avoid mixing the two conventions:

* C++: `pernix::decompression_scale_from_bmax`, `pernix::compression_scale_from_bmax`, `pernix::inverse_scale`
* C: `pernix_decompression_scale_f32`, `pernix_compression_scale_f32`, `pernix_inverse_scale_f32`
* C f64: the same names with `_f64`

Compatibility aliases `pernix::scale_from_bmax`, `pernix_scale_f32`, and `pernix_scale_f64` compute the forward
decompression scale.

Scale arguments passed to compression and decompression must be finite and greater than zero. Zero, negative, NaN, or
infinite scales return `PERNIX_STATUS_INVALID_ARGUMENT`.

## API Contracts

All public compression and decompression calls return `pernix_status` (`pernix::Status` in C++).

Status values:

* `PERNIX_STATUS_OK`: success
* `PERNIX_STATUS_INVALID_ARGUMENT`: null pointer, invalid span size, zero block count, or invalid scale
* `PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH`: bit width outside `1..24`
* `PERNIX_STATUS_UNSUPPORTED_BACKEND`: backend enum value is unknown
* `PERNIX_STATUS_UNSUPPORTED_BLOCK_SIZE`: block size is not supported
* `PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION`: backend is not compiled in or is unavailable on this CPU

Use `pernix::status_string(status)` or `pernix_status_string(status)` for readable names.

For one block:

* compression reads at least `(block_size * 8) / bit_width` `float` or `double` values
* compression writes exactly `block_size` bytes
* decompression reads exactly `block_size` bytes
* decompression writes at least `(block_size * 8) / bit_width` `float` or `double` values

For `*_blocks` calls, multiply those sizes by `blocks`. `blocks == 0` is invalid. The C ABI validates null pointers and
basic parameters, but it cannot validate buffer lengths. The C++ `std::span` wrappers validate span sizes before calling
the lower-level kernels.

The `_f32` APIs operate on `float` data and take `float` scale values. The `_f64` APIs operate on `double` data and take
`double` scale values. Both use the same packed integer block format for a given bit width.

Given the same backend, inputs, bit width, scale, and sign mode, behavior is deterministic. Different backends are tested
for compatible results, but exact floating-point details should not be treated as a cross-backend ABI guarantee beyond the
documented quantization model.

## Backends

`PERNIX_BACKEND_FALLBACK` is the portable scalar backend and is always available. `PERNIX_BACKEND_AUTO` selects an
available optimized backend when one is compiled and supported on the host CPU, otherwise it falls back.

Current backend enum values:

* `PERNIX_BACKEND_AUTO`
* `PERNIX_BACKEND_FALLBACK`
* `PERNIX_BACKEND_X86_AVX2`
* `PERNIX_BACKEND_X86_BMI2`
* `PERNIX_BACKEND_X86_AVX512_VBMI`
* `PERNIX_BACKEND_ARM64_NEON`
* `PERNIX_BACKEND_ARM64_SVE`
* `PERNIX_BACKEND_FALLBACK_STDPAR`
* `PERNIX_BACKEND_FALLBACK_SIMD`

x86 SIMD kernels are compiled with per-source ISA flags when enabled. Generic dispatch and fallback code are built for the
baseline target. ARM decompression paths exist, but ARM compression is incomplete/stubbed. The stdpar fallback is currently
a header-only target feature, disabled by default, and the compiled library target does not export stdpar dispatch.

## C++ Example

```cpp
#include <pernix/pernix.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

int main() {
    constexpr u8 bit_width = 16;
    constexpr u32 block_size = pernix::compressed_block_size();
    constexpr std::size_t elements = pernix::elements_per_block(bit_width);

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
        return 2;
    }
    if (pernix::decompress_block(pernix::Backend::Fallback, bit_width, block_size, compressed, scale,
                                 restored) != PERNIX_STATUS_OK) {
        return 3;
    }
}
```

## C Example

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
        return 2;
    }
    if (pernix_decompress_block_f32(PERNIX_BACKEND_FALLBACK, bit_width, block_size, compressed, scale, restored,
                                    true) != PERNIX_STATUS_OK) {
        return 3;
    }
}
```

Complete examples are in `examples/cpp`, `examples/c`, and `examples/fortran`.

## Fortran

Fortran bindings are optional. Enable them with:

```bash
cmake -S . -B build -DPERNIX_ENABLE_FORTRAN_BINDINGS=ON
```

The modules in `bindings/fortran/src` bind directly to the C ABI names and currently expose f32 and f64 compression and
decompression entry points. The bindings do not yet install Fortran module files as a packaged Fortran SDK; they are meant
for in-tree builds and examples.

## Building

Release build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(nproc)"
```

Common options:

* `-DBUILD_SHARED_LIBS=OFF`: build a static library instead of the default shared library
* `-DPERNIX_ENABLE_TESTS=ON`: build tests
* `-DPERNIX_ENABLE_EXAMPLES=ON`: build examples
* `-DPERNIX_ENABLE_INSTALL=ON`: install library, headers, CMake package files, and pkg-config metadata
* `-DPERNIX_ENABLE_INSTALL_CONSUMER_TESTS=ON`: add a CTest that installs PERNIX and builds C/C++ consumers
* `-DPERNIX_ENABLE_FORTRAN_BINDINGS=ON`: build Fortran bindings and Fortran example/test
* `-DPERNIX_ENABLE_X86_AVX2=OFF`, `-DPERNIX_ENABLE_X86_BMI2=OFF`,
  `-DPERNIX_ENABLE_X86_AVX512VBMI=OFF`: disable specific x86 backends
* `-DPERNIX_ENABLE_FALLBACK_STDPAR=OFF|AUTO|ON`: control the header-only stdpar fallback and TBB dependency

Install:

```bash
cmake --install build --prefix /path/to/prefix --config Release
```

CMake consumers can use:

```cmake
find_package(pernix CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE pernix::pernix)
```

## Testing

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPERNIX_ENABLE_TESTS=ON -DPERNIX_ENABLE_EXAMPLES=ON
cmake --build build --config Release -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

Examples are built when `PERNIX_ENABLE_EXAMPLES=ON`:

```bash
./build/examples/pernix_example_cpp
./build/examples/pernix_example_c
```

If Fortran bindings are enabled:

```bash
./build/examples/pernix_example_fortran
```

## Limitations

* Public documentation is centered on fixed 64-byte blocks; larger accepted block sizes are compatibility/internal paths.
* Input values should be finite and within the intended quantization range for portable cross-backend behavior. The scalar
  fallback clamps NaN, infinity, and out-of-range scaled values, but that is not yet specified as a cross-backend policy.
* Scale must be positive and finite.
* ARM64 compression backends are incomplete/stubbed.
* `PERNIX_BACKEND_FALLBACK_STDPAR` is not exported by the compiled library target.
* Fortran bindings are buildable in-tree but not yet packaged for installation.
* The packed format is intended to be stable for a given bit width and sign mode, but PERNIX is still pre-1.0.

## Performance Notes

Decompression is the performance-sensitive path. Prefer `PERNIX_BACKEND_AUTO` for normal use so PERNIX can select an
available optimized backend. Use `PERNIX_BACKEND_FALLBACK` when deterministic portable fallback behavior is more important
than backend selection.

A separate benchmark framework exists at <https://github.com/pc2/pernix-benchmark>.
