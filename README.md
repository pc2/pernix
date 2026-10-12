# PERNIX

PERNIX is a small C/C++ library for CPU-based scientific data compression. It quantizes `float` or `double` values to a
user-chosen bit width and packs the quantized integers into fixed 64-byte blocks. The main goal is fast block
decompression with a portable fallback path and optional SIMD backends.

The public API includes:

* a reusable C++ `pernix::Dispatcher` in `<pernix/pernix.hpp>`
* a plain C ABI in `<pernix/pernix.h>`, including opaque dispatcher creation and `pernix_compress_block_f32` /
  `pernix_decompress_block_f32`
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
zero-fills the output block before packing, so unused tail bits in a block are zero. When `sign_values=true`,
decompression
sign-extends each `N`-bit value before multiplying by scale. When `sign_values=false`, decompression treats the packed
value as unsigned. For `bit_width == 1`, signed fallback compression clamps to binary `0/1`.

The scalar fallback packs and unpacks byte-by-byte, so the serialized 64-byte block is not a native integer dump.
Backends
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
* `PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION`: backend or requested kernel is not built or is unavailable on this CPU
* `PERNIX_STATUS_OUT_OF_MEMORY`: a C dispatcher handle could not be allocated (the compiled C++ dispatcher uses the same handle)

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

Given the same backend, inputs, bit width, scale, and sign mode, behavior is deterministic. Different backends are
tested
for compatible results, but exact floating-point details should not be treated as a cross-backend ABI guarantee beyond
the
documented quantization model.

## Backends

`PERNIX_BACKEND_FALLBACK` is the portable scalar backend and is always available. Create a dispatcher once before
processing data, then reuse it for compression and decompression. `PERNIX_BACKEND_AUTO` resolves to one concrete backend
when the dispatcher is created: AVX512 VBMI, BMI2, AVX2, SVE2, NEON, then scalar fallback, subject to the kernels built
for the library and CPU features available on the host. The selected backend stays fixed for that dispatcher's lifetime.
The C++ `Dispatcher` is move-only. In compiled builds it owns and releases a dispatcher handle; the C API exposes
explicit create and destroy functions. Compiled C++ and C calls share the library registry; the header-only C++ target
uses the same registry definitions inline.

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

Auto resolves once and never switches backends per operation. If the selected backend has no kernel for an operation,
that call returns `PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION`. ARM64 compression kernels are currently incomplete, so use
the fallback backend for a portable compression/decompression round trip on ARM64.

x86 SIMD kernels are compiled with per-source ISA flags when enabled. Generic dispatch and fallback code are built for
the
baseline target. ARM decompression paths exist, but ARM compression is incomplete/stubbed. The stdpar fallback is
disabled
by default. When enabled, both the compiled library and the header-only target export stdpar dispatch. If TBB is
available,
stdpar uses standard parallel execution policies; otherwise it preserves the same 8-value grouping logic while executing
sequentially.

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
    constexpr usize elements = pernix::elements_per_block(bit_width);

    std::array<float, elements> input{};
    for (usize i = 0; i < input.size(); ++i) {
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

    pernix::Dispatcher dispatcher;
    if (pernix::Dispatcher::create(pernix::Backend::Fallback, dispatcher) != PERNIX_STATUS_OK) {
        return 2;
    }
    if (dispatcher.compress_block(bit_width, block_size, input, inverse_scale, compressed) != PERNIX_STATUS_OK) {
        return 3;
    }
    if (dispatcher.decompress_block(bit_width, block_size, compressed, scale, restored) != PERNIX_STATUS_OK) {
        return 4;
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
    pernix_dispatcher* dispatcher = 0;

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

    if (pernix_dispatcher_create(PERNIX_BACKEND_FALLBACK, &dispatcher) != PERNIX_STATUS_OK) {
        return 2;
    }
    if (pernix_compress_block_f32(dispatcher, bit_width, block_size, input, inverse_scale, compressed) != PERNIX_STATUS_OK) {
        pernix_dispatcher_destroy(dispatcher);
        return 3;
    }
    if (pernix_decompress_block_f32(dispatcher, bit_width, block_size, compressed, scale, restored, true) != PERNIX_STATUS_OK) {
        pernix_dispatcher_destroy(dispatcher);
        return 4;
    }
    pernix_dispatcher_destroy(dispatcher);
}
```

Complete examples are in `examples/cpp`, `examples/c`, and `examples/fortran`.

## Fortran

Fortran bindings are optional. Enable them with:

```bash
cmake -S . -B build -DPERNIX_ENABLE_FORTRAN_BINDINGS=ON
```

The modules in `bindings/fortran/src` bind directly to the C ABI names and expose dispatcher creation/destruction plus
f32 and f64 compression and decompression entry points. The bindings do not yet install Fortran module files as a
packaged Fortran SDK; they are meant for in-tree builds and examples.

## Building

Release build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build build --config Release -j"$(nproc)"
```

Common options:

* `-DBUILD_SHARED_LIBS=OFF`: build a static library; shared is the default
* `-DBUILD_TESTING=ON`: configure the test targets; tests are enabled by default
* `-DPERNIX_ENABLE_EXAMPLES=ON`: build the C and C++ examples
* `-DPERNIX_ENABLE_INSTALL=OFF`: skip install rules; installation is enabled by default
* `-DPERNIX_ENABLE_INSTALL_CONSUMER_TESTS=ON`: add a CTest that installs the package and builds C/C++ consumers
* `-DPERNIX_ENABLE_FORTRAN_BINDINGS=ON`: build the optional Fortran bindings and example
* `-DPERNIX_ENABLE_X86_64=OFF`: disable all x86_64 optimized backends
* `-DPERNIX_ENABLE_X86_64_AVX2=OFF`, `-DPERNIX_ENABLE_X86_64_BMI2=OFF`,
  `-DPERNIX_ENABLE_X86_64_AVX512=ON`: select x86_64 backends
* `-DPERNIX_ENABLE_ARM64=OFF`: disable all ARM64 optimized backends
* `-DPERNIX_ENABLE_ARM64_NEON=OFF`, `-DPERNIX_ENABLE_ARM64_SVE2=ON`: select ARM64 backends
* `-DPERNIX_ENABLE_LTO=ON`: enable link-time optimization
* `-DPERNIX_ENABLE_SANITIZER_ADDRESS=ON` or `-DPERNIX_ENABLE_SANITIZER_UNDEFINED=ON`: enable a sanitizer
* `-DPERNIX_ENABLE_DOCS=ON`: generate API documentation with Doxygen
* `-DPERNIX_INSTALL_DOCS=ON`: install generated docs; requires docs and installation to be enabled

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

The test suite uses GoogleTest, CMake, and CTest. On Debian or Ubuntu, install the test dependencies with:

```bash
sudo apt-get install cmake ninja-build libgtest-dev
```

Configure, build, and run the full suite with the checked-in preset:

```bash
cmake --preset tests
cmake --build --preset tests --parallel
ctest --preset tests
```

The matrix covers all bit widths and supported block sizes from 64 through 1024 bytes. To run one part of the suite, use its CTest label:

```bash
ctest --test-dir cmake-build-tests -L header_only --output-on-failure
ctest --test-dir cmake-build-tests -L c-api --output-on-failure
```

The pseudo-random cases use a fixed seed by default. Set `PERNIX_TEST_SEED` to reproduce a different deterministic input set. CI can set `PERNIX_TEST_REQUIRED_BACKENDS` at configure time to require a comma-separated list of host backends, such as `X86Avx2,X86Bmi2`.

Run the sanitizer suite with:

```bash
cmake --preset tests-sanitizers
cmake --build --preset tests-sanitizers --parallel
ctest --preset tests-sanitizers
```

Generate a coverage report with GCC or Clang and `lcov`/`genhtml` installed:

```bash
cmake --preset tests-coverage
cmake --build --preset tests-coverage --parallel
```

The coverage build instruments the compiled library and all test targets. Reports are written to `cmake-build-coverage/coverage_report`.

To configure without presets, the equivalent basic workflow is:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DPERNIX_ENABLE_EXAMPLES=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure --no-tests=error
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

* Public documentation is centered on fixed 64-byte blocks; larger accepted block sizes are compatibility/internal
  paths.
* Input values should be finite and within the intended quantization range for portable cross-backend behavior. The
  scalar
  fallback clamps NaN, infinity, and out-of-range scaled values, but that is not yet specified as a cross-backend
  policy.
* Scale must be positive and finite.
* ARM64 compression backends are incomplete/stubbed.
* `PERNIX_BACKEND_FALLBACK_STDPAR` is not exported by the compiled library target.
* Fortran bindings are buildable in-tree but not yet packaged for installation.
* The packed format is intended to be stable for a given bit width and sign mode, but PERNIX is still pre-1.0.

## Performance Notes

Decompression is the performance-sensitive path. Prefer `PERNIX_BACKEND_AUTO` for normal use so PERNIX can select an
available optimized backend. Use `PERNIX_BACKEND_FALLBACK` when deterministic portable fallback behavior is more
important
than backend selection.

A separate benchmark framework exists at <https://github.com/pc2/pernix-benchmark>.
