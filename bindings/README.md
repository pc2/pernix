## Fortran bindings

Fortran bindings for the library are provided in the `bindings/fortran` directory. They expose the public C ABI symbols
such as `pernix_compress_block_f32`, `pernix_decompress_block_f32`, and the matching f64 variants.

### Building the Fortran bindings

To build the Fortran bindings, you need a Fortran compiler. Enable them from the main CMake build:

```bash
cmake -S . -B build -DPERNIX_ENABLE_FORTRAN_BINDINGS=ON
cmake --build build --config Release -j"$(nproc)"
```

The `pernix_bindings_fortran_test` target performs one f32 64-byte block round trip through the Fortran modules.
