# PERNIX: Floating-Point Number De/Compression on CPUs
PERNIX is a high-throughput floating-point number compression library for CPUs. It provides SIMD-optimized implementations of quantization and bit-packing algorithms to reduce transfer bandwidth requirements for scientific applications.

Compression of floating-point numbers $x_i$ to $N$-bit quantized numbers with scale $\varepsilon$ (M. Guidon, F. Schiffmann, J. Hutter and J. VandeVondele, The Journal of Chemical Physics, 2008, 128, 214104):

$$b_{\max} = \max\left(\lvert x_i \rvert\right) \quad \quad  \varepsilon = \frac{b_{\max}}{2^{N-1}-1}$$

$$x_{i,N} = \mathrm{ANINT}\left(x_i\cdot\varepsilon^{-1}\right)$$

* implemented in C++, easy-to-use C/C++ and Fortran API
* optimized for various SIMD instruction sets (AVX2, AVX-512)
* block-based: 512-bit blocks similar to compression on FPGA (Wu, T. Kenter, R. Schade, T. D. Kühne and C. Plessl, 2023 IEEE 31st Annual International Symposium on Field-Programmable Custom Computing Machines (FCCM), 2023, pp. 162-173)

## Compiling
1. clone repository `git clone https://github.com/pc2/pernix`
2. Build with cmake:
    * `cmake -E make_directory "build"`
    * `cmake -E chdir "build" cmake -DCMAKE_BUILD_TYPE=Release -DPERNIX_ENABLE_TESTS=off ../`
    * `cmake --build "build" --config Release`
3. `libpernix.so` will be in `build/src`

## Benchmarking
A benchmark framework for PERNIX can be found at https://github.com/pc2/pernix-benchmark.

## Usage with Bindings
Bindings and examples can be found in the subdirectory `bindings`.
