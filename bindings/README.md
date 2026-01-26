## Fortran bindings

Fortran bindings for the library are provided in the `bindings/fortran` directory. These bindings allow Fortran programs
to interface with the core functionality of the library.

### Building the Fortran Bindings

To build the Fortran bindings, you need to have a Fortran compiler installed on your system. The build process is
integrated into the main build system of the library. You can enable the Fortran bindings by passing the appropriate
flag to the build configuration command. For example, if you are using CMake, you can enable the Fortran bindings with:

```bash
cmake -DPERNIX_ENABLE_FORTRAN_BINDINGS=ON ..
```
