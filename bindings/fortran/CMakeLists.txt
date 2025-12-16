enable_language(Fortran)

file(GLOB_RECURSE
        SOURCE_FILES
        src/*.f90
)

add_library(libcompression_bindings_fortran SHARED ${SOURCE_FILES})
target_sources(libcompression_bindings_fortran PRIVATE ${SOURCE_FILES})

add_executable(libcompression_bindings_fortran_test main.f90)
target_link_libraries(libcompression_bindings_fortran_test PRIVATE libcompression_bindings_fortran libcompression)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(libcompression_bindings_fortran_test PUBLIC OpenMP::OpenMP_CXX)
endif ()