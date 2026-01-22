enable_language(Fortran)

file(GLOB_RECURSE
        SOURCE_FILES
        src/*.f90
)

add_library(pernix_bindings_fortran SHARED ${SOURCE_FILES})
target_sources(pernix_bindings_fortran PRIVATE ${SOURCE_FILES})

add_executable(pernix_bindings_fortran_test main.f90)
target_link_libraries(pernix_bindings_fortran_test PRIVATE pernix_bindings_fortran pernix)
link_directories( $ENV{LD_LIBRARY_PATH} )

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(pernix_bindings_fortran_test PUBLIC OpenMP::OpenMP_CXX)
endif ()
