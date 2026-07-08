add_executable(pernix_example_cpp cpp/roundtrip.cpp)
target_link_libraries(pernix_example_cpp PRIVATE pernix)

add_executable(pernix_example_c c/roundtrip.c)
target_link_libraries(pernix_example_c PRIVATE pernix)

if (PERNIX_ENABLE_FORTRAN_BINDINGS)
    add_executable(pernix_example_fortran fortran/roundtrip.f90)
    target_link_libraries(pernix_example_fortran PRIVATE pernix_bindings_fortran pernix)
endif ()
