file(GLOB_RECURSE
        SOURCE_FILES
        src/*.f90
)

add_library(pernix_bindings_fortran SHARED ${SOURCE_FILES})
set_target_properties(pernix_bindings_fortran PROPERTIES
        Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mod"
)
target_include_directories(pernix_bindings_fortran PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/mod")
target_link_libraries(pernix_bindings_fortran PUBLIC pernix)

add_executable(pernix_bindings_fortran_test main.f90)
target_link_libraries(pernix_bindings_fortran_test PRIVATE pernix_bindings_fortran pernix)

find_package(OpenMP)
if (OpenMP_Fortran_FOUND)
    target_link_libraries(pernix_bindings_fortran_test PUBLIC OpenMP::OpenMP_Fortran)
endif ()

if (BUILD_TESTING)
    add_test(NAME pernix_fortran_roundtrip COMMAND pernix_bindings_fortran_test)
endif ()
