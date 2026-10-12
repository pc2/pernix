file(GLOB_RECURSE
        SOURCE_FILES
        src/*.f90
)

add_library(pernix_bindings_fortran SHARED ${SOURCE_FILES})
set_target_properties(pernix_bindings_fortran PROPERTIES
        Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mod"
)
target_include_directories(pernix_bindings_fortran PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/mod")
target_link_libraries(pernix_bindings_fortran PUBLIC pernix::pernix)

add_executable(pernix_bindings_fortran_test main.f90)
target_link_libraries(pernix_bindings_fortran_test PRIVATE pernix_bindings_fortran pernix::pernix)

find_package(OpenMP)
if (OpenMP_Fortran_FOUND)
    target_link_libraries(pernix_bindings_fortran_test PUBLIC OpenMP::OpenMP_Fortran)
endif ()

if (BUILD_TESTING AND PROJECT_IS_TOP_LEVEL)
    add_test(NAME pernix_fortran_roundtrip COMMAND pernix_bindings_fortran_test)
    set_tests_properties(pernix_fortran_roundtrip PROPERTIES LABELS "integration;fortran")
endif ()
