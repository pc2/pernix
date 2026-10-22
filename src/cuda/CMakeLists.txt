add_library(pernix_cuda_backend
        runtime.cu
        decompress.cu
)

add_library(pernix::pernix_cuda ALIAS pernix_cuda_backend)

set_target_properties(pernix_cuda_backend PROPERTIES
        VERSION "${PROJECT_VERSION}"
        SOVERSION "1"
        EXPORT_NAME pernix_cuda
        CXX_EXTENSIONS OFF
        CUDA_EXTENSIONS OFF
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME pernix_cuda
)

target_compile_features(pernix_cuda_backend
        PUBLIC cxx_std_20
)

target_include_directories(pernix_cuda_backend
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(pernix_cuda_backend
        PUBLIC CUDA::cudart
)

# Keep compiler-specific host flags out of nvcc options.
target_compile_options(pernix_cuda_backend PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

if (BUILD_SHARED_LIBS)
    target_compile_definitions(pernix_cuda_backend
            PUBLIC PERNIX_CUDA_SHARED=1
            PRIVATE PERNIX_CUDA_BUILD=1
    )
endif ()
