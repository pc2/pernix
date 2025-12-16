find_package(PkgConfig)
pkg_search_module(BENCHMARK REQUIRED benchmark)

file(GLOB_RECURSE
        SOURCE_FILES
        src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
)

add_executable(libcompression_benchmarks)
target_sources(libcompression_benchmarks PRIVATE ${SOURCE_FILES})
target_link_libraries(libcompression_benchmarks PRIVATE libcompression ${BENCHMARK_LDFLAGS})
target_compile_options(libcompression_benchmarks PRIVATE ${BENCHMARK_CFLAGS})
target_include_directories(libcompression_benchmarks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(libcompression_benchmarks PRIVATE -march=native)

#[===[
add_executable(libcompression_benchmarks_fallback src/main.cpp src/benchmark_fallback.cpp)
target_link_libraries(libcompression_benchmarks_fallback PRIVATE libcompression ${BENCHMARK_LDFLAGS})
target_compile_options(libcompression_benchmarks_fallback PRIVATE ${BENCHMARK_CFLAGS})
target_include_directories(libcompression_benchmarks_fallback PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(libcompression_benchmarks_fallback PRIVATE -DLIBCOMPRESSION_FALLBACK_ONLY)
target_compile_options(libcompression_benchmarks_fallback PRIVATE -march=x86-64)


add_executable(libcompression_benchmarks_avx2 src/main.cpp src/benchmark_avx2.cpp)
target_link_libraries(libcompression_benchmarks_avx2 PRIVATE libcompression ${BENCHMARK_LDFLAGS})
target_compile_options(libcompression_benchmarks_avx2 PRIVATE ${BENCHMARK_CFLAGS})
target_include_directories(libcompression_benchmarks_avx2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(libcompression_benchmarks_avx2 PRIVATE -march=x86-64-v3)


add_executable(libcompression_benchmarks_bmi2 src/main.cpp src/benchmark_bmi2.cpp)
target_link_libraries(libcompression_benchmarks_bmi2 PRIVATE libcompression ${BENCHMARK_LDFLAGS})
target_compile_options(libcompression_benchmarks_bmi2 PRIVATE ${BENCHMARK_CFLAGS})
target_include_directories(libcompression_benchmarks_bmi2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(libcompression_benchmarks_bmi2 PRIVATE -march=x86-64-v3)


add_executable(libcompression_benchmarks_avx512vbmi src/main.cpp src/benchmark_avx512vbmi.cpp)
target_link_libraries(libcompression_benchmarks_avx512vbmi PRIVATE libcompression ${BENCHMARK_LDFLAGS})
target_compile_options(libcompression_benchmarks_avx512vbmi PRIVATE ${BENCHMARK_CFLAGS})
target_include_directories(libcompression_benchmarks_avx512vbmi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(libcompression_benchmarks_avx512vbmi PRIVATE -march=x86-64-v4 -mavx512vbmi)
]===]