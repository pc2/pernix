#include <benchmark/benchmark.h>

#include <iostream>
#include <ostream>

#include "benchmark_testset.h"

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}