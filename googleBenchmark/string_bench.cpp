// This program profiles the short string optimization as implemented
// in GCC using google benchmark
// By: Nick from CoffeeBeforeArch

#include <benchmark/benchmark.h>
#include <string>
#include <vector>

using namespace std;

static void stringBench(benchmark::State &s){
    // Extract the string size
    int len = s.range(0);

    // Create a vector to store our strings
    vector<string> v;
    v.reserve(10000);

    // Run the benchmark
    for(auto _ : s){
        for(int i = 0; i < 10000; i++){
            string str(len, 'x');
            v.push_back(str);
        }
    }
}
// Register the benchmark
BENCHMARK(stringBench)->DenseRange(0,32)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
