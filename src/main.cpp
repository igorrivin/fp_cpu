
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <algorithm>
#include <numeric>

#include "fg_sizes_cpu.hpp"
#include "knn_cpu.hpp"

static void die(const char* msg){ std::fprintf(stderr, "%s\n", msg); std::exit(1); }

int main(int argc, char** argv) {
    int n = -1, d = 2, k = 2, threads = -1;
    unsigned seed = 42;
    bool mean_only = false;
    bool compute_next = false;
    bool gen_points = false;
    std::string next_path, points_path, out_points = "points.bin";

    for (int i=1;i<argc;++i){
        std::string a = argv[i];
        if (a=="--n" && i+1<argc) n = std::atoi(argv[++i]);
        else if (a=="--d" && i+1<argc) d = std::atoi(argv[++i]);
        else if (a=="--k" && i+1<argc) k = std::atoi(argv[++i]);
        else if (a=="--threads" && i+1<argc) threads = std::atoi(argv[++i]);
        else if (a=="--next" && i+1<argc) next_path = argv[++i];
        else if (a=="--points" && i+1<argc) points_path = argv[++i];
        else if (a=="--compute-next") compute_next = true;
        else if (a=="--mean_only") mean_only = true;
        else if (a=="--gen-points") gen_points = true;
        else if (a=="--out" && i+1<argc) out_points = argv[++i];
        else if (a=="--seed" && i+1<argc) seed = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
        else if (a=="--help"){
            std::puts("Usage: fg_sizes_cpu --n N [--d D] [--k K] "
                      "[--next next.bin | --points points.bin --compute-next | --gen-points --out points.bin] "
                      "[--threads T] [--mean_only] [--seed S]");
            return 0;
        }
    }
    if (n <= 0) die("Specify --n N");
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#endif

    if (gen_points) {
        if (d <= 0) die("Specify --d D for --gen-points");
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> U(0.f,1.f);
        std::vector<float> X((size_t)n*d);
        for (size_t i=0;i<X.size();++i) X[i] = U(rng);
        std::ofstream f(out_points, std::ios::binary);
        if (!f) die("Could not open output file");
        f.write(reinterpret_cast<const char*>(X.data()), (std::streamsize)X.size()*sizeof(float));
        if (!f) die("Write failed");
        std::printf("Wrote %d points in %dD to %s\n", n, d, out_points.c_str());
        return 0;
    }

    std::vector<int32_t> next;
    if (!next_path.empty()) {
        std::ifstream f(next_path, std::ios::binary);
        if (!f) die("Could not open next.bin");
        next.resize(n);
        f.read(reinterpret_cast<char*>(next.data()), (std::streamsize)n*sizeof(int32_t));
        if (!f) die("Short read in next.bin");
    } else if (compute_next) {
        if (points_path.empty()) die("With --compute-next you must provide --points points.bin");
        std::ifstream f(points_path, std::ios::binary);
        if (!f) die("Could not open points.bin");
        std::vector<float> X((size_t)n*d);
        f.read(reinterpret_cast<char*>(X.data()), (std::streamsize)X.size()*sizeof(float));
        if (!f) die("Short read in points.bin");
        next = kth_neighbor_successor_bruteforce(X.data(), n, d, k);
    } else {
        die("Provide --next next.bin or --points points.bin --compute-next, or use --gen-points");
    }

    auto sizes = reachable_sizes_outdeg1(next);
    long long sum = 0;
    for (auto v: sizes) sum += v;
    double mean = double(sum) / double(n);
    if (mean_only) std::printf("%.9f\n", mean);
    else std::printf("Mean reachable size = %.9f\n", mean);
    return 0;
}
