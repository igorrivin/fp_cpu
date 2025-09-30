
# Functional Graph Reachable Sizes — CPU-only C++ (with generator)

This builds and runs without CUDA/FAISS. It includes:

- O(n) **functional-graph** sizes (exact) from `next[i]`.
- Optional **CPU brute-force kNN** to build `next[i]` from points.
- A **random point generator** to create `points.bin`.

## Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

## Generate random points
```bash
./fg_sizes_cpu --n 100000 --d 2 --gen-points --out points.bin
```

## Compute successors and sizes
```bash
./fg_sizes_cpu --n 100000 --d 2 --k 2 --points points.bin --compute-next
# or if you have next.bin already
./fg_sizes_cpu --n 100000 --next next.bin
```

Options: `--threads T` (OpenMP), `--mean_only` to print only the mean.
