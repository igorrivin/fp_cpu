rm -rf build
    cmake -S . -B build -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ -DUSE_OPENMP=ON
    cmake --build build -j
    Then execute ./build/fg_sizes_cpu …. This rebuild also ensures libomp is linked; time
  should now show multiple cores when OpenMP is available.
