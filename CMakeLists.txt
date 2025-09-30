
cmake_minimum_required(VERSION 3.16)
project(fg_sizes_cpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_OPENMP "Enable OpenMP" ON)

add_executable(fg_sizes_cpu
  src/main.cpp
)

target_include_directories(fg_sizes_cpu PRIVATE src)

if (USE_OPENMP)
  find_package(OpenMP)
  if (OpenMP_CXX_FOUND)
    target_link_libraries(fg_sizes_cpu PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(fg_sizes_cpu PRIVATE -D_OPENMP)
  endif()
endif()

if (MSVC)
  target_compile_options(fg_sizes_cpu PRIVATE /O2)
else()
  target_compile_options(fg_sizes_cpu PRIVATE -O3 -march=native -ffast-math)
endif()
