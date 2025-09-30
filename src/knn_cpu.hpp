
#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif

// CPU brute-force kNN to produce successor next[i] = k-th nearest neighbor of i (excluding self).
// Complexity O(n^2 d), OpenMP-parallel over i.
inline std::vector<int32_t> kth_neighbor_successor_bruteforce(
    const float* X, int n, int d, int k)
{
    std::vector<int32_t> next(n, 0);
    auto less_pair = [](const std::pair<float,int>& a, const std::pair<float,int>& b) {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    };
    if (k == 2 && d == 1) {
        std::vector<int32_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            if (X[a] != X[b]) return X[a] < X[b];
            return a < b;
        });

        const std::pair<float,int> sentinel{
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<int>::max()
        };

        #pragma omp parallel for schedule(static)
        for (int pos=0; pos<n; ++pos) {
            const int idx = order[pos];
            const float xi = X[idx];
            int left = pos - 1;
            int right = pos + 1;
            std::pair<float,int> best1 = sentinel;
            std::pair<float,int> best2 = sentinel;
            auto push = [&](const std::pair<float,int>& cand) {
                if (cand.second < 0) return;
                if (less_pair(cand, best1)) {
                    best2 = best1;
                    best1 = cand;
                } else if (less_pair(cand, best2)) {
                    best2 = cand;
                }
            };
            while (best2.second == sentinel.second && (left >= 0 || right < n)) {
                if (left >= 0) {
                    int id = order[left];
                    float dx = xi - X[id];
                    push({dx*dx, id});
                    --left;
                }
                if (best2.second != sentinel.second) break;
                if (right < n) {
                    int id = order[right];
                    float dx = xi - X[id];
                    push({dx*dx, id});
                    ++right;
                }
            }
            if (best2.second == sentinel.second) best2 = best1;
            next[idx] = best2.second;
        }
        return next;
    }

    if (k == 2) {
        const float INF = std::numeric_limits<float>::infinity();
        const int INF_IDX = std::numeric_limits<int>::max();
        auto better = [](float dist, int idx, float best_dist, int best_idx) {
            if (dist < best_dist) return true;
            if (dist > best_dist) return false;
            return idx < best_idx;
        };

        #pragma omp parallel for schedule(static)
        for (int i=0; i<n; ++i) {
            float best1_dist = INF;
            float best2_dist = INF;
            int best1_idx = INF_IDX;
            int best2_idx = INF_IDX;
            const float* xi = X + (size_t)i*d;
            for (int j=0; j<n; ++j) {
                if (j == i) continue;
                const float* xj = X + (size_t)j*d;
                float dist2 = 0.f;
                for (int t=0; t<d; ++t) {
                    float dx = xi[t] - xj[t];
                    dist2 += dx*dx;
                }
                if (better(dist2, j, best1_dist, best1_idx)) {
                    best2_dist = best1_dist;
                    best2_idx = best1_idx;
                    best1_dist = dist2;
                    best1_idx = j;
                } else if (better(dist2, j, best2_dist, best2_idx)) {
                    best2_dist = dist2;
                    best2_idx = j;
                }
            }
            if (best2_idx == INF_IDX) best2_idx = best1_idx;
            next[i] = best2_idx;
        }
        return next;
    }

    #pragma omp parallel for schedule(static)
    for (int i=0; i<n; ++i) {
        std::vector<std::pair<float,int>> ds;
        ds.reserve(n-1);
        const float* xi = X + (size_t)i*d;
        for (int j=0; j<n; ++j) {
            if (j == i) continue;
            const float* xj = X + (size_t)j*d;
            float dist2 = 0.f;
            for (int t=0; t<d; ++t) {
                float dx = xi[t] - xj[t];
                dist2 += dx*dx;
            }
            ds.emplace_back(dist2, j);
        }
        std::nth_element(ds.begin(), ds.begin() + (k-1), ds.end(), less_pair);
        next[i] = ds[k-1].second;
    }
    return next;
}
