
#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>

// Compute reachable-set sizes for an outdegree-1 graph given by next[i].
// Returns vector<int32_t> sizes, where sizes[i] = |{nodes reachable from i, including i}|.
// Time O(n), space O(n).
inline std::vector<int32_t> reachable_sizes_outdeg1(const std::vector<int32_t>& next) {
    const int n = static_cast<int>(next.size());
    const int8_t UNVIS=0, VIS=1, DONE=2;
    std::vector<int8_t> state(n, UNVIS);
    std::vector<int32_t> size(n, 0);
    std::vector<int32_t> stack; stack.reserve(n);
    std::vector<int32_t> pos(n, -1);

    for (int s=0; s<n; ++s) {
        if (state[s] != UNVIS) continue;
        int u = s;
        stack.clear();
        while (true) {
            if (state[u] == UNVIS) {
                state[u] = VIS;
                pos[u] = static_cast<int>(stack.size());
                stack.push_back(u);
                u = next[u];
                continue;
            }
            if (state[u] == VIS) {
                int start = pos[u];
                int cyc_len = static_cast<int>(stack.size()) - start;
                for (int t=start; t<(int)stack.size(); ++t) {
                    int v = stack[t];
                    size[v] = cyc_len;
                    state[v] = DONE;
                }
                for (int t=start-1; t>=0; --t) {
                    int v = stack[t];
                    size[v] = size[next[v]] + 1;
                    state[v] = DONE;
                }
                break;
            } else { // DONE
                for (int t=(int)stack.size()-1; t>=0; --t) {
                    int v = stack[t];
                    size[v] = size[next[v]] + 1;
                    state[v] = DONE;
                }
                break;
            }
        }
    }
    return size;
}
