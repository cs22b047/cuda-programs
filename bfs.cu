#include <iostream>
#include <vector>
#include <queue>
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <unordered_set>
#include <algorithm>
#include "ECLgraph.h"


void generate_graph(int n, int m, std::vector<int>& nodes, std::vector<int>& adj_nodes) {
    std::vector<std::unordered_set<int>> adjacency(n);

    while (m > 0) {
        int u = rand()%n;
        int v = rand()%n;
        if (u != v && adjacency[u].insert(v).second) {
            --m;
        }
    }

    nodes.resize(n + 1);
    int total_edges = 0;
    for (int i = 0; i < n; ++i) {
        nodes[i] = total_edges;
        total_edges += adjacency[i].size();
        adj_nodes.insert(adj_nodes.end(), adjacency[i].begin(), adjacency[i].end());
    }
    nodes[n] = total_edges;
}

void bfs_cpu(int* nodes, int* adj_nodes, int num_nodes, int source, std::vector<int>& distance) {
    std::queue<int> q;
    std::vector<bool> visited(num_nodes, false);

    q.push(source);
    visited[source] = true;
    distance[source] = 0;

    while (!q.empty()) {
        int node = q.front(); q.pop();
        for (int i = nodes[node]; i < nodes[node + 1]; ++i) {
            int neighbor = adj_nodes[i];
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                distance[neighbor] = distance[node] + 1;
                q.push(neighbor);
            }
        }
    }
}
__global__ void bfs_kernel(int* nodes, int* adj_nodes, int* frontier, int* next_frontier, int* visited, int* distance, int* frontier_size, int level, int num_nodes) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= *frontier_size) return;

    int node = frontier[idx];
    for (int i = nodes[node]; i < nodes[node + 1]; ++i) {
        int neighbor = adj_nodes[i];
        if (!visited[neighbor]) {
            visited[neighbor] = 1;
            distance[neighbor] = level + 1;
            int pos = atomicAdd(frontier_size + 1, 1);
            next_frontier[pos] = neighbor;
        }
    }
}

void bfs_gpu(int* h_nodes, int* h_adj_nodes, int num_nodes, int num_edges, int source, std::vector<int>& h_distance) {
    int *d_nodes, *d_adj_nodes, *d_frontier, *d_next_frontier, *d_visited, *d_distance, *d_frontier_size;

    cudaMalloc(&d_nodes, (num_nodes + 1)*sizeof(int));
    cudaMalloc(&d_adj_nodes, num_edges*sizeof(int));
    cudaMalloc(&d_frontier, num_nodes*sizeof(int));
    cudaMalloc(&d_next_frontier, num_nodes*sizeof(int));
    cudaMalloc(&d_visited, num_nodes*sizeof(int));
    cudaMalloc(&d_distance, num_nodes*sizeof(int));
    cudaMalloc(&d_frontier_size, 2*sizeof(int)); // [curr, next]

    cudaMemcpy(d_nodes, h_nodes, (num_nodes + 1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adj_nodes, h_adj_nodes, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_visited, 0, num_nodes*sizeof(int));
    cudaMemset(d_distance, -1, num_nodes*sizeof(int));

    int h_frontier_size[2] = {1, 0};
    cudaMemcpy(d_frontier_size, h_frontier_size, 2*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_frontier[0], &source, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(&d_visited[source], 1, sizeof(int));
    cudaMemcpy(&d_distance[source], &h_frontier_size[1], sizeof(int), cudaMemcpyHostToDevice);

    int level = 0;
    while (h_frontier_size[0] > 0) {
        int threads_per_block = 1024;
        int num_blocks = (h_frontier_size[0] + threads_per_block - 1) / threads_per_block;

        bfs_kernel<<<num_blocks, threads_per_block>>>(d_nodes, d_adj_nodes, d_frontier, d_next_frontier,
                                              d_visited, d_distance, d_frontier_size, level, num_nodes);

        cudaMemcpy(h_frontier_size, d_frontier_size, 2*sizeof(int), cudaMemcpyDeviceToHost);
        std::swap(d_frontier, d_next_frontier);
        h_frontier_size[0] = h_frontier_size[1];
        h_frontier_size[1] = 0;
        cudaMemcpy(d_frontier_size, h_frontier_size, 2*sizeof(int), cudaMemcpyHostToDevice);
        level++;
    }

    cudaMemcpy(h_distance.data(), d_distance, num_nodes*sizeof(int), cudaMemcpyDeviceToHost);
}

int main(int argc, char* argv[]) {
    
    if (argc < 2) {
        return 1;
    }

    const char* filename = argv[1];

    ECLgraph g = readECLgraph(filename);

    int num_nodes = g.nodes;
    int num_edges = g.edges;
    int source = 0;

    std::vector<int> nodes(g.nindex, g.nindex + num_nodes + 1);
    std::vector<int> adj_nodes(g.nlist, g.nlist + num_edges);

    std::vector<int> cpu_distance(num_nodes, -1);
    std::vector<int> gpu_distance(num_nodes, -1);

    //-----------------CPU-----------------
    auto start_cpu = std::chrono::high_resolution_clock::now();

    bfs_cpu(nodes.data(), adj_nodes.data(), num_nodes, source, cpu_distance);

    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;

    //-----------------GPU-----------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    bfs_gpu(nodes.data(), adj_nodes.data(), num_nodes, adj_nodes.size(), source, gpu_distance);

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    //-----------------Result-----------------
    std::cout << "CPU Time: " << cpu_duration.count()*1000 << " ms\n";
    std::cout << "GPU Time: " << milliseconds << " ms\n";

    bool match = true;
    for (int i = 0; i < num_nodes; ++i) {
        if (cpu_distance[i] != gpu_distance[i]) {
            std::cout << "Mismatch at node " << i << ": CPU=" << cpu_distance[i]
                      << ", GPU=" << gpu_distance[i] << "\n";
            match = false;
            break;
        }
    }
    if(match) std::cout << "Distances match\n";
    else std::cout << "Distances do not match.\n";
    return 0;
}
