#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <set>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include "ECLgraph.h"

struct Edge{
  int u,v,w,idx;
};
class DJ{
  private:
    int V;
    std::vector<int> parent;
    std::vector<int> rank;
  public:
    DJ(int V){
        this->V = V;
        parent.resize(V);
        rank.resize(V);
        for(int i = 0; i < V; i++){
            parent[i] = i;
            rank[i] = 0;
        }
    }
    int find(int v){
        if(parent[v] == v)
            return v;
        return parent[v] = find(parent[v]);
    }
    bool merge(int u, int v){
        int a = find(u);
        int b = find(v);
        if(a == b)
            return false;
        if(rank[a] > rank[b])
            parent[b] = a;
        else if(rank[a] < rank[b])
            parent[a] = b;
        else{
            parent[a] = b;
            rank[b]++;
        }
        return true;
    }
    int countRoots(){
        int count = 0;
        for(int i = 0; i < V; i++){
            if(parent[i] == i)
                count++;
        }
        return count;
    }
};


long long mst_cpu(std::vector<Edge>& edges, int numNodes) {
    long long mst_weight = 0;
    int trees = numNodes;
    DJ dj(numNodes);
    std::vector<Edge> cheapest(numNodes, {-1, -1, -1});

    while (trees > 1) {
        for (auto edge : edges) {
            int u = edge.u;
            int v = edge.v;
            int w = edge.w;

            int set1 = dj.find(u);
            int set2 = dj.find(v);

            if (set1 != set2) {
                if (cheapest[set1].w == -1 || cheapest[set1].w > w) {
                    cheapest[set1] = edge;
                }
                if (cheapest[set2].w == -1 || cheapest[set2].w > w) {
                    cheapest[set2] = edge;
                }
            }
        }

        for (int i = 0; i < numNodes; i++) {
            if (cheapest[i].w != -1) {
                int u = cheapest[i].u;
                int v = cheapest[i].v;
                int w = cheapest[i].w;
                int set1 = dj.find(u);
                int set2 = dj.find(v);
                if (set1 != set2) {
                    mst_weight += w;
                    dj.merge(set1, set2);
                    trees--;
                }
            }
        }

        for (int i = 0; i < numNodes; i++) {
            cheapest[i].w = -1;
        }
    }

    return mst_weight;
}

__device__ int find(int* parent, int v) {
    while (parent[v] != v) {
        v = parent[v];
    }
    return v;
}

__device__ void join(int arep, int brep, int* parent)
{
  int mrep;
  do {
    mrep = max(arep, brep);
    arep = min(arep, brep);
  } while ((brep = atomicCAS(&parent[mrep], mrep, arep)) != mrep);
}

__global__ void pointer_jump(int* parent, int* numNodes){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx>=*numNodes) return;

  if(parent[idx]==idx || parent[parent[idx]] == parent[idx]) return;
  while(parent[parent[idx]] != parent[idx]){
    parent[idx] = parent[parent[idx]];
  }
  return;
}

__global__ void reduce_graph(Edge* edges, int* parent, int* cheapest_idx, int* in_mst, int* E, int* d_mst_weight) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= *E) return;

    Edge e = edges[tid];
    int uRoot = find(parent,e.u);
    int vRoot = find(parent,e.v);

    if (uRoot == vRoot) return;

    if (cheapest_idx[uRoot] == tid || cheapest_idx[vRoot] == tid) {
        join(uRoot, vRoot, parent);
        in_mst[tid] = 0;
        atomicAdd(d_mst_weight, e.w);
    }
}

__global__ void find_cheapest_edges(Edge* d_edges, int* d_cheapest, int* d_cheapest_idx, int* d_total_edges, int *d_mst_weight, int* in_mst, int* parent, int* d_changed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *d_total_edges) return;
    if (in_mst[idx] == 0) return;
    int u = d_edges[idx].u;
    int v = d_edges[idx].v;
    int w = d_edges[idx].w;



    // use pointer jumping here
    int u_root = find(parent, u);
    int v_root = find(parent, v);

    if(v_root == u_root) return;


    if(d_cheapest[u_root] == -1 || d_edges[d_cheapest[u_root]].w > w || d_cheapest[v_root] == -1 || d_edges[d_cheapest[v_root]].w > w){
        *d_changed = 1;
    }

    int old_idx, assumed_idx;
    int new_idx = idx;
    do {
        old_idx = d_cheapest[u_root];
        assumed_idx = old_idx;

        // If no edge assigned yet OR new edge has lower weight
        if (old_idx == -1 || d_edges[old_idx].w > w || (d_edges[old_idx].w == w && old_idx > idx)) {
            old_idx = atomicCAS(&d_cheapest[u_root], assumed_idx, new_idx);
        } else {
            break;  // Existing edge is better
        }
    } while (old_idx != assumed_idx);


    new_idx = idx;

    do {
        old_idx = d_cheapest[v_root];
        assumed_idx = old_idx;
        // If no edge assigned yet OR new edge has lower weight
        if (old_idx == -1 || d_edges[old_idx].w > w || (d_edges[old_idx].w == w && old_idx > idx)) {
            old_idx = atomicCAS(&d_cheapest[v_root], assumed_idx, new_idx);
        } else {
            break;  // Existing edge is better
        }
    } while (old_idx != assumed_idx);

}

long long mst_gpu(Edge* edges, int numNodes, int totalEdges){

  Edge* d_edges;
  int* d_in_mst;
  int *d_cheapest, *d_cheapest_idx, *d_parent, *d_changed, *d_mst_weight, *d_total_edges, *d_num_nodes;

  cudaMalloc(&d_total_edges, sizeof(int));
  cudaMemcpy(d_total_edges, &totalEdges, sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc(&d_edges, sizeof(Edge) * totalEdges);
  cudaMalloc(&d_in_mst, sizeof(int) * totalEdges);
  cudaMalloc(&d_cheapest, sizeof(int) * numNodes);
  cudaMalloc(&d_cheapest_idx, sizeof(int)* numNodes);
  cudaMalloc(&d_parent, sizeof(int) * numNodes);
  cudaMalloc(&d_changed, sizeof(int));
  cudaMalloc(&d_mst_weight, sizeof(int));
  cudaMalloc(&d_num_nodes, sizeof(int));
  cudaMemset(d_num_nodes,numNodes,sizeof(int));



  cudaMemcpy(d_edges, edges, totalEdges * sizeof(Edge), cudaMemcpyHostToDevice);

  int* h_parent = new int[numNodes];
  for (int i = 0; i < numNodes; i++) h_parent[i] = i;
  cudaMemcpy(d_parent, h_parent, numNodes * sizeof(int), cudaMemcpyHostToDevice);



  cudaMemset(d_in_mst,1,sizeof(int)*totalEdges);


  int h_changed = 1;
  long long h_mst_weight_total = 0;
  int threads_per_block = 1024;
  int num_blocks = (totalEdges + threads_per_block - 1) / threads_per_block;
  int num_blocks_nodes = (numNodes + threads_per_block - 1) / threads_per_block;

  int h_mst_weight = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  do {
      cudaMemset(d_cheapest, -1, sizeof(int) * numNodes);
      h_mst_weight = 0;

      h_changed = 0;
      cudaMemset(d_mst_weight, 0, sizeof(int));
      cudaMemset(d_changed, 0, sizeof(int));

      pointer_jump<<<num_blocks_nodes, threads_per_block>>>(d_parent, d_num_nodes);
      cudaDeviceSynchronize();


      find_cheapest_edges<<<num_blocks, threads_per_block>>>(d_edges, d_cheapest, d_cheapest_idx, d_total_edges, d_mst_weight, d_in_mst, d_parent,d_changed);
      cudaDeviceSynchronize();

      reduce_graph<<<num_blocks, threads_per_block>>>(d_edges, d_parent, d_cheapest, d_in_mst, d_total_edges, d_mst_weight);
      cudaDeviceSynchronize();


      cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&h_mst_weight, d_mst_weight, sizeof(int), cudaMemcpyDeviceToHost);
      h_mst_weight_total += h_mst_weight;

  } while(h_changed != 0);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float gpu_time = 0;
  cudaEventElapsedTime(&gpu_time, start, stop);

  std::cout << "GPU Time: " << gpu_time << " ms" << std::endl;

  cudaFree(d_changed);
  cudaFree(d_mst_weight);
  cudaFree(d_total_edges);
  cudaFree(d_parent);
  cudaFree(d_cheapest_idx);
  cudaFree(d_cheapest);
  cudaFree(d_in_mst);
  cudaFree(d_edges);

  delete[] h_parent;

  return h_mst_weight_total;
}

int main(int argc, char** argv) {

    if (argc < 2) {
        return 1;
    }

    int numNodes = 0;
    int totalEdges = 0;
    const char* filename = argv[1];
    ECLgraph g = readECLgraph(filename);

    std::cout << "Graph loaded: " << g.nodes << " nodes, " << g.edges << " edges\n";
    numNodes = g.nodes;
    totalEdges = g.edges;

    Edge* edges = new Edge[totalEdges];

    // Print or save edge list
    for (int u = 0; u < g.nodes; u++) {
        for (int i = g.nindex[u]; i < g.nindex[u + 1]; i++) {
            int v = g.nlist[i];
            int w = g.eweight ? g.eweight[i] : 1; // default weight = 1
            edges[i] = {u, v, w, i};
        }
    }
    freeECLgraph(g);

    std::cout << "totalEdges: " << totalEdges << "\n";
    std::cout << "numNodes: " << numNodes << "\n";

    // CPU Execution
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Edge> edgeVec(edges, edges + totalEdges); // convert to vector temporarily
    long long weight = mst_cpu(edgeVec, numNodes);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;


    // Output
    std::cout << "CPU Weight: " << weight << std::endl;
    std::cout << "CPU Time: " << elapsed.count() * 100 << " ms" << std::endl;

    long long gpu_weight = mst_gpu(edges,numNodes,totalEdges);
    std::cout << "GPU Weight: " << gpu_weight << std::endl;
    delete[] edges;

    return 0;
}
