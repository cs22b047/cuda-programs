#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void blockScan(float* d_in, float* d_out, float* d_blockSums, int N) {
    __shared__ float temp[1024];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < N) {
        temp[tid] = d_in[idx];
    } else {
        temp[tid] = 0;
    }

    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        float val = 0;
        if (tid >= offset) val = temp[tid - offset];
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    if (idx < N) {
        d_out[idx] = temp[tid];
    }

    if (d_blockSums && tid == blockDim.x - 1) {
        d_blockSums[blockIdx.x] = temp[tid];  // Save last element of block
    }
}

__global__ void addBlockSums(float* d_out, float* d_blockSumsScan, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && idx < N) {
        d_out[idx] += d_blockSumsScan[blockIdx.x - 1];
    }
}

void prefixSumCPU(float* idata, float* odata, int N) {
    odata[0] = idata[0];
    for (int i = 1; i < N; i++) {
        odata[i] = idata[i] + odata[i - 1];
    }
}

int main() {
    int N = 1000000;
    size_t size = N * sizeof(float);

    float* h_in = (float*)malloc(size);
    float* h_out = (float*)malloc(size);
    float* h_outCPU = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_in[i] = 1.0f;
    }

    //---------------CPU execution------------------

    auto start_cpu = std::chrono::high_resolution_clock::now();
    prefixSumCPU(h_in, h_outCPU, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

    //---------------GPU execution------------------

    float *d_in, *d_out, *d_blockSums, *d_blockSumsScan;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);


    int threadsPerBlock = 1024;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t blockSumSize = numBlocks * sizeof(float);

    cudaMalloc(&d_blockSums, blockSumSize);
    cudaMalloc(&d_blockSumsScan, blockSumSize);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // cuda event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    //block-wise scan(assuming numBlocks<1024)
    //works for arrays with size < 1024*1024; for larger arrays, we can repeat this step until numBlocks<1024;
    blockScan<<<numBlocks, threadsPerBlock>>>(d_in, d_out, d_blockSums, N);

    //scan block sums
    int threadsPerBlock2 = 1024;
    blockScan<<<1, threadsPerBlock2>>>(d_blockSums, d_blockSumsScan, nullptr, numBlocks);

    //add scanned block sums to each block result
    addBlockSums<<<numBlocks, threadsPerBlock>>>(d_out, d_blockSumsScan, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    //--------------results---------------
    for (int i = 0; i < N; i++) {
        if (h_out[i] != h_outCPU[i]) {
            std::cout << "Error at index " << i << "\n";
            break;
        }
    }
    std::cout << "CPU Time: " << cpu_time.count() << " ms";
    std::cout << "\nGPU Time: " << gpu_time << " ms\n";

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_blockSums);
    cudaFree(d_blockSumsScan);

    return 0;
}
