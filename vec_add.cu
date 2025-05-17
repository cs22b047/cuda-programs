#include <iostream>
#include <cuda_runtime.h>
#include <chrono>


__global__ void vecAdd(float* A, float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

void vecAddCPU(float* A, float* B, float* C, int N){
  for(int i=0;i<N;i++){
    C[i] = A[i] + B[i];
  }
}

int main(){
    int N = 10000000;
    size_t size = N * sizeof(float);

    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    float* h_C_CPU = (float*)malloc(size);


    for(int i=0;i<N;i++){
      h_A[i] = i;
      h_B[i] = i;
    }

    //------------CPU Execution--------------

    auto start_cpu = std::chrono::high_resolution_clock::now();
    vecAddCPU(h_A, h_B, h_C_CPU, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;


    //--------------GPU Execution------------

    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);

    //cuda event intialization
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);


    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    //-------------Results------------
    for(int i=0;i<N;i++){
      if(h_C[i]!=h_C_CPU[i]) std::cout<<"incorrect"<<std::endl;
    }
    std::cout << "CPU time: " << cpu_time.count() << " ms\n";
    std::cout << "GPU time: " << milliseconds << " ms\n";
    return 0;
}
