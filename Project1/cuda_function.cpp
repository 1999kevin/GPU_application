
#include "cuda_function.h"

__global__ void test(float* H) {
	*H = 2;
}


void test_wrapper(float* H) {
	float* d_H;
	cudaMalloc((void**)&d_H, 1 * sizeof(float));
	test << <1, 1 >> > (d_H);
	cudaDeviceSynchronize();
	cudaMemcpy(H, d_H, 1 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_H);
	// print("H = ")
}