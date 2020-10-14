#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
using namespace std;

#define length 320
#define tile_width 10


#define TOTALN 64*32*1000
#define THREADS_PerBlock 64
#define BLOCKS_PerGrid 32


static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}


#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


int cpu_sum(int *data, int len){
    int result = 0;
    for (int i = 0; i<len; i++){
        result += data[i];
    }
    return result;
}


// __global__ void gpu_sum(int *d_data, int * result, int len){
//     // int thread_num = ceil(length/tile_width);
//     __shared__ int sub_result[100];

//     int tx = threadIdx.x;
    
//     sub_result[tx] = 0;
//     __syncthreads();
//     for (int i = 0; i < tile_width; i++){
//         sub_result[tx] += d_data[tx*tile_width+i];
//     }
//     __syncthreads();

//     if(tx == 0){
//         for (int i = 0; i < 100; i++){
//             *result += sub_result[i];
//         }
//     }

// }

__global__ void gpu_sum(int *d_data, int * result, int len){
    // int thread_num = ceil(length/tile_width);
    int sub_result = 0;

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    
    // sub_result[tx] = 0;
    // __syncthreads();
    for (int i = 0; i < tile_width; i++){
        atomicAdd(&sub_result, d_data[bx*8*tile_width+tx*tile_width+i]);
    }
    __syncthreads();

    atomicAdd(result, sub_result);

}


__global__ void SumArray(int *c, int *a) {
    __shared__ unsigned int mycache[THREADS_PerBlock];
    int i = threadIdx.x+blockIdx.x*blockDim.x;
    int j = gridDim.x*blockDim.x;
    int cacheN;
    unsigned sum,k;

    sum=0;
    cacheN=threadIdx.x; 
    while(i<TOTALN) {
        sum += a[i];
        i = i+j;
    }

    mycache[cacheN]=sum;

    __syncthreads();

    k=THREADS_PerBlock>>1;
    while(k) {
        if(cacheN<k) {
            mycache[cacheN] += mycache[cacheN+k];
        }
        __syncthreads();
        k=k>>1;
    }
    if(cacheN==0) {
        c[blockIdx.x]=mycache[0];
    }
}

int main()
{
    int data[TOTALN] = {0};
    int cpu_result;
    int i;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* get data */
    for (i = 0; i < TOTALN; i++)
    {
        data[i] = rand()%100;
        // printf("%d",a);
    }
    // printf(" %d %d %d\n",data[0],data[1],data[2]);

    cudaEventRecord(start, 0);
    cpu_result = cpu_sum(data,TOTALN);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    float cpu_time;
    cudaEventElapsedTime(&cpu_time, start, stop);
    printf("cpu result: %d, time:%f(ms)\n", cpu_result, cpu_time);
    

    /* for gpu */

    int h_result = 0;
    int *d_data;
    // int *d_result;
    int c[BLOCKS_PerGrid];
    int *d_c;

    HANDLE_ERROR( cudaMalloc((void **)&d_data, TOTALN*sizeof(int)));
    // HANDLE_ERROR( cudaMalloc((void **)&d_result, 1*sizeof(int)) );
    HANDLE_ERROR( cudaMemcpy(d_data, data, TOTALN*sizeof(int), cudaMemcpyHostToDevice));
    // HANDLE_ERROR( cudaMemcpy(d_result, &h_result, 1*sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR( cudaMalloc((void**)&d_c, BLOCKS_PerGrid * sizeof(int)));


    /* Launch the GPU kernel */
    cudaEventRecord(start, 0);
    // gpu_sum<<< BLOCKS_PerGrid, THREADS_PerBlock >>>(d_data,d_result,length);
    SumArray<<< BLOCKS_PerGrid, THREADS_PerBlock >>>(d_c,d_data);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // cudaMemcpy(&h_result, d_result, 1*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, BLOCKS_PerGrid * sizeof(int), cudaMemcpyDeviceToHost);
    h_result = cpu_sum(c, BLOCKS_PerGrid);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_c);


    printf("gpu result: %d, time:%f(ms)\n", h_result, gpu_time);
    return 0;

}