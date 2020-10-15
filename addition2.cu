#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
// #include <time.h>
using namespace std;

#define length 320
#define tile_width 10


#define TOTALN 64*32
#define THREADS_PerBlock 64
#define BLOCKS_PerGrid 32
#define EPOCH 10

int data[TOTALN] = {0};

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

float avg(float *data, int len){
    float result = 0;
    for (int i = 0; i<len; i++){
        result += data[i];
    }
    result /= len;
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

__global__ void gpu_sum(int *c, int *a){
    __shared__ unsigned int mycache[THREADS_PerBlock];
    int i = threadIdx.x+blockIdx.x*blockDim.x;
    int j = gridDim.x*blockDim.x;
    int cacheN;
    unsigned sum;

    sum=0;
    cacheN=threadIdx.x; 
    while(i<TOTALN) {
        sum += a[i];
        i = i+j;
    }

    mycache[cacheN]=sum;
    __syncthreads();

    c[blockIdx.x] = 0;
    atomicAdd(&c[blockIdx.x],mycache[threadIdx.x]);
    __syncthreads();



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


__global__ void SumArray2(int *c, int *a) {
    __shared__ unsigned int mycache[THREADS_PerBlock];


    int numberPerThread = TOTALN/THREADS_PerBlock/BLOCKS_PerGrid;
    int i = threadIdx.x+blockIdx.x*blockDim.x;

    int cacheN;
    unsigned sum,k;

    sum=0;
    cacheN=threadIdx.x; 

    int start_pos = i*numberPerThread;
    for (int count=0; count < numberPerThread; count++){
        if(start_pos + count < TOTALN){
            sum+=a[start_pos+count];
        }
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

    int count;
    cudaGetDeviceCount(&count);
    printf("We have %d devices\n", count);

    cudaSetDevice(1);



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

    float cpu_time[EPOCH];
    for (int t = 0; t < EPOCH; t++){
        cudaEventRecord(start, 0);
        cpu_result = cpu_sum(data,TOTALN);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        // float cpu_time;
        cudaEventElapsedTime(&cpu_time[t], start, stop);
        // printf("cpu result: %d, time:%f(ms)\n", cpu_result, cpu_time[t]);
    }
    printf("cpu average time: %f\n", avg(cpu_time, EPOCH));
    

    /* for gpu */

    int h_result = 0;
    int *d_data;
    // int *d_result;
    int c[BLOCKS_PerGrid];
    int *d_c;

    float gpu_time[EPOCH];

    HANDLE_ERROR( cudaMalloc((void **)&d_data, TOTALN*sizeof(int)));
    // HANDLE_ERROR( cudaMalloc((void **)&d_result, 1*sizeof(int)) );
    HANDLE_ERROR( cudaMemcpy(d_data, data, TOTALN*sizeof(int), cudaMemcpyHostToDevice));
    // HANDLE_ERROR( cudaMemcpy(d_result, &h_result, 1*sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR( cudaMalloc((void**)&d_c, BLOCKS_PerGrid * sizeof(int)));
    for (int t = 0; t < EPOCH; t++){
        /* Launch the GPU kernel */
        cudaEventRecord(start, 0);
        gpu_sum<<< BLOCKS_PerGrid, THREADS_PerBlock >>>(d_c,d_data);
        // SumArray<<< BLOCKS_PerGrid, THREADS_PerBlock >>>(d_c,d_data);
    
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        // float gpu_time;
        cudaEventElapsedTime(&gpu_time[t], start, stop);
    
        // cudaMemcpy(&h_result, d_result, 1*sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, BLOCKS_PerGrid * sizeof(int), cudaMemcpyDeviceToHost);
        h_result = cpu_sum(c, BLOCKS_PerGrid);
    
    

        // cudaFree(d_result);
        // printf("gpu result: %d, time:%f(ms)\n", h_result, gpu_time[t]);
    }

    cudaFree(d_data);
    cudaFree(d_c);
    printf("gpu1 average time: %f\n", avg(gpu_time, EPOCH));
    



    /* algorithm 2 */

    int h_result2 = 0;
    int *d_data2;
    // int *d_result;
    int c2[BLOCKS_PerGrid];
    int *d_c2;

    float gpu_time2[EPOCH];

    HANDLE_ERROR( cudaMalloc((void **)&d_data2, TOTALN*sizeof(int)));
    // HANDLE_ERROR( cudaMalloc((void **)&d_result, 1*sizeof(int)) );
    HANDLE_ERROR( cudaMemcpy(d_data2, data, TOTALN*sizeof(int), cudaMemcpyHostToDevice));
    // HANDLE_ERROR( cudaMemcpy(d_result, &h_result, 1*sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR( cudaMalloc((void**)&d_c2, BLOCKS_PerGrid * sizeof(int)));

    for(int t=0; t<EPOCH; t++){

        cudaEventRecord(start, 0);
        // gpu_sum<<< BLOCKS_PerGrid, THREADS_PerBlock >>>(d_c,d_data);
        SumArray<<< BLOCKS_PerGrid, THREADS_PerBlock >>>(d_c2,d_data2);
    
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        // float gpu_time2;
        cudaEventElapsedTime(&gpu_time2[t], start, stop);
    
        // cudaMemcpy(&h_result, d_result, 1*sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cudaMemcpy(c2, d_c2, BLOCKS_PerGrid * sizeof(int), cudaMemcpyDeviceToHost);
        h_result2 = cpu_sum(c2, BLOCKS_PerGrid);
        // printf("gpu2 result: %d, time:%f(ms)\n", h_result2, gpu_time2[t]);
    
    }
    printf("gpu2 average time: %f\n", avg(gpu_time2, EPOCH));

    cudaFree(d_data2);
    cudaFree(d_c2);
    // cudaFree(d_result);



/* gpu algorithm 3 */

int h_result3 = 0;
int *d_data3;
// int *d_result;
int c3[BLOCKS_PerGrid];
int *d_c3;

float gpu_time3[EPOCH];

HANDLE_ERROR( cudaMalloc((void **)&d_data3, TOTALN*sizeof(int)));
// HANDLE_ERROR( cudaMalloc((void **)&d_result, 1*sizeof(int)) );
HANDLE_ERROR( cudaMemcpy(d_data3, data, TOTALN*sizeof(int), cudaMemcpyHostToDevice));
// HANDLE_ERROR( cudaMemcpy(d_result, &h_result, 1*sizeof(int), cudaMemcpyHostToDevice));
HANDLE_ERROR( cudaMalloc((void**)&d_c3, BLOCKS_PerGrid * sizeof(int)));
for (int t = 0; t < EPOCH; t++){
    /* Launch the GPU kernel */
    cudaEventRecord(start, 0);
    SumArray2<<< BLOCKS_PerGrid, THREADS_PerBlock >>>(d_c3,d_data3);
    // SumArray<<< BLOCKS_PerGrid, THREADS_PerBlock >>>(d_c,d_data);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    // float gpu_time;
    cudaEventElapsedTime(&gpu_time3[t], start, stop);

    // cudaMemcpy(&h_result, d_result, 1*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaMemcpy(c3, d_c3, BLOCKS_PerGrid * sizeof(int), cudaMemcpyDeviceToHost);
    h_result3 = cpu_sum(c3, BLOCKS_PerGrid);



    // cudaFree(d_result);
    // printf("gpu3 result: %d, time:%f(ms)\n", h_result3, gpu_time3[t]);
}

cudaFree(d_data3);
cudaFree(d_c3);
printf("gpu3 average time: %f\n", avg(gpu_time3, EPOCH));




cudaEventDestroy(start);
cudaEventDestroy(stop);
// printf("gpu result: %d, time:%f(ms)\n", h_result2, gpu_time2);
return 0;


}