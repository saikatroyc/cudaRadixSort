#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "support.h"
#include "defs.h"
#include "kernel_radix.cu"

int compare(const void *a, const void *b) {
    int a1 = *((unsigned int*)a);
    int b1 = *((unsigned int*)b);
    if (a1 == b1) return 0;
    else if (a1 < b1) return -1; 
    else return 1;
}


int main(int argc, char* argv[])
{
    Timer timer;

    unsigned int *in_h;
    unsigned int *out_h;
    unsigned int *out_d;
    unsigned int *in_d;
    unsigned int *out_scan_d;
    unsigned int num_elements;
    cudaError_t cuda_ret;

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);
    if(argc == 1) {
        num_elements = 1000000;
    } else if(argc == 2) {
        num_elements = atoi(argv[1]);
    }
    in_h = (unsigned int*) malloc(num_elements*sizeof(unsigned int));
    out_h = (unsigned int*) malloc(num_elements*sizeof(unsigned int));
    //only for test
    unsigned int *out_scan_h = (unsigned int *)malloc(num_elements * sizeof(unsigned int));
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    
    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMalloc((void**)&in_d, num_elements * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&out_d, num_elements * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&out_scan_d, num_elements * sizeof(unsigned int ));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate scan memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    //init array
    for(int i = 0;i < num_elements;i++) {
        in_h[i] = num_elements - 1 - i;
        #ifdef TEST_MODE
        printf("%u,", in_h[i]);
        #endif
    }
    // Copy host variables to device ------------------------------------------

    printf("\nCopying data from host to device..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(in_d, in_h, num_elements * sizeof(unsigned int),
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);

    radix_sort(in_d, out_d, out_scan_d, in_h, out_scan_h, num_elements);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

    stopTime(&timer); printf("GPU Sort time: %f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(out_h, out_d, num_elements * sizeof(unsigned int),
        cudaMemcpyDeviceToHost);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    #ifdef TEST_MODE
    for (int i = 0; i< num_elements;i++) {
        printf("%u,",out_h[i]);    
    }
    #endif

    printf("\nCPU sort"); fflush(stdout);
    startTime(&timer);
    qsort(in_h, num_elements, sizeof(unsigned int), compare);
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    // Verify correctness -----------------------------------------------------
    int flag = 0;
    for (int i = 0;i < num_elements;i++) {
        if (in_h[i] != out_h[i]) {
            flag = 1;
            break; 
        }
    }
    if (flag == 1) {
        printf("test failed\n");
    } else
        printf("test passed\n");
    // Free memory ------------------------------------------------------------
    cudaFree(in_d);
    cudaFree(out_scan_d);
    cudaFree(out_d);
    free(in_h); 
    free(out_h);

    return 0;
}

