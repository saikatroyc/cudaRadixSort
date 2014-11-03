#include "defs.h"
#include "kernel_prescan.cu"
__global__ void splitGPU(unsigned int*in_d, unsigned int *out_d, unsigned int in_size, int bit_shift) {
    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
    int bit = 0;
    if (index < in_size) {
        bit = in_d[index] & (1 << bit_shift);
        bit = (bit > 0) ? 1 : 0;
        out_d[index] = 1 - bit;
    }

}
__global__ void indexDefine(unsigned int *in_d, unsigned int *rev_bit_d, unsigned int in_size, unsigned int last_input) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    int total_falses = in_d[in_size - 1] + last_input;
    __syncthreads();
    if (index < in_size) {
        if (rev_bit_d[index] == 0) {
            int val = in_d[index];
            in_d[index] = index + 1 - val + total_falses;
        }
    }

}

__global__ void scatterElements(unsigned int *in_d, unsigned int *index_d, unsigned int *out_d, unsigned int in_size) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < in_size) {
        unsigned int val = index_d[index];
        if (val < in_size) {
            out_d[val] = in_d[index];
        }
    }

}
void radix_sort(unsigned int *in_d, unsigned int *out_d, unsigned int *out_scan_d, unsigned int *in_h,unsigned int *out_scan_h, int num_elements) {
    cudaError_t ret;
    unsigned int *temp;
    dim3 dimThreadBlock;
    dimThreadBlock.x = BLOCK_SIZE;
    dimThreadBlock.y = 1;
    dimThreadBlock.z = 1;

    dim3 dimGrid;
    dimGrid.x =(int)(ceil(num_elements/(1.0 * dimThreadBlock.x)));
    dimGrid.y = 1;
    dimGrid.z = 1; 
    
    for (int i =0;i<32;i++) {
        splitGPU<<<dimGrid, dimThreadBlock>>>(in_d,out_d,num_elements,i);
        ret = cudaDeviceSynchronize();
        if(ret != cudaSuccess) FATAL("Unable to launch kernel:splitGPU");

        preScan(out_scan_d, out_d, num_elements);
        ret = cudaDeviceSynchronize();
        if(ret != cudaSuccess) FATAL("Unable to launch kernel");
        #ifdef TEST_MODE 
        cudaMemcpy(out_scan_h, out_scan_d, num_elements * sizeof(unsigned int),
        cudaMemcpyDeviceToHost);
        ret = cudaDeviceSynchronize();
        if(ret != cudaSuccess) FATAL("Unable to copy memory fromthe device");
        printf("after exclusive scan:\n");
        for (int i = 0; i< num_elements;i++) {
            printf("%u,",out_scan_h[i]);    
        }
        printf("\n");
        #endif

        indexDefine<<<dimGrid, dimThreadBlock>>>(out_scan_d, out_d, num_elements, in_h[num_elements - 1]);
        ret = cudaDeviceSynchronize();
        if(ret != cudaSuccess) FATAL("Unable to launch kernel");
        #ifdef TEST_MODE 
        cudaMemcpy(out_scan_h, out_scan_d, num_elements * sizeof(unsigned int),
        cudaMemcpyDeviceToHost);
        ret = cudaDeviceSynchronize();
        if(ret != cudaSuccess) FATAL("Unable to copy memory fromthe device");
        printf("after index define:\n");
        for (int i = 0; i< num_elements;i++) {
            printf("%u,",out_scan_h[i]);    
        }
        printf("\n");
        #endif

        scatterElements<<<dimGrid, dimThreadBlock>>>(in_d, out_scan_d, out_d, num_elements);
        ret = cudaDeviceSynchronize();
        if(ret != cudaSuccess) FATAL("Unable to launch kernel");

        // swap pointers
        temp = in_d;
        in_d = out_d;
        out_d = temp;
    }
}
