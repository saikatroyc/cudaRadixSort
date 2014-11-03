/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/
#include "defs.h"

// Define your kernels in this file you may use more than one kernel if you
// need to

// INSERT KERNEL(S) HERE

__global__ void exclusiveScan(unsigned int *out, unsigned int* in, unsigned int*sum, unsigned int inputSize) {
    __shared__ unsigned int temp[2 * BLOCK_SIZE];
    int start = 2 * blockIdx.x * blockDim.x;
    int tx = threadIdx.x;
    int index = 0;
    if (start + tx < inputSize) {
        temp[tx] = in[start + tx];
    } else {
        temp[tx] = 0;
    }
    if (start + tx + blockDim.x < inputSize) {
        temp[tx + blockDim.x] = in[start + tx + blockDim.x];
    } else {
        temp[tx + blockDim.x] = 0;
    }

    __syncthreads();
    // reduction step
    int stride = 1;
    while(stride <= blockDim.x) {
        index = (tx + 1) * 2 * stride - 1;
        if (index < (2 * blockDim.x)) {
              temp[index] += temp[index - stride];
        }
        stride *= 2;
        __syncthreads();
    }
    // first store the reduction sum in sum array
    // make it zero since it is exclusive scan
    if (tx == 0) {
        // sum array contains the prefix sum of each
        // 2*blockDim blocks of element..
        if (sum != NULL) { 
            sum[blockIdx.x] = temp[2*blockDim.x - 1];
        }
        temp[2*blockDim.x - 1] = 0; 
    }
    //wait for thread zero to write
    __syncthreads();
    // post scan step
    stride = blockDim.x;
    index = 0;
    unsigned int var = 0;
    while(stride > 0) {
        index = (2 * stride * (tx + 1)) - 1;
        if (index < 2 * blockDim.x) {
            var = temp[index];
            temp[index] += temp[index - stride];
            temp[index-stride] = var;
        }
        stride >>= 1;
        __syncthreads();
    }

    // now write the temp array to output
    if (start + tx < inputSize) {
        out[start + tx] = temp[tx];   
    }
    if(start + tx + blockDim.x < inputSize) {
        out[start + tx + blockDim.x] = temp[tx + blockDim.x];    
    }
}

__global__ void mergeScanBlocks(unsigned int *sum, unsigned int* output, int opSize) {
    int index = 2 * blockDim.x * blockIdx.x + threadIdx.x;
    if (index < opSize) {
        output[index] += sum[blockIdx.x]; 
    }
    if (index + blockDim.x < opSize) {
        output[index + blockDim.x] += sum[blockIdx.x];    
    } 

}





/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void preScan(unsigned int *out, unsigned int *in, unsigned int in_size)
{
    // INSERT CODE HERE
    cudaError_t ret;
    unsigned int numBlocks1 = in_size / BLOCK_SIZE;
    if (in_size % BLOCK_SIZE) numBlocks1++;
   
    int numBlocks2 = numBlocks1 / 2;
    if(numBlocks1 % 2) numBlocks2++;
    dim3 dimThreadBlock;
    dimThreadBlock.x = BLOCK_SIZE;
    dimThreadBlock.y = 1;
    dimThreadBlock.z = 1;

    dim3 dimGrid;
    dimGrid.x = numBlocks2;
    dimGrid.y = 1;
    dimGrid.z = 1;

    unsigned int*sumArr_d = NULL;
    if (in_size > (2*BLOCK_SIZE)) {
        // we need the sum auxilarry  array only if numblocks2 > 1
        ret = cudaMalloc((void**)&sumArr_d, numBlocks2 * sizeof(unsigned int));
        if (ret != cudaSuccess) FATAL("unable to create sum array");
    }
    exclusiveScan<<<dimGrid, dimThreadBlock>>>(out, in, sumArr_d, in_size);
    ret = cudaDeviceSynchronize();
    if (ret != cudaSuccess) FATAL("unable to launch scan kernel");

    if (in_size <= (2*BLOCK_SIZE)) {
        // out has proper exclusive scan. just return
        return;
    } else {
        // now we need to perform exclusive scan on the auxilliary sum array
        unsigned int *sumArr_scan_d;
        ret = cudaMalloc((void**)&sumArr_scan_d, numBlocks2 * sizeof(unsigned int));
        if (ret != cudaSuccess) FATAL("unable to create sum scan array");
        preScan(sumArr_scan_d, sumArr_d, numBlocks2);
        // sumAdd_scan_d now contains the exclusive scan op of individual blocks
        // now just do a one-one addition of blocks
        mergeScanBlocks<<<dimGrid,dimThreadBlock>>>(sumArr_scan_d, out, in_size);
        ret = cudaDeviceSynchronize();
        if (ret != cudaSuccess) FATAL("unable to launch merge scan kernel");
        cudaFree(sumArr_d); 
        cudaFree(sumArr_scan_d); 
    }

}

