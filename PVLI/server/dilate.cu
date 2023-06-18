#include "dilate.h"
#include <iostream>

using namespace std;
using namespace glm;

static __global__ void dilateKernel(u8vec3* color, int width, int thickness) {
    extern __shared__ u8vec3 block[];
    int blockSize = blockDim.x + 2;

    //global index
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.x + threadIdx.y;
    int gi = iy * width + ix;

    //center index
    int index = (threadIdx.y + 1) * blockSize + threadIdx.x + 1;

    //surrounding indices
    unsigned I[4] = {
        (threadIdx.y + 1) * blockSize + threadIdx.x + 2,
        (threadIdx.y + 1) * blockSize + threadIdx.x + 0,
        (threadIdx.y + 2) * blockSize + threadIdx.x + 1,
        (threadIdx.y + 0) * blockSize + threadIdx.x + 1
    };

    //copy to shared memory
    block[index] = color[gi];
    if(threadIdx.x == 0) block[index-1] = u8vec3(0);
    else if(threadIdx.x == blockDim.x-1) block[index+1] = u8vec3(0);
    if(threadIdx.y == 0) block[index-blockSize] = u8vec3(0);
    else if(threadIdx.y == blockDim.x-1) block[index+blockSize] = u8vec3(0);
    __syncthreads();

    //dilate
    for(int j = 0; j<thickness; j++) {
        uvec3 sum = uvec3(0);
        unsigned n = 0;

        u8vec3 cur = block[index];
        if (cur == u8vec3(0))
            for (unsigned int i : I) {
                u8vec3 c = block[i];
                if(c != u8vec3(0)){
                    sum += c;
                    n++;
                }
            }

        __syncthreads();
        if(n > 0) block[index] = u8vec3(sum / n);
        __syncthreads();
    }

    //copy back
    color[gi] = block[index];
}

void dilate(u8vec3* cuColor, int width, int height, int blockSize, int thickness) {
    if (!cuColor || width <= 0 || height <= 0 || blockSize <= 0) return;
    if (thickness <= 0) thickness = blockSize * 2;
    dilateKernel <<< dim3(width/blockSize, height/blockSize),
        dim3(blockSize, blockSize),
        (blockSize + 2) * (blockSize + 2) * sizeof(u8vec3)>>>
        (cuColor, width, thickness);
}