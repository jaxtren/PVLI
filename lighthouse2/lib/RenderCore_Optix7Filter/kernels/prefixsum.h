/**
 * compute prefix sum in shared memory
 * @param data __shared__ data of size threadDim.x * 2
 * @param sum sum of all elements
 */
static __device__ void prefixSumSharedMem(unsigned int* data, unsigned int& sum) {
    int n = blockDim.x * 2;
    int id = threadIdx.x;
    int offset = 1;
    int id1 = id * 2 + 1;
    int id2 = id * 2 + 2;

    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (id < d)
            data[offset * id2 - 1] += data[offset * id1 - 1];
        offset *= 2;
    }

    if (id == 0) {
        sum = data[n-1];
        data[n-1] = 0;
    }

    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (id < d) {
            int l = offset * id1 - 1;
            int r = offset * id2 - 1;
            unsigned int t = data[l];
            data[l] = data[r];
            data[r] += t;
        }
    }
    __syncthreads();
}

static __global__ void prefixSumKernel(
    int n,
    unsigned int* __restrict__ data,
    unsigned int* __restrict__ sum) {

    extern __shared__ unsigned int temp[];
    int lid = threadIdx.x * 2;
    int gid = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    temp[lid] = gid < n ? data[gid] : 0;
    temp[lid+1] = gid+1 < n ? data[gid+1] : 0;
    prefixSumSharedMem(temp, sum[blockIdx.x]);
    if(gid < n) data[gid] = temp[lid];
    if(gid+1 < n) data[gid+1] = temp[lid+1];
}

static __global__ void prefixSumFinalize(
    int n,
    unsigned int* __restrict__ sum,
    unsigned int* __restrict__ sum2,
    unsigned int* __restrict__ sum3) {

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = blockDim.x*2;
    int dim2 = dim * dim;
    if(id < n) sum[id] += sum2[id / dim] + sum3[id / dim2];
}


template<typename T>
static inline cudaError_t cudaAlloc(T* &p, size_t c = 1) {
    return cudaMalloc((void **) &p, c * sizeof(T));
}

template<typename T>
static inline cudaError_t cudaClear(T* p, size_t c = 1, int v = 0) {
    return cudaMemset(p, v, c * sizeof(T));
}

unsigned int prefixSum(unsigned int* cuData, int n, int threadCount){
    int blockSize = threadCount * 2;
    int bN = n / blockSize + 1;
    int bN2 = bN / blockSize + 1;
    int n2 = bN2 * blockSize;
    int n3 = blockSize;

    long bs = blockSize;
    if(bs*bs*bs < n){
        cerr << "Small block size: " << blockSize << " for size: " << n << endl;
        return 0;
    }

    //initialize data
    unsigned int *sum2, *sum3, *sum4;
    cudaAlloc(sum2,n2);
    cudaAlloc(sum3,n3);
    cudaAlloc(sum4);
    cudaClear(sum2, n2);
    cudaClear(sum3, n3);
    cudaClear(sum4);

    //max 3 layers of prefix sum
    if(bN == 1)
        prefixSumKernel <<< bN,  threadCount, blockSize * sizeof(int) >>> (n,  cuData, sum4);
    else {
        prefixSumKernel <<< bN,  threadCount, blockSize * sizeof(int) >>> (n,  cuData, sum2);
        if(bN2 == 1)
            prefixSumKernel <<< bN2,  threadCount, blockSize * sizeof(int) >>> (n2,  sum2, sum4);
        else {
            prefixSumKernel <<< bN2,  threadCount, blockSize * sizeof(int) >>> (n2,  sum2, sum3);
            prefixSumKernel <<< 1,  threadCount, blockSize * sizeof(int) >>> (n3,  sum3, sum4);
        }
    }

    //finalize
    prefixSumFinalize<<< bN*2, threadCount >>> (n, cuData, sum2, sum3);

    unsigned int ret = 0;
    cudaMemcpy(&ret, sum4, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //free device data
    cudaFree(sum2);
    cudaFree(sum3);
    cudaFree(sum4);

    return ret;
}