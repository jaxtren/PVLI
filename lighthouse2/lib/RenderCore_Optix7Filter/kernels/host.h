#pragma once
// host-side access points for the CUDA kernels
// define CUDA_HOST_IMPL to generate definitions (requires to be included from *.cu) otherwise it generates declarations
// parameter 'in' is index of source buffer when using algorithm with swapping/dual buffers (destination index is 1-in)

inline int blockCount(int n, int b) { return (n + b - 1) / b; }
inline int blockSize() { return 128; }
inline int blockCount(int n) { return blockCount(n, blockSize()); }

inline dim3 blockSize2D() { return dim3(16, 16, 1); }
inline dim3 blockCount2D(const int2& s) {
    auto b = blockSize2D();
    return dim3(blockCount(s.x, b.x), blockCount(s.y, b.y), 1);
}

// prefixsum.h
unsigned int prefixSum(unsigned int* cuData, int size, int threadCount = 64);

__host__ void setCoreData(const CoreData& d)
#ifdef CUDA_HOST_IMPL
{
    hostData = d;
    cudaMemcpyToSymbol(data, &hostData, sizeof(CoreData));
}
#endif
;

__host__ void* getCoreDataAddress()
#ifdef CUDA_HOST_IMPL
{
    void* address = 0;
    cudaGetSymbolAddress(&address, data);
    return address;
}
#endif
;

__host__ void updateFilterFragments(int sample, bool accumulated)
#ifdef CUDA_HOST_IMPL
{ updateFilterFragmentsKernel <<< blockCount(*hostData.count), blockSize() >>> (sample, accumulated); }
#endif
;

__host__ void reorderFragmentsPrepare()
#ifdef CUDA_HOST_IMPL
{ reorderFragmentsPrepareKernel <<< blockCount(hostData.scrsize.x * hostData.scrsize.y), blockSize() >>> (); }
#endif
;

__host__ void reorderFragments()
#ifdef CUDA_HOST_IMPL
{ reorderFragmentsKernel <<< blockCount(hostData.scrsize.x * hostData.scrsize.y), blockSize() >>> (); }
#endif
;

__host__ void reorderFragmentsUpdate()
#ifdef CUDA_HOST_IMPL
{ reorderFragmentsUpdateKernel <<< blockCount(hostData.scrsize.x * hostData.scrsize.y), blockSize() >>> (); }
#endif
;

__host__ void prepareFilterFragments(int phase)
#ifdef CUDA_HOST_IMPL
{ prepareFilterFragmentsKernel <<< blockCount(*hostData.count), blockSize() >>> (phase); }
#endif
;

__host__ void applyLayeredFilter(const int phase, int mode)
#ifdef CUDA_HOST_IMPL
{
    auto& d = hostData;
    if (mode == 0) // all layers (1D blocks)
         applyLayeredFilterKernel <<< blockCount(*d.count), blockSize() >>> (phase, mode);
    else if (mode == 1) // only first layer (2D blocks)
        applyLayeredFilterKernel <<< blockCount2D(d.scrsize), blockSize2D() >>> (phase, mode);
    else if (mode == 2 && *d.count > d.scrsize.x * d.scrsize.y) // skip first layer (1D blocks)
        applyLayeredFilterKernel <<< blockCount(*d.count - d.scrsize.x * d.scrsize.y), blockSize() >>> (phase, mode);
}
#endif
;

__host__ void finalizeAllLayers(void* fragments, int filter) //FIXME lighthouse2::Fragment* <-> void*
#ifdef CUDA_HOST_IMPL
{ finalizeAllLayersKernel <<< blockCount(*hostData.count), blockSize() >>> ((Fragment*)fragments, filter); }
#endif
;

__host__ void shade(int sample, int pathLength, int pathCount, int R0)
#ifdef CUDA_HOST_IMPL
{ shadeKernel <<< blockCount(pathCount), blockSize() >>> (sample, pathLength, pathCount, R0); }
#endif
;
