#ifndef CMAKE_AND_CUDA_CUDA_INCLUDE_H
#define CMAKE_AND_CUDA_CUDA_INCLUDE_H

#ifdef WIN32
#include <Windows.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector>
#include "debug.h"

#ifndef __CUDACC__

#ifdef __CLION_IDE__
#define __CUDACC__ 1
#endif

// used for IDE code completion
inline void __syncthreads() {}
inline void __threadfence_block() {}
template<class T> inline T __clz(const T val) { return val; }
extern const dim3 blockDim;
extern const uint3 threadIdx;
extern const uint3 blockIdx;

#endif

/* error checking */

cudaError_t cudaHandleError( cudaError_t error, const char *file, int line );
#define cuEC( error ) ( cudaHandleError( error, __FILE__, __LINE__ ) )

#ifdef NDEBUG
#define cuCheck
#else
#define cuCheck { cuEC(cudaDeviceSynchronize()); cuEC(cudaGetLastError()); }
#endif

/* helper functions */

#include <iostream>
using namespace std;

template<typename T>
inline cudaError_t cudaAlloc(T* &p, size_t c = 1) {
    DEBUG_GPU_RAM << "CUDA alloc " << (c * sizeof(T)) << ' ' << sizeof(T) << endl;
    return cudaMalloc((void **) &p, c * sizeof(T));
}

template<typename T>
inline cudaError_t cudaClear(T* p, size_t c = 1, int v = 0) {
    return cudaMemset(p, v, c * sizeof(T));
}

template<typename T>
inline cudaError_t cudaCopy(T *dst, const T *src, size_t count, enum cudaMemcpyKind kind) {
    return cudaMemcpy(dst, src, count * sizeof(T), kind);
}

/* reference can be used only on host -> does not to pass cudaMemcpuKind */

template<typename T>
inline cudaError_t cudaCopy(T& dst, const T *src) {
    return cudaMemcpy(&dst, src, sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
inline cudaError_t cudaCopy(T* dst, const T& src) {
    return cudaMemcpy(dst, &src, sizeof(T), cudaMemcpyHostToDevice);
}

/* std::vector */

template<typename T>
inline cudaError_t cudaCopy(std::vector<T>& dst, const T *src) {
    return cudaMemcpy(dst.data(), src, dst.size() * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
inline cudaError_t cudaCopy(T* dst, const std::vector<T>& src) {
    return cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
struct CuBuffer {
    T* data = nullptr;
    size_t size = 0;
    size_t allocated_size = 0;

    inline CuBuffer(size_t s = 0) { alloc(s); }
    inline ~CuBuffer() { free(true); }

    //move
    inline CuBuffer(CuBuffer&& b) {
        swap(data, b.data);
        swap(size, b.size);
        swap(allocated_size, b.allocated_size);
    }
    inline CuBuffer& operator=(CuBuffer&& b) {
        swap(data, b.data);
        swap(size, b.size);
        swap(allocated_size, b.allocated_size);
        return *this;
    }

    //weak reference to simplify passing to CUDA kernels
    inline CuBuffer(const CuBuffer& b) : data(b.data), size(b.size), allocated_size(0) {
        if (size == 0) data = nullptr;
    }
    inline CuBuffer& operator=(const CuBuffer& b) {
        if (&b == this) return *this;
        free(true);
        size = b.size;
        data = size > 0 ? b.data : nullptr;
        allocated_size = 0;
        return *this;
    }
    inline bool isWeakRef() { return size > 0 && allocated_size == 0; }

    inline void alloc(size_t s, bool keepData = false, bool clear = false) {
        if (s > allocated_size || (isWeakRef() && s > 0)) {
            if (keepData && size > 0) {
                auto oldData = data;
                cuEC(cudaAlloc(data, s));
                cuEC(cudaCopy(data, oldData, size, cudaMemcpyDeviceToDevice));
                if (clear && s > size) cuEC(cudaClear(data + size, s - size));
                if(!isWeakRef()) cuEC(cudaFree(oldData));
            } else {
                if(!isWeakRef()) cuEC(cudaFree(data));
                cuEC(cudaAlloc(data, s));
                if (clear) cuEC(cudaClear(data, s));
            }
            allocated_size = size = s;
            DEBUG_GPU_RAM << "CuBuffer alloc " << size << " / " << allocated_size << ' ' << sizeof(T) << endl;
        } else {
            if (clear) {
                if (keepData && s > size)
                    cuEC(cudaClear(data + size, s - size));
                else if (s > 0)
                    cuEC(cudaClear(data, s));
            }
            size = s;
            DEBUG_GPU_RAM << "CuBuffer keep " << size << " / " << allocated_size << ' ' << sizeof(T) << endl;
        }
    }

    inline void resize(size_t s, bool clear = false) { alloc(s, true, clear); }

    inline void free(bool deallocate = false) {
        if (deallocate) {
            if (!isWeakRef()) cuEC(cudaFree(data));
            data = nullptr;
            allocated_size = 0;
        } else if(isWeakRef()) data = nullptr;
        size = 0;
    }

    template<typename T2>
    inline CuBuffer<T2> stack(size_t& offset, size_t s) {
        CuBuffer<T2> ret;
        if (s > 0) {
            ret.data = (T2*) (data + offset);
            ret.size = s;
            offset += s * sizeof(T2) / sizeof(T);
        }
        return ret;
    }

    inline void clear(int value = 0) {
        if(size > 0)
            cuEC(cudaClear(data, size, value));
    }

    inline void get(T* dst) const {
        if (size > 0)
            cuEC(cudaCopy(dst, data, size, cudaMemcpyDeviceToHost));
    }

    inline void set(const T* src, size_t s = 0) {
        alloc(s);
        if (size > 0)
            cuEC(cudaCopy(data, src, size, cudaMemcpyHostToDevice));
    }

    inline void set(const CuBuffer& src) {
        alloc(src.size);
        if (size > 0)
            cuEC(cudaCopy(data, src.data, size, cudaMemcpyDeviceToDevice));
    }

    template<typename T2 = T>
    inline std::vector<T2> get() const {
        std::vector<T2> ret(size * sizeof(T) / sizeof(T2));
        if (size > 0) get((T*) ret.data());
        return ret;
    }
    inline void set(const std::vector<T>& src) { set(src.data(), src.size()); }

    __host__ __device__ operator bool() const { return data && size > 0; }
    __host__ __device__ operator T*() const { return size > 0 ? data : nullptr; }

    __device__ T& operator[](size_t i) { return data[i]; }
    __device__ const T& operator[](size_t i) const { return data[i]; }
};

#endif
