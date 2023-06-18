#pragma once

enum { NOT_ALLOCATED = 0, ON_HOST = 1, ON_DEVICE = 2, STAGED = 4 };
enum { POLICY_DEFAULT = 0, POLICY_COPY_SOURCE };

/**
 * Buffer for data device-host management and synchronization
 * Can be directly passed to CUDA kernel
 * Copy constructor and assignment creates weak reference
 * @tparam T type
 */

template <class T> class CoreBuffer
{
public:
    #ifndef __CUDACC__ //TODO replace with __CUDA_ARCH__
    CoreBuffer() : dev(0), host(0), size(0), owner(0) { }
    CoreBuffer( int elements, int location, const void* source = 0, const int policy = POLICY_DEFAULT )
        : CoreBuffer() { Allocate(elements, location, source, policy); }
    void Allocate( int elements, int location, const void* source = 0, const int policy = POLICY_DEFAULT )
    {
        //do not reallocate with sufficient buffer size
        if ((location & owner) == location && elements <= GetSize())
            return;

        Free();
        size = elements * sizeof(T);
        if (size > 0)
        {
            if (location & ON_DEVICE)
            {
                CHK_CUDA(cudaMalloc(&dev, size));
                owner |= ON_DEVICE;
            }

            if (location & ON_HOST)
            {
                if (source)
                {
                    if (policy == POLICY_DEFAULT) host = (T*)source; // use supplied pointer
                    else // POLICY_COPY_SOURCE: pointer was supplied, and we are supposed to copy it
                    {
                        host = (T*)MALLOC64( size );
                        owner |= ON_HOST;
                        memcpy( host, source, size );
                    }

                    if (location & ON_DEVICE) CopyToDevice();
                }
                else
                {
                    host = (T*)MALLOC64( size );
                    owner |= ON_HOST;
                }
            }
            else if (source && (location & ON_DEVICE) && !(location & STAGED)) // location is ON_DEVICE only, and we have data, so send the data over
                CHK_CUDA( cudaMemcpy( dev, source, size, cudaMemcpyHostToDevice ) );
        }
    }
    inline CoreBuffer( const CoreBuffer& b ) : CoreBuffer() { *this = b; }; //weak reference
    inline CoreBuffer( CoreBuffer&& b ) : CoreBuffer() { *this = std::move(b); }
    inline CoreBuffer& operator=(const CoreBuffer& b) { //weak reference
        Free();
        dev = b.dev;
        host = b.host;
        size = b.size;
        return *this;
    }
    inline CoreBuffer& operator=(CoreBuffer&& b){
        std::swap(dev, b.dev);
        std::swap(host, b.host);
        std::swap(size, b.size);
        std::swap(owner, b.owner);
        return *this;
    }
    inline ~CoreBuffer() { Free(); }
    void Free()
    {
        #ifndef __CUDACC__
        if (owner & ON_HOST) FREE64( host );
        if (owner & ON_DEVICE) CHK_CUDA( cudaFree( dev ) );
        dev = host = 0;
        size = 0;
        owner = 0;
        #endif
    }
    void* CopyToDevice()
    {
        if (size > 0)
        {
            if (!dev)
            {
                CHK_CUDA( cudaMalloc( &dev, size ) );
                owner |= ON_DEVICE;
            }
            CHK_CUDA( cudaMemcpy( dev, host, size, cudaMemcpyHostToDevice ) );
        }
        return dev;
    }
    void* CopyToDevice(const T& v){
        *host = v;
        CopyToDevice();
        return dev;
    }
    void* StageCopyToDevice()
    {
        if (size > 0)
        {
            if (!dev)
            {
                CHK_CUDA( cudaMalloc( &dev, size ) );
                owner |= ON_DEVICE;
            }
            lh2core::stageMemcpy( dev, host, size );
        }
        return dev;
    }
    void* MoveToDevice()
    {
        CopyToDevice();
        if (owner & ON_HOST) FREE64( host );
        owner &= ~ON_HOST;
        host = 0;
        return dev;
    }
    T* CopyToHost()
    {
        if (size > 0)
        {
            if (!host)
            {
                host = (T*)MALLOC64( size );
                owner |= ON_HOST;
            }
            CHK_CUDA( cudaMemcpy( host, dev, size, cudaMemcpyDeviceToHost ) );
        }
        return host;
    }
    void Clear(int location, int overrideSize = -1 )
    {
        if (size > 0)
        {
            int bytesToClear = overrideSize == -1 ? size : overrideSize;
            if (location & ON_HOST && host) memset( host, 0, bytesToClear );
            if (location & ON_DEVICE && dev) CHK_CUDA( cudaMemset( dev, 0, bytesToClear ) );
        }
    }
    #endif
    #ifdef __CUDA_ARCH__
    __host__ __device__  inline T* Ptr() { return dev; }
    __host__ __device__  inline const T* Ptr() const { return dev; }
    #else
    __host__ __device__  inline T* Ptr() { return host; }
    __host__ __device__  inline const T* Ptr() const { return host; }
    #endif
    __host__ __device__  inline int GetSizeInBytes() const { return size; }
    __host__ __device__  inline int GetSize() const { return size / sizeof(T); }
    __host__ __device__  inline T* DevPtr() { return dev; }
    __host__ __device__  inline const T* DevPtr() const { return dev; }
    __host__ __device__  inline T** DevPtrPtr() { return &dev; /* Optix7 wants an array of pointers; this returns an array of 1 pointers. */ }
    __host__ __device__  inline T* HostPtr() { return host; }
    __host__ __device__  inline const T* HostPtr() const { return host; }
    __host__ __device__  inline void SetHostData( T* hostData ) { host = hostData; }
    //operators
    __host__ __device__  inline T* operator->() { return Ptr(); }
    __host__ __device__  inline const T* operator->() const { return Ptr(); }
    __host__ __device__  inline T& operator*() { return *Ptr(); }
    __host__ __device__  inline const T& operator*() const { return *Ptr(); }
    __host__ __device__  inline T& operator[](size_t i) { return Ptr()[i]; }
    __host__ __device__  inline const T& operator[](size_t i) const { return Ptr()[i]; }
    __host__ __device__  inline operator bool() const { return Ptr(); }
    // member data
private:
    T* dev, *host;
    int size;
    char owner;
};