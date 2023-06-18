#pragma once

#include "graphic.h"
#include "cudaHelpers.h"

template<typename T>
struct GlCuBuffer {
    GLenum type = GL_NONE, usage = GL_NONE;
    GLuint gl = 0;
    cudaGraphicsMapFlags mapMode;
    int size = 0;
    T* cu = nullptr;
    cudaGraphicsResource_t res;

    GlCuBuffer() = default;
    ~GlCuBuffer() { free(); }

    void init(GLenum t, int s, T* data = nullptr, GLenum u = GL_STATIC_DRAW, cudaGraphicsMapFlags m = cudaGraphicsMapFlagsNone) {
        //do not reallocate with almost same buffer size
        if(gl && type == t && usage == u && mapMode == m &&
           data == nullptr && s <= size && size * 2 < s * 3){
            DEBUG_GPU_RAM << "GlCuBuffer keep " << s << ' ' << sizeof(T) << endl;
            return;
        }
        DEBUG_GPU_RAM << "GlCuBuffer alloc " << s << ' ' << sizeof(T) << endl;

        free();
        type = t;
        size = s;
        usage = u;
        mapMode = m;
        glGenBuffers(1, &gl);
        glBindBuffer(type, gl);
        glBufferData(type, size * sizeof(T), (void*)data, usage);
        cuEC(cudaGraphicsGLRegisterBuffer(&res, gl, mapMode));
        glBindBuffer(type, 0);
    }

    void free() {
        if(gl) {
            unmap();
            cuEC(cudaGraphicsUnregisterResource(res));
            glDeleteBuffers(1, &gl);
            gl = 0;
            type = GL_NONE;
        }
    }

    inline void bind() {
        glBindBuffer(type, unmap());
    }

    inline void unbind() {
        glBindBuffer(type, 0);
    };

    T* map(){
        if(gl && !cu) {
            cuEC(cudaGraphicsMapResources(1, &res));
            size_t size = 0;
            cuEC(cudaGraphicsResourceGetMappedPointer((void**)&cu, &size, res));
        }
        return cu;
    }

    GLuint unmap(){
        if(!cu) return gl;
        cuEC(cudaGraphicsUnmapResources(1, &res));
        cu = nullptr;
        return gl;
    }

    inline void get(T* dst) { cuEC(cudaCopy(dst, map(), size, cudaMemcpyDeviceToHost)); }
    inline void set(const T* src, size_t s = 0) {
        if (s == size)
            cuEC(cudaCopy(map(), src, size, cudaMemcpyHostToDevice));
    }

    template<typename T2 = T>
    inline std::vector<T2> get() {
        std::vector<T2> ret(size * sizeof(T) / sizeof(T2));
        get((T*) ret.data());
        return ret;
    }
    inline void set(const std::vector<T>& src) { set(src.data(), src.size()); }

    operator bool() const { return size > 0; }
};