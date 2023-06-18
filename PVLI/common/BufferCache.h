#pragma once

#include <GL/glew.h>
#include <map>
#include <mutex>
#include <future>
#include "Config.h"

namespace gl {

    class BufferCache;

    struct Buffer {
        GLenum type = 0;
        GLuint buffer = 0;
        GLuint stagingBuffer = 0;
        GLenum usage = GL_STREAM_DRAW;
        GLenum stagingUsage = GL_STREAM_DRAW;
        char *data = nullptr;
        size_t size = 0;
        BufferCache* cache = nullptr;

        Buffer() = default;
        Buffer(const Buffer&) = delete;
        inline Buffer(Buffer&& m) : Buffer() { swap(m); }
        inline ~Buffer() { free(); }
        inline Buffer& operator=(const Buffer&) = delete;
        inline Buffer& operator=(Buffer&& m) { swap(m); return *this; }

        inline bool hasStagingBuffer() { return stagingBuffer != 0; }

        inline void swap(Buffer& m) {
            std::swap(type, m.type);
            std::swap(buffer, m.buffer);
            std::swap(stagingBuffer, m.stagingBuffer);
            std::swap(usage, m.usage);
            std::swap(stagingUsage, m.stagingUsage);
            std::swap(data, m.data);
            std::swap(size, m.size);
            std::swap(cache, m.cache);
        }

        inline operator GLuint() { return buffer; }

        void allocate(GLenum type, size_t size, GLenum usage = GL_STREAM_DRAW, GLenum stagingBufferUsage = GL_NONE);
        void free(bool useCache = true);

        inline void map() {
            if (data || !buffer) return;
            auto t = stagingBuffer ? GL_COPY_READ_BUFFER : type;
            glBindBuffer(t, stagingBuffer ? stagingBuffer : buffer);
            data = (char *) glMapBufferRange(t, 0, size, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
        }

        inline void unmap(bool copy = true) {
            if (!data || !buffer) return;
            glBindBuffer(type, buffer);
            if (stagingBuffer && copy) {
                glBindBuffer(GL_COPY_READ_BUFFER, stagingBuffer);
                glUnmapBuffer(GL_COPY_READ_BUFFER);
                glCopyBufferSubData(GL_COPY_READ_BUFFER, type, 0, 0, size);
                glBindBuffer(GL_COPY_READ_BUFFER, 0);
            } else
                glUnmapBuffer(type);
            data = nullptr;
        }
    };

    class BufferCache {
    public:
        using Lock = std::unique_lock<std::mutex>;

    private:
        GLenum type = GL_ARRAY_BUFFER;
        size_t frame = 0;
        std::mutex m;
        std::multimap <size_t, std::pair<size_t, Buffer>> cache;
        std::multimap <size_t, std::promise<Buffer>> pendingRequests;

    public:
        BufferCache(GLenum t = GL_ARRAY_BUFFER) { setType(t); }

        //settings
        size_t framesNotUsed = 3;
        double allocationSizeMultiplier = 1.20;
        double reuseSmallestSizeMultiplier = 0.6;

        inline void setType(GLenum t) {
            Lock l(m);
            type = t;
            cache.clear();
        };

        inline void clear() { Lock l(m); cache.clear(); };
        inline void nextFrame() { Lock l(m); frame++; }
        std::future <Buffer> request(size_t size);
        void returnBuffer(Buffer &buffer);
        void provideBuffers();
        void GC();
        size_t getSize();
        size_t getCount() { Lock l(m); return cache.size(); }
    };

    struct BufferCaches {
        BufferCache array;
        BufferCache elementArray;
        BufferCache pixelUnpack;

        inline BufferCaches() :
            array(GL_ARRAY_BUFFER),
            elementArray(GL_ELEMENT_ARRAY_BUFFER),
            pixelUnpack(GL_PIXEL_UNPACK_BUFFER) {}

        bool updateConfig(const Config &cfg) {
            bool ret = cfg.get("FramesNotUsed", array.framesNotUsed) |
                       cfg.get("AllocationSizeMultiplier", array.allocationSizeMultiplier) |
                       cfg.get("ReuseSmallestSizeMultiplier", array.reuseSmallestSizeMultiplier);
            array.allocationSizeMultiplier = std::max(1.0, array.allocationSizeMultiplier);
            elementArray.framesNotUsed = pixelUnpack.framesNotUsed = array.framesNotUsed;
            elementArray.allocationSizeMultiplier = pixelUnpack.allocationSizeMultiplier = array.allocationSizeMultiplier;
            elementArray.reuseSmallestSizeMultiplier = pixelUnpack.reuseSmallestSizeMultiplier = array.reuseSmallestSizeMultiplier;
            return ret;
        }

        inline void clear() {
            array.clear();
            elementArray.clear();
            pixelUnpack.clear();
        };

        inline void nextFrame() {
            array.nextFrame();
            elementArray.nextFrame();
            pixelUnpack.nextFrame();
        }

        inline void provideBuffers() {
            array.provideBuffers();
            elementArray.provideBuffers();
            pixelUnpack.provideBuffers();
        }

        inline void GC() {
            array.GC();
            elementArray.GC();
            pixelUnpack.GC();
        }
    };
}