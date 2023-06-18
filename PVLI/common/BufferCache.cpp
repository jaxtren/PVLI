#include "BufferCache.h"
#include <iostream>

using namespace std;

//#define DEBUG_BUFFER_CACHE

namespace gl {

    void Buffer::allocate(GLenum type, size_t size, GLenum usage, GLenum stagingUsage) {
        free();
        this->type = type;
        this->size = size;
        this->usage = usage;
        this->stagingUsage = stagingUsage;
        if (!buffer) glGenBuffers(1, &buffer);
        if (stagingUsage && !stagingBuffer) glGenBuffers(1, &stagingBuffer);
        glBindBuffer(type, buffer);
        glBufferData(type, size, nullptr, usage);
        glBindBuffer(type, 0);
        if (stagingBuffer) {
            glBindBuffer(GL_COPY_READ_BUFFER, stagingBuffer);
            glBufferData(GL_COPY_READ_BUFFER, size, nullptr, stagingUsage);
            glBindBuffer(GL_COPY_READ_BUFFER, 0);
        }
        #ifdef DEBUG_BUFFER_CACHE
        if (size > 0) cout << "Buffer::allocate " << size << endl;
        #endif
    }

    void Buffer::free(bool useCache) {
        if (!buffer) return;
        if (useCache && cache) cache->returnBuffer(*this);
        else {
            #ifdef DEBUG_BUFFER_CACHE
            if (size > 0) cout << "Buffer::free " << size << endl;
            #endif
            unmap(false);
            glDeleteBuffers(1, &buffer);
            glDeleteBuffers(1, &stagingBuffer);
            buffer = 0;
            stagingBuffer = 0;
            type = 0;
            size = 0;
        }
    }

    future<Buffer> BufferCache::request(size_t size) {
        Lock l(m);
        auto prom = promise<Buffer>();
        auto fut = prom.get_future();
        if(size == 0)
            prom.set_value(Buffer());
        else {
            auto it = cache.lower_bound(size);
            if (it != cache.end() && size >= it->first * reuseSmallestSizeMultiplier) {
                #ifdef DEBUG_BUFFER_CACHE
                cout << "BufferCache::request reuse " << size << endl;
                #endif
                it->second.second.cache = this;
                prom.set_value(move(it->second.second));
                cache.erase(it);
            }
            else pendingRequests.emplace(size, move(prom));
        }
        return fut;
    }

    void BufferCache::returnBuffer(Buffer& buffer) {
        if(!buffer) return;
        Lock l(m);

        Buffer local;
        swap(local, buffer);
        local.cache = this;
        local.map();
        size_t s = local.size;

        if (!pendingRequests.empty()) {
            auto it = pendingRequests.lower_bound(s);
            if (it == pendingRequests.end() || it->first > s) it--;
            if (it != pendingRequests.end() && it->first <= s) {
                local.cache = this;
                it->second.set_value(move(local));
                pendingRequests.erase(it);
                return;
            }
        }
        local.cache = nullptr;
        cache.emplace(s, make_pair(frame, move(local)));
    }

    void BufferCache::provideBuffers() {
        Lock l(m);
        std::multimap<size_t, std::promise<Buffer>> pending;
        swap(pending, pendingRequests);
        l.unlock();

        for(auto& p : pending){
            Buffer mapped;
            mapped.allocate(type, (size_t)(p.first * allocationSizeMultiplier));
            mapped.cache = this;
            mapped.map();
            p.second.set_value(move(mapped));
        }
    }

    size_t BufferCache::getSize() {
        Lock l(m);
        size_t s = 0;
        for(auto& c : cache)
            s += c.first;
        return s;
    }

    void BufferCache::GC() {
        if (framesNotUsed <= 0) return;

        Lock l(m);
        size_t lastFrame = frame - framesNotUsed;
        for(auto it = cache.begin(); it != cache.end();) {
            if (it->second.first < lastFrame) {
                it->second.second.cache = nullptr;
                it = cache.erase(it);
            } else it++;
        }
    }
}
