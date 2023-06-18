#pragma once
#include <cstddef>

struct Buffer {
    unsigned char* data = nullptr;
    size_t size = 0;

    Buffer() = default;

    template<typename T>
    Buffer(const T* v, size_t s){
        data = (unsigned char*)v;
        size = s * sizeof(T);
    }

    template<typename T>
    Buffer(const std::vector<T>& v){
        data = (unsigned char*)v.data();
        size = v.size() * sizeof(T);
    }

    template<typename T>
    Buffer& operator = (const std::vector<T>& v){
        data = (unsigned char*)v.data();
        size = v.size() * sizeof(T);
        return *this;
    }

    inline unsigned char& operator[](size_t i){
        return data[i];
    }

    //reading
    template<typename T = unsigned char>
    inline T read() {
        T ret = *reinterpret_cast<T*>(data);
        data += sizeof(T);
        size -= sizeof(T);
        return ret;
    }

    template<typename T = unsigned char>
    inline bool canRead() {
        return sizeof(T) <= size;
    }

    inline void skip(size_t s){
        s = std::min(s, size);
        data += s;
        size -= s;
    }

};
using ConstBuffer = Buffer; //TODO