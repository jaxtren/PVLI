#include <cassert>
#include <vector>
#include <cstring>
#include <cstdint>
#include "RLE.h"

using namespace std;

namespace Compression { namespace RLE {

    template<typename T>
    inline static void add(vector<unsigned char>& v, const T& d) {
        auto start = v.size();
        auto size = sizeof(T);
        v.resize(start + size);
        memcpy(reinterpret_cast<void*>(v.data() + start), reinterpret_cast<const void*>(&d), size);
    }

    // requires little-endian byte ordering
    inline static void addCount(vector<unsigned char>& data, int v) {
        if (v < 128) data.push_back(v << 1);
        else add<uint16_t>(data, (v << 1) | 1);
    }

    // requires little-endian byte ordering
    inline static int getCount(ConstBuffer& data) {
        if (data[0] & 1) return data.read<uint16_t>() >> 1;
        else return data.read<uint8_t>() >> 1;
    }

    static const int maxCount = (1 << 15) - 1;

    size_t compressMask(ConstBuffer src, vector<unsigned char>& dst) {
        if (!src.size) return 0;
        auto osize = dst.size();
        dst.reserve(osize + src.size / 2);
        add(dst, (int)src.size);
        unsigned char value = 0;
        int count = 0;
        for(int i=0; i<src.size; i++){
            auto u = src[i];
            if (u != value && u < 2) {
                addCount(dst, count);
                value = u;
                count = 0;
            } else if (count == maxCount) {
                addCount(dst, count);
                dst.push_back(0);
                count = 0;
            }
            count++;
        }
        addCount(dst, count);
        return dst.size() - osize;
    }

    bool decompressMask(ConstBuffer src, Buffer dst) {
        int o = 0, size = src.read<int>();
        if(dst.size < size) return false;
        memset(dst.data, 0, size);
        bool value = false;
        while (src.size) {
            size = getCount(src);
            if (value) memset(&dst[o], 1, size);
            value = !value;
            o += size;
        }
        return true;
    }

    vector<Range> decompressMaskRange(ConstBuffer src) {
        vector<Range> ranges;
        int o = 0, size = src.read<int>();
        ranges.reserve(size / 4);
        unsigned char value = 0;
        while (src.size) {
            size = getCount(src);
            if(size > 0)
                ranges.push_back({o, size, value});
            value = 1 - value;
            o += size;
        }
        return ranges;
    }

    size_t compress(ConstBuffer src, vector<unsigned char>& dst) {
        if (!src.size) return 0;
        auto osize = dst.size();
        dst.reserve(osize + src.size / 2);
        add(dst, (int)src.size);
        unsigned char value = src[0];
        int count = 0;
        for(int i=0; i<src.size; i++){
            auto u = src[i];
            if (u != value || count == maxCount) {
                addCount(dst, count);
                dst.push_back(value);
                count = 0;
            }
            value = u;
            count++;
        }
        addCount(dst, count);
        dst.push_back(value);
        return dst.size() - osize;
    }

    bool decompress(ConstBuffer src, Buffer dst) {
        int o = 0, size = src.read<int>();
        if(dst.size < size) return false;
        memset(dst.data, 0, size);
        while (src.size) {
            size = getCount(src);
            memset(&dst[o], src.read(), size);
            o += size;
        }
        return true;
    }

    vector<Range> decompressRange(ConstBuffer src) {
        vector<Range> ranges;
        int o = 0, size = src.read<int>();
        ranges.resize(size / 4);
        while (src.size) {
            size = getCount(src);
            auto value = src.read();
            if (size > 0)
                ranges.push_back({o, size, value});
            o += size;
        }
        return ranges;
    }
} }