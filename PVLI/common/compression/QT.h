#pragma once

#include <vector>
#include "../Buffer.h"

// quad tree based compression for 2D binary images

namespace Compression {  namespace QT {

    struct Header {
        int width, height, block;
    };

    inline Header header(ConstBuffer src){
        return src.read<Header>();
    }

    size_t compress(ConstBuffer src, std::vector<unsigned char>& dst, int w, int h, int block, int part = 0, int partCount = 1);
    bool decompress(ConstBuffer src, Buffer dst);
    int decompress(ConstBuffer src, Buffer dst, int part, int partCount, int startBit);

    // wrappers

    inline std::vector<unsigned char> compress(ConstBuffer src, int w, int h, int block){
        std::vector<unsigned char> dst;
        compress(src, dst, w, h, block);
        return dst;
    }

    inline std::vector<unsigned char> decompress(ConstBuffer src) {
        auto head = header(src);
        std::vector<unsigned char> dst(head.width * head.height);
        decompress(src, dst);
        return dst;
    }
} }
