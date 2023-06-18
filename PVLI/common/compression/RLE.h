#pragma once

#include "../Buffer.h"
#include <vector>

namespace Compression { namespace RLE {

    struct Range {
        int start, count;
        unsigned char value;
    };

    inline int size(ConstBuffer src){
        return src.read<int>();
    }

    size_t compressMask(ConstBuffer src, std::vector<unsigned char>& dst);
    bool decompressMask(ConstBuffer src, Buffer dst);
    std::vector<Range> decompressMaskRange(ConstBuffer);

    size_t compress(ConstBuffer src, std::vector<unsigned char>& dst);
    bool decompress(ConstBuffer src, Buffer dst);
    std::vector<Range> decompressRange(ConstBuffer);

    // wrappers

    inline std::vector<unsigned char> compressMask(ConstBuffer src){
        std::vector<unsigned char> dst;
        compressMask(src, dst);
        return dst;
    }

    inline std::vector<unsigned char> decompressMask(ConstBuffer src){
        std::vector<unsigned char> dst(size(src));
        decompressMask(src, dst);
        return dst;
    }

    inline std::vector<unsigned char> compress(ConstBuffer src){
        std::vector<unsigned char> dst;
        compress(src, dst);
        return dst;
    }

    inline std::vector<unsigned char> decompress(ConstBuffer src){
        std::vector<unsigned char> dst(size(src));
        decompress(src, dst);
        return dst;
    }
} }