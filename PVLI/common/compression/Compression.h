#pragma once
#include "boostHelpers.h"
#include <vector>
#include <utility>
#include <functional>
#include "../Buffer.h"

//TODO safe processing (asserts/exceptions)

namespace Compression {

    enum class Method : unsigned char {
        NONE, RLE8, RLE1, QT1, HUFFMAN, FSE
    };

    struct Header {
        int size = 0; // fully uncompressed
        int size2 = 0; // decompressed entropy (HUFFMAN), compressed RLE
        Method entropy = Method::NONE;
        Method method = Method::NONE;
    };

    std::vector<unsigned char> compress(Method entropy, Method rle, ConstBuffer src);
    std::vector<unsigned char> compress(ConstBuffer src, int w, int h, int block, int part = 0, int partCount = 1);
    bool decompress(ConstBuffer src, Buffer dst, bool skipRLE = false);

    inline Header header(ConstBuffer src){
        return src.canRead<Header>() ? src.read<Header>() : Header();
    }

    // wrapper

    inline std::vector<unsigned char> decompress(ConstBuffer src, bool skipRLE = false){
        std::vector<unsigned char> data(skipRLE ? header(src).size2 : header(src).size);
        decompress(src, data, skipRLE);
        return data;
    }
}

ENUM_STRINGS(Compression::Method, NONE, (NONE) (RLE8) (RLE1) (HUFFMAN) (FSE));
ENUM_STREAM_OPERATORS(Compression::Method);
