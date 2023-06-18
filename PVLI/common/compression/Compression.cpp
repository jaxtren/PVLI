#include "Compression.h"
#include <cstring>
#include <cassert>
#include "../common.h"
#include "RLE.h"
#include "QT.h"

extern "C" {
#include "huf.h"
};

using namespace std;

namespace Compression {

    template<typename T>
    inline size_t add(vector<unsigned char>& v, const T& d) {
        auto start = v.size();
        auto size = sizeof(T);
        v.resize(start + size);
        *reinterpret_cast<T*>(v.data() + start) = d;
        return start;
    }

    template<typename T>
    inline void set(unsigned char* v, const T& d) {
        *reinterpret_cast<T*>(v) = d;
    }

    std::vector<unsigned char> compress(ConstBuffer src, int w, int h, int block, int part, int partCount) {
        vector<unsigned char> dst;
        if (part == 0) {
            Header head = {(int) src.size, (int) src.size, Method::NONE, Method::QT1};
            add(dst, head);
        }
        QT::compress(src, dst, w, h, block, part, partCount);
        return dst;
    }

    vector<unsigned char> compress(Method entropy, Method rle, ConstBuffer src) {
        Header head = {(int) src.size, (int) src.size, entropy, rle};
        vector<unsigned char> dst;
        add(dst, head);

        vector<unsigned char> rleData;
        if(rle == Method::RLE1) {
            RLE::compressMask(src, rleData);
            src = rleData;
        } if(rle == Method::RLE8) {
            RLE::compress(src, rleData);
            src = rleData;
        }

        head.size2 = (int)src.size;
        set(dst.data(), head);

        size_t start = dst.size();
        if (entropy == Method::HUFFMAN) {
            size_t sizeLimit = 128 * 1024; //RLE library has limit for HUFFMAN compression
            for (; src.size > 0; src.skip(sizeLimit)) {
                auto size = min(src.size, sizeLimit);

                //reserve location for compressed size of this part
                auto sizeLoc = start;
                start += sizeof(int);

                //compress
                dst.resize(start + HUF_compressBound(size));
                auto csize = HUF_compress(dst.data() + start, dst.size() - start, src.data, size);

                //copy if cannot compress
                if (csize <= 0) {
                    csize = 0;
                    memcpy(dst.data() + start, src.data, size);
                    start += size;
                } else start += csize;

                set(dst.data() + sizeLoc, (int)csize); //compressed size of this part or 0
            }
            dst.resize(start);
            return dst;
        } else if (entropy == Method::FSE) {
            throw string_exception("FSE not implemented"); //TODO
        }

        dst.resize(dst.size() + src.size);
        memcpy(dst.data() + sizeof(Header), src.data, src.size);
        return dst;
    }

    bool decompress(ConstBuffer src, Buffer dst, bool skipRLE) {
        auto head = src.read<Header>();

        if(head.method == Method::QT1){
            QT::decompress(src, dst);
            return true;
        }

        bool useRLE = head.method != Method::NONE && !skipRLE;

        vector<unsigned char> data;
        if (head.entropy == Method::HUFFMAN) {
            ConstBuffer src2 = src;
            Buffer dst2 = dst;
            size_t size2 = head.size2;

            if (useRLE) {
                //decompress to intermediate buffer
                data.resize(size2);
                src = dst2 = data;
            }

            size_t sizeLimit = 128 * 1024; //RLE library has limit for HUFFMAN compression
            for (size_t i = 0; i < size2; i += sizeLimit) {
                auto size = min(size2 - i, sizeLimit);

                size_t csize = src2.read<int>();
                if (csize == 0) {
                    memcpy(dst2.data, src2.data, size);
                    src2.skip(size);
                } else {
                    auto dsize = HUF_decompress(dst2.data, size, src2.data, csize);
                    assert(dsize == size);
                    src2.skip(csize);
                }
                dst2.skip(size);
            }
        } else if (head.entropy == Method::FSE) {
            throw string_exception("FSE not implemented"); //TODO
        }

        if (!useRLE) {
            if (head.entropy == Method::NONE)
                memcpy(dst.data, src.data, head.size2);
            return true;
        }

        //RLE
        if(head.method == Method::RLE1) RLE::decompressMask(src, dst);
        else RLE::decompress(src, dst);
        return true;
    }
}
