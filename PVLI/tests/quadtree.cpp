#include "catch.hpp"

#include <vector>
#include <cstdlib>
#include <iostream>
#include <cstring>

#include "compression/QT.h"

using namespace std;
using namespace Compression::QT;

static void compareBin(const vector<unsigned char>& d1, const vector<unsigned char>& d2) {
    REQUIRE(d1.size() == d2.size());
    bool eq = true;
    int i = 0;
    for (; i < d1.size() && eq; i++)
        if (d1[i] < 2 && d2[i] < 2 && d1[i] != d2[i]) eq = false;
    CAPTURE(i);
    REQUIRE(eq);
}

TEST_CASE( "Compression::QT", "[Compression::QT]" ) {

    vector<unsigned char> data = {
        1, 1, 1, 1, 0, 0, 0, 0, 1, 1,
        1, 1, 1, 1, 0, 0, 0, 0, 1, 1,
        1, 1, 1, 1, 0, 0, 2, 2, 0, 0,
        1, 1, 1, 1, 0, 0, 2, 2, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 2, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    };

    vector<unsigned char> data2;
    for(int i=0; i<40000; i++)
        data2.insert( data2.end(), data.begin(), data.end() );

    SECTION( "Small" ) {
        auto compressed = compress(data, 10, 9, 4);
        auto decompressed = decompress(compressed);
        compareBin(data, decompressed);
    }

    SECTION( "Big" ) {
        auto compressed = compress(data2, 2000, 1800, 512);
        auto decompressed = decompress(compressed);
        compareBin(data2, decompressed);
    }

    SECTION( "Small, partial compression" ) {
        vector<unsigned char> compressed;

        auto partCount = 2;
        for (auto i = 0; i < partCount; i++)
        {
            vector<unsigned char> dst;
            const auto bits = compress(data, dst, 10, 9, 4, i, partCount);
            compressed.reserve(compressed.size() + dst.size());
            compressed.insert(compressed.end(), dst.begin(), dst.end());
        }

        auto decompressed = decompress(compressed);
        compareBin(data, decompressed);
    }

    SECTION( "Small, partial decompression" ) {
        auto compressed = compress(data, 10, 9, 4);

        vector<unsigned char> decompressed;
        decompressed.resize(10 * 9);
        memset(decompressed.data(), 0, 10 * 9);

        auto partCount = 2;
        int readBits = 0;
        for (auto i = 0; i < partCount; i++)
            readBits = decompress(compressed, decompressed, i, partCount, readBits);

        compareBin(data, decompressed);
    }

    SECTION( "Big, partial decompression" ) {
        auto compressed = compress(data2, 2000, 1800, 512);

        vector<unsigned char> decompressed;
        decompressed.resize(2000 * 1800);
        memset(decompressed.data(), 0, 2000 * 1800);

        auto partCount = 8;
        int readBits = 0;
        for (auto i = 0; i < partCount; i++)
            readBits = decompress(compressed, decompressed, i, partCount, readBits);

        compareBin(data2, decompressed);
    }
}