#include "catch.hpp"

#include <vector>
#include <cstdlib>
#include <iostream>

#include "compression/Compression.h"

using namespace std;
using namespace Compression;

static void compare(const vector<unsigned char>& d1, const vector<unsigned char>& d2) {
    REQUIRE(d1.size() == d2.size());
    bool eq = true;
    int i = 0;
    for (; i < d1.size() && eq; i++)
        if (d1[i] != d2[i]) eq = false;
    CAPTURE(i);
    REQUIRE(eq);
}

TEST_CASE( "Compression", "[compression]" ) {

    vector<unsigned char> data1(260000);
    for(int i=0; i<data1.size(); i++)
        data1[i] = (i%100 + rand()%20) < 50 ? 1 : 0;

    vector<unsigned char> data2(470000);
    for(int i=0; i<data2.size(); i++)
        data2[i] = rand()%50 + rand()%50 + rand()%50;

    SECTION( "RLE1" ) {
        compare(data1, decompress(compress(Method::NONE, Method::RLE1, data1)));
    }

    SECTION( "RLE8" ) {
        compare(data1, decompress(compress(Method::NONE, Method::RLE8, data1)));
        compare(data2, decompress(compress(Method::NONE, Method::RLE8, data2)));
    }

    SECTION( "HUFFMAN" ) {
        compare(data1, decompress(compress(Method::HUFFMAN, Method::NONE, data1)));
        compare(data2, decompress(compress(Method::HUFFMAN, Method::NONE, data2)));
    }

    SECTION( "HUFFMAN + RLE1" ) {
        compare(data1, decompress(compress(Method::HUFFMAN, Method::RLE1, data1)));
    }

    SECTION( "HUFFMAN + RLE8" ) {
        compare(data1, decompress(compress(Method::HUFFMAN, Method::RLE8, data1)));
        compare(data2, decompress(compress(Method::HUFFMAN, Method::RLE8, data2)));
    }
}