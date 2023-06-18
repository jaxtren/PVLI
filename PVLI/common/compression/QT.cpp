#include "QT.h"

#include <iostream>
#include <cstring>

using namespace std;

#define ANY 2
#define BOTH 3

namespace Compression { namespace QT {

    template<typename T>
    inline size_t add(vector<unsigned char>& v, const T& d) {
        auto start = v.size();
        auto size = sizeof(T);
        v.resize(start + size);
        *reinterpret_cast<T*>(v.data() + start) = d;
        return start;
    }

    inline static void setBit(unsigned char* data, int i, unsigned char b) {
        data[i >> 3] |= b << (i & 7);
    }

    inline static unsigned char getBit(unsigned char* data, int i) {
        return (data[i >> 3] >> (i & 7)) & 1;
    }

    inline static void alignToByte(int& bit) {
        if (bit > 0) bit = (((bit - 1) >> 3) + 1) << 3;
    }

    size_t compress(ConstBuffer data, vector<unsigned char>& dst, int w, int h, int block, int part, int partCount) {
        vector<vector<char>> buf;
        for (int i = 0, d = block; d > 0; i++, d /= 2)
            buf.push_back(vector<char>(d * d));

        bool addHeader = part == 0;
        if (addHeader) {
            Header head = {w, h, block};
            add(dst, head);
        }

        auto start = dst.size();
        dst.resize(start + w * h);
        auto bits = dst.data() + start;
        int bit = 0;
        bool OUT = false;
        int blockCount = ((w - 1) / block + 1) * ((h - 1) / block + 1) / partCount;
        int firstBlock = part * blockCount, currentBlock = 0;

        for (int Y = 0; Y < h; Y += block) {
            for (int X = 0; X < w; X += block, currentBlock++) {
                if (currentBlock < firstBlock ||
                    (part + 1 < partCount && currentBlock >= firstBlock + blockCount)) continue;

                if (OUT) cout << X << ' ' << Y << endl;

                // init block
                auto cur = buf[0].data();
                int ey = Y + block, ex = X + block;
                for (int y = Y; y < ey; y++) {
                    int yw = y * w;
                    for (int x = X; x < ex; x++) {
                        char v = x < w && y < h ? data[yw + x] : ANY;
                        *cur++ = v <= 1 ? v : ANY;
                        if (OUT) cout << (int) buf[0][y * block + x];
                    }
                    if (OUT) cout << endl;
                }

                // mipmap
                for (int i = 1, d = block >> 1; d > 0; i++, d >>= 1) {
                    int W = d * 2;
                    auto cur = buf[i].data();
                    auto& src = buf[i - 1];
                    for (int y = 0; y < W; y += 2) {
                        int Y0 = y * W, Y1 = Y0 + W;
                        for (int x = 0; x < W; x += 2) {
                            int V[4] = {0, 0, 0, 0};
                            V[src[Y0 + x + 0]] = 1;
                            V[src[Y0 + x + 1]] = 1;
                            V[src[Y1 + x + 1]] = 1;
                            V[src[Y1 + x + 0]] = 1;

                            auto& O = *cur++;
                            int V01 = V[0] + V[1];
                            if (V[BOTH] || V01 == 2) O = BOTH;
                            else if (V01 == 0) O = ANY;
                            else O = V[1];

                            if (OUT) cout << (int) O;
                        }
                        if (OUT) cout << endl;
                    }
                }

                // compress
                int i = (int)buf.size() - 1, x = 0, y = 0, s = i;
                do {
                    char v = buf[i][(y << (s - i)) + x];
                    if (v != BOTH) {
                        if (i > 0) setBit(bits, bit++, 0);
                        setBit(bits, bit++, v & 1);
                        while (true) {
                            if (!(x & 1)) x++;
                            else if (y & 1) {
                                i++;
                                x >>= 1;
                                y >>= 1;
                                continue;
                            } else {
                                x--;
                                y++;
                            }
                            break;
                        }
                    } else {
                        setBit(bits, bit++, 1);
                        i--;
                        x <<= 1;
                        y <<= 1;
                    }
                } while (i < buf.size() - 1);

                alignToByte(bit);
            }
        }

        auto size = bit >> 3;
        dst.resize(start + size);
        return + (int)addHeader * sizeof(Header) + size;
    }

    bool decompress(ConstBuffer src, Buffer dst) {
        auto head = src.read<Header>();
        int bit = 0, w = head.width, h = head.height;
        memset(dst.data, 0, w * h);

        for (int Y = 0; Y < h; Y += head.block) {
            for (int X = 0; X < w; X += head.block) {
                int x = 0, y = 0, i = head.block;
                do {
                    if (i > 1 && getBit(src.data, bit++)) {
                        i >>= 1;
                        x <<= 1;
                        y <<= 1;
                    } else {
                        auto v = getBit(src.data, bit++);

                        if (v) {
                            if (i == 1) dst[(Y + y) * w + X + x] = 1;
                            else //fill quad
                                for (int k = (Y + y * i) * w, K = min(h * w, k + i * w); k < K; k += w)
                                    for (int j = k + X + x * i, J = min(k + w, j + i); j < J; j++)
                                        dst[j] = 1;
                        }

                        while (true) {
                            if (!(x & 1)) x++;
                            else if (y & 1) {
                                i <<= 1;
                                x >>= 1;
                                y >>= 1;
                                continue;
                            } else {
                                x--;
                                y++;
                            }
                            break;
                        }
                    }
                } while (i < head.block);

                alignToByte(bit);
            }
        }

        return true;
    }

    int decompress(ConstBuffer src, Buffer dst, int part, int partCount, int startBit) {
        auto head = src.read<Header>();
        int bit = 0, w = head.width, h = head.height;

        auto wBlockCount = (w - 1) / head.block + 1;
        auto hBlockCount = (h - 1) / head.block + 1;
        int blockCount =  wBlockCount * hBlockCount / partCount;
        int firstBlock = part * blockCount, currentBlock = 0;
        bit = startBit;

        for (int Y = 0; Y < h; Y += head.block) {
            for (int X = 0; X < w; X += head.block, currentBlock++) {
                if (currentBlock < firstBlock ||
                    (part + 1 < partCount && currentBlock >= firstBlock + blockCount)) continue;

                int x = 0, y = 0, i = head.block;
                do {
                    if (i > 1 && getBit(src.data, bit++)) {
                        i >>= 1;
                        x <<= 1;
                        y <<= 1;
                    } else {
                        auto v = getBit(src.data, bit++);

                        if (v) {
                            if (i == 1) dst[(Y + y) * w + X + x] = 1;
                            else //fill quad
                                for (int k = (Y + y * i) * w, K = min(h * w, k + i * w); k < K; k += w)
                                    for (int j = k + X + x * i, J = min(k + w, j + i); j < J; j++)
                                        dst[j] = 1;
                        }

                        while (true) {
                            if (!(x & 1)) x++;
                            else if (y & 1) {
                                i <<= 1;
                                x >>= 1;
                                y >>= 1;
                                continue;
                            } else {
                                x--;
                                y++;
                            }
                            break;
                        }
                    }
                } while (i < head.block);

                alignToByte(bit);
            }
        }

        return bit;
    }

} }