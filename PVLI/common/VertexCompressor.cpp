#include "VertexCompressor.h"
#include <cstring>
#include <iostream>
#include <map>
#include <algorithm>
#ifndef DISABLE_PARALLEL_ALGORITHMS
#include <execution>
#endif
#include "Timer.h"

template<typename D, typename C>
inline void sortVector(D& data, C cmp) {
    #ifdef DISABLE_PARALLEL_ALGORITHMS
    std::sort(data.begin(), data.end(), cmp);
    #else
    std::sort(std::execution::par_unseq, data.begin(), data.end(), cmp);
    #endif
}

using namespace std;
using namespace glm;

//#define DEBUG_VERTEX_COMPRESSOR_PROTOCOL
//#define DEBUG_VERTEX_COMPRESSOR_TIMES
//#define DEBUG_VERTEX_COMPRESSOR_STATS
//#define VERTEX_COMPRESSOR_OLD_VARINT

#ifdef DEBUG_VERTEX_COMPRESSOR_STATS
#define DEBUG_VCS(x) x
#else
#define DEBUG_VCS(x)
#endif

template<typename T, int size = sizeof(T)>
inline static void add(vector<unsigned char>& data, const T& v, const string& info = "") {
    auto start = data.size();
    data.resize(start + size);
    memcpy(reinterpret_cast<void *>(&data[start]), reinterpret_cast<const void *>(&v), size);
    #ifdef DEBUG_VERTEX_COMPRESSOR_PROTOCOL
    if (size == 1) cout << info << ' ' << (int)*(unsigned char*)&v << endl;
    else cout << info << ' ' << v << endl;
    #endif
}

template<typename T, int size = sizeof(T)>
inline static void get(const vector<unsigned char>& data, int& pos, T& v, const string& info = "") {
    assert(pos + size <= data.size());
    memcpy(reinterpret_cast<void *>(&v), reinterpret_cast<const void *>(&data[pos]), size);
    pos += size;
    #ifdef DEBUG_VERTEX_COMPRESSOR_PROTOCOL
    if (size == 1) cout << info << ' ' << (int)*(unsigned char*)&v << endl;
    else cout << info << ' ' << v << endl;
    #endif
}

template<typename T, int size = sizeof(T)>
inline static T get(const vector<unsigned char>& data, int& pos, const string& info = "") {
    T v = (T)0;
    get<T, size>(data, pos, v, info);
    return v;
}
inline static void addNumber(vector<unsigned char>& data, int v, const string& info) {
    #ifdef VERTEX_COMPRESSOR_OLD_VARINT
    if (v < (1 << 15))
        add<uint16_t>(data, v << 1, info);
    else add(data, (v << 1) | 1, info);
    #else
    if (v < (1 << 15))
        add<uint16_t>(data, v << 1, info);
    else if (v < (1 << 22))
        add<uint32_t, 3>(data, (v << 2) | 1, info);
    else add<uint32_t>(data, (v << 2) | 3, info);
    #endif
}

inline static void getNumber(const vector<unsigned char>& data, int& pos, int& v, const string& info = "") {
    #ifdef VERTEX_COMPRESSOR_OLD_VARINT
    if (data[pos] & 1) get(data, pos, v, info);
    else {
        uint16_t count;
        get(data, pos, count, info);
        v = count;
    }
    v = v >> 1;
    #else
    if (!(data[pos] & 1))
        v = get<uint16_t>(data, pos, info) >> 1;
    else if (!(data[pos] & 2))
        v = get<uint32_t, 3>(data, pos, info) >> 2;
    else v = get<uint32_t>(data, pos, info) >> 2;
    #endif
}

inline static int getNumber(const vector<unsigned char>& data, int& pos, const string& info = "") {
    int n;
    getNumber(data, pos, n, info);
    return n;
}

inline static void addModeCount(vector<unsigned char>& data, unsigned char mode, int count) {
    #ifdef VERTEX_COMPRESSOR_OLD_VARINT
    add(data, mode, "mode");
    add(data, count, "count");
    #else
    int v = (count << 4) | (mode & 3), size = 0;
    if (count < (1 << 4))
        add<uint8_t>(data, v, "modeCount");
    else if (count < (1 << 12))
        add<uint16_t>(data, v | (1 << 2), "modeCount");
    else if (count < (1 << 20))
        add<uint32_t, 3>(data, v | (2 << 2), "modeCount");
    else add<uint32_t>(data, v | (3<<2), "modeCount");
    #endif
}

inline static void getModeCount(const vector<unsigned char>& data, int& pos, unsigned char& mode, int& count) {
    #ifdef VERTEX_COMPRESSOR_OLD_VARINT
    get(data, pos, mode);
    get(data, pos, count);
    #else
    auto v = data[pos];
    mode = v & 3;
    auto size = (v >> 2) & 3;
    if (size == 0) count = get<uint8_t>(data, pos, "modeCount") >> 4;
    else if (size == 1) count = get<uint16_t>(data, pos, "modeCount") >> 4;
    else if (size == 2) count = get<uint32_t, 3>(data, pos, "modeCount") >> 4;
    else count = get<uint32_t>(data, pos, "modeCount") >> 4;
    #endif
}

static const unsigned char MODE_START = -1;
static const unsigned char MODE_END = -2;
static const unsigned char MODE_INSERT = 0;
static const unsigned char MODE_REPLACE = 1;
static const unsigned char MODE_KEEP = 2;
static const unsigned char MODE_REMOVE = 3;

static const unsigned int FLAG_RESET = 1;
static const unsigned int FLAG_HAS_UV = 2;
static const unsigned int FLAG_HAS_MATERIAL = 4;
static const unsigned int FLAG_REUSE_DATA = 8;

static const unsigned char DATA_NEW = 0;
static const unsigned char DATA_REUSE_NEW = 1;
static const unsigned char DATA_REUSE_PREVIOUS = 2;
static const unsigned char DATA_MATERIAL_CHANGED = 128;

/*
varInt: number (X) with variable size inspired by UTF8 coding, uses little-endian byte order:
    2B if value < 2^15: XXXXXXX0|XXXXXXXX
    3B if value < 2^22: XXXXXX01|XXXXXXXX|XXXXXXXX
    4B if value < 2^30: XXXXXX11|XXXXXXXX|XXXXXXXX|XXXXXXXX

modeVarCount: mode (M) with variable sized count (C), similar to varInt
    1B if count < 2^4:  CCCC00MM
    2B if count < 2^12: CCCC01MM|CCCCCCCC
    3B if count < 2^20: CCCC10MM|CCCCCCCC|CCCCCCCC
    4B if count < 2^28: CCCC11MM|CCCCCCCC|CCCCCCCC|CCCCCCCC

Protocol:
    unsigned int flags
    int triangleCount
    while true:
        modeVarCount mode, count
        if count == 0: break
        if mode == MODE_INSERT || mode == MODE_REPLACE:
            if FLAG_REUSE_DATA:
                for count:
                    byte type: //DATA_* for vertex 1/2/3 (bits 0-1, 2-3. 4-5) | DATA_MATERIAL_CHANGED
                    if FLAG_HAS_MATERIAL && type | DATA_MATERIAL_CHANGED: byte material
                    if FLAG_HAS_UV: byte uvType // same format as type but for uv
                    for t in 3:
                        if vertexType[t] == DATA_NEW: vec3 vertex
                        else: varInt vertexIndex
                        if FLAG_HAS_UV
                            if uvType[t] == DATA_NEW: vec2 uv
                            else: varInt uvIndex
            else:
                for count:
                    if FLAG_HAS_MATERIAL: byte material
                    for t in 3:
                        vec3 vertex
                        if FLAG_HAS_UV: vec2 uv
 */

unsigned int VertexCompressor::getFlags(const std::vector<unsigned char>& patch) {
    int pos = 0;
    unsigned int f = 0;
    get(patch, pos, f);
    return f;
}

size_t VertexCompressor::getTriangleCount(const std::vector<unsigned char>& patch) {
    int pos = sizeof(flags);
    int count = 0;
    get(patch, pos, count);
    return count;
}

const std::vector<glm::vec3>& VertexCompressor::applyPatch(const vector<unsigned char>& patch) {
    unsigned char m;
    int p = 0, src = 0, triangleCount = 0;

    get(patch, p, flags, "flags");
    get(patch, p, triangleCount, "size");
    if (flags & FLAG_RESET) reset();

    vector<vec3> prevVertices;
    vector<vec2> prevUv;
    vector<int> prevMaterial;
    swap(vertices, prevVertices);
    swap(uv, prevUv);
    swap(material, prevMaterial);

    vertices.reserve(triangleCount * 3);
    bool hasUV = flags & FLAG_HAS_UV;
    bool hasMat = flags & FLAG_HAS_MATERIAL;
    bool reuseData = flags & FLAG_REUSE_DATA;
    if (hasUV) uv.reserve(triangleCount * 3);
    if (hasMat) material.reserve(triangleCount);

    int curMat = -1;
    while (p < patch.size()) {
        getModeCount(patch, p, m, triangleCount);
        if (triangleCount == 0) break;
        if (m == MODE_REMOVE)
            src += triangleCount;
        else if (m == MODE_KEEP) {
            for (int e = src + triangleCount; src < e; src++) {
                if (hasMat)
                    material.push_back(prevMaterial[src]);
                for (int i = 0; i < 3; i++) {
                    vertices.push_back(prevVertices[src * 3 + i]);
                    if (hasUV)
                        uv.push_back(prevUv[src * 3 + i]);
                }
            }
        } else {
            if (m == MODE_REPLACE) src += triangleCount;

            if (reuseData) {
                for (int t = 0; t < triangleCount; t++) {
                    int type = get<unsigned char>(patch, p, "vt");
                    if (hasMat) {
                        if (type & DATA_MATERIAL_CHANGED)
                            curMat = get<unsigned char>(patch, p, "m");
                        material.push_back(curMat);
                    }
                    int uvType = hasUV ? get<unsigned char>(patch, p, "ut") : 0;

                    for (int i = 0; i < 3; i++) {

                        // vertex
                        int t = type & 3;
                        if (t == DATA_NEW)
                            vertices.push_back(get<vec3>(patch, p, "vd"));
                        else {
                            int loc = getNumber(patch, p, "vi");
                            if (t == DATA_REUSE_NEW)
                                vertices.push_back(vertices[loc]);
                            else // DATA_REUSE_PREVIOUS
                                vertices.push_back(prevVertices[loc]);
                        }
                        type >>= 2;

                        if (hasUV) {
                            int t = uvType & 3;
                            if (t == DATA_NEW)
                                uv.push_back(get<vec2>(patch, p, "ud"));
                            else {
                                int loc = getNumber(patch, p, "ui");
                                if (t == DATA_REUSE_NEW)
                                    uv.push_back(uv[loc]);
                                else // DATA_REUSE_PREVIOUS
                                    uv.push_back(prevUv[loc]);
                            }
                            uvType >>= 2;
                        }
                    }
                }

            } else {
                for (int t = 0; t < triangleCount; t++) {
                    if (hasMat)
                        material.push_back(get<unsigned char>(patch, p, "m"));
                    for (int i = 0; i < 3; i++) {
                        vertices.push_back(get<vec3>(patch, p, "v"));
                        if (hasUV)
                            uv.push_back(get<vec2>(patch, p, "u"));
                    }
                }
            }
        }
    }

    return vertices;
}

struct vec2Less {
    inline bool operator()(const vec2& a, const vec2& b) const {
        if (a.x != b.x) return a.x < b.x;
        return a.y < b.y;
    };
};

struct vec3Less {
    inline bool operator()(const vec3& a, const vec3& b) const {
        if (a.x != b.x) return a.x < b.x;
        if (a.y != b.y) return a.y < b.y;
        return a.z < b.z;
    };
};

struct vec2intLess {
    inline bool operator()(const pair<vec2, int>& a, const pair<vec2, int>& b) const {
        if (a.first.x != b.first.x) return a.first.x < b.first.x;
        if (a.first.y != b.first.y) return a.first.y < b.first.y;
        return a.second < b.second;
    };
};

struct vec3intLess {
    inline bool operator()(const pair<vec3, int>& a, const pair<vec3, int>& b) const {
        if (a.first.x != b.first.x) return a.first.x < b.first.x;
        if (a.first.y != b.first.y) return a.first.y < b.first.y;
        if (a.first.z != b.first.z) return a.first.z < b.first.z;
        return a.second < b.second;
    };
};

struct vec2intLessInd {
    const vector<vec2>& data;
    inline vec2intLessInd(const vector<vec2>& d) : data(d) {}
    inline bool operator()(const int& A, const int& B) const {
        auto a = data[A], b = data[B];
        if (a.x != b.x) return a.x < b.x;
        if (a.y != b.y) return a.y < b.y;
        return A < B;
    };
};

struct vec3intLessInd {
    const vector<vec3>& data;
    inline vec3intLessInd(const vector<vec3>& d) : data(d) {}
    inline bool operator()(const int& A, const int& B) const {
        auto a = data[A], b = data[B];
        if (a.x != b.x) return a.x < b.x;
        if (a.y != b.y) return a.y < b.y;
        if (a.z != b.z) return a.z < b.z;
        return A < B;
    };
};

std::vector<unsigned char> VertexCompressor::createPatch(const vector<glm::vec3>& v, const vector<glm::vec2>& u,
                                                         const vector<glm::ivec2>& t, const vector<int>& mat, int reuseData) {
    TimerCPU timer;
    timer("Init");

    auto int2Less = [](const ivec2& t1, const ivec2& t2) {
        return t1.x != t2.x ? t1.x < t2.x : t1.y < t2.y;
    };

    // flags
    flags = 0;
    if (triangles.empty() || u.empty() != uv.empty() || mat.empty() != material.empty()) {
        reset();
        flags |= FLAG_RESET;
    }
    bool hasUV = !u.empty(), hasMat = !mat.empty();
    if (hasUV) flags |= FLAG_HAS_UV;
    if (hasMat) flags |= FLAG_HAS_MATERIAL;

    // for reuseData
    vector<int> sortedVertices, sortedUv;
    vector<int> iv, iu; // 0: new; >0: previous; <0: current;

    if (reuseData > 0) {
        flags |= FLAG_REUSE_DATA;

        auto locate = [](const auto& sorted, const auto& prevSorted,
                         const auto& data, const auto& prevData,
                         auto& dst, auto less) {
            dst.resize(data.size());
            for (int i=0, j=0; i<sorted.size();) {
                auto cur = sorted[i];
                auto curData = data[cur];

                // try to find in previous data
                while (j < prevSorted.size() && less(prevData[prevSorted[j]], curData)) j++;
                int loc = j < prevSorted.size() && curData == prevData[prevSorted[j]] ? (prevSorted[j] + 1) : 0;

                // first data
                dst[cur] = loc;

                // another data with same location
                if (!loc) loc = -(cur + 1); // point to first data, if not found in previous
                while (++i<sorted.size() && data[sorted[i]] == curData) dst[sorted[i]] = loc;
            }
        };

        // for reuseData > 1
        vector<int> remap, add, addUV;

        // merge previous with new vertices (for reuseData == 2)
        auto merge = [&](auto& dst, const auto& previous, const auto& add, auto cmp) {
            dst.reserve(v.size());
            int i1 = 0, i2 = 0;
            while (i1 < previous.size() && i2 < add.size()) {
                auto loc = remap[previous[i1]];
                if (loc < 0) i1++;
                else {
                    int loc2 = add[i2];
                    if (cmp(loc, loc2)) {
                        dst.push_back(loc);
                        i1++;
                    } else {
                        dst.push_back(loc2);
                        i2++;
                    }
                }
            }
            while (i1 < previous.size()) {
                auto loc = remap[previous[i1++]];
                if (loc >= 0) dst.push_back(loc);
            }
            while (i2 < add.size())
                dst.push_back(add[i2++]);
        };

        // merge previous with new vertices and simultaneously locate (for reuseData == 3)
        auto mergeAndLocate = [&](auto& mergeDest, auto& locateDest,
                                   const auto& prevSorted, const auto& addSorted,
                                   const auto& prevData, const auto& newData, auto less) {
            if (newData.empty()) return;
            mergeDest.reserve(newData.size());
            locateDest.resize(newData.size(), 0);
            int i1 = 0, i2 = 0, lastLocate = 0;
            auto lastLocateData = newData.front();

            while (i1 < prevSorted.size() && i2 < addSorted.size()) {
                auto prevLoc = prevSorted[i1], addLoc = addSorted[i2];
                auto data = prevData[prevLoc], addData = newData[addLoc];
                bool equalData = data == addData;

                // location to previous data
                if (equalData) {
                    lastLocate = prevLoc + 1;
                    lastLocateData = prevData[prevLoc];
                }

                // merge
                auto newLoc = prevLoc >= 0 ? remap[prevLoc] : -1;
                if (equalData ? (newLoc < 0 || newLoc < addLoc) : less(data, addData)) {
                    if (newLoc >= 0)
                        mergeDest.push_back(newLoc);
                    i1++;
                } else {
                    if (equalData || (lastLocate != 0 && addData == lastLocateData))
                        locateDest[addLoc] = lastLocate;
                    else {
                        // location to new data
                        lastLocate = -(addLoc + 1);
                        lastLocateData = addData;
                    }
                    mergeDest.push_back(addLoc);
                    i2++;
                };
            }
            while (i1 < prevSorted.size()) {
                auto loc = remap[prevSorted[i1++]];
                if (loc >= 0) mergeDest.push_back(loc);
            }
            while (i2 < addSorted.size()) {
                auto addLoc = addSorted[i2++];
                auto addData = newData[addLoc];
                if (lastLocate !=0 && addData == lastLocateData)
                    locateDest[addLoc] = lastLocate;
                else {
                    lastLocate = -(addLoc + 1);
                    lastLocateData = newData[addLoc];
                }
                mergeDest.push_back(addLoc);
            }
        };

        // find all new vertices and create mapping table for previous vertices
        if (reuseData > 1) {
            timer("Diff + Remap");
            int i1 = 0, i2 = 0;
            remap.resize(vertices.size());
            while (true) {
                unsigned char m = MODE_END;
                if (i1 < triangles.size() && i2 < t.size()) {
                    auto t1 = triangles[i1], t2 = t[i2];
                    m = int2Less(t1, t2) ? MODE_REMOVE : (int2Less(t2, t1) ? MODE_INSERT : MODE_KEEP);
                } else if (i1 < triangles.size()) m = MODE_REMOVE;
                else if (i2 < t.size()) m = MODE_INSERT;
                else break;
                for (int i = 0; i < 3; i++) {
                    if (m == MODE_KEEP) remap[i1 * 3 + i] = i2 * 3 + i;
                    else if (m == MODE_REMOVE) remap[i1 * 3 + i] = -1;
                    else if (m == MODE_INSERT) add.push_back(i2 * 3 + i);
                }
                if (m == MODE_KEEP || m == MODE_REMOVE) i1++;
                if (m == MODE_KEEP || m == MODE_INSERT) i2++;
            }
            if (hasUV) addUV = add;
        }

        // vertices
        if (reuseData == 3) {
            timer("Sort");
            sortVector(add, vec3intLessInd(v));
            timer("Merge + Locate");
            mergeAndLocate(sortedVertices, iv, sorted.vertices, add, vertices, v, vec3Less());
        } else if (reuseData == 2) {
            timer("Sort");
            sortVector(add, vec3intLessInd(v));
            timer("Merge");
            merge(sortedVertices, sorted.vertices, add, vec3intLessInd(v));
            timer("Locate");
            locate(sortedVertices, sorted.vertices, v, vertices, iv, vec3Less());
        } else {
            timer("Copy");
            sortedVertices.resize(v.size());
            for (int i = 0; i < v.size(); i++) sortedVertices[i] = i;
            timer("Sort");
            sortVector(sortedVertices, vec3intLessInd(v));
            timer("Locate");
            locate(sortedVertices, sorted.vertices, v, vertices, iv, vec3Less());
        }

        // uv
        if (hasUV) {
            if (reuseData == 3) {
                timer("UV.Sort");
                sortVector(addUV, vec2intLessInd(u));
                timer("UV.Merge + Locate");
                mergeAndLocate(sortedUv, iu, sorted.uv, addUV, uv, u, vec2Less());
            } else if (reuseData == 2) {
                timer("UV.Sort");
                sortVector(addUV, vec2intLessInd(u));
                timer("UV.Merge");
                merge(sortedUv, sorted.uv, addUV, vec2intLessInd(u));
                timer("UV.Locate");
                locate(sortedUv, sorted.uv, u, uv, iu, vec2Less());
            } else {
                timer("UV.Copy");
                sortedUv.resize(u.size());
                for (int i = 0; i < u.size(); i++) sortedUv[i] = i;
                timer("UV.Sort");
                sortVector(sortedUv, vec2intLessInd(u));
                timer("UV.Locate");
                locate(sortedUv, sorted.uv, u, uv, iu, vec2Less());
            }
        }
    }

    #ifdef DEBUG_VERTEX_COMPRESSOR_STATS
    struct {
        struct Diff {
            int insert = 0, keep = 0, remove = 0, replace = 0;
        };

        struct Data {
            int data = 0, indices = 0;
        };

        int materials = 0;
        Diff triangles, mode;
        Data vertices, uv;
    } stats;
    #endif

    timer("Compress");
    std::vector<unsigned char> ret;
    add(ret, flags, "flags");
    add(ret, (int)t.size(), "size");
    unsigned char mode = MODE_START;
    int i1 = 0, i2 = 0, l = 0, prevMat = -1;
    while (true) {
        unsigned char m = MODE_END;
        if (i1 < triangles.size() && i2 < t.size()) {
            auto t1 = triangles[i1], t2 = t[i2];
            m = int2Less(t1, t2) ? MODE_REMOVE : (int2Less(t2, t1) ? MODE_INSERT : MODE_KEEP);
        } else if (i1 < triangles.size()) m = MODE_REMOVE;
        else if (i2 < t.size()) m = MODE_INSERT;

        if (mode != MODE_START && (mode != m || m == MODE_END)) {
            #ifdef DEBUG_VERTEX_COMPRESSOR_STATS
            if (mode == MODE_INSERT) stats.mode.insert++;
            else if (mode == MODE_KEEP) stats.mode.keep++;
            else if (mode == MODE_REMOVE) stats.mode.remove++;
            else if (mode == MODE_REPLACE) stats.mode.replace++;
            #endif
            if (mode == MODE_INSERT) {
                addModeCount(ret, mode, i2 - l);
                DEBUG_VCS(stats.triangles.insert += i2 - l);

                if (reuseData) {
                    for (int i = l; i < i2; i++) {

                        // types
                        int type = 0, uvType = 0;;
                        for (int j=0; j<3; j++) {
                            int curLoc = i * 3 + j;

                            // vertex
                            if (iv[curLoc]) type |= (iv[curLoc] > 0 ? DATA_REUSE_PREVIOUS : DATA_REUSE_NEW) << j * 2;

                            // uv
                            if (hasUV && iu[curLoc])
                                uvType |= (iu[curLoc] > 0 ? DATA_REUSE_PREVIOUS : DATA_REUSE_NEW) << j * 2;
                        }

                        // material
                        int curMat = -1;
                        if (hasMat && mat[i] != prevMat) {
                            curMat = mat[i];
                            type |= DATA_MATERIAL_CHANGED;
                            prevMat = curMat;
                        }

                        // add
                        add<unsigned char>(ret, type, "vt");
                        if (curMat >= 0) {
                            add<unsigned char>(ret, curMat,"m");
                            DEBUG_VCS(stats.materials++);
                        }
                        if (hasUV)
                            add<unsigned char>(ret, uvType, "ut");

                        // data
                        for (int j=0; j<3; j++) {
                            int curLoc = i * 3 + j;
                            if (!iv[curLoc]) {
                                add(ret, v[curLoc], "vd");
                                DEBUG_VCS(stats.vertices.data++);
                            } else {
                                addNumber(ret, abs(iv[curLoc]) - 1, "vi");
                                DEBUG_VCS(stats.vertices.indices++);
                            }
                            if (hasUV) {
                                if (!iu[curLoc]) {
                                    add(ret, u[curLoc], "ud");
                                    DEBUG_VCS(stats.uv.data++);
                                } else {
                                    addNumber(ret, abs(iu[curLoc]) - 1, "ui");
                                    DEBUG_VCS(stats.uv.indices++);
                                }
                            }
                        }

                    }
                } else {
                    for (int i = l; i < i2; i++) {
                        if (hasMat) {
                            add<unsigned char>(ret, mat[i], "m");
                            DEBUG_VCS(stats.materials++);
                        }
                        for (int j = 0; j < 3; j++) {
                            add(ret, v[i * 3 + j], "v");
                            DEBUG_VCS(stats.vertices.data++);
                            if (hasUV) {
                                add(ret, u[i * 3 + j], "u");
                                DEBUG_VCS(stats.uv.data++);
                            }
                        }
                    }
                }
            } else if (mode == MODE_KEEP || mode == MODE_REMOVE) {
                addModeCount(ret, mode, i1 - l);
                #ifdef DEBUG_VERTEX_COMPRESSOR_STATS
                if (mode == MODE_KEEP) stats.triangles.keep += i1 - l;
                else stats.triangles.remove += i1 - l;
                #endif
            }

            l = m == MODE_INSERT ? i2 : i1;
        }

        if (m == MODE_END) {
            addModeCount(ret, 0, 0);
            break;
        }

        if (m == MODE_KEEP || m == MODE_REMOVE) i1++;
        if (m == MODE_KEEP || m == MODE_INSERT) i2++;
        mode = m;
    }

    timer("Finish");
    triangles = t;
    vertices = v;
    uv = u;
    material = mat;
    swap(sorted.vertices, sortedVertices);
    swap(sorted.uv, sortedUv);
    timer.finish();

    #ifdef DEBUG_VERTEX_COMPRESSOR_TIMES
    for(auto e : timer.getEntries())
        cout << e.what << ' ' << e.elapsed / 1000 << endl;
    #endif

    #ifdef DEBUG_VERTEX_COMPRESSOR_STATS
    auto& S = this->stats;
    S.clear();
    S.emplace_back("Materials", stats.materials);
    S.emplace_back("Triangles.Insert", stats.triangles.insert);
    S.emplace_back("Triangles.Keep", stats.triangles.keep);
    S.emplace_back("Triangles.Remove", stats.triangles.remove);
    S.emplace_back("Triangles.Replace", stats.triangles.replace);
    S.emplace_back("Mode.Insert", stats.mode.insert);
    S.emplace_back("Mode.Keep", stats.mode.keep);
    S.emplace_back("Mode.Remove", stats.mode.remove);
    S.emplace_back("Mode.Replace", stats.mode.replace);
    S.emplace_back("Vertices.Data", stats.vertices.data);
    S.emplace_back("Vertices.Indices", stats.vertices.indices);
    S.emplace_back("UV.Data", stats.uv.data);
    S.emplace_back("UV.Indices", stats.uv.indices);
    #endif

    return ret;
}