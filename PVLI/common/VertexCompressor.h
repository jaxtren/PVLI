#pragma once

#include <vector>
#include "glmHelpers.h"

/**
 * Vertex compression using difference (patch) from previous vertices
 */
class VertexCompressor {
public:
    unsigned int flags = 0;
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec2> uv;
    std::vector<int> material;

    // compressor only
    std::vector<glm::ivec2> triangles; //object id, triangle id
    struct {
        std::vector<int> vertices;
        std::vector<int> uv;
    } sorted;
    std::vector<std::pair<std::string, int>> stats;

    inline void reset(){
        vertices.clear();
        uv.clear();
        triangles.clear();
        material.clear();
        sorted.vertices.clear();
        sorted.uv.clear();
    }

    //decompression
    unsigned int getFlags(const std::vector<unsigned char>& patch);
    size_t getTriangleCount(const std::vector<unsigned char>& patch);
    const std::vector<glm::vec3>& applyPatch(const std::vector<unsigned char>& patch);

    std::vector<unsigned char> createPatch(const std::vector<glm::vec3>& vertices, const std::vector<glm::vec2>& uv,
                                           const std::vector<glm::ivec2>& triangles, const std::vector<int>& material, int reuseData = 0);
};
