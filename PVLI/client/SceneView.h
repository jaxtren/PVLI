#pragma once

#include <GL/glew.h>
#include <string>
#include <vector>
#include "px_sched.h"
#include "common.h"
#include "types.h"
#include "glmHelpers.h"
#include "BufferCache.h"
#include "DepthPeeling.h"
#include "asioHelpers.h"
#include "Timer.h"
#include "SceneData.h"

struct Material;
class SceneTexture;

class SceneView : public SceneData {
public:

    // base
    int flags;
    int priority;
    glm::mat4 projection;
    glm::mat4 view;
    float blendFactor;
    int layerCount;
    glm::ivec2 layerSize;
    glm::ivec2 subdivide;
    glm::ivec2 subdivideOffset;
    glm::vec3 skipLayers;

    // settings
    bool extendedBlocks = false;

    struct {
        int count;
        glm::ivec2 offset;
        std::string textureID;
        SceneTexture* texture = nullptr;
    } fullLayers;

    struct {
        glm::ivec3 size;
        int layerCount = 0;
        int offset = 0;
        int tileCount = 0;
        std::string textureID;
        std::vector<unsigned char> rawBlockCounts;
        std::vector<unsigned char> rawBlockIndices;

        // block-first order: one layer with x: location, y: count;
        // layer-first order: first layer: count, other layers: location
        gl::Buffer blocksBuf;
        GLuint blocksTexture = 0;

        SceneTexture* texture = nullptr;
    } blocks;

    struct {
        std::vector<unsigned char> rawSubset; // subset of triangles to generate shading for or empty to use all triangles
        std::vector<unsigned char> rawLayerMask; // subset of triangleSubset for every layer (for depth peeling optimization) or empty, vector[6] if type == CUBEMAP

        int count = 0;
        bool hasData = false;
        gl::Buffer vertexIndices;
        std::vector<std::pair<int, int>> materialRanges; // material, vertex count

        std::vector<int> triangleLayerCount; // vertex count of triangleLayer for every layer
        gl::Buffer triangleLayer; // element buffer to scene vertices for every layer
    } triangles;

    GLuint vao = 0, depth = 0;

    struct {
        px_sched::Sync vertexBuf, triangleLayerBuf, blocksBuf, vertices,  blocks;
    } sync;

    std::string getStatsID();
    void process(SocketReader&);
    void received();
    void beforeFirstRender();
    void beforeReuse();
    void free();

    void processTriangles(Scene* pvsSource);
    void processBlocks();

    void generateBlocksTexture();
    void generateDepthLayers();

    glm::mat4 cubemapView(int face);
    glm::mat4 cubemapProjection(int face);

    //debug
    bool render = true;
    struct {
        bool renderView = true;
    } debug;
};