#pragma once

#include <GL/glew.h>
#include <vector>
#include "cudaHelpers.h"
#include "glmHelpers.h"
#include "types.h"
#include "Scene.h"
#include "common.h"
#include "PVS.h"
#include "CameraPrediction.h"
#include "Config.h"
#include "structs.h"
#include "Shader.h"

struct PackConfig {
    int fullLayers = 1;
    int maxLayers = 100;
    int fragmentCountThreshold = 0;
    bool relocatePixels = false;
    bool layerFirstBlockOrder = false;
    bool triangleLayerMask = false;

    bool updateConfig(const Config& cfg);
    void provideConfig(Config cfg);
};

class LinkedDepthLV {
public:
    struct Fragment {
        glm::u8vec4 color;
        float depth;
        int instID;
        int primID;
        int skip;
        int next;

        __host__ __device__ inline glm::u8vec3 getColor() {
            return glm::max(glm::u8vec3(1), glm::u8vec3(color)); //FIXME black is empty
        }

        __host__ __device__ inline void setColor(const glm::u8vec4& c) {
            color = glm::u8vec4(glm::max(glm::u8vec3(1), glm::u8vec3(c)), c.w); //FIXME black is empty
        }
    };

private:
    Scene* scene;

    void allocFrame(glm::ivec2 size, int fragmentCount);
    void freeFrame();

    int maxFragmentCount = 0;
    glm::ivec2 maxFrameSize = {0, 0};
    GLuint fbo = 0, headTex = 0, dataBuf = 0;

    cudaGraphicsResource_t cuHeadRes = nullptr, cuDataRes = nullptr;
    cudaSurfaceObject_t cuHeadTex = 0;
    int* cuCount = nullptr;
    Fragment* cuFragments = nullptr;
    void cuMap();
    void cuUnmap();

    // temporary data
    CuBuffer<int> layerCount;
    CuBuffer<int> blocksPrefixes;

    // render program
    GLuint vao = 0;
    gl::Shader program;

public:

    //configuration
    unsigned cudaBlock = 256;
    unsigned cudaBlock2D = 16;

    LinkedDepthLV(const Config&, Scene*);
    ~LinkedDepthLV();

    void updateConfig(const Config&);
    void provideConfig(Config cfg);
    bool loadShaders();

    struct ViewContext {
        glm::mat4 globalToView = glm::mat4(1), viewToGlobal = glm::mat4(1);
        glm::vec3 globalViewOrigin = glm::vec3(0);
        float spreadDistance = 0;
        glm::ivec3 size = glm::ivec3(0); // width, height, layer count without blocks

        CuBuffer<glm::vec3> pos;
        CuBuffer<glm::vec3> prevPos; // debug
        CuBuffer<int> indices; // indices to blocks texture

        void setViewProjection(const glm::mat4& view, const glm::mat4& projection);
        void reset();
    };

    struct PackOutput {
        glm::ivec2 layerSize = {0, 0}, layerOffset = {0, 0}, blocksSize = {0, 0};
        PackConfig config;
        int layerCount = 0;
        int tileOffset = 0, tileCount = 0 /* without full layers */; // in blocks texture
        int fragmentCount = 0;
        CuBuffer<unsigned char> triangleLayerMask;
        CuBuffer<int> blocks; // tile count per block without full layers

        // for tracking
        int reprojectedTileCount = 0;
        CuBuffer<int> tileIndices; // index for tile outside of full layers to blocks texture
    } output;

    struct FullLayers {
        glm::ivec2 size;
        CuBuffer<glm::u8vec3> color;
        CuBuffer<unsigned char> mask;
        bool hasMask = false;

        void init(const glm::ivec2& size);
    } fullLayers;

    struct Blocks {
        friend class LinkedDepthLV;
    protected:
        int extend(int count);
        void clearEmptyBlocks();
    public:
        int blockSize = 8;
        int tileCount = 0; // occupied tile count
        glm::ivec2 size;
        CuBuffer<glm::u8vec3> color;
        CuBuffer<unsigned char> mask;
        bool hasMask = false;
        CuBuffer<char> occupied; // for every block

        // track and reprojected blocks from previous views
        struct Track {
            bool use = false;
            int lastAllocated = 0; // index of last allocated block

            // config
            int reservationAttempts = 1;
            int mode = 0;
            bool cycle = 0;
            float reprojMaxDistFactor = 1;
            bool debug = false;

            bool updateConfig(const Config& cfg);
            void provideConfig(Config cfg);
        } track;

        int tileCapacity() const {
            return size.x * size.y / (blockSize * blockSize);
        }

        void init(int width, int block, bool track = false);
        void nextFrame() {
            tileCount = 0;
        }
    } blocks;

    void render(const glm::ivec2& size, const glm::mat4& projection, const glm::mat4& view,
                const std::vector<Scene::Instance>& instances, GLuint mask, int fragmentCount); //internal renderer, TODO move outside class

    void setData(const glm::ivec2& size, LinkedFragment* cuExtFragments, int extFragmentCount,
                 float toneMapping, float expToneMapping, float gamma,
                 glm::vec2 colorMapping, bool fullFirstLayer = false);

    // pack data set from setData() or render()
    void pack(const glm::ivec2& srcOffset, const glm::ivec2& dstOffset, const glm::ivec2& size, const PackConfig&,
              int triangleCount, bool sort = false, int** cuTriangleLayerMap = nullptr,
              ViewContext* ctx = nullptr, ViewContext* prevCtx = nullptr);

    void finalizeBlocks(int minHeight = 0, int dilateSize = -1);
    void packFirstLayer(bool flipY = false);

    // debug
    TimerGL timerGL;
    TimerCUDA timerCUDA;

    inline void resetTimers () {
        timerGL.reset();
        timerCUDA.reset();
    }
};
