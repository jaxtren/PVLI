#pragma once

#include <GL/glew.h>
#include "cudaHelpers.h"
#include "glmHelpers.h"
#include "types.h"
#include "Scene.h"
#include "common.h"
#include "CameraPrediction.h"
#include "TimerGL.h"
#include "TimerCUDA.h"
#include "Shader.h"

/**
 * Generates potentially visible set of triangles using rasterization (GL) and CUDA
 */
class PVS {
public:
    template<typename T> using vector = std::vector<T>;

    struct CuInstance {
        glm::mat4 transform;
        Mesh::VertexData* cuVertices; // from Mesh::cuVertices
        int offset;
    };

    struct View {
        glm::mat4 projection;
        glm::mat4 camera;
        glm::ivec2 size;
    };

    static inline glm::ivec2 minRange(int i) { return {i, std::numeric_limits<int>::max()}; }
    static inline glm::ivec2 maxRange(int i) { return {std::numeric_limits<int>::min(), i}; }
    static inline glm::ivec2 invertRange(int x, int y) {
        const int imax = std::numeric_limits<int>::max(), imin = std::numeric_limits<int>::min();
        return {y == imax ? imax : y + 1, x == imin ? imin : x - 1};
    }
    static inline glm::ivec2 invertRange(glm::ivec2 range) { return invertRange(range.x, range.y); }

    struct State {
        PVS* pvs = nullptr;

        GlCuBuffer<int> data; // data for every triangle: data[instance.firstID + triangleID] = value (value = prefix or -1 when using as mapping)
        CuBuffer<int*> indirect; // indirect access to remap: remap[instanceID][triangleID] = value

        State() = default;
        inline State(State& state) : State() { *this = state; }
        State& operator = (State& state);

        void init(PVS* pvs, bool clear = true);
        void set(int value);
        int** updateIndirect(); //map data and update cuIndirect if necessary
        void max(State& state); // data = max(data, state.data)
        void replace(State& state, glm::ivec2 range, int value); // data = value if state.data is in / outside of range
        int compact(PVS::State& state, glm::ivec2 range); //prefixes for every state.data in range, return number of prefixes
        int count(glm::ivec2 range); // count number values in range <min, max>

        struct Geometry {
            std::vector<glm::vec3> vertices;
            std::vector<glm::vec2> uv;
            std::vector<glm::ivec2> triangles;
            std::vector<int> material;
        };

        Geometry geometry(int count, bool vertices, bool uv, bool  triangles, bool material);

        // filter marked triangle based on theirs visibility:
        // camera: projection * camera transformation
        // cullface: cull none (0), back (-1) or front (1) faces
        // data as mask, bits:
        //   0 - partially inside
        //   1 - fully inside
        //   2 - facing forwards (slope > slopeLimit and not culled out)
        // e.g.:
        //   -00 - fully outside
        //   001 - partially inside, perpendicular
        //   110 - fully inside
        //   ..
        void filter(State& state, glm::ivec2 range, const glm::mat4& camera, const glm::mat4& projection, float slopeLimit, int cullface = 0);

        std::vector<unsigned char> mask(int count);
    };

private:
    glm::ivec2 frameSize = {0, 0};
    GLuint fbo = 0, tex = 0, depth = 0, vao = 0;
    gl::Shader rasterShader, markShader, rasterMarkShader;
    cudaGraphicsResource_t cuTexRes;

    bool allocFrame(glm::ivec2 size);
    void freeFrame();

    // instance state data
    CuBuffer<CuInstance> cuInstances;
    CuBuffer<int> triangleToInstance;

    CuBuffer<unsigned char> tempData;

public:
    // configuration
    int cuBlock1D = 64;
    int cuBlock2D = 8;

    struct {
        bool use = false;
        bool prepass = false;
        bool presnap = true;
        int subpixel = 0;
        glm::vec2 polygonOffset = {-1, -1};
    } conservativeRaster;

    void updateConfig(const Config& cfg);
    void provideConfig(Config);

    PVS();
    ~PVS();

    bool loadShaders();
    void setInstances(const std::vector<Scene::Instance>& instances);
    void generate(const std::vector<View>& views, State& state);

    // public state data set with setInstances()
    int allTriangleCount = 0;
    vector<Scene::Instance> instances;

    //debug
    TimerGL timerGL;
    TimerCUDA timerCUDA;

    inline void resetTimers () {
        timerGL.reset();
        timerCUDA.reset();
    }
};
