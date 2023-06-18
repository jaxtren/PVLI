#pragma once

#include <GL/glew.h>
#include <list>
#include <algorithm>
#include "glmHelpers.h"
#include "types.h"
#include "ServerConnection.h"
#include "Scene.h"
#include "graphic.h"
#include "TimerGL.h"
#include "Pipe.h"
#include "Config.h"
#include "BlockPixelRelocator.h"
#include "Shader.h"
#include "Inpainter.h"

class Renderer {
    GLuint vao = 0;
    gl::Shader texPlot;

public:
    static const int INPAINT_PULL_IN_COMPOSITE = 1;
    static const int INPAINT_SKIP_INIT = 2;

    // settings
    glm::vec2 depthOffset = {0, 0}; // x: constant, y: normal
    glm::vec2 depthRange = {0, 0};  // x: constant, y: normal
    bool wireframe = false;
    bool cullface = true;
    bool composite = true;
    bool alphaMask = true; // use alpha from local texture to discard fragments
    bool blendEdges = true; // blend between primary view and cubemaps
    bool blendScenes = true; // blend require 2 scenes
    bool optimizePOT = true; // optimize power of two arithmetic (for old shader)
    bool optimizeCubemapAA = true; // remove rotation when cubemap is axis-aligned
    bool optimizeSharedOrigin = true; // for combineViews: some optimizations when primary view and cubemap have same origin
    bool combineViews = false; // render primary view and cubemap in one pass using new shader, doesn't work work with subdivide
    bool useOldShader = false; // use old shader if possible
    bool deferredRemote = false; // render remote lighting using screen-space reprojection instead of rasterization
    int inpaint = 2; // 0: disabled; 1: last scene; 2: both scenes (when blendScenes = true)
    int inpaintFlags = INPAINT_PULL_IN_COMPOSITE | INPAINT_SKIP_INIT;

    // default
    GLuint fallbackColor = 0, fallbackDepth = 0, fallbackBlock = 0;
    GLuint nearestSampler = 0;

    // shaders
    std::map<std::string, bool> shaderFlags;

    using ShaderParams = const std::vector<std::pair<std::string, bool>>;
    struct ShaderCacheComp {
        inline bool operator()(const std::pair<std::string, ShaderParams>& a,
                               const std::pair<std::string, ShaderParams>& b) const {
            auto cmp = a.first.compare(b.first);
            if (cmp != 0) return cmp < 0;
            return std::lexicographical_compare(a.second.begin(), a.second.end(),
                                                b.second.begin(), b.second.end());
        }
    };

    std::map<std::pair<std::string, ShaderParams>, gl::Shader, ShaderCacheComp> shaders;
    gl::Shader& getShader(const std::string& name, const ShaderParams&);

    // FBO
    glm::ivec2 size = {0, 0};
    GLuint fbo = 0, depth = 0, localColor = 0, remoteColor[2] = {0, 0}, normals = 0;
    GLuint remoteColorLayer2[2] = {0, 0};
    void initFBO(const glm::ivec2& size);
    void destroyFBO();

    Inpainter inpainter;

    struct {
        gl::Shader cubeWireframe, circle, simpleShader;
        bool render = false;
        bool sphereAxis = false;
        float viewScale = 0.5f;
        GLuint linesVAO = 0, linesBuffer = 0;
    } debug;

    struct {
        bool render = false;
        glm::mat4 view;
    } debugLocal;

    void renderDebug(const std::list<Scene*>& scenes, const glm::mat4& projection, const glm::mat4& view);

    inline bool requireAdditionalDepthLayer () {
        return shaderFlags["CLOSEST_DEPTH"] && !shaderFlags["CLOSEST_DEPTH_RANGE"];
    }

    Renderer();
    ~Renderer();
    void updateConfig(const Config&);
    bool loadShaders();
    void render(const glm::ivec2& frameSize, const Viewpoint& view, const std::list<ServerConnection>& connections, bool synchronized);
    void renderTexture(const glm::ivec2& frameSize, GLuint texture);
    void GUI();

    TimerGL timer;
};
