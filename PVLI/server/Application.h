#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <set>
#include <atomic>
#include <GL/glew.h>
#include "cudaHelpers.h"
#include "glmHelpers.h"
#include "asioHelpers.h"
#include "px_sched.h"
#include "common.h"
#include "Config.h"
#include "types.h"
#include "Scene.h"
#include "PVS.h"
#include "PVScache.h"
#include "LinkedDepthLV.h"
#include "JpegEncoder.h"
#include "VertexCompressor.h"
#include "GLbackend.h"
#include "Pipe.h"
#include "Timer.h"
#include "TimerGL.h"
#include "TimerCUDA.h"
#include "StatCounter.h"
#include "AsyncWriter.h"
#include "compression/Compression.h"
#include "Lighthouse.h"
#include "structs.h"
#include "SceneState.h"
#include "VideoCodingUtil.h"

class Application {
public:
    std::unique_ptr<GLbackend> backend;
    ConfigMapper cfgMap;
    std::string configFile = "config.cfg";
    std::string sceneFile;
    int port = 8080;
    glm::ivec2 frameSize = {1000, 600};
    glm::vec3 projParams = {60, 0.1, 1000};
    int fragmentsMultiple = 5;
    bool TCPNoDelay = true;
    bool serverOnly3D = false;
    bool sendAllVertices = false;

    std::unique_ptr<VideoEncoderBase> videoEncoderFactory(const VideoEncoderConfig& c, int w, int h);

    struct ViewConfig {
        bool use = true;
        int cullface = 0; // 0: disable, <0: cull back, >0: cull front
        float slopeLimit = 0;
        float skipLayers = 0;
        glm::ivec2 subdivide = {1, 1};
        glm::ivec2 subdivideOffset = {0, 0};
        bool subdivideChecker = false;
        PackConfig pack;
        LighthouseConfig lighthouse;
        VideoEncoderConfig video; // config provided from the client
        VideoEncoderConfig currentVideo; // real config, derived from video with optional adaptive settings
        bool updateConfig(const Config& cfg);
        void provideConfig(Config cfg);
        bool isSubdivided () const { return subdivide.x > 1 || subdivide.y > 1; }
    };

    struct {
        ViewConfig config;
        LinkedDepthLV::ViewContext packContext, prevPackContext; // TODO move to state
        float fovExpand = 1;
        float frameSizeMultiple = 1;
        float pvsSizeMultiple = 1;
        float blendFactor = 0.1f;
    } primaryView;

    struct {
        ViewConfig config;
        glm::vec3 offset = {1, 1, 0};
    } auxiliaryViews;

    struct {
        ViewConfig config;
        bool rotate = true;
        bool mergeFullLayers = true; // merge full layers of all views to one texture
        bool mergeViews = true; // create cubemap view if possible (ViewFlag::CUBEMAP) instead of 6 perspective views
        bool mergeRender = true;  // lighthouse only: render and filter whole cubemap at once
        bool enableFallback = false; // lighthouse only: use cubemap as reprojection fallback for primary view (requires mergeViews)
        bool renderAll = true; // render all or only triangles not fully inside primary view
        int size = 512;
        float pvsSizeMultiple = 2;
    } cubemap;

    struct {
        VideoEncoderConfig video, currentVideo;
        int blockSize = 16;
        int textureWidth = 2048;
        int dilateSize = -1;
        int minTextureHeight = 64;
        int textureHeightStep = 0;
        bool track = false;
    } blocks;

    struct {
        float avgBitrateFactor = 1;
        float maxBitrateFactor = 1;

        int ratioMinPrimary = 6;
        int ratioMinCubemap = 2;
        int ratioMinBlocks = 32;
        int ratioMinAux = 0;
        int ratioMaxPrimary = 10;
        int ratioMaxCubemap = 3;
        int ratioMaxAux = 0;
    } video;

    ViewConfig referenceViewConfig;

    struct {
        bool use = false;
        bool illuminationOnly = false;
        float toneMapping = 0; // FIXME incorrect with illuminationOnly
        float expToneMapping = 0; // FIXME incorrect with illuminationOnly
        float gamma = 1;
        glm::vec2 colorMapping = {10, 5}; // for illuminationOnly
        std::string backend;
        Lighthouse* renderer = nullptr;
        LighthouseConfig config;
        CuBuffer<LinkedFragment> fragments;

        inline void allocate(int size) {
            if (fragments.size < size)
                fragments.alloc(size);
        }
    } lighthouse;

    struct {
        bool rle = false;
        int qt = 0;
        int qtParts = 1;
        int reuseData = 1;
        Compression::Method entropy = Compression::Method::NONE;

        inline std::vector<unsigned char> compress(Compression::Method rleMethod, ConstBuffer src) {
            return Compression::compress(entropy, rle ? rleMethod : Compression::Method::NONE, src);
        }
    } compression;

    struct {
        bool send = true;
        bool minMax = false;
        int maxSamples = 0;
        float EMA = 0;
        std::string prefix;
        StatCounters<double> update, benchmark; // accessed only inside response.call()

        inline std::string realPrefix(){
            return !prefix.empty() && prefix.back() != '.' ? prefix + '.' : prefix;
        }
    } stats;

    struct {
        bool running = false;
    } benchmark;

    struct {
        bool send = false;
        #ifdef NDEBUG
        bool glDebugOutput = false;
        #else
        bool glDebugOutput = true;
        #endif
        bool glDebugOutputSynchronous = true;
    } debug;

    Scene scene;
#ifdef ENABLE_NVJPEG
    JpegEncoder jpegEncoder;
#endif

    // scene state data
    std::shared_ptr<SceneState> state, previousState;
    bool hasAllVertices = false;
    CameraPrediction cameraPrediction;
    std::unique_ptr<PVS> pvs;
    int pvsTriangleCount = 0;
    PVScache pvsCache;
    std::unique_ptr<LinkedDepthLV> lv;
    PVS::State renderState, allState, directState, cubeState, cubeViewState[6];
    PVS::State allMapping, tempMapping;
    PVS::State auxLeft, auxState;

    // request state
    Viewpoint lastViewpoint;
    bool allowUpdate = true;

    /* Notations:
        dynamic range value: x,y,z =  min, max, weight (e.g. size = max(x, min(y, speed * duration * z)))
        prediction weight: value is computed at predicted time = duration * weight
     */

    struct {
        float extrapolation = 1;
        float primaryView = 0; // prediction weight for primary view camera
    } prediction;

    struct PVSConfig {
        int predictionSamples = 3;
        glm::vec3 radius = {0, 100, 1}; // dynamic range value
        float center = 0; // prediction weight for center
        float corners = 1; // distance scale for corners

        struct View {
            // raster mode for views
            static const int DIRECT = 1;
            static const int CUBEMAP = 2;
            int primary = DIRECT | CUBEMAP;
            int prediction = CUBEMAP;
            int corners = DIRECT;
        } view;

        struct Cache {
            int size = 100;
            int predictionSamples = 15;

            struct Radius {
                // all variables are dynamic range values
                glm::vec3 cache = {10, 10, 1};
                glm::vec3 merge = {0.1, 0.1, 1};;
                glm::vec3 render = {0, 0, 0.5};
            } radius;

        } cache;

    } pvsConfig;

    void renderScene(const Viewpoint&);

    /**
     * same as renderScene() but renders only on server, doesn't update PVS and doesn't send data using UPDATE_SCENE
     * used for simulated benchmark to get higher quality of filter in lighthouse
     */
    void renderSceneServerOnly(const Viewpoint&);

    enum RenderFlags : unsigned int {
        CUBEMAP = 1,
        CLASSIC_MODE = 8,
        DONT_SAVE_STATS = 16
    };

    void renderView(const glm::mat4& camera, glm::vec3 projParams, glm::ivec2 projSize, PVS::State& mask,
                    const ViewConfig& cfg, int id, int fallback, unsigned flags, const string& name);

    void packView(const glm::ivec2& srcOffset, const glm::ivec2& dstOffset, const glm::ivec2& size,
                  const ViewConfig& cfg, const Viewpoint& viewpoint, PVS::State& subset,
                  const glm::mat4& camera, glm::vec3 projParams, float blend, int flags, int priority,
                  const string& viewName, const string& fullLayersName, const string& blocksName,
                  LinkedDepthLV::ViewContext* packContext = nullptr,
                  LinkedDepthLV::ViewContext* prevPackContext = nullptr);

    void renderReferenceScene(const Viewpoint&);
    void resetState(bool resetHasAllVertices = true);
    void resetBlocksState();

    // threading
    Pipe<std::function<void()>> requests;
    AsyncWriter response;
    std::atomic_int sendingScenes = 0;
    px_sched::Scheduler scheduler;

    std::set<std::string> sceneExtensions = { "gltf", "lights", "aao" };
    void requestListener(boost::asio::ip::tcp::socket& socket);
    void requestHandler();
    void sceneList();

    // config management
    void loadConfig(bool first, const std::string& cfg = "");
    void updateConfig(const Config& cfg, bool first = false);
    void getCurrentConfig(Config& cfg);

    Application(const std::string& config = "");
    ~Application();

    void run();
};
