#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>
#include <vector>
#include <list>
#include <queue>
#include <thread>
#include <atomic>
#include <iostream>
#include <functional>
#include "asioHelpers.h"
#include "glmHelpers.h"
#include "common.h"
#include "graphic.h"
#include "FreeView.h"
#include "Pipe.h"
#include "Renderer.h"
#include "Config.h"
#include "CameraPath.h"
#include "StatCounter.h"
#include "ByteRateCounter.h"
#include "AsyncWriter.h"
#include "px_sched.h"
#include "VertexCompressor.h"
#include "VideoCodingUtil.h"
#include "ffmpegDecoder.h"
#include "DepthPeeling.h"
#include "Scene.h"
#include "TextureManager.h"
#include "ServerConnection.h"
#include "Benchmark.h"

#ifdef ENABLE_TURBOJPEG
#include <turbojpeg.h>
#endif

enum GPUTask : int {
    OTHER = 0, COMPUTE = 1, UPLOAD = 2
};

class Application {
public:
    using ConfigParameter = std::pair<std::string, std::string>;

    TimerCPU::TimePoint frameTime, startTime;
    bool vsync = true;
    bool showGUI = true;
    bool TCPNoDelay = true;
    #ifdef NDEBUG
    bool glDebugOutput = false;
    #else
    bool glDebugOutput = true;
    #endif
    bool glDebugOutputSynchronous = true;
    float frameDeltaTime = 0;
    float frameGPUtimeWithoutTasks = 0;
    float textureScale = 1;
    int frameID = 0;
    int port = 8080;

    std::string configFile = "config.cfg";
    std::string outputPath;

    // connections
    bool syncSceneUpdate = false;
    bool asyncInterleave = true;
    int viewpointID = 1;
    std::string syncSourceForPVS;
    std::list<ServerConnection> connections;

    template<typename T>
    inline void requestAll(const T& v) {
        for (auto& c : connections)
            c.request.write(v);
    }

    void allocateSynchronizedScenes(const std::vector<Viewpoint>&);
    void sendViewpoint(const glm::mat4& transform, bool allowUpdate);

    // scene update
    int framesFromUpdate = 0;
    TimerCPU::TimePoint prevUpdateTime;
    Viewpoint lastUpdateViewpoint;
    void onSceneUpdate(Scene*);

    inline void resetTime() {
        startTime = frameTime = prevUpdateTime = TimerCPU::now();
        lastUpdateViewpoint.time = 0;
        framesFromUpdate = 0;
    }
    inline double elapsedTime () {
        return TimerCPU::diff(startTime, frameTime) / 1000000;
    }

    inline std::string realOutputPath(){
        return outputPath.empty() ? "./" : (outputPath.back() == '/' ? outputPath : outputPath + '/');
    }

    void updateConfig(bool first = false);

	// configuration presets
    std::string presetFile = "preset.cfg";
    pt::ptree serverConfigPresets;

    // scene settings
    bool render = true;
    bool updateScene = true;
    bool updateViewpoint = true;
    bool relocatePixels = true;
    int maxCachedScenes = 0;
    int maxProcessingScenes = 0;
    int maxReuseScenes = 0; // limit for reuseScenes
    bool reuseTextures = false; // cache and reuse also textures
    int uploadChunkSize = 0; // number of pixels for texture upload at once (<=0 - unlimited)
    double bufferSizeMultiple = 1;
    bool useStagingBuffers = false;

    const std::map<std::string, GLenum> bufferUsages = {
        {"STATIC_DRAW", GL_STATIC_DRAW},
        {"DYNAMIC_DRAW", GL_DYNAMIC_DRAW},
        {"STREAM_DRAW", GL_STREAM_DRAW},
        {"STATIC_COPY", GL_STATIC_COPY},
        {"DYNAMIC_COPY", GL_DYNAMIC_COPY},
        {"STREAM_COPY", GL_STREAM_COPY},
    };
    GLenum bufferUsage = GL_STREAM_DRAW;
    GLenum stagingBufferUsage = GL_STREAM_DRAW;

    inline bool allowRender() {
        return render && !clientSceneRecord.running() && !connections.empty();
    }

    inline bool allowViewpointUpdate() {
        bool allow = updateViewpoint && !clientSceneRecord.running() &&
            !serverSceneRecord.running() && !benchmark.simulated.isRunning();
        for (auto& c : connections)
            allow = allow && c.connected;
        return allow;
    }

    FreeView view;
    Renderer renderer;
    BlockPixelRelocator pixelRelocator;
    px_sched::Scheduler scheduler;
    TextureManager textureManager;
    bool prevMouseButton3 = false;

#ifdef ENABLE_TURBOJPEG
    mutexed<tjhandle> jpegDecoder;
#endif

    // for depth peeling
    struct {
        DepthPeeling peeling;
        float epsilon = 0;
        bool useTriangleLayerMask = true;
    } depth;

    // gpu async processing
    struct GPU {
        using Task = std::pair<std::function<void()>, double>;

        // all times are in ms
        std::queue<std::pair<GLuint,GLuint>> queries;
        Pipe<Task> tasks, upload, compute;
        TimerCPU::TimePoint prevUpdate;
        StatCounter<double> elapsed, elapsedLeft, deltaTime;

        // config
        int maxTasks = 0;
        double maxUpload = 0; // "virtual" time
        double maxCompute = 0; //size in kB
        int queryDelay = 0;
        double elapsedMax = 0;
        double elapsedAdaptive = 0;

        // stats
        struct {
            double elapsedMax = 0;
            int tasks = 0;
            double compute = 0;
            double upload = 0;
        } stats;

        double nextElapsed();
        void processTasks();
        void processInstantTasks();
    } gpu;

    template<typename Job>
    inline void run(GPUTask type, Job&& job, double t = 1) {
        if (type == GPUTask::COMPUTE) gpu.compute.send({std::forward<Job>(job), t});
        else if (type == GPUTask::UPLOAD) gpu.upload.send({std::forward<Job>(job), t});
        else gpu.tasks.send({std::forward<Job>(job), t});
    }

    template<typename Job>
    inline void run(GPUTask type, Job&& job, px_sched::Sync *s, double t = 1) {
        if (!s) run(type, std::forward<Job>(job));
        else {
            scheduler.incrementSync(s);
            run(type, [this, s, job = std::forward<Job>(job)]() mutable {
                job();
                scheduler.decrementSync(s);
            }, t);
        }
    }

    // tasks on main thread
    using Task = std::function<void()>;
    Pipe<Task> tasks;
    template<typename Job>
    inline void run(Job&& job) {
        tasks.send(std::forward<Job>(job));
    }
    inline void processTasks()  {
        Task task;
        while (tasks.receive(task, false)) task();
    }

    // server config
    bool syncServerConfigRequest = true;

    inline void requestServerConfig(ServerConnection* c = nullptr) {
        if (c && !syncServerConfigRequest) {
            c->request.write(RequestType::SET_SETTINGS);
            c->request.write(pt::to_string_info(c->config));
            c->request.write(RequestType::GET_ALL_SETTINGS); // read back server config
        } else
            for (auto& c : connections) {
                c.request.write(RequestType::SET_SETTINGS);
                c.request.write(pt::to_string_info(c.config));
                c.request.write(RequestType::GET_ALL_SETTINGS); // read back server config
            }
    }

    template<typename T>
    inline void setServerConfig(const std::string& k, const T& v, ServerConnection* c = nullptr) {
        if (c && !syncServerConfigRequest)
            Config(c->config).set(k, v);
        else for (auto& c : connections)
                Config(c.config).set(k, v);
    }

    inline void setServerConfig(const std::string& k, const pt::ptree& t, ServerConnection* c = nullptr) {
        if (c && !syncServerConfigRequest) {
            auto ct = c->config.get_child_optional(k);
            if (ct) pt::merge(*ct, t);
            else c->config.put_child(k, t);
        } else
            for (auto& c : connections) {
                auto ct = c.config.get_child_optional(k);
                if (ct) pt::merge(*ct, t);
                else c.config.put_child(k, t);
            }
    }

    // window
    GLFWwindow* window = nullptr;
    glm::vec2 cursorPos;
    glm::ivec2 frameSize;
    int unfocusedWindowSleep = 0;
    void framebufferSizeChanged(int width, int height);
    void keyCallback(int key, int scancode, int action, int mods);

    struct {
        double multiplier = 1, minRate = 0, maxRate = 0;
        double overrideLatencyInMs = -1;
    } updatePrediction;

    struct {
        bool minMax = false;
        StatCounters<double> frame;
        StatCounters<double> update;
    } counters;

    // adaptive video settings
    struct {
        struct {
            bool available = false;
            bool primCubeM = false;
            bool blocks = false;
            bool inUsePrimCubeM = false;
            bool inUseBlocks = false;
        } hwaccel;
        struct {
            int bitrate = -1; // <0: original, 0: adaptive, >0: constant;
            int min = 0;
            float timeFactor = 1.2f;
            float timeOffsetInMs = 0;
            int stepUp = 1000000;
            int stepDown = 1000000;
        } bitrate;
        struct {
            float factor = 1; // <=0: original, >0: multiplication of average framerate
            float offset = 0;
            int min = 0;
        } framerate;
    } video;

    std::shared_ptr<hw_accel::Interop> hwAccelInterop = nullptr;

    struct {
        pt::ptree frame;
        pt::ptree update;
    } stats;

    struct ViewpointRecord {
        enum Type : int {
            RENDER_FRAME = 0,
            SEND_VIEWPOINT,
            UPDATE_SCENE,
            UPDATE_SCENE_SYNC
        };

        Type type;
        Viewpoint viewpoint;
        std::string server;

        bool isUpdate() const { return type == UPDATE_SCENE || type == UPDATE_SCENE_SYNC; };
    };

    struct SceneRecord {
        int index = 0;
        std::vector<ViewpointRecord> data;
        std::string file;
        bool load(const std::string&);
        bool save(const std::string&);

        // for client/server scene record
        static const int STOP = -2;
        static const int START = -1;
    } sceneRecord;

    // client scene record
    struct {
        std::string file = "#.ppm";
        std::string fileThirdPerson = "#.ppm";
        bool correctionTest = false;
        int index = SceneRecord::STOP;
        int targetIndex = -1;
        inline bool running() { return index >= SceneRecord::START; }
    } clientSceneRecord;

    inline void clientSceneRecordStart(int targetIndex = -1) {
        clientSceneRecord.index = SceneRecord::START;
        clientSceneRecord.targetIndex = targetIndex;
        requestAll(RequestType::RESET_STATE);
        run([this]() { clientSceneRecordProcess(); });
    }
    inline void clientSceneRecordStop() {
        clientSceneRecord.index = SceneRecord::STOP;
        clientSceneRecord.targetIndex = -1;
        if (benchmark.automation.use) {
            if (benchmark.automation.isInState(Benchmark::Automation::State::eSaveImgClientRunning))
                benchmark.automation.state = Benchmark::Automation::State::eSaveImgClientThirdPersonStart;
            if (benchmark.automation.isInState(Benchmark::Automation::State::eSaveImgClientThirdPersonRunning))
                benchmark.automation.state = Benchmark::Automation::State::eSaveImgServerStart;
        }

    }
    void clientSceneRecordProcess(const Viewpoint& viewpoint = {}, ServerConnection* connection = nullptr);
    void clientSceneRecordSaveImage(int index);

    // server scene record
    struct {
        std::string file = "#.ppm";
        int index = SceneRecord::STOP;
        std::string server;
        inline bool running() { return index >= SceneRecord::START; }
    } serverSceneRecord;

    void serverSceneRecordStart() {
        serverSceneRecord.index = SceneRecord::START;
        requestAll(RequestType::RESET_STATE);
        run([this]() { serverSceneRecordProcess(); });
    }
    void serverSceneRecordStop() {
        serverSceneRecord.index = SceneRecord::STOP;
        if (benchmark.automation.use && benchmark.automation.isInState(Benchmark::Automation::State::eSaveImgServerRunning))
            benchmark.automation.state = Benchmark::Automation::State::eExit;
    }
    std::pair<float, glm::mat4> getReferenceFrameData(int i) const;
    void serverSceneRecordProcess(float time = 0, glm::mat4 view = {}, const std::vector<unsigned char>& data = {});

    Benchmark benchmark = {this};

    struct {
        std::string file;
        bool play = false;
        bool record = false;
        bool updateOnly = false;
        float time = 0;
        float start = 0;
        float stop = 0;
        float speed = 1;
        CameraPath path;
        inline float realStop() {
            return stop > 0 ? stop : path.duration();
        }

        inline void step(float frameDeltaTime) {
            if (play || record)
                time += frameDeltaTime * speed;
        }

        bool inRange() {
            return time >= start && time < realStop();
        }

        void cycle() {
            if (time > realStop()) time = start;
            else if (time < start) time = realStop();
        }

        void clamp() {
            if (time < start) time = start;
            else if (time > realStop()) time = realStop();
        }

        glm::mat4 view() {
            return inverse(path.sample(time).mat());
        }

    } camera;

    // benchmark path
    bool benchmarkPath = false;

    // saved camera locations
    std::vector<FreeView::Location> cameraLocations;

    struct {
        pt::ptree state;
        std::string stateFile = "gui.ini";
    } gui;

    Application(GLFWwindow*, const std::string& config = "");
    void run();
    void GUI();
};
