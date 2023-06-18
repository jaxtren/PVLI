#pragma once

#include <vector>
#include <list>
#include <GL/glew.h>
#include "px_sched.h"
#include "common.h"
#include "types.h"
#include "glmHelpers.h"
#include "BufferCache.h"
#include "Timer.h"
#include "TimerGL.h"
#include "asioHelpers.h"
#include "SceneView.h"
#include "SceneTexture.h"
#include "StatCounter.h"
#include "VertexCompressor.h"
#include "compression/Compression.h"
#include "TextureManager.h"

class Application;
class ServerConnection;
enum GPUTask : int;

struct Material {
    std::string textureName;
    glm::vec3 color;
    bool opaque = true;
    Texture::Ref texture;
};

class Scene {
public:
    Scene(ServerConnection*, const Viewpoint&);

    struct Vertex { glm::vec3 pos; glm::u8vec4 normal; };
    struct VertexUV { glm::vec3 pos; glm::u8vec4 normal; glm::vec2 uv; };

    ServerConnection* connection = nullptr;
    Application* app = nullptr;
    VertexCompressor vertexCompressor;

    /**
     * previous scene
     * notes:
     *  can be empty, if current scene is first
     *  doesn't have to be fully decoded before processing of current scene (use sync objects to ensure correct order)
     */
    Scene* previous = nullptr;

    /**
     * scene for reusing GL buffers/textures (can be empty)
     */
    Scene* reuse = nullptr;

    /*
     * looped linked list of siblings from other connections for synchronized multi-server updates
     */
    Scene* sibling = nullptr;

    // check if all synchronized scenes are ready
    bool allReady () {
        if (!sibling) return ready;
        for (auto s = sibling; s != this; s = s->sibling)
            if (!s->ready) return false;
        return ready;
    }

    bool isSynchronized() { return sibling != nullptr; }
    bool isVideoStream() { return !views.empty() && views.front().flags & ViewFlag::VIDEO_STREAM; }

    Viewpoint viewpoint;
    glm::vec3 projParams;
    glm::ivec2 frameSize = { 1000, 600 };
    int vertexCount = 0;
    bool oneTimeVertices = false;
    bool benchmark = false;
    bool ready = false; // when is in scene list and ready to render
    float gamma = 1;
    glm::vec2 colorMapping;
    Material background;
    std::vector<Material> materials;

    // PVS data, synchronized scenes can share PVS: one scene has data, others use DONT_SEND_PVS and have no data
    // use findSourceForPVS() to get scene with PVS (can be this scene)
    bool hasUV = false;
    GLuint vao = 0;
    std::shared_ptr<gl::Buffer> vertices;
    std::vector<int> materialPerTriangle;
    std::vector<std::pair<int, int>> materialRanges; // material, vertex count

    inline bool hasPVS() const { return !(viewpoint.flags & Viewpoint::DONT_SEND_PVS); }

    inline Scene* findSourceForPVS() {
        if (hasPVS() || !isSynchronized()) return this;
        for (auto s = sibling; s != this; s = s->sibling)
            if (hasPVS()) return s;
        return this;
    }

    std::list<SceneView> views;
    std::list<SceneTexture> textures;

    inline SceneView* findView(const std::string& n) {
        if (n.empty()) return nullptr;
        for (auto& v : views)
            if (v.name == n) return &v;
        return nullptr;
    }

    inline SceneTexture* findTexture(const std::string& n) {
        if (n.empty()) return nullptr;
        for (auto& t : textures)
            if (t.name == n) return &t;
        return nullptr;
    }

    struct {
        px_sched::Sync vertexBuf, vertices, finish;
    } sync;

    struct {
        std::vector<unsigned char> vertices;
    } raw;

    // monitoring / timers / stats
    TimerCPU::TimePoint receiveStarted, receiveFinished;
    size_t receivedBytes = 0, receivedVideoBytes = 0;
    float serverProcessingTime = 0;

    struct {
        ParallelStopwatch cpu;
        ParallelStopwatch gpu;
    } stopwatch;

    struct {
        TimerCPU cpu;
        struct {
            TimerGL tasks;
            float frames = 0;
            int lastFrameID = -1;
            int usedFrames = 0;
            int skippedFrames = 0;
        } gpu;
    } timer;

    struct {
        // all sizes are in kB and times in ms
        std::string server;
        StatCounters<double>::Entries local, size;
        mutexed<StatCounters<double>::Entries> async;

        inline void addSize(const std::string& n, const std::vector<unsigned char>& v, int uncompressed = -1) {
            size.emplace_back("Size.Compressed." + n, (double) v.size()  * 0.001);
            if(uncompressed < 0) // detect size
                size.emplace_back("Size.Uncompressed." + n, (double) Compression::header(v).size * 0.001);
            else if(uncompressed > 0) // provided size
                size.emplace_back("Size.Uncompressed." + n, (double) uncompressed * 0.001);
            else // use same size
                size.emplace_back("Size.Uncompressed." + n, (double) v.size() * 0.001);
        }

        inline void addSize(const std::string& n, const std::vector<std::vector<unsigned char>>& v, int uncompressed = -1) {
            size_t compressedSize = 0;
            int uncompressedSize = 0;
            for (const auto& p : v) {
                compressedSize += p.size();
                if (uncompressed < 0)
                    uncompressedSize += Compression::header(p).size;
            }
            size.emplace_back("Size.Compressed." + n, (double) compressedSize  * 0.001);
            if(uncompressed < 0) // detect size
                size.emplace_back("Size.Uncompressed." + n, (double) uncompressedSize * 0.001);
            else if(uncompressed > 0) // provided size
                size.emplace_back("Size.Uncompressed." + n, (double) uncompressed * 0.001);
            else // use same size
                size.emplace_back("Size.Uncompressed." + n, (double) compressedSize * 0.001);
        }

        inline void addSize(const std::string& n, const std::string& v) {
            size.emplace_back("Size.Compressed." + n, (double) v.size() * 0.001);
            size.emplace_back("Size.Uncompressed." + n, (double) v.size() * 0.001);
        }

        template<typename T>
        inline void addSize(const std::string& n, const std::vector<T>& v) {
            size.push_back({"Size.Compressed." + n, (double) v.size() * sizeof(T) * 0.001});
            size.push_back({"Size.Uncompressed." + n, (double) v.size() * sizeof(T) * 0.001});
        }

        inline void addTime(const std::string& n, TimerCPU::TimePoint start) {
            async.lock()->emplace_back("Time.Update." + n, TimerCPU::diff(start, TimerCPU::now()) * 0.001);
        }
    } stats;

    struct {
        bool renderRequestView = true;
        bool renderAABB = false;
        bool renderViews = true;
        bool renderSpheres = true;
        bool renderLines = true;
        std::vector<Debug::AABB> aabb;
        std::vector<Debug::View> views;
        std::vector<Debug::Sphere> spheres;
        std::vector<Debug::Vertex> lines;
        std::vector<Debug::PVSCacheSample> PVSCacheSamples;
    } debug;

    void process(boost::asio::ip::tcp::socket& socket);
    void processVertices();
    void beforeReuse();
    void free();

    void updateVAO(GLuint& vao);

    template<typename Job>
    inline void runAfter(px_sched::Sync sync, Job&& job, px_sched::Sync *out_sync_obj = nullptr, const std::string& n = "", bool syncFinish = true);

    template<typename Job>
    inline void runAfter(px_sched::Sync sync, Job&& job, const std::string& n, bool syncFinish = true) {
        runAfter(sync, std::forward<Job>(job), nullptr, n, syncFinish);
    }

    template<typename Job>
    inline void run(Job&& job, px_sched::Sync *out_sync_obj = nullptr, const std::string& n = "", bool syncFinish = true) {
        runAfter(px_sched::Sync(), std::forward<Job>(job), out_sync_obj, n, syncFinish);
    }

    template<typename Job>
    inline void run(GPUTask type, Job&& job, px_sched::Sync *s = nullptr, const std::string& n = "", double t = 1, bool syncFinish = true);

    inline bool isBufferSuitable(gl::Buffer& buffer, size_t size);
    inline void allocateBuffer(gl::Buffer& buffer, GLenum type, size_t size);
};
