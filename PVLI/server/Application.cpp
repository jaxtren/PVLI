#include "Application.h"
#include <filesystem>
#include <memory>
#include "SceneState.inl"
#include "ffmpegEncoder.h"
#include "NvEncoderCuda.h"

using namespace std;
using namespace glm;
using boost::asio::ip::tcp;
namespace fs = std::filesystem;

static inline vec3 position(const mat4& m) { return vec3(m[3]); }

static ivec2 adjustForStream(const ivec2& frameSize, const ivec2& subdivide = {1, 1}) {
    return ((frameSize - 1) / (subdivide * 2) + 1) * (subdivide * 2);
}

static void expandProjection(vec3& projParams, ivec2& frameSize, float fovExpand = 1.0f) {
    if (frameSize.x <= 0 || frameSize.y <= 0) return;
    if (fovExpand != 1.0f) {
        float fovy = radians(projParams.x) / 2;
        vec2 fov = {atan(tan(fovy) * frameSize.x / frameSize.y), fovy};
        frameSize = vec2(frameSize) / tan(fov) * tan(fov * fovExpand);
    }
    projParams.x *= fovExpand;
}

static void scaleProjection(vec3& projParams, ivec2& frameSize, float scale = 1.0f) {
    frameSize = vec2(frameSize) * scale;
    if (scale != 1.0f) {
        float fovy = radians(projParams.x) / 2;
        projParams.x = degrees(atan(tan(fovy) * scale) * 2);
    }
}

inline static mat4 computeProjection(vec3 projParams, ivec2 frameSize) {
    return perspective<float>(radians(projParams.x), (float) frameSize.x / frameSize.y, projParams.y, projParams.z);
}

bool Application::ViewConfig::updateConfig(const Config& cfg) {
    bool videoChanged = video.updateConfig(cfg["Video"]);
    bool subdivideChanged = cfg.get("Subdivide", subdivide);
    if (videoChanged) currentVideo = video;
    if (subdivideChanged) subdivide = glm::max(ivec2(1), subdivide);
    return cfg.get("Use", use) |
        cfg.get("Cullface", cullface) |
        cfg.get("SlopeLimit", slopeLimit) |
        cfg.get("SkipLayers", skipLayers) |
        cfg.get("SubdivideOffset", subdivideOffset) |
        cfg.get("SubdivideChecker", subdivideChecker) |
        pack.updateConfig(cfg) |
        lighthouse.updateConfig(cfg["Lighthouse"]) |
        videoChanged | subdivideChanged;
}

void Application::ViewConfig::provideConfig(Config cfg) {
    cfg.set("Use", use);
    cfg.set("Cullface", cullface);
    cfg.set("SlopeLimit", slopeLimit);
    cfg.set("SkipLayers", skipLayers);
    cfg.set("Subdivide", subdivide);
    cfg.set("SubdivideOffset", subdivideOffset);
    cfg.set("SubdivideChecker", subdivideChecker);
    pack.provideConfig(cfg);
    lighthouse.provideConfig(cfg.create("Lighthouse"));
    video.provideConfig(cfg.create("Video"));
}

Application::Application(const std::string& config){

    // register config parameters
    cfgMap.reg("Port", port);
    cfgMap.reg("Scene", sceneFile);

    cfgMap.reg("FrameSize", frameSize);
    cfgMap.reg("ProjParams", projParams);
    cfgMap.reg("FragmentsMultiple", fragmentsMultiple);
    cfgMap.reg("TCPNoDelay", TCPNoDelay);

    cfgMap.reg("ServerOnly3D", serverOnly3D);
    cfgMap.reg("SendAllVertices", sendAllVertices);

    cfgMap.reg("PrimaryView.FovExpand", primaryView.fovExpand);
    cfgMap.reg("PrimaryView.FrameSizeMultiple", primaryView.frameSizeMultiple);
    cfgMap.reg("PrimaryView.PvsSizeMultiple", primaryView.pvsSizeMultiple);
    cfgMap.reg("PrimaryView.BlendFactor", primaryView.blendFactor);

    cfgMap.reg("Lighthouse.Use", lighthouse.use);
    cfgMap.reg("Lighthouse.Backend", lighthouse.backend);
    cfgMap.reg("Lighthouse.IlluminationOnly", lighthouse.illuminationOnly);
    cfgMap.reg("Lighthouse.ToneMapping", lighthouse.toneMapping);
    cfgMap.reg("Lighthouse.ExpToneMapping", lighthouse.expToneMapping);
    cfgMap.reg("Lighthouse.Gamma", lighthouse.gamma);
    cfgMap.reg("Lighthouse.ColorMapping", lighthouse.colorMapping);

    cfgMap.reg("Stats.MaxSamples", stats.maxSamples);
    cfgMap.reg("Stats.EMA", stats.EMA);
    cfgMap.reg("Stats.Send", stats.send);
    cfgMap.reg("Stats.MinMax", stats.minMax);
    cfgMap.reg("Stats.Prefix", stats.prefix);

    cfgMap.reg("Compression.RLE", compression.rle);
    cfgMap.reg("Compression.QT", compression.qt);
    cfgMap.reg("Compression.QTParts", compression.qtParts);
    cfgMap.reg("Compression.Entropy", compression.entropy);
    cfgMap.reg("Compression.ReuseData", compression.reuseData);

    cfgMap.reg("Cubemap.Rotate", cubemap.rotate);
    cfgMap.reg("Cubemap.Size", cubemap.size);
    cfgMap.reg("Cubemap.PvsSizeMultiple", cubemap.pvsSizeMultiple);
    cfgMap.reg("Cubemap.MergeFullLayers", cubemap.mergeFullLayers);
    cfgMap.reg("Cubemap.MergeViews", cubemap.mergeViews);
    cfgMap.reg("Cubemap.MergeRender", cubemap.mergeRender);
    cfgMap.reg("Cubemap.EnableFallback", cubemap.enableFallback);
    cfgMap.reg("Cubemap.RenderAll", cubemap.renderAll);

    cfgMap.reg("AuxiliaryViews.Offset", auxiliaryViews.offset);

    cfgMap.reg("Blocks.BlockSize", blocks.blockSize);
    cfgMap.reg("Blocks.TextureWidth", blocks.textureWidth);
    cfgMap.reg("Blocks.MinTextureHeight", blocks.minTextureHeight);
    cfgMap.reg("Blocks.TextureHeightStep", blocks.textureHeightStep);
    cfgMap.reg("Blocks.DilateSize", blocks.dilateSize);
    cfgMap.reg("Blocks.Track", blocks.track);

    cfgMap.reg("Video.AvgBitrateFactor", video.avgBitrateFactor);
    cfgMap.reg("Video.MaxBitrateFactor", video.maxBitrateFactor);
    cfgMap.reg("Video.RatioMinPrimary", video.ratioMinPrimary);
    cfgMap.reg("Video.RatioMinCubemap", video.ratioMinCubemap);
    cfgMap.reg("Video.RatioMinBlocks", video.ratioMinBlocks);
    cfgMap.reg("Video.RatioMinAux", video.ratioMinAux);
    cfgMap.reg("Video.RatioMaxPrimary", video.ratioMaxPrimary);
    cfgMap.reg("Video.RatioMaxCubemap", video.ratioMaxCubemap);
    cfgMap.reg("Video.RatioMaxAux", video.ratioMaxAux);

    cfgMap.reg("CameraPrediction.Extrapolation", prediction.extrapolation);
    cfgMap.reg("CameraPrediction.PrimaryView", prediction.primaryView);
    cfgMap.reg("CameraPrediction.Rotation", cameraPrediction.rotation);

    cfgMap.reg("PVS.Corners", pvsConfig.corners);
    cfgMap.reg("PVS.PredictionSamples", pvsConfig.predictionSamples);
    cfgMap.reg("PVS.Radius", pvsConfig.radius);
    cfgMap.reg("PVS.Center", pvsConfig.center);
    cfgMap.reg("PVS.Corners", pvsConfig.corners);

    cfgMap.reg("PVS.View.Primary", pvsConfig.view.primary);
    cfgMap.reg("PVS.View.Prediction", pvsConfig.view.prediction);
    cfgMap.reg("PVS.View.Corners", pvsConfig.view.corners);

    cfgMap.reg("PVS.Cache.Size", pvsConfig.cache.size);
    cfgMap.reg("PVS.Cache.PredictionSamples", pvsConfig.cache.predictionSamples);

    cfgMap.reg("PVS.Cache.Radius.Cache", pvsConfig.cache.radius.cache);
    cfgMap.reg("PVS.Cache.Radius.Merge", pvsConfig.cache.radius.merge);
    cfgMap.reg("PVS.Cache.Radius.Render", pvsConfig.cache.radius.render);

    cfgMap.reg("Debug.Send", debug.send);
    cfgMap.reg("Debug.GL.Output", debug.glDebugOutput);
    cfgMap.reg("Debug.GL.OutputSynchronous", debug.glDebugOutputSynchronous);

    referenceViewConfig.pack.maxLayers = 1;

    // init
    if (!config.empty()) configFile = config;
    loadConfig(true);
    px_sched::SchedulerParams params;
    scheduler.init(params);
    cout << "Application init" << endl;
}

Application::~Application() {
    if (backend) backend->terminate();
}

void Application::loadConfig(bool first, const std::string& cfg) {
    pt::ptree c, c2;
    pt::read_info_ext(configFile, c);
    if (!cfg.empty()) {
        pt::read_info_string(cfg, c2);
        pt::merge(c, c2);
    }
    updateConfig(Config(c), first);
}

void Application::updateConfig(const Config& cfg, bool first) {
    // TODO real update config (merge new config with previous, handle back update)

    cfgMap.updateFromConfig(cfg);

    // init backend
    if (first) {

        vector<string> backends;

        std::string requestedBackend;
        cfg.get("Backend", requestedBackend);
        if (!requestedBackend.empty())
            backends.push_back(requestedBackend);

        #ifdef ENABLE_EGL
        if (requestedBackend != "EGL")
            backends.push_back("EGL");
        #endif

        #ifdef ENABLE_GLFW
        if (requestedBackend != "GLFW")
            backends.push_back("GLFW");
        #endif

        for (auto& backendName : backends) {
            if (backendName == "EGL") {
                #ifdef ENABLE_EGL
                backend = std::make_unique<EGLbackend>();
                #else
                cerr << "Compiled without EGL" << endl;
                #endif
            } else if (backendName == "GLFW") {
                #ifdef ENABLE_GLFW
                backend = std::make_unique<GLFWbackend>();
                #else
                cerr << "Compiled without GLFW" << endl;
                #endif
            } else cerr << "Unknown backend " + backendName << endl;

            if (backend && backend->init(frameSize.x, frameSize.y, "Server")) {
                cout << "Backend: " << backendName << endl;
                break;
            }

            cerr << "Cannot init backend " + backendName << endl;
            backend.reset();
        }

        if (!backend) throw init_error("Cannot init any backend");

#ifdef ENABLE_NVJPEG
        if (!jpegEncoder.init())
            throw init_error("Application: cannot init jpeg encoder");
#endif
    }

    // GL debug
    if (cfgMap.changed(debug.glDebugOutput) ||
        cfgMap.changed(debug.glDebugOutputSynchronous) ||
        (first && debug.glDebugOutput)) {
        if (debug.glDebugOutput) {
            cout << "Enable GL Debug Output" << endl;
            glEnable(GL_DEBUG_OUTPUT);
            if (debug.glDebugOutputSynchronous)
                glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
            else glDisable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
            glDebugMessageCallback(gl::debugMessageCallback, nullptr);
            glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_TRUE);

            // ignore some unimportant and annoying messages, e.g. in PVS:
            // Rasterization usage warning: Dithering is enabled, but is not supported for integer framebuffers
            glDebugMessageControl(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_OTHER, GL_DEBUG_SEVERITY_LOW, 0, NULL, GL_FALSE);
        } else {
            cout << "Disable GL Debug Output" << endl;
            glDisable(GL_DEBUG_OUTPUT);
            glDisable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        }
    }

    if (lighthouse.use) {
        if (!lighthouse.renderer) {

            cout << "Lighthouse init" << endl;
            lighthouse.renderer = new Lighthouse(cfg["Lighthouse"], lighthouse.backend, sceneFile);
            lighthouse.renderer->initScene(scene);
            resetState();

        } else {

            if (cfgMap.changed(lighthouse.backend)) {
                cout << "Lighthouse does not support changing of backend" << endl;
                lighthouse.backend = lighthouse.renderer->rendererName;
            }

            // load back lighthouse scene for PVS
            if (cfgMap.changed(lighthouse.use)) {
                sceneFile = lighthouse.renderer->sceneName;
                lighthouse.renderer->initScene(scene);
                resetState();
            }

            if (cfgMap.changed(lighthouse.illuminationOnly))
                resetState();

            if (!lighthouse.renderer->updateConfig(cfg["Lighthouse"])) {
                // Lighthouse currently does not support changing of the scene
                exit(111); // exit with special code, allowing watch-dog to restart server
            }
        }

        if (lighthouse.renderer->sceneName != sceneFile) {
            // Lighthouse currently does not support changing of the scene
            exit(111); // exit with special code, allowing watch-dog to restart server
        }

    } else if (cfgMap.changed(sceneFile) || !scene.loaded() || cfgMap.changed(lighthouse.use)) {
        resetState();
#ifdef ENABLE_ASSIMP
        scene.load(sceneFile);
#else
        cerr << "Compilled without Assimp: cannot load scene without Lighthouse2 " << endl;
#endif
    }

    if (!lighthouse.use) lighthouse.fragments.free(true);
    if (!pvs) pvs = std::make_unique<PVS>();
    if (pvs) pvs->updateConfig(cfg["PVS"]);
    if (!lv) lv = std::make_unique<LinkedDepthLV>(cfg, &scene);
    else lv->updateConfig(cfg["LV"]);

    // stats
    if (first || cfgMap.changed(stats.maxSamples) || cfgMap.changed(stats.EMA)) {
        stats.maxSamples = std::max(1, stats.maxSamples);
        if (first) {
            stats.update.maxSamples(stats.maxSamples);
            stats.update.EMA(stats.EMA);
        } else response.call([this, maxSamples = stats.maxSamples, ema = stats.EMA](auto&) {
                stats.update.maxSamples(maxSamples);
                stats.update.EMA(ema);
                return 0;
            });
    }

    bool blocksVideoChanged = blocks.video.updateConfig(cfg["Blocks.Video"]);
    if (blocksVideoChanged) blocks.currentVideo = blocks.video;
    if (lighthouse.config.updateConfig(cfg["Lighthouse"], true) |
        primaryView.config.updateConfig(cfg["PrimaryView"]) |
        cubemap.config.updateConfig(cfg["Cubemap"]) |
        auxiliaryViews.config.updateConfig(cfg["AuxiliaryViews"]) |
        blocksVideoChanged |
        cfgMap.changed(sendAllVertices) |
        cfgMap.changed(serverOnly3D))
        resetState();

    referenceViewConfig.updateConfig(cfg["ReferenceView"]);
    lv->blocks.track.updateConfig(cfg["Blocks.Track"]);

    // reset blocks
    if (blocks.track && (first || cfgMap.changed(blocks.track) ||
        cfgMap.changed(blocks.blockSize) || cfgMap.changed(blocks.textureWidth)))
        resetBlocksState();

}

void Application::getCurrentConfig(Config& cfg) {
    cfgMap.storeToConfig(cfg);

    if (lv) {
        lv->provideConfig(cfg.create("LV"));
        lv->blocks.track.provideConfig(cfg.create("Blocks.Track"));
    }
    if (pvs) pvs->provideConfig(cfg.create("PVS"));

    if (lighthouse.renderer)
        lighthouse.renderer->provideConfig(cfg.create("Lighthouse"));

    lighthouse.config.provideConfig(cfg.create("Lighthouse"));
    primaryView.config.provideConfig(cfg.create("PrimaryView"));
    cubemap.config.provideConfig(cfg.create("Cubemap"));
    auxiliaryViews.config.provideConfig(cfg.create("AuxiliaryViews"));
    blocks.video.provideConfig(cfg.create("Blocks.Video"));
    referenceViewConfig.provideConfig(cfg.create("ReferenceView"));

    cfg.set("Benchmark.Running", benchmark.running);
}

void Application::run() {
    try {
        boost::asio::io_context io_context;
        tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), port));
        cout << "Listening on port " << port << endl;
        while (true) {
            thread listenerThread;
            try {
                tcp::socket socket(io_context);
                acceptor.accept(socket);
                socket.set_option(tcp::no_delay(TCPNoDelay));
                cout << "Session start" << endl;

                // init
                sendingScenes = 0;
                response.close();
                response.setSocket(socket);
                response.run();
                requests.clear();
                requests.start();

                // request listener
                listenerThread = thread([this,&socket]() {
                    try { requestListener(socket); }
                    catch (const boost::system::system_error& error) {
                        cout << "Listener error (" << error.what() << ")" << endl;
                    }
                    requests.close();
                });

                // request handler
                try { requestHandler(); }
                catch (const boost::system::system_error& error) {
                    cout << "Handler error (" << error.what() << ")" << endl;
                }

                // finish
                response.wait();
                response.close(true);
                requests.close();
                cout << "Session end" << endl;
            }
            catch (const boost::system::system_error &error) {
                cout << "Session error (" << error.what() << ")" << std::endl;
            }
            if (listenerThread.joinable())
                listenerThread.join();
        }
    }
    catch (const boost::system::system_error &error) {
        cout << "Application error (" << error.what() << ")" << std::endl;
    }
}

void Application::requestListener(tcp::socket& socket) {
    SocketReader reader(socket);
    int updateViewpointCount = 0;
    while (socket.is_open()) {
        auto type = reader.read<RequestType>();

        if (type != RequestType::UPDATE_VIEWPOINT || ++updateViewpointCount > 1000) {
            requests.send([=]() {
                if (updateViewpointCount > 0)
                    cout << "Request handle: UPDATE_VIEWPOINT " << updateViewpointCount << endl;
                if (type != RequestType::UPDATE_VIEWPOINT)
                    cout << "Request handle: " << type << endl;
            });
            updateViewpointCount = 0;
        }

        Viewpoint viewpoint;
        switch(type) {
            case RequestType::UPDATE_VIEWPOINT: {
                reader.read(viewpoint);
                auto updateMode = reader.read<UpdateMode>();
                requests.send([=]() {
                    cameraPrediction.add(viewpoint.time, inverse(viewpoint.view));
                    if (updateMode == UpdateMode::FORCE) {
                        cout << "Request handle: UPDATE_SCENE (by client)" << endl;
                        renderScene(viewpoint);
                    } else if (updateMode == UpdateMode::SERVER_ONLY) {
                        cout << "Request handle: UPDATE_SCENE (server only)" << endl;
                        renderSceneServerOnly(viewpoint);
                    }
                    lastViewpoint = viewpoint;
                    allowUpdate = updateMode == UpdateMode::ALLOW;
                });
                break;
            }
            case RequestType::RENDER_REFERENCE:
                viewpoint = {};
                reader.read(viewpoint.time);
                reader.read(viewpoint.view);
                requests.send([=]() { renderReferenceScene(viewpoint); });
                break;
            case RequestType::SET_SETTINGS: {
                auto cfg = reader.read<string>();
                requests.send([=]() { loadConfig(false, cfg); });
                break;
            }
            default:
                requests.send([=]() {
                    switch (type) {
                        case RequestType::GET_SCENE_LIST:
                            sceneList();
                            break;
                        case RequestType::GET_ALL_SETTINGS: {
                            pt::ptree tree;
                            Config cfg(tree);
                            getCurrentConfig(cfg);
                            response.write(RequestType::GET_ALL_SETTINGS);
                            response.write(pt::to_string_info(tree));
                            break;
                        }
                        case RequestType::START_BENCHMARK:
                            response.write(RequestType::START_BENCHMARK);
                            if (benchmark.running)
                                cout << "Benchmark error: already running" << endl;
                            else {
                                benchmark.running = true;
                                resetState(false); // do not reset vertices when sendAllVertices = true
                                response.call([this](auto&) {
                                    stats.benchmark.reset();
                                    stats.benchmark.maxSamples(0);
                                    return 0;
                                });
                            }
                            break;
                        case RequestType::STOP_BENCHMARK:
                            response.write(RequestType::STOP_BENCHMARK);
                            if (!benchmark.running) {
                                cout << "Benchmark error: not running" << endl;
                                response.write(string());
                            } else {
                                benchmark.running = false;
                                response.call([this, minMax = stats.minMax, prefix = stats.realPrefix()](auto& socket) {
                                    ::write(socket, pt::to_string_info(stats.benchmark.report(minMax, prefix)));
                                });
                            }
                            break;
                        case RequestType::RESET_STATE:
                            resetState();
                        case RequestType::RELOAD_CONFIG:
                            loadConfig(false);
                            break;
                        case RequestType::RELOAD_SHADERS:
                            if (pvs) pvs->loadShaders();
                            if (lv) lv->loadShaders();
                            break;
                        default:
                            cout << "Unknown request type: " << type << endl;
                    }
                });
        }
    }
}

void Application::requestHandler() {
    resetState();
    std::function<void()> request;
    while(requests.receive(request)) {

        // handle all requests
        do { request(); } while (requests.receive(request, false));

        // render scene if need
        if (allowUpdate && sendingScenes.load() <= 1 &&
            (!state || state->renderStart + TimerCPU::dur(lastViewpoint.deltaTime * 1000000) <= TimerCPU::now())) {
            cout << "Request handle: UPDATE_SCENE (by server)" << endl;
            lastViewpoint.flags |= Viewpoint::AUTOMATIC_UPDATE;
            renderScene(lastViewpoint);
            allowUpdate = false; // do not render multiple times same request
        }
    }
}

void Application::resetState(bool resetHasAllVertices) {
    cout << "Reset State" << endl;

    // for NVENC: force destroying of encoders here to prevent crash
    // when creating new encoders, because state can be still referenced elsewhere
    if (state) {
        scheduler.waitFor(state->finished);
        state->streams.clear();
    }
    if (previousState) {
        scheduler.waitFor(previousState->finished);
        previousState->streams.clear();
    }

    state.reset();
    previousState.reset();
    cameraPrediction.reset();
    pvsCache.data.clear();
    lastViewpoint = {};
    allowUpdate = false;
    pvsTriangleCount = 0;
    resetBlocksState();
    if (resetHasAllVertices)
        hasAllVertices = false;
    if (lighthouse.renderer)
        lighthouse.renderer->resetViews();
    response.call([this](auto&) {
        stats.update.reset();
        return 0;
    });
}

void Application::resetBlocksState() {
    if (lv) lv->blocks.init(blocks.textureWidth, blocks.blockSize, blocks.track);
    primaryView.packContext.reset();
    primaryView.prevPackContext.reset();
}

void Application::renderScene(const Viewpoint& viewpoint) {
    previousState = state;
    auto hasPrevState = (bool)previousState;
    if (!previousState) {
        previousState = make_shared<SceneState>();
        previousState->app = this;
        previousState->renderStart = TimerCPU::now();
    }
    state = make_shared<SceneState>();
    state->app = this;
    state->renderStart = TimerCPU::now();
    double renderDeltaTime = TimerCPU::diff(previousState->renderStart, state->renderStart) / 1000000;
    std::vector<Debug::AABB> debugAABB;
    std::vector<Debug::View> debugViews;
    std::vector<Debug::Sphere> debugSpheres;
    std::vector<Debug::Vertex> debugLines;
    TimerCPU timer;
    TimerCUDA timerCUDA;
    pvs->resetTimers();
    response.resetSentBytes();
    sendingScenes++;
    scheduler.incrementSync(&state->finished);
    state->stopwatch.start();

    // video framerate
    if (viewpoint.videoFramerate > 0) {
        primaryView.config.currentVideo.framerate = viewpoint.videoFramerate;
        cubemap.config.currentVideo.framerate = viewpoint.videoFramerate;
        auxiliaryViews.config.currentVideo.framerate = viewpoint.videoFramerate;
        blocks.currentVideo.framerate = viewpoint.videoFramerate;
    } else {
        primaryView.config.currentVideo.framerate = primaryView.config.video.framerate;
        cubemap.config.currentVideo.framerate = cubemap.config.video.framerate;
        auxiliaryViews.config.currentVideo.framerate = auxiliaryViews.config.video.framerate;
        cubemap.config.currentVideo.framerate = cubemap.config.video.framerate;
        blocks.currentVideo.framerate = blocks.video.framerate;
    }

    // video bitrate
    if (viewpoint.videoBitrate > 0)
    {
        const auto encPrimary = primaryView.config.use;
        const auto encCubemap = cubemap.config.use && !serverOnly3D;
        const auto encAuxView = auxiliaryViews.config.use && !serverOnly3D;
        const auto encBlocks = primaryView.config.pack.maxLayers > primaryView.config.pack.fullLayers && !serverOnly3D;

        int sumBitrate = 0;
        if (encPrimary) sumBitrate += primaryView.config.video.maxBitrate;
        if (encCubemap) sumBitrate += cubemap.config.video.maxBitrate;
        if (encAuxView) sumBitrate += auxiliaryViews.config.video.maxBitrate;
        if (encBlocks) sumBitrate += blocks.video.maxBitrate;
        float factor = (float)viewpoint.videoBitrate / (float)sumBitrate;

        int ratioMinSum = video.ratioMinPrimary;
	    if (encCubemap) 
            ratioMinSum += video.ratioMinCubemap;
	    if (encBlocks)
            ratioMinSum += video.ratioMinBlocks;
        if (encAuxView)
            ratioMinSum += video.ratioMinAux;

        int primaryTextureSize = (int)(frameSize.x * frameSize.y * primaryView.frameSizeMultiple);
        int boundMin = primaryTextureSize * 4 / ratioMinSum;
        int base = std::max(boundMin, viewpoint.videoBitrate / ratioMinSum);
        //std::cout << "base:\t" << base << "\n";
        //std::cout << "bound:\t" << boundMin << "\n";

        int bitratePrimary = video.ratioMinPrimary * base;
        int bitrateCubemap = video.ratioMinCubemap * base;
        int bitrateAux = video.ratioMinAux * base;

        if (video.ratioMaxPrimary > video.ratioMinPrimary)
            bitratePrimary = std::min(bitratePrimary, video.ratioMaxPrimary * boundMin);
        if (video.ratioMaxCubemap > video.ratioMinCubemap)
            bitrateCubemap = std::min(bitrateCubemap, video.ratioMaxCubemap * boundMin);
        if (video.ratioMaxAux > video.ratioMinAux)
            bitrateAux = std::min(bitrateAux, video.ratioMaxAux * boundMin);

        int bitrateBlocks = viewpoint.videoBitrate - bitratePrimary;
        if (encCubemap)
            bitrateBlocks -= bitrateCubemap;
        if (encAuxView)
            bitrateBlocks -= bitrateAux;

        // if the budget is insufficient, assign primary bitrate as a minimum
        bitrateBlocks = std::max(bitrateBlocks, bitratePrimary);

        primaryView.config.currentVideo.maxBitrate = primaryView.config.currentVideo.avgBitrate = bitratePrimary;
        cubemap.config.currentVideo.maxBitrate = cubemap.config.currentVideo.avgBitrate = bitrateCubemap;
        auxiliaryViews.config.currentVideo.maxBitrate = auxiliaryViews.config.currentVideo.avgBitrate = bitrateAux;
        blocks.currentVideo.maxBitrate = blocks.currentVideo.avgBitrate = bitrateBlocks;
        	
        //// OLD DISTRIBUTION
        //// max bitrate
        //float maxFactor = factor * video.maxBitrateFactor;
        //primaryView.config.currentVideo.maxBitrate = maxFactor * primaryView.config.video.maxBitrate;
        //cubemap.config.currentVideo.maxBitrate = maxFactor * cubemap.config.video.maxBitrate;
        //auxiliaryViews.config.currentVideo.maxBitrate = maxFactor * auxiliaryViews.config.video.maxBitrate;
        //blocks.currentVideo.maxBitrate = maxFactor * blocks.video.maxBitrate;

        //// avg bitrate
        //float avgFactor = factor * video.avgBitrateFactor;
        //primaryView.config.currentVideo.avgBitrate = avgFactor * primaryView.config.video.avgBitrate;
        //cubemap.config.currentVideo.avgBitrate = avgFactor * cubemap.config.video.avgBitrate;
        //auxiliaryViews.config.currentVideo.avgBitrate = avgFactor * auxiliaryViews.config.video.avgBitrate;
        //blocks.currentVideo.avgBitrate = avgFactor * blocks.video.avgBitrate;
    } else {

        // max bitrate
        primaryView.config.currentVideo.maxBitrate = primaryView.config.video.maxBitrate;
        cubemap.config.currentVideo.maxBitrate = cubemap.config.video.maxBitrate;
        auxiliaryViews.config.currentVideo.maxBitrate = auxiliaryViews.config.video.maxBitrate;
        cubemap.config.currentVideo.maxBitrate = cubemap.config.video.maxBitrate;
        blocks.currentVideo.maxBitrate = blocks.video.maxBitrate;

        // avg bitrate
        primaryView.config.currentVideo.avgBitrate = primaryView.config.video.avgBitrate;
        cubemap.config.currentVideo.avgBitrate = cubemap.config.video.avgBitrate;
        auxiliaryViews.config.currentVideo.avgBitrate = auxiliaryViews.config.video.avgBitrate;
        cubemap.config.currentVideo.avgBitrate = cubemap.config.video.avgBitrate;
        blocks.currentVideo.avgBitrate = blocks.video.avgBitrate;
    }
    //std::cout << "P\t" << primaryView.config.currentVideo.framerate << "\t" << primaryView.config.currentVideo.avgBitrate << "\t" << primaryView.config.currentVideo.maxBitrate << "\n";
    //std::cout << "C\t" << cubemap.config.currentVideo.framerate << "\t" << cubemap.config.currentVideo.avgBitrate << "\t" << cubemap.config.currentVideo.maxBitrate << "\n";
    //std::cout << "B\t" << blocks.currentVideo.framerate << "\t" << blocks.currentVideo.avgBitrate << "\t" << blocks.currentVideo.maxBitrate << "\n";
    //std::cout << "A\t" << auxiliaryViews.config.currentVideo.framerate << "\t" << auxiliaryViews.config.currentVideo.avgBitrate << "\t" << auxiliaryViews.config.currentVideo.maxBitrate << "\n";
    //std::cout << "sum\t\t\t" << primaryView.config.currentVideo.maxBitrate + cubemap.config.currentVideo.maxBitrate + blocks.currentVideo.maxBitrate + auxiliaryViews.config.currentVideo.maxBitrate << "\n";
    //std::cout << "viewpoint video bitrate:\t" << viewpoint.videoBitrate << "\n\n";

    // prediction
    float extrapolation = (viewpoint.latency + viewpoint.deltaTime) * prediction.extrapolation;

    // TODO add dependency on dynamic range value speed? (using cameraPrediction.speed)
    auto computeRadius = [&] (const vec3& v) { return std::max(v.x, std::min(v.y, v.z * extrapolation)); };

    // primary view settings
    bool hasPrimaryView = frameSize.x > 0 && frameSize.y > 0 && primaryView.frameSizeMultiple > 0 && primaryView.fovExpand > 0;
    mat4 requestCamera = inverse(viewpoint.view);
    mat4 primCamera = cameraPrediction.predict(extrapolation * prediction.primaryView);

    // extend FOV
    vec3 primProjParams = projParams;
    ivec2 primFrameSize = frameSize;
    expandProjection(primProjParams, primFrameSize, primaryView.fovExpand);

    // extend sizes
    primFrameSize = adjustForStream((vec2)primFrameSize * primaryView.frameSizeMultiple, primaryView.config.subdivide);
    mat4 primProjection = hasPrimaryView ? computeProjection(primProjParams, primFrameSize) : mat4(1);

    // cube views settings
    ivec2 cubeFrameSize = adjustForStream(ivec2(cubemap.size), cubemap.config.subdivide);
    vec3 cubeProjParams = { 90, projParams.y, projParams.z };
    mat4 cubeProjection = frustum<float>(-projParams.y, projParams.y, -projParams.y, projParams.y, projParams.y, projParams.z);
    mat4 cubeStaticCamera =  mat4( 1, 0,  0, 0, 0, 0,  1, 0,  0, -1,  0, 0, 0, 0, 0, 1); // Z - up
    mat4 cubeCamera = cubemap.rotate ? primCamera : glm::translate(mat4(1), vec3(primCamera[3])) * cubeStaticCamera;
    string cubeNames[] = {"Left", "Front", "Right", "Back", "Top", "Bottom"};
    vector<mat4> cubeStaticViews = {
            mat4( 0, 0, -1, 0, 0, 1,  0, 0,  1,  0,  0, 0, 0, 0, 0, 1), // left
            mat4(1), // front
            mat4( 0, 0,  1, 0, 0, 1,  0, 0, -1,  0,  0, 0, 0, 0, 0, 1), // right
            mat4(-1, 0,  0, 0, 0, 1,  0, 0,  0,  0, -1, 0, 0, 0, 0, 1), // back
            mat4( 1, 0,  0, 0, 0, 0,  1, 0,  0, -1,  0, 0, 0, 0, 0, 1), // top
            mat4( 1, 0,  0, 0, 0, 0, -1, 0,  0,  1,  0, 0, 0, 0, 0, 1), // bottom
    };
    vector<mat4> cubeViews(6);
    for (int i=0; i<6; i++) cubeViews[i] = cubeCamera * cubeStaticViews[i];

    // init PVS
    if (!serverOnly3D) {
        auto pvsProjection = hasPrimaryView ? primProjection : cubeProjection;
        auto instances = scene.collectForCameras({});
        pvs->setInstances(instances);
        if (debug.send)
            for (auto& i : instances)
                debugAABB.push_back({i.aabb.min, i.aabb.max, {1, 1, 1}}); // debug

        // render areas
        float pvsRadius = std::max(computeRadius(pvsConfig.radius), 1e-4f);
        std::vector<vec4> renderAreas;
        {
            float e = computeRadius(pvsConfig.cache.radius.render);
            float c = pvsConfig.center;

            // center
            renderAreas.emplace_back(vec4(vec3(cameraPrediction.predict(extrapolation * c)[3]), pvsRadius));

            // extrapolation
            int i = 0;
            auto cachePrediction = cameraPrediction.predict(0, extrapolation, pvsConfig.cache.predictionSamples);
            for (auto& p : cachePrediction) {
                float w = (float) i / (float) (cachePrediction.size() - 1);
                w = w > c ? w = (w - c) / (1.0f - c) : (c - w) / c;
                float r = pvsRadius * (1.0f - w) + e * w;
                if (r > 0) renderAreas.emplace_back(vec4(vec3(p[3]), r));
                i++;
            }

            // debug
            if (debug.send)
                for (auto& a : renderAreas)
                    debugSpheres.push_back({vec3(a), a.w, {0, 1, 1}});
        }

        // clear whole PVS cache or by areas
        if (pvsConfig.cache.size == 0) pvsCache.clear();
        else {
            // keep everything in current render areas and cache area
            auto cacheArea = vec4(vec3(requestCamera[3]), computeRadius(pvsConfig.cache.radius.cache));
            auto keepAreas = renderAreas;
            keepAreas.emplace_back(cacheArea);
            pvsCache.clear(keepAreas);

            // debug
            if (debug.send)
                debugSpheres.push_back({vec3(cacheArea), cacheArea.w, {1, 1, 1}});
        };

        // generate new PVS samples and cache
        timer("PVS.Generate");
        int newSampleCount = 0;
        float mergeDistance = computeRadius(pvsConfig.cache.radius.merge);
        auto addView = [&](auto c, int viewMode) {
            vector<PVS::View> views;

            // direct
            if (viewMode & PVSConfig::View::DIRECT && hasPrimaryView && primaryView.pvsSizeMultiple > 0 &&
                (primaryView.config.use || viewpoint.flags & Viewpoint::SYNCHRONIZED))
                views.push_back({primProjection, c, (vec2) frameSize * primaryView.pvsSizeMultiple});

            // cubemap
            if (viewMode & PVSConfig::View::CUBEMAP && cubemap.config.use && cubemap.size > 0 && cubemap.pvsSizeMultiple > 0) {
                mat4 cam = cubemap.rotate ? c : glm::translate(mat4(1), vec3(c[3])) * cubeStaticCamera;
                for (auto& cv : cubeStaticViews)
                    views.push_back({cubeProjection, cam * cv, (vec2) cubeFrameSize * cubemap.pvsSizeMultiple});
            }

            // generate and add
            pvs->generate(views, renderState);
            pvsCache.add(c, pvsProjection, renderState, mergeDistance);
            newSampleCount++;
        };
        auto corners = cameraPrediction.predictCorners(extrapolation * pvsConfig.center, pvsRadius / sqrtf(2) * pvsConfig.corners);
        for (auto& c : corners) addView(c, pvsConfig.view.corners);
        auto pvsPrediction = cameraPrediction.predict(0, extrapolation, pvsConfig.predictionSamples);
        for (auto& c : pvsPrediction)
            if (distance(position(primCamera), position(c)) > mergeDistance)
                addView(c, pvsConfig.view.prediction);
        addView(primCamera, pvsConfig.view.primary);

        timer("PVS.Cache");

        // clear cache by count
        if (pvsConfig.cache.size > 0)
            pvsCache.clear(std::max(newSampleCount, pvsConfig.cache.size));

        // collect for render
        renderState.init(pvs.get());
        auto renderSamples = pvsCache.collect(0, renderAreas);
        for (auto& s : renderSamples) renderState.max(s->pvs);

        // collect all for vertex patch
        allState.init(pvs.get());
        if (sendAllVertices) allState.set(1);
        else for (auto& s : pvsCache.data) allState.max(s.pvs);

        // debug views
        if (debug.send) {
            vec3 colorCached(0.8, 0.8, 0.8), colorRender(0, 0.8, 0.8), colorNew(0.8, 0.8, 0);
            for (auto& s : pvsCache.data) debugViews.push_back({s.transform, s.projection, colorCached});
            for (auto& s : renderSamples) debugViews.push_back({s->transform, s->projection, colorRender});
            for (auto& c : corners) debugViews.push_back({c, pvsProjection, colorNew});
            for (auto& c : pvsPrediction)
                if (distance(position(primCamera), position(c)) > mergeDistance)
                    debugViews.push_back({c, pvsProjection, colorNew});
            debugViews.push_back({primCamera, pvsProjection, colorNew});
        }
    }

    // base
    response.write(RequestType::UPDATE_SCENE);
    response.write(viewpoint);
    response.write(projParams);
    response.write(frameSize);
    response.write(lighthouse.gamma);
    response.write(lighthouse.colorMapping);

    // background
    if (!serverOnly3D && lighthouse.use) {
        response.write(lighthouse.renderer->skyTexture);
        response.write(lighthouse.renderer->skyColor);
    } else {
        response.write<string>("");
        response.write(vec3(1));
    }

    // materials
    int matCount = 0;
    if (!serverOnly3D && lighthouse.use && lighthouse.illuminationOnly)
        matCount = hasPrevState ? -1 : (int)scene.materials.size();
    response.write(matCount);
    if (matCount > 0)
        for (auto m : scene.materials) {
            response.write(m.textureName);
            response.write(m.color);
            response.write(m.opaque);
        }

    // send PVS
    if (!serverOnly3D) {
        if (!sendAllVertices || pvsTriangleCount != pvs->allTriangleCount)
            pvsTriangleCount = allMapping.compact(allState, PVS::minRange(1));
        if (viewpoint.flags & Viewpoint::DONT_SEND_PVS || (sendAllVertices && hasAllVertices)) {
            // vertices are not streamed or already processed and sent once
            response.write(sendAllVertices);
            response.write<int>(pvsTriangleCount * 3);
            state->asyncEmptyVector("Time.PVS.Compress");
        } else {
            // create patch
            timer("PVS.Compact");
            hasAllVertices = sendAllVertices;
            response.write(sendAllVertices);
            response.write<int>(pvsTriangleCount * 3);
            bool ext = lighthouse.use && lighthouse.illuminationOnly;
            auto geometry = allMapping.geometry(pvsTriangleCount, true, ext, true, ext);
            state->asyncVectorAfter(previousState->verticesFinished,
                [this, previous = previousState, state = state, g = move(geometry)](auto prom) mutable {
                    swap(previous->vertexCompressor, state->vertexCompressor);
                    previous.reset();
                    prom->set_value(compression.compress(Compression::Method::NONE,
                        state->vertexCompressor.createPatch(g.vertices, g.uv, g.triangles, g.material, compression.reuseData)));
                    // stats
                    auto& src = state->vertexCompressor.stats;
                    if (!src.empty()) {
                        auto dst = state->stats.background.lock();
                        for (auto& s : src)
                            dst->emplace_back("Vertex Compressor." + s.first, s.second);
                    }
                }, &state->verticesFinished, "Time.PVS.Compress");
        }
        timer();
    }

    // render
    const int NO_FALLBACK = -1;
    int cameraFilterCullface = 0; // TODO configurable
    unsigned renderFlags = 0; // set CULL_*_FACE from cameraFilterCullface

    timer("Update Lighthouse");
    if (lighthouse.use && lighthouse.renderer)
        lighthouse.renderer->update(viewpoint.time);
    timer();

    if (serverOnly3D) {
        pvsTriangleCount = 0;
        auto& cfg = primaryView.config;
        auto renderFrameSize = adjustForStream(frameSize, cfg.subdivide);

        // empty vertices
        response.write<bool>(false);
        response.write<int>(0);
        response.write(vector<unsigned char>());

        // view
        response.write(DataType::VIEW);
        response.write(string("Screen"));
        response.write<int>(ViewFlag::VIDEO_STREAM | (cfg.subdivideChecker ? ViewFlag::SUBDIVIDE_CHECKER : 0));
        response.write<int>(0); // priority
        response.write(computeProjection(projParams, renderFrameSize));
        response.write(inverse(requestCamera)); // view
        response.write<float>(0); // blend
        response.write<int>(1); // layerCount
        response.write(renderFrameSize);
        response.write(cfg.subdivide);
        response.write(cfg.subdivideOffset);
        response.write(vec3(0)); // skipLayers

        // full layers
        response.write<int>(1); // fullLayers
        response.write(ivec2(0)); // fullLayersOffset
        response.write(string("Screen")); // fullLayersTexture

        // blocks
        response.write(ivec3(0)); // blocksSize
        response.write<int>(0); // blocksOffset
        response.write(string()); // blocksTexture
        response.write(vector<unsigned char>()); // blockCounts
        response.write(vector<unsigned char>()); // blockIndices

        // triangles
        response.write<int>(0); // triangleCount
        response.write(vector<unsigned char>()); // triangleSubset
        response.write(vector<unsigned char>()); // triangleLayerMask

        // render
        timer("View.Primary");
        renderView(requestCamera, projParams, renderFrameSize, directState,
                   cfg, 0, 0, renderFlags | RenderFlags::CLASSIC_MODE, "Screen");

        // texture
        timer("View.Primary.Pack");
        lv->fullLayers.init(renderFrameSize / cfg.subdivide);
        lv->packFirstLayer();
        state->packFullLayers("Screen", cfg.currentVideo);

    } else {
        if (!blocks.track || !lv->blocks.track.use)
            resetBlocksState();
        lv->blocks.nextFrame();
    }

    // primary view
    if (!serverOnly3D && primaryView.config.use && hasPrimaryView) {
        timer("View.Primary");
        timerCUDA("PVS.Filter");

        // filter by camera projection
        directState = renderState;
        directState.filter(directState, PVS::minRange(1), primCamera, primProjection, primaryView.config.slopeLimit, cameraFilterCullface);

        // remove slope triangles and keep them for auxiliary views
        auxLeft = directState;
        auxLeft.replace(auxLeft, PVS::invertRange(1, 2), 0);
        directState.replace(directState, PVS::invertRange(5, 6), 0);
        timerCUDA();

        // render
        int fallback = cubemap.enableFallback && cubemap.mergeRender && cubemap.size > 0 ? 1 : -1;
        renderView(primCamera, primProjParams, primFrameSize, directState,
                   primaryView.config, 0, fallback, renderFlags, "Primary");

        // pack
        packView({0, 0}, {0, 0}, primFrameSize, primaryView.config, viewpoint,
                 directState, primCamera, primProjParams, primaryView.blendFactor, 0, 10,
                 "Primary", "Primary", "Blocks", &primaryView.packContext, &primaryView.prevPackContext);
    }

    if (!serverOnly3D && auxiliaryViews.config.use) {
        timer("View.Auxiliary");
        vector<vec3> offsets = {
            { 1,  1, 1},
            {-1,  1, 1},
            {-1, -1, 1},
            { 1, -1, 1}
        };

        int i = 0;
        for (auto& o : offsets) {
            timerCUDA("PVS.Filter");
            auto auxName = "Auxiliary " + to_string(i++);
            auto auxCamera = primCamera * translate(mat4(1), o * auxiliaryViews.offset);

            // filter by camera
            auxState = auxLeft;
            auxState.filter(auxState, PVS::minRange(1), auxCamera, primProjection, auxiliaryViews.config.slopeLimit, cameraFilterCullface);

            // divide triangles
            auxLeft.replace(auxState, {5, 6}, 0);
            auxState.replace(auxState, PVS::invertRange(5, 6), 0);
            timerCUDA();

            // generate view
            if (auxState.count(PVS::minRange(1)) > 0) {

                // render
                renderView(auxCamera, primProjParams, primFrameSize, auxState,
                           auxiliaryViews.config, -1, -1, renderFlags, auxName);

                // pack
                packView({0, 0}, {0, 0}, primFrameSize, auxiliaryViews.config, viewpoint,
                         auxState, auxCamera, primProjParams, 0, 0, 0, auxName, auxName, "Blocks");
            }
        }
    }

    // cubemap
    if (!serverOnly3D && cubemap.config.use && cubemap.size > 0) {
        timer("View.Cubemap");
        timerCUDA("PVS.Filter");
        bool mergeRender = cubemap.mergeRender && lighthouse.use;
        bool mergeFullLayers = cubemap.mergeFullLayers;
        int fullLayers = cubemap.config.pack.fullLayers;
        bool filterOutTriangles = primaryView.config.use && hasPrimaryView &&
            !cubemap.renderAll && primaryView.blendFactor < 1;

        // filter by camera projection scaled by blending factor
        if (filterOutTriangles && primaryView.blendFactor > 0) {
            vec3 tempProjParams = primProjParams;
            ivec2 tempFrameSize = primFrameSize;
            scaleProjection(tempProjParams, tempFrameSize, 1.0f - primaryView.blendFactor);
            directState.filter(directState, PVS::minRange(1), primCamera,
                               computeProjection(tempProjParams, tempFrameSize),
                               0, cameraFilterCullface);
        }
        timerCUDA();

        if (!mergeRender && !mergeFullLayers) {
            int i = 0;
            for (auto& cam : cubeViews) {
                timerCUDA("PVS.Filter");
                cubeState = renderState;
                cubeState.filter(cubeState, PVS::minRange(1), cam, cubeProjection, cubemap.config.slopeLimit, cameraFilterCullface);
                cubeState.replace(cubeState, PVS::invertRange(5, 6), 0); // filter by slope
                if (filterOutTriangles)
                    cubeState.replace(directState, {6, 6}, 0); // filter out fully visible in primary view
                timerCUDA();
                renderView(cam, cubeProjParams, cubeFrameSize, cubeState, cubemap.config,
                           1 + i, NO_FALLBACK, renderFlags, cubeNames[i]);

                packView({0, 0}, {0, 0}, cubeFrameSize, cubemap.config, viewpoint,
                         cubeState, cam, cubeProjParams, 0, 0, -1, cubeNames[i], cubeNames[i], "Blocks");
                i++;
            }
        } else if (mergeRender && !mergeFullLayers) {
            timerCUDA("PVS.Filter");
            for (int i=0; i<6; i++) {
                cubeViewState[i] = renderState;
                cubeViewState[i].filter(cubeViewState[i], PVS::minRange(1), cubeViews[i], cubeProjection, cubemap.config.slopeLimit, cameraFilterCullface);
                cubeViewState[i].replace(cubeViewState[i], PVS::invertRange(5, 6), 0); // filter by slope
                if (filterOutTriangles)
                    cubeViewState[i].replace(directState, {6, 6}, 0); // filter out fully visible in primary view
                if (i == 0) cubeState = cubeViewState[i];
                else cubeState.max(cubeViewState[i]);
            }
            timerCUDA();
            renderView(cubeViews[1], cubeProjParams, cubeFrameSize, cubeState, cubemap.config,
                       1, NO_FALLBACK, renderFlags | RenderFlags::CUBEMAP, "Cubemap");

            for (int i=0; i<6; i++)
                packView({i * cubeFrameSize.x, 0}, {0, 0}, cubeFrameSize, cubemap.config, viewpoint,
                         cubeViewState[i], cubeViews[i], cubeProjParams, 0, 0, -1, cubeNames[i], cubeNames[i], "Blocks");
        } else if (!mergeRender && mergeFullLayers) {
            int i = 0;
            if (fullLayers > 0)
                lv->fullLayers.init(cubeFrameSize * ivec2(6, fullLayers) / cubemap.config.subdivide);
            for (auto& cam : cubeViews) {
                timerCUDA("PVS.Filter");
                cubeState = renderState;
                cubeState.filter(cubeState, PVS::minRange(1), cam, cubeProjection, cubemap.config.slopeLimit, cameraFilterCullface);
                cubeState.replace(cubeState, PVS::invertRange(5, 6), 0); // filter by slope
                if (filterOutTriangles)
                    cubeState.replace(directState, {6, 6}, 0); // filter out fully visible in primary view
                timerCUDA();
                renderView(cam, cubeProjParams, cubeFrameSize, cubeState, cubemap.config,
                           1 + i, NO_FALLBACK, renderFlags, cubeNames[i]);
                packView({0, 0}, {i * cubeFrameSize.x, 0}, cubeFrameSize, cubemap.config, viewpoint,
                         cubeState, cam, cubeProjParams, 0, 0, -1, cubeNames[i], "Cubemap", "Blocks");
                i++;
            }
            if (fullLayers > 0)
                state->packFullLayers("Cubemap", cubemap.config.currentVideo);

        } else if (mergeRender && mergeFullLayers) {
            timerCUDA("PVS.Filter");
            for (int i=0; i<6; i++) {
                cubeViewState[i] = renderState;
                cubeViewState[i].filter(cubeViewState[i], PVS::minRange(1), cubeViews[i], cubeProjection, cubemap.config.slopeLimit, cameraFilterCullface);
                cubeViewState[i].replace(cubeViewState[i], PVS::invertRange(5, 6), 0); // filter by slope
                if (filterOutTriangles)
                    cubeViewState[i].replace(directState, {6, 6}, 0); // filter out fully visible in primary view
                if (i == 0) cubeState = cubeViewState[i];
                else cubeState.max(cubeViewState[i]);
            }
            timerCUDA();
            renderView(cubeViews[1], cubeProjParams, cubeFrameSize, cubeState, cubemap.config,
                       1, NO_FALLBACK, renderFlags | RenderFlags::CUBEMAP, "Cubemap");

            if (cubemap.mergeViews)
                packView({0, 0}, {0, 0}, cubeFrameSize, cubemap.config, viewpoint,
                         cubeState, cubeViews[1], cubeProjParams, 0, ViewFlag::CUBEMAP, -1, "Cubemap", "Cubemap", "Blocks");
            else {
                if (fullLayers > 0)
                    lv->fullLayers.init(cubeFrameSize * ivec2(6, fullLayers) / cubemap.config.subdivide);
                for (int i = 0; i < 6; i++)
                    packView({i * cubeFrameSize.x, 0}, {i * cubeFrameSize.x, 0}, cubeFrameSize, cubemap.config,
                             viewpoint, cubeViewState[i], cubeViews[i], cubeProjParams, 0, 0, -1, cubeNames[i], "Cubemap", "Blocks");
                if (fullLayers > 0)
                    state->packFullLayers("Cubemap", cubemap.config.currentVideo);
            }
        }
    }

    timer("Blocks");
    if (!serverOnly3D) {
        if (blocks.track)
            state->packBlocks("Blocks", blocks.currentVideo, blocks.minTextureHeight, blocks.dilateSize);
        else {
            int prevHeight = previousState->blocksSize.x == lv->blocks.size.x ? previousState->blocksSize.y : 0;
            int newHeight = lv->blocks.size.y + blocks.textureHeightStep;
            if (abs(newHeight - prevHeight) < blocks.textureHeightStep)
                newHeight = prevHeight;

            newHeight = std::max(blocks.minTextureHeight, newHeight);
            state->packBlocks("Blocks", blocks.currentVideo, newHeight, blocks.dilateSize);
        }
    }

    response.write(DataType::NONE); // no more data

    // debug
    timer("Stats");
    if (!debug.send) {
        debugAABB.clear();
        debugViews.clear();
        debugSpheres.clear();
        debugLines.clear();
    } else if (lv->blocks.track.use) {
        auto pos = primaryView.packContext.pos.get();
        auto prevPos = primaryView.packContext.prevPos.get();
        for (int i = 0; i < pos.size(); i++) {
            bool reprojected = prevPos[i] != vec3(0);
            if (reprojected) {
                vec3 color = {0.5, 0.5, 1};
                debugLines.push_back({pos[i], color});
                debugLines.push_back({prevPos[i], color});
            }
            debugSpheres.push_back({pos[i], 0.1f, reprojected ? vec3(0, 1, 0) : vec3(1, 0, 0)});
        }
    }
    response.write(move(debugAABB));
    response.write(move(debugViews));
    response.write(move(debugSpheres));
    response.write(move(debugLines));

    vector<Debug::PVSCacheSample> debugPVSCacheSamples;
    if (debug.send)
        for(auto& s : pvsCache.data)
            debugPVSCacheSamples.push_back({s.transform, s.projection});
    response.write(move(debugPVSCacheSamples));

    // stats
    state->stats.add("Request.ID", viewpoint.id);
    state->stats.add("Request.Time", viewpoint.time);
    state->stats.add("Request.DeltaTime", viewpoint.deltaTime * 1000);
    state->stats.add("Request.Latency", viewpoint.latency * 1000);
    state->stats.add("Request.VideoKBps", (float)viewpoint.videoBitrate / 8000);
    state->stats.add("Request.VideoFPS", (float)viewpoint.videoFramerate);
    state->stats.add("Rate", renderDeltaTime > 0 ? (1.0f / renderDeltaTime) : 0);
    state->stats.add("Delta Time", renderDeltaTime * 1000);
    state->stats.add("Data.Triangle Count", pvsTriangleCount);
    state->stats.add("Data.Triangle Count.All", pvs->allTriangleCount);
    state->stats.add("Data.PVS.Cached States", (double)pvsCache.data.size());
    state->stats.add("Time.PVS.Details", compact(pvs->timerGL.getEntries()));
    state->stats.add("Time.PVS.Details", compact(pvs->timerCUDA.getEntries()));
    state->stats.add("Time", timerCUDA.getEntries());
    state->stats.add("Time.Main Thread", TimerCPU::diff(state->renderStart, TimerCPU::now()) / 1000);
    state->stats.add("Time.Main Thread", timer.getEntries());
    pvs->resetTimers();
    response.resetSentBytes();

    // finish stats on the end to have collected State::stats.background
    response.call([this, state = state, send = stats.send, benchmark = benchmark.running,
                   minMax = stats.minMax, prefix = stats.realPrefix()]
                   (auto& socket) {

        scheduler.waitFor(state->finished);
        sendingScenes--;

        // overall time
        double processingTime = TimerCPU::diff(state->stopwatch.started, state->stopwatch.stopped);
        state->stats.add("Time", processingTime / 1000);

        // background stats
        auto background = state->stats.background.lock();
        std::sort(background->begin(), background->end());
        auto& data = state->stats.data;
        data.reserve(data.size() + background->size());
        data.insert(data.end(), background->begin(), background->end());

        // background stopwatches (multipart tasks)
        for (auto& sw : state->stats.stopwatches)
            data.emplace_back(sw.first, TimerCPU::diff(sw.second.started, sw.second.stopped) / 1000);

        // local stats -> application stats
        stats.update.add(data);
        if (benchmark) stats.benchmark.add(data);

        // send
        auto size = ::write(socket, send ? pt::to_string_info(stats.update.stats(minMax, prefix)) : ""); // stats
        size += ::write(socket, (float)(processingTime / 1000000)); // processingTime
        return size;
    });
    state->stopwatch.stop();
    scheduler.decrementSync(&state->finished);
    backend->swapBuffers();
}

void Application::renderSceneServerOnly(const Viewpoint& viewpoint) {

    // prediction
    float extrapolation = (viewpoint.latency + viewpoint.deltaTime) * prediction.extrapolation;

    // primary view settings
    bool hasPrimaryView = frameSize.x > 0 && frameSize.y > 0 && primaryView.frameSizeMultiple > 0 && primaryView.fovExpand > 0;
    mat4 requestCamera = inverse(viewpoint.view);
    mat4 primCamera = cameraPrediction.predict(extrapolation * prediction.primaryView);

    // extend FOV
    vec3 primProjParams = projParams;
    ivec2 primFrameSize = frameSize;
    expandProjection(primProjParams, primFrameSize, primaryView.fovExpand);

    // extend sizes
    primFrameSize = adjustForStream((vec2)primFrameSize * primaryView.frameSizeMultiple, primaryView.config.subdivide);
    mat4 primProjection = hasPrimaryView ? computeProjection(primProjParams, primFrameSize) : mat4(1);

    // cube views settings
    ivec2 cubeFrameSize = adjustForStream(ivec2(cubemap.size), cubemap.config.subdivide);
    vec3 cubeProjParams = { 90, projParams.y, projParams.z };
    mat4 cubeProjection = frustum<float>(-projParams.y, projParams.y, -projParams.y, projParams.y, projParams.y, projParams.z);
    mat4 cubeStaticCamera =  mat4( 1, 0,  0, 0, 0, 0,  1, 0,  0, -1,  0, 0, 0, 0, 0, 1); // Z - up
    mat4 cubeCamera = cubemap.rotate ? primCamera : glm::translate(mat4(1), vec3(primCamera[3])) * cubeStaticCamera;
    string cubeNames[] = {"Left", "Front", "Right", "Back", "Top", "Bottom"};
    vector<mat4> cubeStaticViews = {
        mat4( 0, 0, -1, 0, 0, 1,  0, 0,  1,  0,  0, 0, 0, 0, 0, 1), // left
        mat4(1), // front
        mat4( 0, 0,  1, 0, 0, 1,  0, 0, -1,  0,  0, 0, 0, 0, 0, 1), // right
        mat4(-1, 0,  0, 0, 0, 1,  0, 0,  0,  0, -1, 0, 0, 0, 0, 1), // back
        mat4( 1, 0,  0, 0, 0, 0,  1, 0,  0, -1,  0, 0, 0, 0, 0, 1), // top
        mat4( 1, 0,  0, 0, 0, 0, -1, 0,  0,  1,  0, 0, 0, 0, 0, 1), // bottom
    };
    vector<mat4> cubeViews(6);
    for (int i=0; i<6; i++) cubeViews[i] = cubeCamera * cubeStaticViews[i];

    // render
    const int NO_FALLBACK = -1;
    int cameraFilterCullface = 0;
    unsigned renderFlags = RenderFlags::DONT_SAVE_STATS;

    if (lighthouse.use && lighthouse.renderer)
        lighthouse.renderer->update(viewpoint.time);

    // primary view
    if (!serverOnly3D && primaryView.config.use && hasPrimaryView) {

        // filter by camera projection
        directState = renderState;
        directState.filter(directState, PVS::minRange(1), primCamera, primProjection, primaryView.config.slopeLimit, cameraFilterCullface);

        // remove slope triangles and keep them for auxiliary views
        auxLeft = directState;
        auxLeft.replace(auxLeft, PVS::invertRange(1, 2), 0);
        directState.replace(directState, PVS::invertRange(5, 6), 0);

        // render
        int fallback = cubemap.enableFallback && cubemap.mergeRender && cubemap.size > 0 ? 1 : -1;
        renderView(primCamera, primProjParams, primFrameSize, directState,
                   primaryView.config, 0, fallback, renderFlags, "Primary");
    }

    if (!serverOnly3D && auxiliaryViews.config.use) {
        vector<vec3> offsets = {
            { 1,  1, 1},
            {-1,  1, 1},
            {-1, -1, 1},
            { 1, -1, 1}
        };

        int i = 0;
        for (auto& o : offsets) {
            auto auxName = "Auxiliary " + to_string(i++);
            auto auxCamera = primCamera * translate(mat4(1), o * auxiliaryViews.offset);

            // filter by camera
            auxState = auxLeft;
            auxState.filter(auxState, PVS::minRange(1), auxCamera, primProjection, auxiliaryViews.config.slopeLimit, cameraFilterCullface);

            // divide triangles
            auxLeft.replace(auxState, {5, 6}, 0);
            auxState.replace(auxState, PVS::invertRange(5, 6), 0);

            // generate view
            if (auxState.count(PVS::minRange(1)) > 0) {

                // render
                renderView(auxCamera, primProjParams, primFrameSize, auxState,
                           auxiliaryViews.config, -1, -1, renderFlags, auxName);
            }
        }
    }

    // cubemap
    if (!serverOnly3D && cubemap.config.use && cubemap.size > 0) {
        bool mergeRender = cubemap.mergeRender && lighthouse.use;
        bool mergeFullLayers = cubemap.mergeFullLayers;
        int fullLayers = cubemap.config.pack.fullLayers;
        bool filterOutTriangles = primaryView.config.use && hasPrimaryView &&
                                  !cubemap.renderAll && primaryView.blendFactor < 1;

        // filter by camera projection scaled by blending factor
        if (filterOutTriangles && primaryView.blendFactor > 0) {
            vec3 tempProjParams = primProjParams;
            ivec2 tempFrameSize = primFrameSize;
            scaleProjection(tempProjParams, tempFrameSize, 1.0f - primaryView.blendFactor);
            directState.filter(directState, PVS::minRange(1), primCamera,
                               computeProjection(tempProjParams, tempFrameSize),
                               0, cameraFilterCullface);
        }

        if (mergeRender) {
            for (int i=0; i<6; i++) {
                cubeViewState[i] = renderState;
                cubeViewState[i].filter(cubeViewState[i], PVS::minRange(1), cubeViews[i], cubeProjection, cubemap.config.slopeLimit, cameraFilterCullface);
                cubeViewState[i].replace(cubeViewState[i], PVS::invertRange(5, 6), 0); // filter by slope
                if (filterOutTriangles)
                    cubeViewState[i].replace(directState, {6, 6}, 0); // filter out fully visible in primary view
                if (i == 0) cubeState = cubeViewState[i];
                else cubeState.max(cubeViewState[i]);
            }
            renderView(cubeViews[1], cubeProjParams, cubeFrameSize, cubeState, cubemap.config,
                       1, NO_FALLBACK, renderFlags | RenderFlags::CUBEMAP, "Cubemap");
        } else {
            int i = 0;
            for (auto& cam : cubeViews) {
                cubeState = renderState;
                cubeState.filter(cubeState, PVS::minRange(1), cam, cubeProjection, cubemap.config.slopeLimit, cameraFilterCullface);
                cubeState.replace(cubeState, PVS::invertRange(5, 6), 0); // filter by slope
                if (filterOutTriangles)
                    cubeState.replace(directState, {6, 6}, 0); // filter out fully visible in primary view
                renderView(cam, cubeProjParams, cubeFrameSize, cubeState, cubemap.config,
                           1 + i, NO_FALLBACK, renderFlags, cubeNames[i]);
                i++;
            }
        }
    }

    backend->swapBuffers();
}

void Application::renderView(const mat4& camera, vec3 projParams, ivec2 projSize, PVS::State& mask,
                             const ViewConfig& cfg, int id, int fallback, unsigned int flags, const string& name) {
    bool saveStats = !(flags & RenderFlags::DONT_SAVE_STATS);
    lv->resetTimers();
    if (saveStats)
        state->stats.setPrefix("Time.Render");

    if (lighthouse.use) {
        bool classicMode = flags & RenderFlags::CLASSIC_MODE;
        bool illuminationOnly = !classicMode && lighthouse.illuminationOnly;
        TimerCUDA timer;
        timer(name);

        ivec2 dataSize = projSize / cfg.subdivide;
        if (flags & RenderFlags::CUBEMAP) dataSize.x *= 6;
        lighthouse.renderer->fovy = projParams.x;
        lighthouse.allocate(dataSize.x * dataSize.y * fragmentsMultiple);

        lighthouse.renderer->set(lighthouse.config); // common config for all views
        lighthouse.renderer->set(cfg.lighthouse); // override config per view
        if (classicMode) lighthouse.renderer->set("storeBackground", 1);
        lighthouse.renderer->set("cubemap", flags & RenderFlags::CUBEMAP ? 1 : 0);
        lighthouse.renderer->set("viewID", id);
        lighthouse.renderer->set("fallbackID", fallback);
        lighthouse.renderer->set("demodulateAlbedo", illuminationOnly ? 1 : 0);
        lighthouse.renderer->set("maxLayers", cfg.pack.maxLayers);
        lighthouse.renderer->set("RNGseed", cfg.subdivideOffset.x + cfg.subdivideOffset.y * cfg.subdivide.x);
        lighthouse.renderer->set("cullface", cfg.cullface);
        lighthouse.renderer->set("skipLayers", cfg.skipLayers);

        // subdivide
        vec2 subpixelOffset = (vec2(cfg.subdivideOffset) + 0.5f) / vec2(cfg.subdivide);
        lighthouse.renderer->set("subpixelOffsetX", subpixelOffset.x);
        lighthouse.renderer->set("subpixelOffsetY", subpixelOffset.y);
        lighthouse.renderer->set("width", projSize.x / cfg.subdivide.x);
        lighthouse.renderer->set("height", projSize.y / cfg.subdivide.y);
        if (cfg.subdivideChecker && (cfg.subdivide == ivec2(2,1) || cfg.subdivide == ivec2(1,2))) {
            lighthouse.renderer->set("evenPixelsOffsetX", cfg.subdivide.x == 2 ? (cfg.subdivideOffset.x == 0 ? 0.5f : -0.5f) : 0.0f);
            lighthouse.renderer->set("evenPixelsOffsetY", cfg.subdivide.y == 2 ? (cfg.subdivideOffset.y == 0 ? 0.5f : -0.5f) : 0.0f);
        } else {
            lighthouse.renderer->set("evenPixelsOffsetX", 0);
            lighthouse.renderer->set("evenPixelsOffsetY", 0);
        }

        int count = lighthouse.renderer->render(projSize, camera, classicMode ? nullptr : mask.updateIndirect(),
                                                lighthouse.fragments, (int)lighthouse.fragments.size, false);
        lv->setData(dataSize, lighthouse.fragments, count,
                    lighthouse.toneMapping, lighthouse.expToneMapping, lighthouse.gamma,
                    illuminationOnly ? lighthouse.colorMapping : vec2(0),
                    classicMode || cfg.pack.fullLayers > 0);

        if (saveStats) {
            state->stats.add("", timer.getEntries());
            lighthouse.renderer->stats(state->stats.data, "Lighthouse." + name);
        }
    } else {
        // subdivide not supported
        if (cfg.subdivide != ivec2(1,1))
            cout << "Error: subdivide not supported with rasterization" << endl;

        TimerGL timer;
        timer(name);
        glEnable(GL_CULL_FACE);
        if (cfg.cullface > 0) glCullFace(GL_FRONT);
        else if (cfg.cullface < 0) glCullFace(GL_BACK);
        else glDisable(GL_CULL_FACE);
        lv->render(projSize, computeProjection(projParams, projSize), camera, pvs->instances,
                   mask.data.unmap(), projSize.x * projSize.y * fragmentsMultiple);

        if (saveStats)
            state->stats.add("", timer.getEntries());
    }

    if (saveStats) {
        state->stats.add(name, lv->timerGL.getEntries());
        state->stats.add(name, lv->timerCUDA.getEntries());
        state->stats.setPrefix();
    }
    lv->resetTimers();
}

void Application::packView(const ivec2& srcOffset, const ivec2& dstOffset, const ivec2& size,
                           const ViewConfig& cfg, const Viewpoint& viewpoint, PVS::State& subset,
                           const mat4& camera, vec3 projParams, float blend, int flags, int priority,
                           const string& viewName, const string& fullLayersName, const string& blocksName,
                           LinkedDepthLV::ViewContext* packContext, LinkedDepthLV::ViewContext* prevPackContext) {
    TimerCPU timer;
    TimerCUDA timerCUDA;
    state->stats.setPrefix("Time.View." + viewName);
    lv->resetTimers();
    bool packFullLayersVideo = cfg.pack.fullLayers > 0 && viewName == fullLayersName;

    // mapping from subset
    tempMapping = allMapping;
    tempMapping.replace(subset, {0, 0}, -1);
    int subsetTriangleCount = pvsTriangleCount > 0 ? tempMapping.count(PVS::minRange(0)) : 0;

    // pack
    if (packContext && prevPackContext)
        swap(*packContext, *prevPackContext);
    auto projection = computeProjection(projParams, size);
    auto layerSize = size;
    if (flags & ViewFlag::CUBEMAP) layerSize.x *= 6;
    auto view = inverse(camera);
    if (packContext) {
        packContext->setViewProjection(view, projection);
        packContext->spreadDistance = sin(radians(projParams.x) / (float)(size.y / blocks.blockSize / cfg.subdivide.y));
    }
    timerCUDA("Pack");
    if (packFullLayersVideo) lv->fullLayers.init(layerSize * ivec2(1, cfg.pack.fullLayers) / cfg.subdivide);
    lv->pack(srcOffset / cfg.subdivide, dstOffset / cfg.subdivide, layerSize / cfg.subdivide,
             cfg.pack, pvsTriangleCount, !lighthouse.use,
             cfg.pack.triangleLayerMask && subsetTriangleCount > 0 && !cfg.isSubdivided() ? tempMapping.updateIndirect() : nullptr,
             packContext, prevPackContext);
    timerCUDA();

    // base
    response.write(DataType::VIEW); // next view
    response.write(viewName);
    response.write<int>(flags | (cfg.subdivideChecker ? ViewFlag::SUBDIVIDE_CHECKER : 0) |
                        (lv->output.config.layerFirstBlockOrder || lv->blocks.track.use ? ViewFlag::LAYER_FIRST_BLOCK_ORDER : 0) |
                        (lv->output.config.relocatePixels ? ViewFlag::RELOCATED_PIXELS : 0));
    response.write(priority);
    response.write(projection);
    response.write(view);
    response.write(blend);
    response.write(lv->output.layerCount);
    response.write(layerSize);
    response.write(cfg.subdivide);
    response.write(cfg.subdivideOffset);
    response.write(vec3(cfg.skipLayers, projParams.y, projParams.z));

    // full layers
    response.write(lv->output.config.fullLayers);
    response.write(dstOffset); // offset
    response.write(cfg.pack.fullLayers > 0 ? fullLayersName : ""); // texture

    // blocks
    response.write(ivec3(lv->output.blocksSize, lv->blocks.blockSize));
    response.write<int>(lv->output.tileOffset); // offset
    response.write(lv->output.tileCount > 0 ? blocksName : ""); // texture

    // block counts
    timer("BlockCounts.Copy");
    auto blockCounts = lv->output.blocks.get();
    timer();
    state->asyncVector([=, blockCounts = move(blockCounts)](auto prom) {
        vector<unsigned char> byteBlockCounts(blockCounts.begin(), blockCounts.end()); // int to byte
        prom->set_value(compression.compress(Compression::Method::RLE8, byteBlockCounts));
    }, "Block Counts");

    // block indices
    if (lv->blocks.track.use && packContext) {
        timer("Block Indices.Copy");
        auto blockIndices = packContext->indices.get();
        timer();

        // compression test
        /*{
            // int
            state->stats.setPrefix("Data.View." + viewName);
            vector<int> reducedBlockIndices;
            reducedBlockIndices.reserve(blockIndices.size());
            for (auto i : blockIndices)
                if (i >= 0) // skip unused blocks
                    reducedBlockIndices.push_back(i);
            state->stats.add("Test.Size._32", compression.compress(Compression::Method::NONE, reducedBlockIndices).size());

            // int diff
            vector<int> reducedBlockDiffIndices;
            reducedBlockDiffIndices.resize(reducedBlockIndices.size());
            for(int i=0; i < reducedBlockIndices.size(); i++) {
                int prev = i > 0 ? reducedBlockIndices[i-1] : 0;
                reducedBlockDiffIndices[i] = reducedBlockIndices[i] - prev;
            }
            state->stats.add("Test.Size.d32", compression.compress(Compression::Method::NONE, reducedBlockDiffIndices).size());

            // 24bit
            vector<unsigned char> reducedBlockIndices3;
            for (auto i : reducedBlockIndices) {
                reducedBlockIndices3.push_back(i & 0xff);
                reducedBlockIndices3.push_back((i>>8) & 0xff);
                reducedBlockIndices3.push_back((i>>16) & 0xff);
            }
            state->stats.add("Test.Size._24", compression.compress(Compression::Method::NONE, reducedBlockIndices3).size());

            // 24bit diff
            vector<unsigned char> reducedBlockDiffIndices3;
            for (auto i : reducedBlockDiffIndices) {
                reducedBlockDiffIndices3.push_back(i & 0xff);
                reducedBlockDiffIndices3.push_back((i>>8) & 0xff);
                reducedBlockDiffIndices3.push_back((i>>16) & 0xff);
            }
            state->stats.add("Test.Size.d24", compression.compress(Compression::Method::NONE, reducedBlockDiffIndices3).size());


            // short
            vector<short> reducedBlockIndices2(reducedBlockIndices.begin(), reducedBlockIndices.end());
            state->stats.add("Test.Size._16", compression.compress(Compression::Method::NONE, reducedBlockIndices2).size());
            // short diff
            vector<short> reducedBlockDiffIndices2(reducedBlockDiffIndices.begin(), reducedBlockDiffIndices.end());
            state->stats.add("Test.Size.d16", compression.compress(Compression::Method::NONE, reducedBlockDiffIndices2).size());

            state->stats.setPrefix("Time.View." + viewName);
        }*/

        state->asyncVector([=, blockIndices = move(blockIndices)](auto prom) {
            vector<int> reducedBlockIndices;
            reducedBlockIndices.reserve(blockIndices.size());
            for (auto i : blockIndices)
                if (i >= 0) // skip unused blocks
                    reducedBlockIndices.push_back(i);
            prom->set_value(compression.compress(Compression::Method::NONE, reducedBlockIndices));
        }, "Block Indices");
    } else state->asyncEmptyVector("Block Indices");

    // triangles
    response.write(subsetTriangleCount);

    // triangle subset
    if (subsetTriangleCount > 0 && (!cfg.isSubdivided() || cfg.subdivideOffset == ivec2(0))) {
        timer("Triangle Subset.Copy");
        auto mask = tempMapping.mask(pvsTriangleCount); // mask of subset to PVS
        timer();
        state->asyncVector([=, mask = move(mask)](auto prom) {
            prom->set_value(compression.compress(Compression::Method::RLE1, mask));
        }, "Triangle Subset");
    } else state->asyncEmptyVector("Triangle Subset");

    // triangle-layer mask
    if (lv->output.triangleLayerMask) { //TODO disable for subdivide
        timer("Triangle-Layer Mask.Copy");
        auto mask = lv->output.triangleLayerMask.get();
        timer();
        state->asyncVector([=, mask = move(mask)](auto prom) {
            prom->set_value(compression.compress(Compression::Method::RLE1, mask));
        }, "Triangle-Layer Mask");
    } else state->asyncEmptyVector("Triangle-Layer Mask");

    // stats
    state->stats.add("", timer.getEntries());
    state->stats.add("", timerCUDA.getEntries());
    state->stats.add("Pack", lv->timerGL.getEntries());
    state->stats.add("Pack", lv->timerCUDA.getEntries());
    lv->resetTimers();

    auto& o = lv->output;
    state->stats.setPrefix("Data.View." + viewName);
    state->stats.add("Layout Count", o.layerCount);
    state->stats.add("Triangle Count", subsetTriangleCount);
    state->stats.add("Tile Count", o.tileCount);
    if (lv->blocks.track.use) {
        state->stats.add("Reprojected Tiles", o.reprojectedTileCount);
        state->stats.add("Reprojected Tiles.Ratio", (float)o.reprojectedTileCount / (float)o.tileCount);
    }
    state->stats.add("Fragments / Pixel", (float) o.fragmentCount / (o.layerSize.x * o.layerSize.y));
    state->stats.add("Fragments Overhead",
         (float) ((o.layerSize.x * o.layerSize.y) * cfg.pack.fullLayers + o.tileCount * lv->blocks.blockSize * lv->blocks.blockSize) / o.fragmentCount - 1);
    state->stats.setPrefix();

    if (packFullLayersVideo) state->packFullLayers(fullLayersName, cfg.currentVideo);
}

void Application::renderReferenceScene(const Viewpoint& viewpoint) {
    response.write(RequestType::RENDER_REFERENCE);
    response.write(viewpoint.time);
    response.write(viewpoint.view);

    if (lighthouse.use && lighthouse.renderer)
        lighthouse.renderer->update(viewpoint.time);

    renderView(inverse(viewpoint.view), projParams, frameSize, renderState, referenceViewConfig,
               0, -1, RenderFlags::CLASSIC_MODE | RenderFlags::DONT_SAVE_STATS, "Reference");
    lv->fullLayers.init(frameSize);
    lv->packFirstLayer(true);
    const auto data = lv->fullLayers.color.get<unsigned char>();

    // ppm
    stringstream header;
    header << "P6" << endl << frameSize.x << ' ' << frameSize.y << endl << "255" << endl;
    response.write<int>(-2); // multipart vector
    response.write(header.str());
    response.write(data);

    backend->swapBuffers();
}

void Application::sceneList() {
    vector<string> scenes;
    for (auto file : fs::recursive_directory_iterator("./")) {
        if (!file.is_regular_file()) continue;
        string e = file.path().extension().string();
        if (!e.empty() && sceneExtensions.find(e.substr(1)) != sceneExtensions.end())
            scenes.push_back(file.path().string());
    }
    std::sort(scenes.begin(), scenes.end());
    response.write(RequestType::GET_SCENE_LIST);
    response.write(scenes);
}

std::unique_ptr<VideoEncoderBase> Application::videoEncoderFactory(const VideoEncoderConfig& c, int w, int h)
{
    h = VideoEncoderBase::getMinimalValidDim(h);

    if (c.backend == "nv") {
        CUcontext ctx;
        CUDA_DRVAPI_CALL(cuCtxGetCurrent(&ctx));
        return std::make_unique<NvEncoderCuda>(ctx, c, w, h);
    }

    // fallback to sw ffmpeg
    return std::make_unique<ffmpegEncoder>(c, w, h);
}
