#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>
#include <thread>
#include <atomic>
#include <iostream>
#include <functional>
#include "asioHelpers.h"
#include "glmHelpers.h"
#include "common.h"
#include "Pipe.h"
#include "Config.h"
#include "StatCounter.h"
#include "ByteRateCounter.h"
#include "AsyncWriter.h"
#include "px_sched.h"
#include "Scene.h"

class Application;

class ServerConnection {
public:
    Application* app = nullptr;

    ServerConnection(Application* app, const std::string& name, const std::string& server, int port);
    void updateConfig(const Config& cfg);

    bool updateScene = true;
    std::string name = "server";
    std::string server = "localhost";
    int port = 8080;
    bool TCPNoDelay = true;

    std::thread communicationThread;
    bool enabled = false;
    bool connected = false;
    bool reconnected = false;
    void connect();
    void disconnect();
    void receiver(boost::asio::ip::tcp::socket& socket);

    AsyncWriter request;
    mutexed<std::queue<Viewpoint>> viewpointHistory;
    void sendViewpoint(const Viewpoint&, UpdateMode mode);

    int framesFromUpdate = 0;
    TimerCPU::TimePoint prevUpdateTime, prevRequestTime;
    std::list<Scene*> scenes; // last scene + cached scenes
    mutexed<std::list<Scene*>> syncScenes; // main thread -> communication, for Viewpoint::syncID > 0
    std::atomic_int processingScenes = 0;
    Pipe<Scene*> reuseScenes; // main thread -> communication (see Scene::reuse)
    void addScene(Scene*);
    void removeScene();
    bool allowSceneRemove();
    void onFrameUpdate();

    // stats
    struct { // in us
        StatCounter<double> receive, cpu, gpu, server, latency, deltaTime;
        struct { StatCounter<double> cpu, gpu; } started;
        struct { StatCounter<double> cpu, gpu; } waiting;
    } updatePrediction;

    struct {
        ByteRateCounter byterate, videoByterate;
        StatCounters<double> update;
    } counters;

    pt::ptree config;

    struct {
        pt::ptree update;
        pt::ptree server;
    } stats;

    struct {
        bool running = false; // communication
        StatCounters<double> update;
        std::string server;
    } benchmark;

    Viewpoint createViewpoint(const glm::mat4& view);
    double minDeltaTime(bool includeTransfer = true);
    bool allowSceneRequest();
    bool isBenchmarkRunning();

    std::vector<std::string> sceneList;
};
