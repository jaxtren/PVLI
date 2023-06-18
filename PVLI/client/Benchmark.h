#pragma once

#include <Config.h>
#include <StatCounter.h>

class Application;

struct Benchmark
{
    Application* app = nullptr;
    Benchmark(Application* app) : app(app) {}

    bool running = false; // main
    StatCounters<double> frame;
    StatCounters<double> update;
    pt::ptree report;
    std::string outputPrefix;
    double nextImgSampleTime = 0.0;

    bool updateConfig(const Config& cfg)
    {
        return Config::anyChanged(
            cfg.get("OutputPrefix", outputPrefix),
            automation.updateConfig(cfg),
            simulated.updateConfig(cfg["Simulated"])
        );
    }

    struct Simulated
    {
        // config
        bool use = false;
        float FPS = 60;
        int serverDivideFPS = 1;
        float latencyInMs = 0;
        bool renderAllViewpointsOnServer = true;
        bool renderClientImages = true;
        bool sceneRecordOnly = false;

        // state
        int frame = -1;
        int serverWaitingForID = -1;
        bool serverProcessed = false;

        bool isRunning() const
        {
            return frame >= 0;
        }
        bool updateServer() const
        {
            return (frame % serverDivideFPS) == 0;
        }
        bool waitingForServer() const
        {
            return serverWaitingForID >= 0;
        }
        float serverFPS() const
        {
            return FPS / serverDivideFPS;
        }
        float frameTime() const
        {
            return (float)frame / FPS;
        }
        float latency() const
        {
            return latencyInMs / 1000;
        }

        bool updateConfig(const Config& cfg)
        {
            return Config::anyChanged(
                cfg.get("Use", use),
                cfg.get("FPS", FPS),
                cfg.get("ServerDivideFPS", serverDivideFPS),
                cfg.get("Latency", latencyInMs),
                cfg.get("RenderAllViewpointsOnServer", renderAllViewpointsOnServer),
                cfg.get("RenderClientImages", renderClientImages),
                cfg.get("SceneRecordOnly", sceneRecordOnly));
        }
    } simulated;

    void start();
    void stop() const;
    void process();
    void save();
    bool frames() const;
    void generateSimulatedSceneRecord();
    void startSimulated();
    void startPath();
    void automationStep(float time);

    struct Automation
    {
        enum class State
        {
            eBenchmarkStart,
            eBenchmarkWait,
            eBenchmarkFinish,
            eSaveImgClientStart,
            eSaveImgClientRunning,
            eSaveImgClientThirdPersonStart,
            eSaveImgClientThirdPersonRunning,
            eSaveImgServerStart,
            eSaveImgServerRunning,
            eExit,
        } state = State::eBenchmarkStart;

        bool use = false;
        bool renderImages = false;
        bool renderClientImages = true;
        bool renderClientThirdPersonImages = false;
        bool renderServerImages = true;
        float delayInMs = 0.0f;
        float imgSamplingFPS = 0.0f;

        bool isInState(const State& s) const
        {
            return state == s;
        }

        bool updateConfig(const Config& cfg)
        {
            return Config::anyChanged(
                cfg.get("FullAutomation", use),
                cfg.get("StartDelay", delayInMs),
                cfg.get("GenImages", renderImages),
                cfg.get("RenderClientImages", renderClientImages),
                cfg.get("RenderClientThirdPersonImages", renderClientThirdPersonImages),
                cfg.get("RenderServerImages", renderServerImages),
                cfg.get("ImgSamplingFPS", imgSamplingFPS));
        }
    } automation;
};