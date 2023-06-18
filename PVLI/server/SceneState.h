#pragma once

#include <iostream>
#include <string>
#include <list>
#include <map>
#include <GL/glew.h>
#include "px_sched.h"
#include "asioHelpers.h"
#include "cudaHelpers.h"
#include "glmHelpers.h"
#include "common.h"
#include "types.h"
#include "VideoCodingUtil.h"
#include "VertexCompressor.h"
#include "Pipe.h"
#include "Timer.h"
#include "TimerGL.h"
#include "TimerCUDA.h"
#include "StatCounter.h"
#include "AsyncWriter.h"
#include "structs.h"

class Application;

struct SceneState {

    struct Stream {
        std::string name;
        std::unique_ptr<VideoEncoderBase> videoEncoder;
        px_sched::Sync videoFinished;
    };

    Application* app;
    std::list<Stream> streams;
    VertexCompressor vertexCompressor;
    px_sched::Sync verticesFinished, finished;
    TimerCPU::TimePoint renderStart;
    ParallelStopwatch stopwatch;
    glm::ivec2 blocksSize = {0, 0};

    inline Stream* findStream(const std::string& n) {
        if (n.empty()) return nullptr;
        for (auto& s : streams)
            if(s.name == n) return &s;
        return nullptr;
    }

    // texture packing
    void packTexture(const std::string& name, glm::ivec2 size, const VideoEncoderConfig& cfg, glm::u8vec3* cuColor, unsigned char* cuMask);
    inline void packFullLayers(const std::string& name, const VideoEncoderConfig& config);
    inline void packBlocks(const std::string& name, const VideoEncoderConfig& config, int height = 0, int dilateSize = -1);

    struct Stats {

        std::string prefix;
        inline void setPrefix(const std::string& s = "") {
            prefix = s;
            if (!prefix.empty() && prefix.back() != '.') prefix += '.';
        }

        StatCounters<double>::Entries data; // for main thread (order specified)
        mutexed<StatCounters<double>::Entries > background; // for background tasks
        std::map<std::string, ParallelStopwatch> stopwatches; // for multipart background tasks

        inline void add(const string& name, double v) { data.emplace_back(prefix + name, v); }
        inline void add(const string& name, const Timer::Entries& entries, bool addSum = false);

    } stats;

    // base async methods

    template<typename Job>
    inline void runAfter(px_sched::Sync sync, Job&& job, px_sched::Sync *outSync = nullptr, const std::string& name = "");

    template<typename Job>
    inline void asyncVectorAfter(px_sched::Sync sync, Job&& job, px_sched::Sync *outSync = nullptr, const std::string& name = "");

    inline void asyncEmptyVector(const std::string& name, std::shared_ptr<std::promise<std::vector<unsigned char>>> prom = nullptr);

    // variants of async methods

    template<typename Job>
    inline void run(Job&& job, px_sched::Sync *outSync = nullptr, const std::string& name = "") {
        runAfter(px_sched::Sync(), std::forward<Job>(job), outSync, name);
    }
    template<typename Job>
    inline void asyncVector(Job&& job, px_sched::Sync *outSync = nullptr, const std::string& name = "") {
        asyncVectorAfter(px_sched::Sync(), std::forward<Job>(job), outSync, name);
    }
    template<typename Job>
    inline void run(Job&& job, const std::string& name) {
        run(std::forward<Job>(job), nullptr, name);
    }
    template<typename Job>
    inline void runAfter(px_sched::Sync sync, Job&& job, const std::string& name) {
        runAfter(sync, std::forward<Job>(job), nullptr, name);
    }
    template<typename Job>
    inline void asyncVector(Job&& job, const std::string& name) {
        asyncVector(std::forward<Job>(job), nullptr, name);
    }
    template<typename Job>
    inline void asyncVectorAfter(px_sched::Sync sync, Job&& job, const std::string& name){
        asyncVectorAfter(sync, std::forward<Job>(job), nullptr, name);
    }
};