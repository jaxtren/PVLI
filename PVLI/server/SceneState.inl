#include "SceneState.h"
#include "Application.h"

using namespace std;

void SceneState::packFullLayers(const string& name, const VideoEncoderConfig& config) {
    auto& lv = app->lv;
    if (name.empty() || lv->fullLayers.size.y == 0) return;
    packTexture(name, lv->fullLayers.size, config, lv->fullLayers.color, lv->fullLayers.hasMask ? lv->fullLayers.mask.data : nullptr);
}

void SceneState::packBlocks(const string& name, const VideoEncoderConfig& config, int height, int dilateSize) {
    auto& lv = app->lv;
    if (name.empty() || lv->blocks.size.y == 0) return;
    TimerCUDA timerCUDA;
    lv->resetTimers();
    timerCUDA("Finalize");
    lv->finalizeBlocks(height, dilateSize);
    blocksSize = lv->blocks.size;
    timerCUDA();
    packTexture(name, lv->blocks.size, config, lv->blocks.color, lv->blocks.hasMask ? lv->blocks.mask.data : nullptr);
    stats.add("Time.Texture." + name, timerCUDA.getEntries());
    stats.add("Time.Texture." + name + ".Finalize", lv->timerCUDA.getEntries());
    lv->resetTimers();
}

template<typename Job>
inline void SceneState::runAfter(px_sched::Sync sync, Job&& job, px_sched::Sync *outSync, const std::string& name) {
    app->scheduler.incrementSync(&finished);
    ParallelStopwatch* sw = nullptr;
    auto n = stats.prefix + name;

    // multipart task
    if (!n.empty() && n.back() == '*') {
        n = n.substr(0, n.size() - 1) + ".Async";
        sw = &stats.stopwatches[n];
        n += ".Serial";
    } else n += ".Async";

    app->scheduler.runAfter(sync, [=, job = std::forward<Job>(job)]() mutable {
        stopwatch.start();
        if (sw) sw->start();
        auto start = TimerCPU::now();
        job();
        double elapsed = TimerCPU::diff(start, TimerCPU::now()) / 1000;
        if (!n.empty()) stats.background.lock()->emplace_back(n, elapsed);
        if (sw) sw->stop();
        stopwatch.stop();
        app->scheduler.decrementSync(&finished);
    }, outSync);
}

template<typename Job>
inline void SceneState::asyncVectorAfter(px_sched::Sync sync, Job&& job, px_sched::Sync *outSync, const std::string& name) {
    auto prom = app->response.writeLater<std::vector<unsigned char>>();
    runAfter(sync, [prom, job = std::forward<Job>(job)]() mutable { job(prom); }, outSync, name);
}

inline void SceneState::asyncEmptyVector(const std::string& name, std::shared_ptr<std::promise<std::vector<unsigned char>>> prom) {
    if(prom) prom->set_value({});
    else app->response.write(vector<unsigned char>());
}

inline void SceneState::Stats::add(const string& name, const Timer::Entries& entries, bool addSum) {
    double sum = 0;
    auto p = prefix + name;
    if (!p.empty() && p.back() != '.') p += '.';
    for (auto& e : entries) {
        data.emplace_back(p + e.what, e.elapsed * 0.001);
        sum += e.elapsed;
    }
    if (addSum && !name.empty())
        data.emplace_back(name, sum * 0.001);
};
