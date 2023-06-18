#include "ServerConnection.h"
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <boost/algorithm/string.hpp>
#include "Application.h"
#include "asioHelpers.h"
#include "imguiHelper.h"
#include "common.h"
#include "graphic.h"
#include "Timer.h"
#include "TimerGL.h"

namespace fs = std::filesystem;
using namespace std;
using namespace glm;
using boost::asio::ip::tcp;

ServerConnection::ServerConnection(Application* app, const std::string& name, const std::string& server, int port) {
    this->app = app;
    this->name = name;
    this->server = server;
    this->port = port;

    prevUpdateTime = prevRequestTime = TimerCPU::now();
    counters.update.maxSamples(1);
    updatePrediction.deltaTime.reset(1);
    updatePrediction.server.reset(1);
    updatePrediction.receive.reset(1);
    updatePrediction.cpu.reset(1);
    updatePrediction.gpu.reset(1);
    updatePrediction.latency.reset(1);
    updatePrediction.started.cpu.reset(1);
    updatePrediction.started.gpu.reset(1);
    updatePrediction.waiting.cpu.reset(1);

    cout << '[' << name << "] Connecting: "<< server << ':' << port << endl;
    connect();
}

void ServerConnection::updateConfig(const Config& cfg) {

    // update stats
    int samples = counters.update.maxSamples();
    if (cfg.get("Stats.MaxSamples", samples))
        counters.update.maxSamples(std::max(1, samples));
    double ema = counters.update.EMA();
    if (cfg.get("Stats.EMA", ema))
        counters.update.EMA(ema);

    // update prediction
    samples = updatePrediction.receive.getSampleCount();
    if (cfg.get("UpdatePrediction.MaxSamples", samples)) {
        samples = std::max(1, samples);
        updatePrediction.deltaTime.reset(samples);
        updatePrediction.server.reset(samples);
        updatePrediction.receive.reset(samples);
        updatePrediction.cpu.reset(samples);
        updatePrediction.gpu.reset(samples);
        updatePrediction.latency.reset(samples);
        updatePrediction.started.cpu.reset(samples);
        updatePrediction.started.gpu.reset(samples);
        updatePrediction.waiting.cpu.reset(samples);
    }
    ema = updatePrediction.receive.getEMAFactor();
    if (cfg.get("UpdatePrediction.EMA", ema)) {
        updatePrediction.deltaTime.setEMAFactor(ema);
        updatePrediction.server.setEMAFactor(ema);
        updatePrediction.receive.setEMAFactor(ema);
        updatePrediction.cpu.setEMAFactor(ema);
        updatePrediction.gpu.setEMAFactor(ema);
        updatePrediction.latency.setEMAFactor(ema);
        updatePrediction.started.cpu.setEMAFactor(ema);
        updatePrediction.started.gpu.setEMAFactor(ema);
        updatePrediction.waiting.cpu.setEMAFactor(ema);
    }

    // byterates
    double dur = counters.byterate.hardMaxDuration();
    if (cfg.get("Byterate.HardMaxDuration", dur)) {
        counters.byterate.hardMaxDuration(dur);
        counters.videoByterate.hardMaxDuration(dur);
    }
    dur = counters.byterate.softMaxDuration();
    if (cfg.get("Byterate.SoftMaxDuration", dur)) {
        counters.byterate.softMaxDuration(dur);
        counters.videoByterate.softMaxDuration(dur);
    }
    samples = (int)counters.byterate.softMinSamples();
    if (cfg.get("Byterate.SoftMinSamples", samples)) {
        counters.byterate.softMinSamples(samples);
        counters.videoByterate.softMinSamples(samples);
    }
}

void ServerConnection::connect() {
    if (enabled) return;
    request.close();
    enabled = true;
    communicationThread = thread([this]() {
        while (enabled) {
            try {
                // connect
                boost::asio::io_context io_context;
                tcp::resolver resolver(io_context);
                tcp::resolver::query query(server, to_string(port));
                auto endpoints = resolver.resolve(query);
                tcp::socket socket(io_context);
                boost::asio::connect(socket, endpoints);
                // TODO try TCP_CORK or accumulate some data to one buffer to prevent small packed when using TCP_NODELAY

                // start writer
                socket.set_option(tcp::no_delay(TCPNoDelay));
                request.setSocket(socket);
                request.run();
                reconnected = true;

                // receive
                connected = true;
                receiver(socket);
                connected = false;
            }
            catch (const boost::system::system_error& error) {

                // close writer
                request.close();
                connected = false;

                cerr << name << ": " << error.what() << std::endl;
                this_thread::sleep_for(1s);
            }
        }
    });
}

void ServerConnection::disconnect() {
    if (!enabled) return;
    enabled = false;
    request.wait();
    request.close(true);
    communicationThread.join();
    connected = false;
}

void ServerConnection::receiver(tcp::socket& socket) {
    RequestType type = RequestType::NONE;
    int sameTypeCount = 0;
    Scene* lastProcessedScene = nullptr;
    while (socket.is_open() && enabled) {
        auto prevType = type;
        read(socket, type);
        if (type != prevType) {
            if (sameTypeCount > 0)
                cout << '[' << name << "] Receive " << prevType << " .. " << sameTypeCount << endl;
            cout << '[' << name << "] Receive " << type << endl;
            sameTypeCount = 0;
        } else sameTypeCount++;
        switch(type) {
            case RequestType::UPDATE_SCENE: {
                // FIXME memory leak when connection fails
                Viewpoint viewpoint;
                Scene* scene = nullptr;
                read(socket, viewpoint);
                if (viewpoint.flags & Viewpoint::AUTOMATIC_UPDATE)
                    processingScenes++;

                // try to find preallocated synchronized scene
                auto scenes = syncScenes.lock();
                for (auto it = scenes->begin(); it != scenes->end(); it++)
                    if ((*it)->viewpoint.id == viewpoint.id) {
                        scene = *it;
                        scenes->erase(it);
                        break;
                    }
                scenes.unlock();

                if (!scene)
                    scene = new Scene(this, viewpoint);
                scene->previous = lastProcessedScene;
                scene->process(socket);
                lastProcessedScene = scene;
                break;
            }
            case RequestType::RENDER_REFERENCE: {
                float time;
                mat4 view;
                std::vector<unsigned char> data;
                read(socket, time);
                read(socket, view);
                read(socket, data);
                app->run([this, time, view, data = move(data)]() {
                    app->serverSceneRecordProcess(time, view, data);
                });
                break;
            }
            case RequestType::GET_SCENE_LIST: {
                vector<string> list;
                read(socket, list);
                sceneList = list;
                break;
            }
            case RequestType::GET_ALL_SETTINGS: {
                string configString;
                read(socket, configString);
                app->run([=]() {
                    pt::ptree tree;
                    if (pt::read_info_string(configString, tree))
                        pt::merge(config, tree);
                });
                break;
            }
            case RequestType::STOP_BENCHMARK: {
                string benchmarkStats;
                read(socket, benchmarkStats);
                benchmark.running = false;
                app->run([this, benchmarkStats = move(benchmarkStats)] {
                    benchmark.server = benchmarkStats;
                });
                break;
            }
            case RequestType::START_BENCHMARK: {
                benchmark.running = true;
                break;
            }
        }
    }
}

void ServerConnection::sendViewpoint(const Viewpoint& viewpoint, UpdateMode mode) {
    if (app->benchmark.running)
        viewpointHistory.lock()->push(viewpoint);
    if (mode == UpdateMode::FORCE) {
        prevRequestTime = app->frameTime;
        processingScenes++;
    }
    request.write(RequestType::UPDATE_VIEWPOINT);
    request.write(viewpoint);
    request.write(mode);
}

bool ServerConnection::allowSceneRemove() {
    return scenes.size() > std::max(1, app->maxCachedScenes + 1);
}

void ServerConnection::removeScene() {
    auto s = scenes.back();
    if (app->maxReuseScenes < 0 || reuseScenes.size() < app->maxReuseScenes) {
        s->beforeReuse();
        reuseScenes.send(s);
    } else {
        s->free();
        delete s;
    }
    scenes.pop_back();
}

void ServerConnection::addScene(Scene* scene) {

    // add scene
    scenes.push_front(scene);

    // debug - transfer configuration
    for (auto it = scenes.begin(); it != scenes.end(); it++) {
        auto it2 = it;
        if (++it2 == scenes.end()) it2 = scenes.begin();
        auto& cur = **it, &prev = **it2;

        cur.debug.renderRequestView = prev.debug.renderRequestView;
        cur.debug.renderAABB = prev.debug.renderAABB;
        cur.debug.renderViews = prev.debug.renderViews;
        cur.debug.renderSpheres = prev.debug.renderSpheres;
        cur.debug.renderLines = prev.debug.renderLines;

        for (auto& c : cur.views) {
            auto p = cur.views.begin();
            for (auto i = prev.views.begin(); i != prev.views.end(); i++)
                if (c.name == i->name) p = i;

            c.render = p->render;
            c.debug.renderView = p->debug.renderView;
        }
    }

    // remove last scenes
    while (allowSceneRemove()) {
        auto s = scenes.back();
        if (!s->isSynchronized()) {
            removeScene();
            continue;
        }

        if (!s->allReady()) break;

        // remove only if all connections allow it and synchronized scene is last in every connection
        bool remove = true;
        for (auto s2 = s->sibling; s2 != s; s2 = s2->sibling)
            if (s2 != s2->connection->scenes.back() || !s2->connection->allowSceneRemove()) {
                remove = false;
                break;
            }
        if (!remove) break;

        // remove all synchronized scenes at once
        for (auto s2 = s->sibling; s2 != s;) {
            auto s3 = s2;
            s2 = s2->sibling;
            s3->connection->removeScene();
        }
        removeScene();
    }

    // resolution change
    if (app->frameSize != scene->frameSize) {
        app->frameSize = scene->frameSize;
        glfwSetWindowSize(app->window, app->frameSize.x, app->frameSize.y);
    }

    // stats
    counters.update.add(scene->stats.local);
    stats.update = counters.update.stats(app->counters.minMax);
    pt::read_info_string(scene->stats.server, stats.server);
    if (scene->benchmark) benchmark.update.add(scene->stats.local);

    processingScenes--;
    app->onSceneUpdate(scene);
}

void ServerConnection::onFrameUpdate(){
    if (reconnected) {
        reconnected = false;
        benchmark.running = false;
        processingScenes = 0;
        request.write(RequestType::GET_SCENE_LIST);
        app->resetTime();
        app->requestAll(RequestType::RESET_STATE);
        app->requestServerConfig();
    }
    framesFromUpdate++;
}

double ServerConnection::minDeltaTime(bool includeTransfer) {
    auto& u = updatePrediction;
    auto& c = app->updatePrediction;
    auto t = std::max(std::max(u.server.average(), includeTransfer ? u.receive.average() : 0.0),
                      std::max(u.cpu.average(), u.gpu.average())) * c.multiplier;
    if (c.minRate > 0) t = std::min(t, 1000000.0 / c.minRate);
    if (c.maxRate > 0) t = std::max(t, 1000000.0 / c.maxRate);
    return t;
}

bool ServerConnection::allowSceneRequest() {
    return updateScene && (app->maxProcessingScenes <= 0 || (int)processingScenes < app->maxProcessingScenes);
}

bool ServerConnection::isBenchmarkRunning(){
    return !scenes.empty() && scenes.front()->benchmark;
}

Viewpoint ServerConnection::createViewpoint(const glm::mat4& view) {
    Viewpoint viewpoint;
    viewpoint.view = view;
    viewpoint.id = app->viewpointID;
    viewpoint.time = (float) app->elapsedTime();

    viewpoint.deltaTime = (float) minDeltaTime() / 1000000;
    if (app->updatePrediction.overrideLatencyInMs >= 0)
        viewpoint.latency =  (float)app->updatePrediction.overrideLatencyInMs / 1000;
    else viewpoint.latency = (float) updatePrediction.latency.average() / 1000000;

    // video framerate
    auto& video = app->video;
    auto avgDeltaTime = updatePrediction.deltaTime.average();
    if (video.framerate.factor > 0 && avgDeltaTime > 0) {
        viewpoint.videoFramerate = std::max(video.framerate.min, (int) std::ceil(
            (1000000.0 / avgDeltaTime) * video.framerate.factor + video.framerate.offset));
    }

    // video bitrate
    auto avgVideoByterate = counters.videoByterate.average();
    auto curMinDeltaTime = minDeltaTime(false);
    if (video.bitrate.bitrate == 0 && avgDeltaTime > 0 && avgVideoByterate > 0 && curMinDeltaTime > 0) { // experimental adaptive bitrate
        auto& v = video.bitrate;
        auto f = ((curMinDeltaTime * v.timeFactor - avgDeltaTime) / 1000 + v.timeOffsetInMs) / 1000; // seconds
        viewpoint.videoBitrate = std::max(v.min, (int)(avgVideoByterate * 8 + f * (float)(f > 0 ? v.stepUp : v.stepDown)));
    } else if (video.bitrate.bitrate > 0)
        viewpoint.videoBitrate = video.bitrate.bitrate; // constant

    return viewpoint;
}