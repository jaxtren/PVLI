#include "Benchmark.h"

#include "Application.h"
#include <filesystem>

void Benchmark::start()
{
    if (running)
        return;
    running = true;

    // reset stats
    frame.maxSamples(0);
    update.maxSamples(0);
    for (auto& c : app->connections)
    {
        c.benchmark.update.maxSamples(0);
        c.benchmark.server = "";
    }

    // reset scene record
    app->clientSceneRecordStop();
    app->serverSceneRecordStop();
    app->sceneRecord.data.clear();

    // clear viewpoint history
    for (auto& c : app->connections)
    {
        std::queue<Viewpoint> q;
        c.viewpointHistory.lock()->swap(q);
    }

    app->resetTime();
    app->render = true;
    app->updateScene = true;
    app->updateViewpoint = true;
    nextImgSampleTime = 0.0;

    app->requestAll(RequestType::START_BENCHMARK);
}

void Benchmark::stop() const
{
    if (running)
        app->requestAll(RequestType::STOP_BENCHMARK);
}

void Benchmark::process()
{
    if (!running)
        return;
    for (auto& c : app->connections)
        if (c.benchmark.server.empty() || c.isBenchmarkRunning()) return; // some connection not finished
    running = false;

    auto frameReport = frame.report(app->counters.minMax);
    pt::ptree updateReport = update.report(app->counters.minMax, "Client");

    // connection updates
    std::vector<pt::ptree> updateReports;
    updateReports.reserve(app->connections.size());
    for (auto& c : app->connections)
    {
        pt::ptree update = c.benchmark.update.report(app->counters.minMax, "Client");
        pt::ptree server;
        pt::read_info_string(c.benchmark.server, server);
        pt::merge(update, server);
        updateReports.push_back(std::move(update));
    }

    report = pt::ptree();

    // stats
    report.put_child("Stats.Frame", frameReport.get_child("Stats"));
    report.put_child("Stats.Update", updateReport.get_child("Stats"));
    int i = 0;
    for (auto& c : app->connections)
        report.put_child("Stats.Connections." + c.name, updateReports[i++].get_child("Stats"));

    // samples
    report.put_child("Samples.Frame", frameReport.get_child("Samples"));
    report.put_child("Samples.Update", updateReport.get_child("Samples"));
    i = 0;
    for (auto& c : app->connections)
        report.put_child("Samples.Connections." + c.name, updateReports[i++].get_child("Samples"));

    // scene record: disable blending with previous scenes in client rendering before second update
    int updateCount = 0;
    for (auto& s : app->sceneRecord.data) {
        if (s.isUpdate() && ++updateCount > 1) break;
        if (s.type == Application::ViewpointRecord::RENDER_FRAME)
            s.viewpoint.latency = 1;
    }

    if (automation.use)
        automation.state = Automation::State::eBenchmarkFinish;
}

bool Benchmark::frames() const
{
    // allow to save benchmark frame stats only when all servers run benchmark
    if (!running)
        return false;
    for (auto& c : app->connections)
        if (!c.isBenchmarkRunning()) return false;
    return true;
}

void Benchmark::save()
{
    const auto prefix = app->realOutputPath() + outputPrefix;
    std::filesystem::create_directories(prefix);

    // scene record
    app->sceneRecord.save(prefix + "scene_record.txt");

    // all
    {
        std::ofstream out(prefix + "all.txt");
        pt::write_info(out, report);
    }

    auto writeValue = [](std::ostream& out, const pt::ptree& t, char delim, int precision = 4) {
        auto i = t.get_value_optional<int>();
        auto v = t.get_value_optional<glm::vec3>();
        auto d = t.get_value_optional<double>();
        if (v) out << std::fixed << std::setprecision(precision) << v->x << delim << v->y << delim << v->z;
        else if (d) {
            if (*d == floor(*d)) out << (long)(*d);
            else out << std::fixed << std::setprecision(precision) << *d;
        }
        else if (!t.data().empty()) out << t.data();
    };

    // stats
    {
        std::ofstream out(prefix + "stats.csv");
        const char delim = ';';
        std::function<void(const pt::ptree&, const std::string&)> writeCSV = [&](const pt::ptree& tree, const std::string& prefix) -> void {
            for (auto& t : tree) {
                auto label = prefix.empty() ? t.first : prefix + '.' + t.first;
                out << label << delim;
                writeValue(out, t.second, delim);
                out << std::endl;
                writeCSV(t.second, label);
            }
        };
        writeCSV(report.get_child("Stats"), "");
    }

    auto writeSamples = [&](std::ostream& out, const pt::ptree& tree, char delim = ';') {
        std::vector<std::string> names;

        // collect names
        std::function<void(const pt::ptree&, const std::string&)> collectNames = [&](const pt::ptree& tree,
            const std::string& prefix) -> void {
                for (auto& t : tree) {
                    auto label = prefix.empty() ? t.first : prefix + '.' + t.first;
                    names.push_back(label);
                    collectNames(t.second, label);
                }
        };
        collectNames(tree.get_child("0"), "");

        // head
        for (const auto& n : names)
            out << n << delim;
        out << std::endl;

        // samples
        for (int i = 0; true; i++) {
            auto t = tree.get_child_optional(std::to_string(i));
            if (!t) break;
            for (auto n : names) {
                writeValue(out, t->get_child(n), ' ');
                out << delim;
            }
            out << std::endl;
        }
    };

    // frame samples
    {
        std::ofstream out(prefix + "frame.csv");
        writeSamples(out, report.get_child("Samples.Frame"));
    }

    // update samples
    {
        std::ofstream out(prefix + "update.csv");
        writeSamples(out, report.get_child("Samples.Update"));
    }

    // connection samples
    {
        for (auto& t : report.get_child("Samples.Connections")) {
            std::ofstream out(prefix + "connection_" + t.first + ".csv");
            writeSamples(out, t.second);
        }
    }
}


void Benchmark::startSimulated()
{
    if (running)
        return;
    start();
    simulated.frame = 0;
    simulated.serverProcessed = false;
    simulated.serverWaitingForID = -1;
}

void Benchmark::startPath()
{
    app->benchmarkPath = true;
    app->camera.time = app->camera.start;
    app->camera.play = true;
    app->camera.record = false;
    start();
}

void Benchmark::automationStep(const float time)
{
    if (automation.isInState(Automation::State::eBenchmarkStart)) {
        app->updateScene = app->updateViewpoint = app->render = true;
        app->showGUI = false;

        bool hasScenes = true;
        for (auto& c : app->connections)
            if (c.scenes.empty()) {
                hasScenes = false;
                break;
            }

        // wait for the first scene updates from all connections and textures to load
        if (hasScenes && !app->textureManager.loading()) {
            app->resetTime();
            automation.state = Automation::State::eBenchmarkWait;
        }
    }
    if (automation.isInState(Automation::State::eBenchmarkWait) && !running &&
        (simulated.use || time*1000 > automation.delayInMs)) {
        app->camera.updateOnly = false;
        if (simulated.use) {
            if (simulated.sceneRecordOnly) {
                generateSimulatedSceneRecord();
                std::filesystem::create_directories(app->realOutputPath() + outputPrefix);
                app->sceneRecord.save(app->realOutputPath() + app->benchmark.outputPrefix + "scene_record.txt");
                if (automation.renderImages)
                    automation.state = Automation::State::eSaveImgClientStart;
                else automation.state = Automation::State::eExit;
            }
            else startSimulated();
        }
        else startPath();
    }
    else if (automation.isInState(Automation::State::eBenchmarkFinish)) {
        save();
        if (automation.renderImages)
            automation.state = Automation::State::eSaveImgClientStart;
        else automation.state = Automation::State::eExit;
    }
    else if (automation.isInState(Automation::State::eSaveImgClientStart)) {
        if (automation.renderClientImages && !(simulated.use && simulated.renderClientImages && !simulated.sceneRecordOnly)) {
            app->camera.updateOnly = false;
            app->clientSceneRecordStart();
            automation.state = Automation::State::eSaveImgClientRunning;
        }
        else automation.state = Automation::State::eSaveImgClientThirdPersonStart;
    }
    else if (automation.isInState(Automation::State::eSaveImgClientThirdPersonStart)) {
        if (automation.renderClientThirdPersonImages) {
            app->camera.updateOnly = true;
            app->clientSceneRecordStart();
            automation.state = Automation::State::eSaveImgClientThirdPersonRunning;
        }
        else automation.state = Automation::State::eSaveImgServerStart;
    }
    else if (automation.isInState(Automation::State::eSaveImgServerStart)) {
        if (automation.renderServerImages) {
            app->serverSceneRecordStart();
            automation.state = Automation::State::eSaveImgServerRunning;
        }
        else automation.state = Automation::State::eExit;
    }

    if (automation.isInState(Automation::State::eExit))
        exit(0);
}

void Benchmark::generateSimulatedSceneRecord()
{
    app->sceneRecord.data.clear();
    std::vector<Viewpoint> viewpointHistory;
    auto& bs = simulated;
    for (bs.frame = 0; bs.frameTime() + app->camera.start < app->camera.realStop(); bs.frame++) {

        // server
        Viewpoint serverViewpoint;
        serverViewpoint.id = app->viewpointID++;
        serverViewpoint.time = bs.frameTime();
        // subtract latency here in camera path instead of adding it to client frame
        // with this way we can render all frames on the path
        serverViewpoint.view = inverse(app->camera.path.sample(std::max(0.0f, bs.frameTime() - bs.latency()) + app->camera.start).mat());
        serverViewpoint.deltaTime = 1.0f / bs.serverFPS();
        serverViewpoint.latency = bs.latency();
        serverViewpoint.videoFramerate = (int)bs.serverFPS();
        serverViewpoint.videoBitrate = app->video.bitrate.bitrate; // constant
        if (bs.updateServer()) {
            for (auto& v : viewpointHistory)
                app->sceneRecord.data.push_back({ Application::ViewpointRecord::SEND_VIEWPOINT, v, app->connections.front().name });
            viewpointHistory.clear();
            app->sceneRecord.data.push_back({ Application::ViewpointRecord::UPDATE_SCENE, serverViewpoint, app->connections.front().name });
        }
        else viewpointHistory.push_back(serverViewpoint);

        // client
        Viewpoint frameViewpoint;
        frameViewpoint.time = bs.frameTime() + bs.latency();
        frameViewpoint.view = inverse(app->camera.path.sample(bs.frameTime() + app->camera.start).mat());
        frameViewpoint.deltaTime = 1.0f / bs.serverFPS();
        // for blending
        if (bs.frame / bs.serverDivideFPS == 0) frameViewpoint.latency = 1; // ignore previous scene for first update
        else frameViewpoint.latency = (float)(bs.frame % bs.serverDivideFPS + 1) / bs.serverDivideFPS;
        app->sceneRecord.data.push_back({ Application::ViewpointRecord::RENDER_FRAME, frameViewpoint, "" });
    }
    bs.frame = -1;
}
