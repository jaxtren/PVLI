#include "Scene.h"
#include "Scene.inl"
#include "Application.h"
#include "asioHelpers.h"
#include "common.h"
#include "graphic.h"
#include "VertexCompressor.h"
#include "compression/Compression.h"
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace glm;
using boost::asio::ip::tcp;

static void addAll(StatCounters<double>::Entries& stats, const string& name, const Timer::Entries& entries, float m = 0.001){
    double sum = 0;
    auto prefix = name.empty() ? "" : name + ".";
    for(auto& e : entries) {
        stats.push_back({prefix + e.what, e.elapsed * m});
        sum += e.elapsed;
    }
    if(!name.empty())
        stats.push_back({name, sum * m});
};

static double sum(const Timer::Entries& entries, const string& s, float m = 0.001){
    double sum = 0;
    for(auto& e : entries)
        if(boost::algorithm::starts_with(e.what, s))
            sum += e.elapsed;
    return sum * m;
};

static double sum(const StatCounters<double>::Entries& entries, const string& s){
    double sum = 0;
    for(auto& e : entries)
        if(boost::algorithm::starts_with(e.first, s))
            sum += e.second;
    return sum;
};


Scene::Scene(ServerConnection* c, const Viewpoint& v) {
    connection = c;
    app = c->app;
    viewpoint = v;

    // increase sync upfront, because in synchronized updates siblings may depends on this vertices
    // and they can start before of starting of processing this scene
    app->scheduler.incrementSync(&sync.vertices);
}

void Scene::process(tcp::socket& socket) {
    auto& sch = app->scheduler;
    sch.incrementSync(&sync.finish);
    receiveStarted = TimerCPU::now();
    benchmark = connection->benchmark.running;
    SocketReader reader(socket);

    // reuse
    connection->reuseScenes.receive(reuse, false);

    timer.cpu("Base");
    reader.read(projParams);
    reader.read(frameSize);
    reader.read(gamma);
    reader.read(colorMapping);

    // background
    reader.read(background.textureName);
    reader.read(background.color);
    if (!background.textureName.empty())
        background.texture = app->textureManager.load(background.textureName);

    // materials
    int materialCount = reader.read<int>();
    if (materialCount == -1) {
        assert(previous);
        materials = previous->materials;
    } else
        for (int i=0; i<materialCount; i++) {
            materials.emplace_back();
            auto& m = materials.back();
            reader.read(m.textureName);
            reader.read(m.color);
            reader.read(m.opaque);
            if (!m.textureName.empty())
                m.texture = app->textureManager.load(m.textureName);
        }

    // vertices
    timer.cpu("Vertices");
    reader.read(oneTimeVertices);
    reader.read(vertexCount);
    size_t bufSize = vertexCount * sizeof(VertexUV); // TODO or sizeof(Vertex)
    if (hasPVS() && !oneTimeVertices) { // streamed vertices: reuse or allocate buffer
        if (reuse && !reuse->oneTimeVertices && reuse->vertices && isBufferSuitable(*reuse->vertices, bufSize))
            swap(vertices, reuse->vertices);
        else if (vertexCount > 0)
            run(GPUTask::OTHER, [=]() {
                vertices = make_shared<gl::Buffer>();
                allocateBuffer(*vertices, GL_ARRAY_BUFFER, bufSize);
            }, &sync.vertexBuf);
    }
    reader.read(raw.vertices);
    stats.addSize("Vertices", raw.vertices, vertexCount * sizeof(vec3));
    if (hasPVS() && vertexCount > 0) {
        if (oneTimeVertices && raw.vertices.empty() && previous) { // one time vertices: use same vertices from previous scene
            runAfter(previous->sync.vertices, [this]() {
                vertices = previous->vertices;
                materialRanges = previous->materialRanges;
                materialPerTriangle = previous->materialPerTriangle; //TODO zero copy?
                hasUV = previous->hasUV;
                app->scheduler.decrementSync(&sync.vertices);
            });
        } else {
            if (oneTimeVertices) // one time vertices: allocate new buffer
                run(GPUTask::OTHER, [=]() {
                    vertices = make_shared<gl::Buffer>();
                    vertices->allocate(GL_ARRAY_BUFFER, bufSize, GL_STATIC_DRAW);
                    vertices->map();
                }, &sync.vertexBuf);
            runAfter(previous ? previous->sync.vertices : px_sched::Sync(), [this]() {
                runAfter(sync.vertexBuf, [this]() {
                    processVertices();
                    app->scheduler.decrementSync(&sync.vertices);
                }, "Vertices");
            });
        }
    } else sch.decrementSync(&sync.vertices);

    // data
    while (true) {
        auto type = reader.read<DataType>();
        if (type == DataType::NONE) break;
        if (type == DataType::VIEW) {
            views.emplace_back();
            auto& v = views.back();
            v.scene = this;
            v.process(reader);
        } if (type == DataType::TEXTURE) {
            textures.emplace_back();
            auto& t = textures.back();
            t.scene = this;
            t.process(reader);
        }
    }

    timer.cpu("Debug");
    reader.read(debug.aabb);
    reader.read(debug.views);
    reader.read(debug.spheres);
    reader.read(debug.lines);
    reader.read(debug.PVSCacheSamples);
    stats.addSize("Debug.AABB", debug.aabb);
    stats.addSize("Debug.Views", debug.views);
    stats.addSize("Debug.Spheres", debug.spheres);
    stats.addSize("Debug.Lines", debug.lines);
    stats.addSize("Debug.PVSCacheSamples", debug.PVSCacheSamples);

    timer.cpu("Stats");
    reader.read(stats.server);
    reader.read(serverProcessingTime);
    stats.addSize("Stats", stats.server);
    timer.cpu();
    receiveFinished = TimerCPU::now();
    receivedBytes = reader.receivedBytes;

    // received
    for (auto& v : views) v.received();
    for (auto& t : textures) t.received();

    // finish
    runAfter(previous ? previous->sync.finish : px_sched::Sync(), [=]() {
        runAfter(sync.finish, [=]() {
            app->run([=]() {
                updateVAO(vao);
                glBindVertexArray(0);

                // reuse
                if (reuse) {
                    reuse->free();
                    delete reuse;
                }

                // before first render
                for (auto& v : views) v.beforeFirstRender();
                for (auto& t : textures) t.beforeFirstRender();

                // GPU timer
                auto gpu = compact(timer.gpu.tasks.getEntries(), true);
                auto gpuElapsed = sum(gpu) + timer.gpu.frames;

                // stats
                auto updateDeltaTime = TimerCPU::diff(connection->prevUpdateTime, app->frameTime);
                auto latency = app->elapsedTime() - viewpoint.time;
                connection->prevUpdateTime = app->frameTime;

                // prediction, skip for first update and oneTimeVertices updates where all vertices are sent at once
                if (previous && (!oneTimeVertices || raw.vertices.empty())) {
                    if (latency > 0) // can be negative after Application::resetTime()
                        connection->updatePrediction.latency.add(latency * 1000000);
                    connection->updatePrediction.receive.add(TimerCPU::diff(receiveStarted, receiveFinished));

                    connection->updatePrediction.cpu.add(stopwatch.cpu.elapsed);
                    connection->updatePrediction.waiting.cpu.add(stopwatch.cpu.paused);
                    connection->updatePrediction.started.cpu.add(TimerCPU::diff(receiveStarted, stopwatch.cpu.started));

                    connection->updatePrediction.gpu.add(gpuElapsed);
                    connection->updatePrediction.waiting.gpu.add(TimerCPU::diff(stopwatch.gpu.started, stopwatch.gpu.stopped) - gpuElapsed);
                    connection->updatePrediction.started.gpu.add(TimerCPU::diff(receiveStarted, stopwatch.gpu.started));

                    connection->updatePrediction.deltaTime.add(updateDeltaTime);
                    connection->updatePrediction.server.add(serverProcessingTime * 1000000);

                    // byterates
                    connection->counters.byterate.add((double)receivedBytes, receiveFinished);
                    connection->counters.videoByterate.add((double)receivedVideoBytes, receiveFinished);
                }

                // stats
                timer.cpu.finish();
                auto receive = timer.cpu.getEntries();
                auto& l = stats.local;

                l.emplace_back("Timepoint", viewpoint.time);
                l.emplace_back("Timepoint.Start", TimerCPU::diff(app->startTime, receiveStarted) / 1000000);
                l.emplace_back("Timepoint.End", app->elapsedTime());
                l.emplace_back("Rate", updateDeltaTime > 0 ? 1000000.0f / updateDeltaTime : 0);
                l.emplace_back("Delta Time", updateDeltaTime / 1000);
                l.emplace_back("Processing Time", TimerCPU::diff(receiveStarted, app->frameTime) / 1000);
                l.emplace_back("Latency", latency > 0 ? latency * 1000 : 0);
                l.emplace_back("Frames/Update", connection->framesFromUpdate);
                connection->framesFromUpdate = 0;

                l.emplace_back("Vertex Count", vertexCount);

                l.emplace_back("Size", (double) receivedBytes / 1000);
                l.emplace_back("Size.kBps", connection->counters.byterate.average() * 0.001);
                l.emplace_back("Size.kBps.Video", connection->counters.videoByterate.average() * 0.001);
                l.emplace_back("Size.Compressed", sum(stats.size, "Size.Compressed"));
                l.emplace_back("Size.Uncompressed", sum(stats.size, "Size.Uncompressed"));

                for (auto& v : views) {
                    auto name = "View." + v.name;
                    l.emplace_back("Time.Receive." + name, sum(receive, name));
                    l.emplace_back("Time.Update." + name, 0); //push empty to ensure correct order
                    l.emplace_back("Time.GPU." + name, sum(gpu, name));

                    l.emplace_back("Size.Compressed." + name, sum(stats.size, "Size.Compressed." + name));
                    l.emplace_back("Size.Uncompressed." + name, sum(stats.size, "Size.Uncompressed." + name));

                    l.emplace_back(name + ".Triangle Count", v.triangles.count);
                    l.emplace_back(name + ".Layer Count", v.layerCount);
                }

                for (auto& t : textures) {
                    auto name = "Texture." + t.name;
                    l.emplace_back("Time.Receive." + name, sum(receive, name));
                    l.emplace_back("Time.Update." + name, 0); //push empty to ensure correct order
                    l.emplace_back("Time.GPU." + name, sum(gpu, name));

                    l.emplace_back("Size.Compressed." + name, sum(stats.size, "Size.Compressed." + name));
                    l.emplace_back("Size.Uncompressed." + name, sum(stats.size, "Size.Uncompressed." + name));

                    l.emplace_back(name + ".Width", t.size.x);
                    l.emplace_back(name + ".Height", t.size.y);
                }

                // merge stats
                addAll(l, "Time.Receive", receive);
                addAll(l, "Time.GPU", gpu);

                l.emplace_back("Time.GPU.Frames", timer.gpu.frames / 1000);
                l.emplace_back("Time.GPU.Frames.Used", timer.gpu.usedFrames);
                l.emplace_back("Time.GPU.Frames.Skipped", timer.gpu.skippedFrames);

                auto a = stats.async.lock();
                std::sort(a->begin(), a->end());
                l.insert(l.end(), a->begin(), a->end());

                l.insert(l.end(), stats.size.begin(), stats.size.end());

                // set scene
                previous = nullptr;
                ready = true;
                connection->addScene(this);
            });
        }, "", false);
    }, "", false);

    sch.decrementSync(&sync.finish);
}

void Scene::processVertices() {
    if (previous) swap(previous->vertexCompressor, vertexCompressor); // get state from previous update
    if (!raw.vertices.empty())
        vertexCompressor.applyPatch(Compression::decompress(raw.vertices));
    materialPerTriangle = vertexCompressor.material;
    assert(vertexCount == vertexCompressor.vertices.size());

    // vertices
    auto& src = vertexCompressor;
    hasUV = !src.uv.empty();
    if (hasUV) {
        auto dst = (VertexUV*)vertices->data;
        for (int i = 0; i < src.vertices.size(); i += 3) {
            auto v0 = src.vertices[i + 0];
            auto v1 = src.vertices[i + 1];
            auto v2 = src.vertices[i + 2];
            auto n = normalize(cross(v1 - v0, v2 - v0)); // generate normal
            auto packedNormal = u8vec4(n * 127.0f + 128.0f, 0); // pack normal
            dst[i + 0] = {v0, packedNormal, hasUV ? src.uv[i + 0] : vec2(0)};
            dst[i + 1] = {v1, packedNormal, hasUV ? src.uv[i + 1] : vec2(0)};
            dst[i + 2] = {v2, packedNormal, hasUV ? src.uv[i + 2] : vec2(0)};
        }
    } else {
        auto dst = (Vertex*)vertices->data;
        for (int i = 0; i < src.vertices.size(); i += 3) {
            auto v0 = src.vertices[i + 0];
            auto v1 = src.vertices[i + 1];
            auto v2 = src.vertices[i + 2];
            auto n = normalize(cross(v1 - v0, v2 - v0)); // generate normal
            auto packedNormal = u8vec4(n * 127.0f + 128.0f, 0); // pack normal
            dst[i + 0] = {v0, packedNormal};
            dst[i + 1] = {v1, packedNormal};
            dst[i + 2] = {v2, packedNormal};
        }
    }


    // material ranges
    if (!materialPerTriangle.empty() && !materials.empty()) {
        int mat = 0, count = 0;
        for (int i=0; i<vertexCount; i++) {
            int m = materialPerTriangle[i/3];
            if (m != mat) {
                if (count > 0) materialRanges.emplace_back(mat, count);
                mat = m;
                count = 1;
            } else count++;
        }
        if (count > 0) materialRanges.emplace_back(mat, count);
    }
}

void Scene::beforeReuse() {
    glDeleteVertexArrays(1, &vao);
    sibling = nullptr;
    vao = 0;
    if (vertices && !oneTimeVertices) vertices->map();
    for (auto& v : views) v.beforeReuse();
    for (auto& t : textures) t.beforeReuse();
}

void Scene::free() {
    glDeleteVertexArrays(1, &vao);
    vao = 0;
    for (auto& v : views) v.free();
    for (auto& t : textures) t.free();
}

void Scene::updateVAO(GLuint& vao) {
    if (!vao) glGenVertexArrays(1, &vao);
    if (!vertices) return;
    glBindVertexArray(vao);

    vertices->unmap();
    glBindBuffer(GL_ARRAY_BUFFER, *vertices);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    if (hasUV) {
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glEnableVertexAttribArray(2);
        gl::vertexAttribPointer(0, 3, GL_FLOAT, sizeof(VertexUV), 0);
        gl::vertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, sizeof(VertexUV), sizeof(vec3), true);
        gl::vertexAttribPointer(2, 2, GL_FLOAT, sizeof(VertexUV), sizeof(vec3) + sizeof(u8vec4));
    } else {
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        gl::vertexAttribPointer(0, 3, GL_FLOAT, sizeof(Vertex), 0);
        gl::vertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, sizeof(Vertex), sizeof(vec3), true);
    }
}