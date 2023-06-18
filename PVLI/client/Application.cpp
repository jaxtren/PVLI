#include "Application.h"
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <boost/algorithm/string.hpp>
#include "asioHelpers.h"
#include "imguiHelper.h"
#include "common.h"
#include "graphic.h"
#include "Timer.h"
#include "TimerGL.h"
#include "VertexCompressor.h"
#include "ffmpegDecoder.h"
#include "Renderer.h"
#include "compression/RLE.h"
#include "Scene.h"
#include "SceneView.h"
#include "TextureManager.h"
#include "stb_image.h"
#include "Scene.inl"
#include "SceneData.inl"

#ifdef DECODE_DXVA2
#include "../platform/dxva2_opengl.h"
#endif


namespace fs = std::filesystem;
using namespace std;
using namespace glm;
using boost::asio::ip::tcp;

static void addAll(StatCounters<double>::Entries& stats, const string& name, const Timer::Entries& entries, bool addSum = true) {
    double sum = 0, m = 0.001;
    auto prefix = name.empty() ? "" : name + ".";
    for (auto& e : entries) {
        stats.push_back({prefix + e.what, e.elapsed * m});
        sum += e.elapsed;
    }
    if (addSum && !name.empty())
        stats.push_back({name, sum * m});
}

bool parseServerString(const string& str, string& server, int& port) {
    auto loc = str.find_last_of(":");
    if (loc == string::npos) {
        server = str;
        return false;
    }
    server = str.substr(0, loc);
    port = stoi(str.substr(loc+1));
    return true;
}

Application::Application(GLFWwindow* w, const string& config) {
    window = w;
    glfwGetFramebufferSize(window, &frameSize.x, &frameSize.y);
    glfwSwapInterval(vsync ? 1 : 0);
    glfwSetWindowUserPointer(window, this);

    glfwSetFramebufferSizeCallback(window, [](GLFWwindow *w, int width, int height) {
        auto app = static_cast<Application*>(glfwGetWindowUserPointer(w));
        app->framebufferSizeChanged(width, height);
    });

    glfwSetCursorPosCallback(window, [](GLFWwindow *w, double x, double y) {
        auto app = static_cast<Application*>(glfwGetWindowUserPointer(w));
        app->cursorPos = {x, y};
    });

    glfwSetKeyCallback(window, [](GLFWwindow* w, int key, int scancode, int action, int mods) {
        auto app = static_cast<Application*>(glfwGetWindowUserPointer(w));
        app->keyCallback(key, scancode, action, mods);
    });

    // ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init();

    // config
    if (!config.empty()) configFile = config;
    updateConfig(true);
    scheduler.init();

#ifdef ENABLE_TURBOJPEG
    jpegDecoder = tjInitDecompress();
#endif
    // texture manager
    stbi_hdr_to_ldr_gamma(1);
    stbi_ldr_to_hdr_gamma(1);
    textureManager.gpuThreadId = this_thread::get_id();
    textureManager.runOnGpuThread = [=](auto f) { run(GPUTask::OTHER, f); };

    // time
    resetTime();
    prevUpdateTime = gpu.prevUpdate = TimerCPU::now();

    video.hwaccel.available = ffmpegHwDecoder::queryHwAccel();
}

void Application::updateConfig(bool first) {

    // presets
    {
        pt::ptree tree;
        pt::read_info_ext(presetFile, tree);
        const auto t = tree.get_child_optional("Server");
        serverConfigPresets = t ? *t : pt::ptree();
    }

    // config
    pt::ptree tree;
    pt::read_info_ext(configFile, tree);
    const Config cfg(tree);

    // GL debug
    if (Config::anyChanged(
        cfg.get("Debug.GL.Output", glDebugOutput),
        cfg.get("Debug.GL.OutputSynchronous", glDebugOutputSynchronous)) ||
        (first && glDebugOutput))
    {
        if (glDebugOutput) {
            cout << "Enable GL Debug Output" << endl;
            glEnable(GL_DEBUG_OUTPUT);
            if (glDebugOutputSynchronous)
                glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
            else glDisable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
            glDebugMessageCallback(gl::debugMessageCallback, nullptr);
            glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_TRUE);
        } else {
            cout << "Disable GL Debug Output" << endl;
            glDisable(GL_DEBUG_OUTPUT);
            glDisable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        }
    }

    cfg.get("OutputPath", outputPath);

    benchmark.updateConfig(cfg["Benchmark"]);

    cfg.get("TextureManager.Path", textureManager.path);

    cfg.get("PixelRelocator.Enable", relocatePixels);
    cfg.get("PixelRelocator.SkipFullLayers", pixelRelocator.skipFullLayers);

    cfg.get("DepthPeeling.Epsilon", depth.epsilon);
    cfg.get("DepthPeeling.UseTriangleLayerMask", depth.useTriangleLayerMask);

    // server connections
    cfg.get("Port", port);
    if (first) {
        auto c = cfg["Servers"];
        if (c.tree) {
            for (auto& t : *c.tree) {
                if (t.second.data().empty())
                    continue;
                string server;
                int port = this->port;
                parseServerString(t.second.data(), server, port);
                if (!server.empty())
                    connections.emplace_back(this, t.first, server, port);
            }
        }

        if (connections.empty()) { // fallback
            string str, server = "localhost";
            int port = this->port;
            if (cfg.get("Server", str))
                parseServerString(str, server, port);
            connections.emplace_back(this, "Server", server, port);
        }
    }

    // server configs
    const auto commonServerConfig = tree.get_child_optional("Server");
    for (auto& c : connections) {
        c.updateConfig(cfg);
        if (commonServerConfig) pt::merge(c.config, *commonServerConfig);
        auto t = tree.get_child_optional("Servers." + c.name);
        if (t) pt::merge(c.config, *t);
    }

    if (cfg.get("VSync", vsync))
        glfwSwapInterval(vsync ? 1 : 0);

    cfg.get("Scene.Update", updateScene);
    cfg.get("Scene.UpdateViewpoint", updateViewpoint);
    cfg.get("Scene.Cache", maxCachedScenes);
    cfg.get("Scene.Process", maxProcessingScenes);
    cfg.get("Scene.Reuse", maxReuseScenes);
    cfg.get("Scene.ReuseTextures", reuseTextures);
    cfg.get("Scene.BufferSizeMultiple", bufferSizeMultiple);
    cfg.get("Scene.UseStagingBuffers", useStagingBuffers);
    cfg.get("Scene.StagingBufferUsage", bufferUsage, bufferUsages);
    cfg.get("Scene.BufferUsage", bufferUsage, bufferUsages);

    cfg.get("Stats.MinMax", counters.minMax);
    cfg.get("Stats.MinMax", counters.minMax);
    cfg.get("UnfocusedWindowSleep", unfocusedWindowSleep);
    cfg.get("UpdatePrediction.Multiplier", updatePrediction.multiplier);
    cfg.get("UpdatePrediction.MinRate", updatePrediction.minRate);
    cfg.get("UpdatePrediction.MaxRate", updatePrediction.maxRate);
    auto size = frameSize;
    if (cfg.get("Resolution", size))
        glfwSetWindowSize(window, size.x, size.y);

    // frame stats
    int samples = counters.frame.maxSamples();
    if (cfg.get("Stats.FrameMaxSamples", samples) || first)
        counters.frame.maxSamples(std::max(1, samples));
    double ema = counters.frame.EMA();
    if (cfg.get("Stats.FrameEMA", ema) || first)
        counters.frame.EMA(ema);

    // update stats
    samples = counters.update.maxSamples();
    if (cfg.get("Stats.MaxSamples", samples) || first)
        counters.update.maxSamples(std::max(1, samples));
    ema = counters.update.EMA();
    if (cfg.get("Stats.EMA", ema) || first)
        counters.update.EMA(ema);

    // video
    cfg.get("Video.Bitrate", video.bitrate.bitrate);
    cfg.get("Video.Bitrate.Min", video.bitrate.min);
    cfg.get("Video.Bitrate.StepUp", video.bitrate.stepUp);
    cfg.get("Video.Bitrate.StepDown", video.bitrate.stepDown);
    cfg.get("Video.Bitrate.TimeFactor", video.bitrate.timeFactor);
    cfg.get("Video.Bitrate.TimeOffset", video.bitrate.timeOffsetInMs);
    cfg.get("Video.Framerate.Factor", video.framerate.factor);
    cfg.get("Video.Framerate.Min", video.framerate.min);
    cfg.get("Video.Framerate.Offset", video.framerate.offset);
    cfg.get("Video.HWAccel.Use", video.hwaccel.primCubeM);
    cfg.get("Video.HWAccel.Blocks", video.hwaccel.blocks);

    // GPU tasks
    cfg.get("GPUTasks.ElapsedMax", gpu.elapsedMax);
    cfg.get("GPUTasks.ElapsedAdaptive", gpu.elapsedAdaptive);
    cfg.get("GPUTasks.QueryDelay", gpu.queryDelay);
    cfg.get("GPUTasks.Tasks", gpu.maxTasks);
    cfg.get("GPUTasks.Upload", gpu.maxUpload);
    cfg.get("GPUTasks.Compute", gpu.maxCompute);

    samples = gpu.elapsed.getSampleCount();
    if (cfg.get("GPUTasks.MaxSamples", samples) || first) {
        samples = std::max(1, samples);
        gpu.elapsed.reset(samples);
        gpu.elapsedLeft.reset(samples);
        gpu.deltaTime.reset(samples);
    }
    ema = gpu.elapsed.getEMAFactor();
    if (cfg.get("GPUTasks.EMA", ema) || first) {
        gpu.elapsed.setEMAFactor(ema);
        gpu.elapsedLeft.setEMAFactor(ema);
        gpu.deltaTime.setEMAFactor(ema);
    }

    if (cfg.get("Path.File", camera.file) || first)
        if (!camera.file.empty())
            camera.path.loadFromFile(camera.file);
    cfg.get("Path.Start", camera.start);
    cfg.get("Path.Stop", camera.stop);
    cfg.get("Path.Speed", camera.speed);
    cfg.get("Path.UpdateOnly", camera.updateOnly);

    if (cfg.get("Record.File", sceneRecord.file) || first)
        sceneRecord.load(sceneRecord.file);
    cfg.get("Record.Images.Client", clientSceneRecord.file);
    cfg.get("Record.Images.ClientThirdPerson", clientSceneRecord.fileThirdPerson);
    cfg.get("Record.Images.Server", serverSceneRecord.file);
    cfg.get("Record.Client.CorrectionTest", clientSceneRecord.correctionTest);
    cfg.get("Record.Server", serverSceneRecord.server);

    renderer.updateConfig(cfg["Renderer"]);
    view.updateConfig(cfg["View"]);

    if (cfg.get("Scene.SyncUpdate", syncSceneUpdate) && !first)
        requestAll(RequestType::RESET_STATE);
    if (cfg.get("Scene.AsyncInterleave", asyncInterleave) && !first && !syncSceneUpdate)
        requestAll(RequestType::RESET_STATE);
    if (cfg.get("Scene.SyncSourceForPVS", syncSourceForPVS) && !first && syncSceneUpdate)
        requestAll(RequestType::RESET_STATE);

    // gui state
    cfg.get("GUI.StateFile", gui.stateFile);
    if (first) pt::read_info_ext(gui.stateFile, gui.state);
}

void Application::keyCallback(int key, int scancode, int action, int mods) {
    if (ImGui::GetIO().WantCaptureKeyboard) return;
    if (action == GLFW_PRESS) {

        if (key == GLFW_KEY_ESCAPE)
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        else if (key == GLFW_KEY_R) {
            if (mods & GLFW_MOD_SHIFT) {
                cout << "Reload config" << endl;
                updateConfig();
                requestServerConfig();
            } else {
                cout << "Reload shaders" << endl;
                renderer.loadShaders();
                depth.peeling.loadShaders();
                pixelRelocator.loadShaders();
            }
        } else if (key == GLFW_KEY_F1 || key == GLFW_KEY_G)
            showGUI = !showGUI;
        else if (key == GLFW_KEY_F2 || key == GLFW_KEY_U)
            updateScene = !updateScene;
        else if (key == GLFW_KEY_F3)
            updateViewpoint = !updateViewpoint;
        else if (key == GLFW_KEY_F4)
            renderer.debug.render = !renderer.debug.render;
        else if (key == GLFW_KEY_F5)
            render = !render;

        // camera location selection
        else if (key >= GLFW_KEY_1 && key <= GLFW_KEY_9) {
            if (key - GLFW_KEY_1 < cameraLocations.size())
                view.location = cameraLocations[key - GLFW_KEY_1];
        } else if (key >= GLFW_KEY_KP_1 && key <= GLFW_KEY_KP_9) {
            if (key - GLFW_KEY_KP_1 < cameraLocations.size())
                view.location = cameraLocations[key - GLFW_KEY_KP_1];
        }
    }
}

void Application::framebufferSizeChanged(int width, int height) {
    ivec2 size = {width, height};
    if (size != frameSize) {
        frameSize = size;
        setServerConfig("FrameSize", frameSize);
        requestServerConfig();
    }
}

double Application::GPU::nextElapsed() {
    if(queries.empty()) return 0;
    auto q = queries.front();
    queries.pop();
    GLuint64 t1 = 0, t2 = 0;
    glGetQueryObjectui64v(q.first, GL_QUERY_RESULT, &t1);
    glGetQueryObjectui64v(q.second, GL_QUERY_RESULT, &t2);
    glDeleteQueries(2, &q.first);
    return t2 < t1 ? 0 : (double) (t2 - t1) / 1000000; // ms
}

void Application::GPU::processInstantTasks()  {
    Task task;
    while (tasks.receive(task, false)) task.first();
}

void Application::GPU::processTasks() {

    // left queries
    double curElapsedLeft = 0;
    while (!queries.empty())
        curElapsedLeft += nextElapsed();

    // adaptive max time
    double curElapsedMax = elapsedMax;
    if(elapsedAdaptive > 0) {
        curElapsedMax = (deltaTime.average() - elapsed.average() - elapsedLeft.average()) * elapsedAdaptive;
        if(elapsedMax > 0) curElapsedMax = std::min(curElapsedMax, elapsedMax);
    }
    if (queryDelay < 0) curElapsedMax = 0;
    stats.elapsedMax = curElapsedMax;

    // tasks
    double curElapsed = 0;
    stats.tasks = 0;
    stats.compute = stats.upload = 0;
    bool again = true, allowCompute = true, allowUpload = true;
    while (again) {

        Task task;
        auto processTask = [&] (bool compute) -> bool {
            stats.tasks++;
            if (compute) stats.compute += task.second;
            else stats.upload += task.second;

            pair <GLuint, GLuint> q;
            glGenQueries(2, &q.first);
            glQueryCounter(q.first, GL_TIMESTAMP);
            task.first();
            glQueryCounter(q.second, GL_TIMESTAMP);
            queries.push(q);

            if (curElapsedMax > 0 && queries.size() >= queryDelay) {
                glFlush();
                curElapsed += nextElapsed();
                if (curElapsed >= curElapsedMax)
                    return false;
            }

            if (maxTasks > 0 && stats.tasks >= maxTasks) return false;
            if (compute && maxCompute > 0 && stats.compute >= maxCompute) {
                allowCompute = false;
                return false;
            }
            if (!compute && maxUpload > 0 && stats.upload >= maxUpload) {
                allowUpload = false;
                return false;
            }
            return true;
        };

        again = false;
        processInstantTasks();
        if (allowCompute && compute.receive(task, false) && processTask(true)) again = true;
        processInstantTasks();
        if (allowUpload && upload.receive(task, false) && processTask(false)) again = true;
    }

    // stats
    if (curElapsedMax > 0) {
        elapsed.add(curElapsed);
        elapsedLeft.add(curElapsedLeft);
    } else {
        elapsed.add(curElapsedLeft);
        elapsedLeft.add(0);
    }
    auto now = TimerCPU::now();
    deltaTime.add(TimerCPU::diff(prevUpdate, now) / 1000);
    prevUpdate = now;
}

bool Application::SceneRecord::load(const std::string& file) {
    if(file.empty()) return false;
    ifstream in(file);
    data.clear();
    int type;
    ViewpointRecord r;
    while(in
            >> type
            >> r.server
            >> r.viewpoint.id
            >> r.viewpoint.flags
            >> r.viewpoint.time
            >> r.viewpoint.deltaTime
            >> r.viewpoint.latency
            >> r.viewpoint.videoBitrate
            >> r.viewpoint.videoFramerate
            >> r.viewpoint.view[0] >> r.viewpoint.view[1]
            >> r.viewpoint.view[2] >> r.viewpoint.view[3]) {
        r.type = static_cast<ViewpointRecord::Type>(type);
        if (r.server == "#") r.server.clear();
        data.push_back(r);
    }
    return (bool)in;
}

bool Application::SceneRecord::save(const std::string& file) {
    ofstream out(file);
    for(auto& r : data)
        out << static_cast<int>(r.type) << ' '
            << (r.server.empty() ? "#" : r.server) << ' '
            << r.viewpoint.id << ' '
            << r.viewpoint.flags << ' '
            << r.viewpoint.time << ' '
            << r.viewpoint.deltaTime << ' '
            << r.viewpoint.latency << ' '
            << r.viewpoint.videoBitrate << ' '
            << r.viewpoint.videoFramerate << ' '
            << r.viewpoint.view[0] << ' ' <<  r.viewpoint.view[1] << ' '
            << r.viewpoint.view[2] << ' ' <<  r.viewpoint.view[3] << endl;
    return (bool)out;
}

void Application::clientSceneRecordProcess(const Viewpoint& viewpoint, ServerConnection* connection) {
    auto& i = clientSceneRecord.index;
    if (i == SceneRecord::STOP) return;
    auto ti = clientSceneRecord.targetIndex;
    auto& record = sceneRecord.data;

    // process current record
    if (i >= 0 && i < record.size()) {
        auto& r = record[i];
        if (r.type == ViewpointRecord::UPDATE_SCENE) {
            if (r.viewpoint.id != viewpoint.id || !connection || r.server != connection->name) return;
            cout << "Client Record: completed update " << i << ' ' << r.viewpoint.time << ' ' << r.server << endl;
        } else if (r.type == ViewpointRecord::UPDATE_SCENE_SYNC) {
            if (r.viewpoint.id != viewpoint.id || connection) return;
            cout << "Client Record: completed update " << i << ' ' << r.viewpoint.time << " sync" << endl;
        } else if (ti < 0) { // render frame
            cout << "Client Record: render " << i << ' ' << r.viewpoint.time << endl;
            if (camera.updateOnly) {
                Viewpoint v = r.viewpoint;
                v.view = view.location.getTransform();
                renderer.debugLocal.render = true;
                renderer.debugLocal.view = r.viewpoint.view;
                renderer.render(frameSize, v, connections, syncSceneUpdate);
            }
            else {
                renderer.debugLocal.render = false;
                renderer.render(frameSize, r.viewpoint, connections, syncSceneUpdate);
            }
            clientSceneRecordSaveImage(i);
            glfwSwapBuffers(window);
        }

        // update data correction test:
        //   compare per update some time independent stats with original benchmark stats
        // requires to send per update stats, config: Server.Stats.Send = true
        // requires to disable averaging for ServerConnection::stats (in StatCounters: maxSamples(1), EMA(0))
        // requires original benchmark data in Application::benchmark.report
        if (clientSceneRecord.correctionTest && r.isUpdate()) {
            auto compareStats = [&](ServerConnection* c, const vector<string>& test) {
                string I;
                for (auto& t : benchmark.report.get_child("Samples.Connections." + c->name)) {
                    auto t2 = t.second.get_child_optional("Server.Request.ID");
                    if (t2 && t2->data() == to_string(viewpoint.id)) {
                        I = t.first;
                        break;
                    }
                }

                if (I.empty()) cout << "Client Record Error: update index " << I << " not found" << endl;

                for (auto& s : test) {
                    auto t1 = c->stats.server.get_child_optional(s);
                    auto t2 = benchmark.report.get_child_optional("Samples.Connections." + c->name + "." + I + "." + s);
//                    if (!t1 && !t2)
//                        cout << "Client Record Error " << c->name << ' ' << i << ' ' << I
//                             << ": " << s << " not found" << endl;
                    if (!t1 && t2)
                        cout << "Client Record Error " << c->name << ' ' << i << ' ' << I
                             << ": " << s << " ? " << t2->data() << endl;
                    else if (t1 && !t2)
                        cout << "Client Record Error " << c->name << ' ' << i << ' ' << I
                             << ": " << s << ' ' << t1->data() << " ?" << endl;
                    else if (t1 && t2 && t1->data() != t2->data())
                        cout << "Client Record Error " << c->name << ' ' << i << ' ' << I
                             << ": " << s << ' ' << t1->data() << ' ' << t2->data() << endl;
                }
            };

            vector<string> test = {
                "Server.Data.Triangle Count",
                "Server.Data.Texture.Primary.Height",
                "Server.Data.Texture.Cubemap.Height",
                "Server.Data.Texture.Blocks.Height",

                "Server.Data.View.Primary.Triangle Count",
                "Server.Data.View.Primary.Fragments / Pixel",
                "Server.Data.View.Primary.Tile Count",

                "Server.Data.View.Front.Triangle Count",
                "Server.Data.View.Front.Fragments / Pixel",
                "Server.Data.View.Front.Tile Count",

                "Server.Data.View.Right.Triangle Count",
                "Server.Data.View.Right.Fragments / Pixel",
                "Server.Data.View.Right.Tile Count",

                "Server.Lighthouse.Primary.Rays",
                "Server.Lighthouse.Primary.Rays.Extension",
                "Server.Lighthouse.Primary.Rays.Shadow",

                "Server.Lighthouse.Cubemap.Rays",
                "Server.Lighthouse.Cubemap.Rays.Extension",
                "Server.Lighthouse.Cubemap.Rays.Shadow"

                // TODO compare transferred data sizes
            };

            if (r.type == ViewpointRecord::UPDATE_SCENE)
                compareStats(connection, test);
            else for (auto& c : connections) compareStats(&c, test);
        }
    }

    // schedule next
    for (i++; i < record.size() && (ti < 0 || i <= ti); i++) {
        auto& r = record[i];
        if (r.type == ViewpointRecord::RENDER_FRAME)
            run([this]() { clientSceneRecordProcess(); });
        else if (r.type == ViewpointRecord::SEND_VIEWPOINT) {
            cout << "Client Record: send viewpoint " << i << ' ' << r.viewpoint.time << ' ' << r.server << endl;
            for (auto& c : connections)
                if (c.name == r.server) {
                    c.sendViewpoint(r.viewpoint,
                                    benchmark.simulated.use && benchmark.simulated.renderAllViewpointsOnServer ?
                                    UpdateMode::SERVER_ONLY : UpdateMode::DISABLE);
                    break;
                }
            continue; // schedule next
        } else if (r.type == ViewpointRecord::UPDATE_SCENE) {
            cout << "Client Record: request update " << i << ' ' << r.viewpoint.time << ' ' << r.server << endl;
            for (auto& c : connections)
                if (c.name == r.server) {
                    c.sendViewpoint(r.viewpoint, UpdateMode::FORCE);
                    break;
                }
        } else if (r.type == ViewpointRecord::UPDATE_SCENE_SYNC) {
            cout << "Client Record: request update " << i << ' ' << r.viewpoint.time << " sync" << endl;

            // find all records of current update
            int j = i;
            while (j < record.size() && record[j].type == ViewpointRecord::UPDATE_SCENE_SYNC &&
                   record[j].viewpoint.id == r.viewpoint.id) j++;
            j--;

            // sort viewpoints by connections
            vector<Viewpoint> viewpoints;
            for (auto& c : connections)
                for (int k = i; k <= j; k++)
                    if (c.name == record[k].server)
                        viewpoints.push_back(record[k].viewpoint);

            allocateSynchronizedScenes(viewpoints);

            int k = 0;
            for (auto& c : connections)
                c.sendViewpoint(viewpoints[k++], UpdateMode::FORCE);

            i = j;
        } else clientSceneRecordStop();
        return; // schedule only one record (with exception for SEND_VIEWPOINT)
    }

    clientSceneRecordStop();
}

void Application::clientSceneRecordSaveImage(int index) {
    // read image
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    vector<unsigned char> image(frameSize.x * frameSize.y * 3);
    glReadPixels(0, 0, frameSize.x, frameSize.y, GL_RGB, GL_UNSIGNED_BYTE, image.data());

    // path
    std::stringstream ss;
    ss << std::setw(5) << std::setfill('0') << index;
    auto fileName = camera.updateOnly ? clientSceneRecord.fileThirdPerson : clientSceneRecord.file;
    string path = realOutputPath() + boost::replace_all_copy(fileName, "#", ss.str());

    // flip vertically
    for (int y = 0; y < frameSize.y / 2; y++)
        for (int x = 0; x < frameSize.x * 3; x++)
            swap(image[y * frameSize.x * 3 + x], image[(frameSize.y - y - 1) * frameSize.x * 3 + x]);

    // save image
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream f(path, ios::binary);
    f << "P6" << endl << frameSize.x << ' ' << frameSize.y << endl << "255" << endl;
    f.write((char*) image.data(), frameSize.x * frameSize.y * 3);
}

pair<float, mat4> Application::getReferenceFrameData(int i) const {
    auto& records = sceneRecord.data;
    auto& r = records[i];
    if (r.type != ViewpointRecord::RENDER_FRAME)
        return make_pair(r.viewpoint.time, r.viewpoint.view);

    // last update
    auto last = i;
    while (--last >= 0 && !records[last].isUpdate());
    if (last < 0)
        return make_pair(r.viewpoint.time, r.viewpoint.view);

    // video stream
    for (auto& c : connections)
        if ((serverSceneRecord.server.empty() || serverSceneRecord.server == c.name) &&
            !c.scenes.empty() && c.scenes.back()->isVideoStream())
                return make_pair(records[last].viewpoint.time, records[last].viewpoint.view);

    // second last update
    auto secondLast = last;
    while (--secondLast >= 0 && (!records[secondLast].isUpdate() || records[last].viewpoint.id == records[secondLast].viewpoint.id));
    if (secondLast < 0)
        return std::make_pair(records[last].viewpoint.time, r.viewpoint.view);

    // interpolate time when blending scenes
    return std::make_pair(glm::mix(records[secondLast].viewpoint.time,
                                   records[last].viewpoint.time,
                                   std::min(1.0f, r.viewpoint.latency)),
                          r.viewpoint.view);
}

void Application::serverSceneRecordProcess(float time, glm::mat4 view, const std::vector<unsigned char>& data) {
    auto& i = serverSceneRecord.index;
    if (i == SceneRecord::STOP) return;

    // process current
    if (i >= 0 && i < sceneRecord.data.size()) {
        const auto [rTime, rView] = getReferenceFrameData(i);
        if (time != rTime || !(view == rView)) {
            cout << "Server Record: wait " << i << ' ' << rTime << ' ' << time << endl;
            return;
        }

        // save image
        cout << "Server Record: save " << i << ' ' << rTime << endl;
        std::stringstream ss;
        ss << std::setw(5) << std::setfill('0') << i;
        string path = realOutputPath() + boost::replace_all_copy(serverSceneRecord.file, "#", ss.str());
        fs::create_directories(fs::path(path).parent_path());
        std::ofstream f(path, ios::binary);
        f.write(reinterpret_cast<const char*>(data.data()), data.size());
    }

    // schedule next
    while (++i < sceneRecord.data.size()) {
        auto& r = sceneRecord.data[i];
        if (r.type == ViewpointRecord::RENDER_FRAME) {
            const auto [rTime, rView] = getReferenceFrameData(i);
            for (auto& c : connections) {
                if (!serverSceneRecord.server.empty() && serverSceneRecord.server != c.name)
                    continue;
                cout << "Server Record: render " << i << ' ' << rTime << endl;
                c.request.write(RequestType::RENDER_REFERENCE);
                c.request.write(rTime);
                c.request.write(rView);
                break;
            }
            break;
        }
    }

    if (i >= sceneRecord.data.size())
        serverSceneRecordStop();
}

void Application::allocateSynchronizedScenes(const vector<Viewpoint>& viewpoints) {
    assert(viewpoints.size() == connections.size());
    vector<Scene*> scenes;

    int i = 0;
    for (auto& c : connections)
        scenes.push_back(new Scene(&c, viewpoints[i++]));

    i = 0;
    for (auto& c : connections) {
        auto s = scenes[i++];
        s->sibling = scenes[i % scenes.size()]; // looped linked list
    }

    i = 0;
    for (auto& c : connections)
        c.syncScenes.lock()->push_back(scenes[i++]);
}

void Application::sendViewpoint(const glm::mat4& transform, bool allowUpdate) {
    if (allowViewpointUpdate()) {
        if (syncSceneUpdate) {
            // synchronize updates: client decides when to render
            viewpointID++; // same id for all viewpoints

            // collect viewpoints, compute synchronized variables and detect if to request scenes
            bool requestScene = allowUpdate;
            float syncDeltaTime = 0, syncLatency = 0;
            vector<Viewpoint> viewpoints;
            for (auto& c : connections) {
                auto viewpoint = c.createViewpoint(transform);
                viewpoint.flags |= Viewpoint::SYNCHRONIZED;
                if (TimerCPU::diff(c.prevRequestTime, frameTime) < viewpoint.deltaTime * 1000000 || !c.allowSceneRequest())
                    requestScene = false;
                syncDeltaTime = std::max(syncDeltaTime, viewpoint.deltaTime);
                syncLatency = std::max(syncLatency, viewpoint.latency);
                viewpoints.push_back(viewpoint);
            }

            if (connections.size() > 1) {

                // synchronize viewpoints
                bool hasSourceForPVS = false;
                int i = 0;
                for (auto& c : connections) {
                    auto& viewpoint = viewpoints[i++];
                    viewpoint.deltaTime = syncDeltaTime;
                    viewpoint.latency = syncLatency;
                    if (c.name != syncSourceForPVS)
                        viewpoint.flags |= Viewpoint::DONT_SEND_PVS;
                    else hasSourceForPVS = true;
                }

                if (!hasSourceForPVS) // fallback when source not found
                    for (auto& v : viewpoints)
                        v.flags &= ~Viewpoint::DONT_SEND_PVS;

                // preallocate synchronized scenes
                if (requestScene)
                    allocateSynchronizedScenes(viewpoints);
            }

            // send viewpoints
            int i = 0;
            for (auto& c : connections)
                c.sendViewpoint(viewpoints[i++], requestScene ? UpdateMode::FORCE : UpdateMode::DISABLE);

        } else if (asyncInterleave) {
            // asynchronous updates: client decides when to render
            // try to interleave requests to minimize delta time between all request from all connections

            // find time of last request and min common delta time
            TimerCPU::TimePoint lastRequestTime = connections.front().prevRequestTime;
            double minDeltaTime = connections.front().minDeltaTime();
            for (auto& c : connections) {
                if (c.prevRequestTime > lastRequestTime)
                    lastRequestTime = c.prevRequestTime;
                minDeltaTime = std::min(minDeltaTime, c.minDeltaTime());
            }

            // forbid update when too soon after last update
            if (TimerCPU::diff(lastRequestTime, frameTime) < minDeltaTime / connections.size())
                allowUpdate = false;

            for (auto& c : connections) {
                viewpointID++;
                auto viewpoint = c.createViewpoint(transform);
                if (allowUpdate && c.allowSceneRequest() &&
                    TimerCPU::diff(c.prevRequestTime, frameTime) > viewpoint.deltaTime * 1000000) {
                    c.sendViewpoint(viewpoint, UpdateMode::FORCE);
                    allowUpdate = false; // max one request per frame
                } else c.sendViewpoint(viewpoint, UpdateMode::DISABLE);
            }
        } else {
            // asynchronous updates: server decides when to render
            for (auto& c : connections) {
                viewpointID++;
                c.sendViewpoint(c.createViewpoint(transform),
                                allowUpdate && c.allowSceneRequest() ? UpdateMode::ALLOW : UpdateMode::DISABLE);
            }
        }
    }
}

void Application::onSceneUpdate(Scene* s) {
    auto c = s->connection;

    // scene record
    if (s->benchmark) {
        if (s->allReady()) {
            auto addViewpointHistory = [&] (auto c) {
                auto h = c->viewpointHistory.lock();
                while (!h->empty() && h->front().id < s->viewpoint.id) {
                    sceneRecord.data.push_back({ViewpointRecord::SEND_VIEWPOINT, h->front(), c->name});
                    h->pop();
                }
                if (!h->empty() && h->front().id == s->viewpoint.id)
                    h->pop(); // without inserting as it is added as scene update record
            };

            // viewpoints
            addViewpointHistory(c);
            if (s->isSynchronized())
                for (auto s2 = s->sibling; s2 != s; s2 = s2->sibling)
                    addViewpointHistory(s2->connection);

            // updates
            if (s->isSynchronized()) {
                // synchronized updates are added at once so they are in a row
                sceneRecord.data.push_back({ViewpointRecord::UPDATE_SCENE_SYNC, s->viewpoint, c->name});
                for (auto s2 = s->sibling; s2 != s; s2 = s2->sibling)
                    sceneRecord.data.push_back({ViewpointRecord::UPDATE_SCENE_SYNC, s2->viewpoint, s2->connection->name});
            } else sceneRecord.data.push_back({ViewpointRecord::UPDATE_SCENE, s->viewpoint, c->name});
        }
    }

    // global stats
    if (s->allReady() && s->viewpoint.id > lastUpdateViewpoint.id) {
        const auto updateDeltaTime = TimerCPU::diff(prevUpdateTime, frameTime);
        const auto latency = std::max(0.0, elapsedTime() - s->viewpoint.time);
        lastUpdateViewpoint = s->viewpoint;
        prevUpdateTime = frameTime;

        StatCounters<double>::Entries e;
        e.emplace_back("Timepoint", s->viewpoint.time);
        e.emplace_back("Timepoint.Frame", elapsedTime());
        e.emplace_back("Rate", updateDeltaTime > 0 ? 1000000.0f / updateDeltaTime : 0);
        e.emplace_back("Delta Time", updateDeltaTime / 1000);
        e.emplace_back("Latency", latency * 1000);
        e.emplace_back("Frames/Update", framesFromUpdate);
        counters.update.add(e);
        if (s->benchmark) benchmark.update.add(e);
        stats.update = counters.update.stats(counters.minMax);
        framesFromUpdate = 0;
    }

    if (clientSceneRecord.running() && s->allReady())
        clientSceneRecordProcess(s->viewpoint, s->isSynchronized() ? nullptr : c);

    if (s->viewpoint.id == benchmark.simulated.serverWaitingForID)
        benchmark.simulated.serverWaitingForID = -1;

    // PVS synchronization correction test
    if (s->allReady() && s->isSynchronized()) {
        for (auto s2 = s->sibling; s2 != s; s2 = s2->sibling) {
            if (s->vertexCount != s2->vertexCount)
                cout << "PVS Sync Error " << s->viewpoint.id << ": vertex count "
                     << s->vertexCount << ' ' << s2->vertexCount << endl;
            if (s->debug.PVSCacheSamples.size() != s2->debug.PVSCacheSamples.size())
                cout << "PVS Sync Error " << s->viewpoint.id << ": cache sample count "
                     << s->debug.PVSCacheSamples.size() << ' ' << s2->debug.PVSCacheSamples.size() << endl;
            else for (int i=0; i<s->debug.PVSCacheSamples.size(); i++) {
                auto& S1 = s->debug.PVSCacheSamples[i], &S2 = s2->debug.PVSCacheSamples[i];
                    if (S1.transform != S2.transform || S1.projection != S2.projection)
                        cout << "PVS Sync Error " << s->viewpoint.id << ": cache sample " << i << endl;
                }
        }
    }
}

void Application::run() {
    vec2 prevCursor;
    TimerCPU timerCPU;
    TimerGL timerGPU;
    TimerGL timerGPUwithoutTasks;

    while (!glfwWindowShouldClose(window)) {
        auto& io = ImGui::GetIO();
        frameID++;
        timerCPU.reset();
        timerGPU.reset();
        timerCPU("Base");

        // time
        auto now = TimerCPU::now();
        frameDeltaTime = (float)(TimerCPU::diff(frameTime, now) / 1000000);
        frameTime = now;

        // cursor
        vec2 relCursor = cursorPos - prevCursor;
        prevCursor = cursorPos;

        // camera
        camera.step(frameDeltaTime);

        // benchmark path
        if (benchmarkPath) {
            if (!camera.inRange()) {
                benchmark.stop();
                camera.play = false;
                benchmarkPath = false;
            } else benchmarkPath = benchmark.running;
        }

        // decoding acceleration
#ifdef DECODE_DXVA2
        if (video.hwaccel.available)
        {
            const auto changedPrimary = video.hwaccel.primCubeM != video.hwaccel.inUsePrimCubeM;
            const auto changedBlocks = video.hwaccel.blocks != video.hwaccel.inUseBlocks;
            const auto changedHWAccel = changedPrimary || changedBlocks;
            const auto useHWAccel = video.hwaccel.primCubeM || video.hwaccel.blocks;
            auto resetState = false;

            if (changedPrimary)
            {
                video.hwaccel.inUsePrimCubeM = video.hwaccel.primCubeM;
                resetState = true;
            }
            if (changedBlocks)
            {
                video.hwaccel.inUseBlocks = video.hwaccel.blocks;
                resetState = true;
            }
            if (useHWAccel && !hwAccelInterop)
            {
                hwAccelInterop = std::make_shared<hw_accel::dxva2_opengl>();
                resetState = true;
            }
            else if (!useHWAccel && hwAccelInterop)
            {
                hwAccelInterop.reset();
                resetState = true;
            }
            if (resetState)
                requestAll(RequestType::RESET_STATE);
        }
#endif

        // view
        bool automationStarting = benchmark.automation.use && benchmark.automation.isInState(Benchmark::Automation::State::eBenchmarkStart);
        bool automationRunning = benchmark.automation.use && benchmark.automation.isInState(Benchmark::Automation::State::eBenchmarkWait);
        mat4 viewMat, updateViewMat;
        if (automationStarting || automationRunning || camera.play) {

            if (automationStarting)
                camera.time = camera.start;
            if (automationRunning)
                camera.clamp();
            else camera.cycle();
            updateViewMat = camera.view();

            if (camera.updateOnly && !automationStarting && !automationRunning) {
                view.update(window, cursorPos, !io.WantCaptureMouse, !io.WantCaptureKeyboard);
                viewMat = view.location.getTransform();
                renderer.debugLocal.render = true;
                renderer.debugLocal.view = updateViewMat;
            } else {
                view.update(window, cursorPos, false, false);
                viewMat = updateViewMat;
                renderer.debugLocal.render = false;
            }

        } else {
            view.update(window, cursorPos, !io.WantCaptureMouse, !io.WantCaptureKeyboard);
            viewMat = updateViewMat = view.location.getTransform();
            renderer.debugLocal.render = false;
        }

        // recording
        if (camera.record)
            camera.path.add(CameraPath::Sample(camera.time, inverse(viewMat)));

        // connections
        for (auto& c : connections)
            c.onFrameUpdate();
        framesFromUpdate++;

        // request scene
        bool mb2 = !io.WantCaptureMouse && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2);
        bool mb3 = !io.WantCaptureMouse && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_3);
        sendViewpoint(updateViewMat, mb2 != updateScene || (mb3 && !prevMouseButton3));
        prevMouseButton3 = mb3;

        // process gpu tasks
        frameGPUtimeWithoutTasks = (float)sum(timerGPUwithoutTasks.getEntries());
        timerCPU("GPU tasks");
        timerGPU("GPU tasks");
        gpu.processTasks();
        timerCPU();
        timerGPU();
        timerGPUwithoutTasks.reset();
        timerGPUwithoutTasks("GPU");

        // process cpu tasks
        timerCPU("CPU tasks");
        processTasks();

        // benchmark
        benchmark.process();

        // render
        timerCPU("Render");
        bool renderFrame = allowRender();

        Viewpoint frameViewpoint;
        frameViewpoint.time = (float)elapsedTime();
        frameViewpoint.view = viewMat;
        frameViewpoint.deltaTime = 0;
        for (auto& c : connections)
            frameViewpoint.deltaTime = std::max(frameViewpoint.deltaTime, (float)c.minDeltaTime());
        if (!syncSceneUpdate) frameViewpoint.deltaTime /= connections.size();
        frameViewpoint.latency = (float)(TimerCPU::diff(prevUpdateTime, frameTime) / frameViewpoint.deltaTime); // for blending
        frameViewpoint.deltaTime /= 1000000;

        // simulated benchmark, requires constant video bitrate and works only with one connection
        auto& bs = benchmark.simulated;
        if (bs.isRunning()) {
            renderFrame = true;
            if (bs.waitingForServer())
                renderFrame = false;
            else {
                // check for ending
                bool ending = bs.frameTime() + camera.start >= camera.realStop();
                if (ending) {
                    if (!benchmark.running)
                        bs.frame = -1; // end
                    else 
                        benchmark.stop(); // request stop
                }
                if (bs.frame > 0 && !benchmark.running)
                    bs.frame = -1;

                // server
                if (bs.isRunning() && !bs.serverProcessed) {
                    Viewpoint serverViewpoint;
                    serverViewpoint.id = viewpointID++;
                    serverViewpoint.time = bs.frameTime();
                    // subtract latency here in camera path instead of adding it to client frame
                    // with this way we can render all frames on the path
                    serverViewpoint.view = inverse(camera.path.sample(std::max(0.0f, bs.frameTime() - bs.latency()) + camera.start).mat()); // TODO missing camera.speed?
                    serverViewpoint.deltaTime = 1.0f / bs.serverFPS();
                    serverViewpoint.latency = bs.latency();
                    serverViewpoint.videoFramerate = (int)bs.serverFPS();
                    serverViewpoint.videoBitrate = video.bitrate.bitrate; // constant

                    if (ending || bs.updateServer()) { // request update and wait
                        connections.front().sendViewpoint(serverViewpoint, UpdateMode::FORCE);
                        bs.serverWaitingForID = serverViewpoint.id;
                        bs.serverProcessed = true;
                    } else connections.front().sendViewpoint(serverViewpoint, bs.renderAllViewpointsOnServer ? UpdateMode::SERVER_ONLY : UpdateMode::DISABLE);
                }

                // client
                if (bs.isRunning() && !bs.waitingForServer()) {
                    frameViewpoint.time = bs.frameTime() + bs.latency();
                    frameViewpoint.view = inverse(camera.path.sample(bs.frameTime() + camera.start).mat()); // TODO missing camera.speed?
                    frameViewpoint.deltaTime = 1.0f / bs.serverFPS();

                    // for blending
                    if (bs.frame / bs.serverDivideFPS == 0) frameViewpoint.latency = 1; // ignore previous scene for first update
                    else frameViewpoint.latency = (float) (bs.frame % bs.serverDivideFPS + 1) / bs.serverDivideFPS;

                    bs.frame++;
                    bs.serverProcessed = false;
                } else renderFrame = false;
            }
        }

        timerGPU("Render");
        if (renderFrame) {
            renderer.render(frameSize, frameViewpoint, connections, syncSceneUpdate);
            if (benchmark.frames() && benchmark.simulated.isRunning() && benchmark.simulated.renderClientImages)
                clientSceneRecordSaveImage((int)sceneRecord.data.size());
        } else {
            glViewport(0, 0, frameSize.x, frameSize.y);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        }
        timerGPU();

        gpu.processInstantTasks();

        // GUI
        timerCPU("GUI");
        GUI::NewFrame(!showGUI);
        // ImGui::ShowDemoWindow();
        GUI();
        ImGui::EndFrame();
        timerGPU("GUI");
        if (showGUI) GUI::Render();
        timerGPU();

        gpu.processInstantTasks();
        textureManager.GC();

        timerCPU("Swap");
        glfwSwapBuffers(window);
        glfwPollEvents();

        timerGPU.finish();
        timerCPU.finish();

        // stats
        StatCounters<double>::Entries e;
        e.emplace_back("Timepoint", frameViewpoint.time);
        e.emplace_back("FPS", 1.0f / frameDeltaTime);
        e.emplace_back("Latency", (frameViewpoint.time - lastUpdateViewpoint.time) * 1000);
        e.emplace_back("Time", frameDeltaTime * 1000);
        e.emplace_back("Time.GPU.Without Tasks", frameGPUtimeWithoutTasks / 1000);
        addAll(e, "Time.CPU", timerCPU.getEntries());
        addAll(e, "Time.GPU", timerGPU.getEntries());
        if (renderFrame) {
            renderer.timer.finish();
            addAll(e, "Time.GPU.Render", renderer.timer.getEntries(), false);
        }
        counters.frame.add(e);
        stats.frame = counters.frame.stats(counters.minMax);

        if (renderFrame && benchmark.frames()) {
            benchmark.frame.add(e);

            // img reconstruction subsampling
            if (frameViewpoint.time >= benchmark.nextImgSampleTime) {
                if (benchmark.automation.imgSamplingFPS > 0.0f && !benchmark.simulated.isRunning())
                    benchmark.nextImgSampleTime += 1.0 / benchmark.automation.imgSamplingFPS;
                sceneRecord.data.push_back({ViewpointRecord::RENDER_FRAME, frameViewpoint, ""});
            }
        }

        if (unfocusedWindowSleep > 0 && !glfwGetWindowAttrib(window, GLFW_FOCUSED))
            this_thread::sleep_for(chrono::milliseconds(unfocusedWindowSleep));

        if (benchmark.automation.use)
            benchmark.automationStep(frameViewpoint.time);
    }

    // CLEANUP

    // ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // gui state
    if (!gui.stateFile.empty())
        pt::write_info(gui.stateFile, gui.state);

#ifdef ENABLE_TURBOJPEG
    tjDestroy(jpegDecoder.get());
#endif
}
