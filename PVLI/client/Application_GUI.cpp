#include "Application.h"
#include <iomanip>
#include <sstream>
#include <boost/algorithm/string/predicate.hpp>
#include "asioHelpers.h"
#include "imguiHelper.h"
#include "common.h"
#include "graphic.h"
#include "Timer.h"
#include "TimerGL.h"
#include "VertexCompressor.h"
#include "ffmpegDecoder.h"
#include "Renderer.h"
#include "compression/Compression.h"
#include "compression/RLE.h"
#include "Scene.h"
#include "SceneView.h"

using namespace std;
using namespace glm;

static void ImGuiDisableBlend(const ImDrawList* parent_list, const ImDrawCmd* cmd) {
    glDisable(GL_BLEND);
}

static void ImGuiEnableBlend(const ImDrawList* parent_list, const ImDrawCmd* cmd) {
    glEnable(GL_BLEND);
}

static void viewTree(const pt::ptree& tree, const string& name, int open = 1) {
    open--;
    auto text = name;
    if (!tree.data().empty())
        text += ' ' + pt::to_string_value(tree);
    if (tree.empty()) {
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetTreeNodeToLabelSpacing());
        ImGui::Text(text.c_str());
    } else if (tree.size() == 1 && tree.data().empty())
        viewTree(tree.front().second, name + '.' + tree.front().first, open);
    else if (ImGui::TreeNodeEx(name.c_str(), open >= 0 ? ImGuiTreeNodeFlags_DefaultOpen : ImGuiTreeNodeFlags_None, text.c_str())) {
        for (auto& t : tree)
            viewTree(t.second, t.first, open);
        ImGui::TreePop();
    };
};

void Application::GUI() {
    ImGui::Begin("Client");
    if (ImGui::BeginTabBar("Client Tab Bar")) {

        if (ImGui::BeginTabItem("Common")) {
            ImGui::Text("Time: %.2f", elapsedTime());
            if (ImGui::Button("Reset time")) resetTime();
            if (ImGui::Checkbox("VSync", &vsync))
                glfwSwapInterval(vsync ? 1 : 0);
            ImGui::Checkbox("TCPNoDelay", &TCPNoDelay);

            if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Checkbox("Render", &render);
                ImGui::Checkbox("Update Scene", &updateScene);
                ImGui::Checkbox("Update Viewpoint", &updateViewpoint);
                if (ImGui::Checkbox("Sync Update", &syncSceneUpdate))
                    requestAll(RequestType::RESET_STATE);
                if (ImGui::Checkbox("Async Interleave", &asyncInterleave) && !syncSceneUpdate)
                    requestAll(RequestType::RESET_STATE);
                ImGui::InputDouble("Buffer Size Multiple", &bufferSizeMultiple, 0, 0, "%g");
                ImGui::InputInt("Cache", &maxCachedScenes);
                ImGui::InputInt("Process", &maxProcessingScenes);
                ImGui::InputInt("Reuse", &maxReuseScenes);
                ImGui::Checkbox("Reuse Textures", &reuseTextures);
                ImGui::InputInt("Upload Chunk Size", &uploadChunkSize);
                ImGui::Checkbox("Use Staging Buffers", &useStagingBuffers);

                auto bufferUsageCombo = [&] (auto name, auto& usage) {
                    string currentUsage;
                    for (auto& u : bufferUsages)
                        if (u.second == usage)
                            currentUsage = u.first;
                    if (ImGui::BeginCombo(name, currentUsage.c_str())) {
                        for (auto& u : bufferUsages)
                            if (ImGui::Selectable(u.first.c_str(), (usage == u.second)))
                                usage = u.second;
                        ImGui::EndCombo();
                    }
                };

                bufferUsageCombo("Staging Buffer Usage", stagingBufferUsage);
                bufferUsageCombo("Buffer Usage", bufferUsage);
            }

            if (ImGui::CollapsingHeader("Update Prediction", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::InputDouble("Multiplier", &updatePrediction.multiplier, 0, 0, "%g");
                ImGui::InputDouble("Min Rate", &updatePrediction.minRate, 0, 0, "%g");
                ImGui::InputDouble("Max Rate", &updatePrediction.maxRate, 0, 0, "%g");
                ImGui::InputDouble("Override Latency [ms]", &updatePrediction.overrideLatencyInMs, 0, 0, "%g");
            }

            if (ImGui::CollapsingHeader("Video", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::InputInt("Bitrate", &video.bitrate.bitrate);
                ImGui::InputInt("Bitrate Min", &video.bitrate.min);
                ImGui::InputFloat("Bitrate Time Factor", &video.bitrate.timeFactor, 0, 0, "%g");
                ImGui::InputFloat("Bitrate Time Offset", &video.bitrate.timeOffsetInMs, 0, 0, "%g");
                ImGui::InputInt("Bitrate Step Up", &video.bitrate.stepUp);
                ImGui::InputInt("Bitrate Step Down", &video.bitrate.stepDown);
                ImGui::InputFloat("Framerate Factor", &video.framerate.factor, 0, 0, "%g");
                ImGui::InputInt("Framerate Min", &video.framerate.min);
                ImGui::InputFloat("Framerate Offset", &video.framerate.offset, 0, 0, "%g");
                ImGui::Checkbox("HW Accel. decoding - Primary + Cube", &video.hwaccel.primCubeM);
                ImGui::Checkbox("HW Accel. decoding - Blocks", &video.hwaccel.blocks);

            }

            if (ImGui::CollapsingHeader("Pixel Relocator", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Checkbox("Enable", &relocatePixels);
                if (ImGui::Checkbox("Skip Full layers", &pixelRelocator.skipFullLayers))
                    pixelRelocator.loadShaders();
            }

            if (ImGui::CollapsingHeader("Depth Peeling", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Checkbox("Use triangle-layer mask", &depth.useTriangleLayerMask);
                ImGui::InputFloat("Epsilon", &depth.epsilon, 0, 0, "%g");
            }

            if (ImGui::CollapsingHeader("GPU Tasks", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::InputInt("Tasks", &gpu.maxTasks);
                ImGui::InputDouble("Compute", &gpu.maxCompute);
                ImGui::InputDouble("Upload", &gpu.maxUpload);
                ImGui::InputInt("Query Delay", &gpu.queryDelay);
                ImGui::InputDouble("Elapsed Max", &gpu.elapsedMax, 0, 0, "%g");
                ImGui::InputDouble("Elapsed Adaptive", &gpu.elapsedAdaptive, 0, 0, "%g");
            }

            if (ImGui::CollapsingHeader("Data", ImGuiTreeNodeFlags_DefaultOpen))
                ImGui::Text("Textures: %.3fM", (float) textureManager.getSize() / 1000000);
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Camera")) {
            ImGui::InputFloat3("Position", &view.location.pos.x, 2);
            ImGui::InputFloat2("Rotation", &view.location.rot.x, 2);
            ImGui::DragFloat("Speed", &view.speed, 0.1f, 0, 100);
            ImGui::DragFloat("Fast Speed", &view.speedFast, 0.05f, 1, 10);

            if (ImGui::CollapsingHeader("Path", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::PushID("Path");

                if (ImGui::Checkbox("Play", &camera.play))
                    if (camera.play)
                        camera.record = false;
                if (ImGui::Checkbox("Record", &camera.record))
                    if (camera.record) {
                        camera.play = false;
                        camera.time = 0;
                        camera.start = 0;
                        camera.stop = 0;
                        camera.speed = 1;
                        camera.path.clear();
                    }

                ImGui::Checkbox("Update Only", &camera.updateOnly);
                ImGui::Text("Path Duration: %.2f", camera.path.duration());
                ImGui::Text("Real Duration: %.2f", (camera.realStop() - camera.start) / camera.speed);
                ImGui::DragFloat("Start", &camera.start, 0.1f);
                ImGui::DragFloat("Stop", &camera.stop, 0.1f);
                ImGui::DragFloat("Time", &camera.time, 0.1f);
                ImGui::DragFloat("Speed###playbackSpeed", &camera.speed, 0.01f);

                auto s = ImVec2(ImGui::GetWindowSize().x * 0.2f, 0);
                if (ImGui::Button("<", s)) camera.speed = -1;
                ImGui::SameLine();
                if (ImGui::Button("||", s)) camera.speed = 0;
                ImGui::SameLine();
                if (ImGui::Button(">", s)) camera.speed = 1;

                ImGui::InputText("File", &camera.file);
                if (ImGui::Button("Save"))
                    camera.path.saveToFile(camera.file);
                ImGui::SameLine();
                if (ImGui::Button("Load"))
                    camera.path.loadFromFile(camera.file);

                ImGui::PopID();
            }

            if (ImGui::CollapsingHeader("Scene Record", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::PushID("Scene Record");
                ImGui::Text("Records: %d, %d / %d", clientSceneRecord.index, serverSceneRecord.index, (int)sceneRecord.data.size());

                if (ImGui::Button("Client"))
                    clientSceneRecordStart();
                ImGui::SameLine();
                if (ImGui::Button("Server"))
                    serverSceneRecordStart();
                ImGui::SameLine();
                if (ImGui::Button("Stop")) {
                    clientSceneRecordStop();
                    serverSceneRecordStop();
                }
                ImGui::SameLine();
                if (ImGui::Button("Simulated"))
                    benchmark.generateSimulatedSceneRecord();

                ImGui::InputText("File", &sceneRecord.file);
                if (ImGui::Button("Save"))
                    sceneRecord.save(sceneRecord.file);
                ImGui::SameLine();
                if (ImGui::Button("Load"))
                    sceneRecord.load(sceneRecord.file);
                ImGui::SameLine();
                if (ImGui::Button("Load Benchmark"))
                    sceneRecord.load(realOutputPath() + benchmark.outputPrefix + "scene_record.txt");

                // scene record camera step
                auto& index = sceneRecord.index;
                auto& data = sceneRecord.data;
                if (!data.empty()) {
                    bool setCamera = false;
                    if (ImGui::InputInt("Index", &sceneRecord.index)) setCamera = true;
                    if (ImGui::Button("--")) { // previous scene update
                        setCamera = true;
                        index--;
                        while (index >= 0 && index < data.size() && !data[index].isUpdate())
                            index--;
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("-")) { // previous render frame
                        setCamera = true;
                        index--;
                        while (index >= 0 && index < data.size() && data[index].type != ViewpointRecord::RENDER_FRAME)
                            index--;
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("+")) { // next render frame
                        setCamera = true;
                        index++;
                        while (index >= 0 && index < data.size() && data[index].type != ViewpointRecord::RENDER_FRAME)
                            index++;
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("++")) { // next scene update
                        setCamera = true;
                        index++;
                        while (index >= 0 && index < data.size() && !data[index].isUpdate())
                            index++;
                    }
                    if (index < 0) index = 0;
                    else if (index >= data.size()) index = (int)data.size() - 1;
                    auto& r = data[index];
                    if (setCamera) view.location.setTransform(r.viewpoint.view);
                    ImGui::SameLine();
                    if (ImGui::Button("Restep Record")) {
                        updateScene = false;
                        clientSceneRecordStart(index);
                    } if (r.type == ViewpointRecord::UPDATE_SCENE) {
                        ImGui::SameLine();
                        if (ImGui::Button("Update Scene")) {
                            updateScene = false;
                            for (auto& c : connections)
                                if (r.server == c.name) {
                                    c.sendViewpoint(r.viewpoint, UpdateMode::FORCE);
                                    break;
                                }
                        }
                    }
                    if (r.type == ViewpointRecord::RENDER_FRAME)
                        ImGui::Text("RENDER_FRAME");
                    else if (r.type == ViewpointRecord::SEND_VIEWPOINT)
                        ImGui::Text("SEND_VIEWPOINT");
                    else if (r.type == ViewpointRecord::UPDATE_SCENE)
                        ImGui::Text("UPDATE_SCENE");
                    else if (r.type == ViewpointRecord::UPDATE_SCENE_SYNC)
                        ImGui::Text("UPDATE_SCENE_SYNC");
                    else ImGui::Text("UNKNOWN");
                    ImGui::SameLine();
                    ImGui::Text((r.server + " " + to_string(r.viewpoint.id)).c_str());
                }

                ImGui::PopID();
            }

            if (ImGui::CollapsingHeader("Locations", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::AlignTextToFramePadding();
                ImGui::Text("Locations");
                ImGui::SameLine();
                if (ImGui::Button("Add"))
                    cameraLocations.push_back(view.location);

                int i = 0, rem = -1;
                for (auto& l : cameraLocations) {
                    ImGui::PushID(i);
                    if (ImGui::Button(" S ")) view.location = l;
                    ImGui::SameLine();
                    ImGui::Text("%.1f %.1f %.1f  %.1f,%.1f", l.pos.x, l.pos.y, l.pos.z, l.rot.x, l.rot.y);
                    ImGui::SameLine();
                    if (ImGui::Button("X")) rem = i;
                    ImGui::PopID();
                    i++;
                }
                if (rem >= 0)
                    cameraLocations.erase(cameraLocations.begin() + rem);
            }
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Renderer")) {
            renderer.GUI();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }
    ImGui::End();

    auto addAllNone = [&](const string& name, auto call) {
        ImGui::AlignTextToFramePadding();
        ImGui::Text((name + ":").c_str());
        ImGui::PushID(name.c_str());
        ImGui::SameLine();
        if (ImGui::Button("All")) call(true);
        ImGui::SameLine();
        if (ImGui::Button("None")) call(false);
        ImGui::PopID();
    };

    ImGui::Begin("Servers");
    if (ImGui::BeginTabBar("Connections")) {
        for (auto& c : connections) {
            if (ImGui::BeginTabItem(c.name.c_str())) {
                if (ImGui::BeginTabBar("Server")) {
                    if (ImGui::BeginTabItem("Config")) {
                        if (ImGui::BeginChild(c.name.c_str())) {
                            ImGui::Checkbox("Sync Config Request", &syncServerConfigRequest);
                            ImGui::Checkbox("Update Scene", &c.updateScene);

                            bool isSourceForPVS = c.name == syncSourceForPVS;
                            if (ImGui::Checkbox("Sync Source For PVS", &isSourceForPVS))
                                syncSourceForPVS = isSourceForPVS ? c.name : "";

                            // render mode switching
                            ImGui::AlignTextToFramePadding();
                            ImGui::Text("Render mode:");
                            ImGui::SameLine();
                            if (ImGui::Button("Standard")) {
                                setServerConfig("Lighthouse.IlluminationOnly", "false", &c);
                                requestServerConfig(&c);
                            }
                            ImGui::SameLine();
                            if (ImGui::Button("Illumination")) {
                                setServerConfig("Lighthouse.IlluminationOnly", "true", &c);
                                requestServerConfig(&c);
                            }

                            // relocate pixels switching
                            ImGui::AlignTextToFramePadding();
                            ImGui::Text("Relocate Pixels:");
                            ImGui::SameLine();
                            if (ImGui::Button("ON")) {
                                setServerConfig("PrimaryView.RelocatePixels", "true", &c);
                                setServerConfig("AuxiliaryViews.RelocatePixels", "true", &c);
                                setServerConfig("Cubemap.RelocatePixels", "true", &c);
                                requestServerConfig(&c);
                            }
                            ImGui::SameLine();
                            if (ImGui::Button("OFF")) {
                                setServerConfig("PrimaryView.RelocatePixels", "false", &c);
                                setServerConfig("AuxiliaryViews.RelocatePixels", "false", &c);
                                setServerConfig("Cubemap.RelocatePixels", "false", &c);
                                requestServerConfig(&c);
                            }

                            // layer-first block order switching
                            ImGui::AlignTextToFramePadding();
                            ImGui::Text("Block order:");
                            ImGui::SameLine();
                            if (ImGui::Button("Layer-first")) {
                                setServerConfig("PrimaryView.LayerFirstBlockOrder", "true", &c);
                                setServerConfig("AuxiliaryViews.LayerFirstBlockOrder", "true", &c);
                                setServerConfig("Cubemap.LayerFirstBlockOrder", "true", &c);
                                requestServerConfig(&c);
                            }
                            ImGui::SameLine();
                            if (ImGui::Button("Block-first")) {
                                setServerConfig("PrimaryView.LayerFirstBlockOrder", "false", &c);
                                setServerConfig("AuxiliaryViews.LayerFirstBlockOrder", "false", &c);
                                setServerConfig("Cubemap.LayerFirstBlockOrder", "false", &c);
                                requestServerConfig(&c);
                            }

                            // commands
                            ImGui::AlignTextToFramePadding();
                            ImGui::Text("Commands:");
                            ImGui::SameLine();
                            if (ImGui::Button("Reset State"))
                                requestAll(RequestType::RESET_STATE);

                            /*ImGui::SameLine();
                            if (ImGui::Button("Reload"))
                                c.request.write(RequestType::GET_ALL_SETTINGS);

                            if (ImGui::Button("RAW")) {
                                setServerConfig("PrimaryView.Video.backend", "raw", &c);
                                setServerConfig("Cubemap.Video.backend", "raw", &c);
                                setServerConfig("AuxiliaryViews.Video.backend", "raw", &c);
                                setServerConfig("Blocks.Video.backend", "raw", &c);
                                requestServerConfig(&c);
                            }

                            ImGui::SameLine();
                            if (ImGui::Button("JPEG 100")) {
                                setServerConfig("PrimaryView.Video.backend", "jpeg", &c);
                                setServerConfig("PrimaryView.Video.crfQuality", "100", &c);
                                setServerConfig("Cubemap.Video.backend", "jpeg", &c);
                                setServerConfig("Cubemap.Video.crfQuality", "100", &c);
                                setServerConfig("AuxiliaryViews.Video.backend", "jpeg", &c);
                                setServerConfig("AuxiliaryViews.Video.crfQuality", "100", &c);
                                setServerConfig("Blocks.Video.backend", "jpeg", &c);
                                setServerConfig("Blocks.Video.crfQuality", "100", &c);
                                requestServerConfig(&c);
                            }

                            ImGui::SameLine();
                            if (ImGui::Button("JPEG 25")) {
                                setServerConfig("PrimaryView.Video.backend", "jpeg", &c);
                                setServerConfig("PrimaryView.Video.crfQuality", "25", &c);
                                setServerConfig("Cubemap.Video.backend", "jpeg", &c);
                                setServerConfig("Cubemap.Video.crfQuality", "25", &c);
                                setServerConfig("AuxiliaryViews.Video.backend", "jpeg", &c);
                                setServerConfig("AuxiliaryViews.Video.crfQuality", "25", &c);
                                setServerConfig("Blocks.Video.backend", "jpeg", &c);
                                setServerConfig("Blocks.Video.crfQuality", "25", &c);
                                requestServerConfig(&c);
                            }*/

                            const auto processPresets = [&](const std::string& key, float offsetFromStart) {
                                const auto presetKey = key + "Preset";
                                const auto node = serverConfigPresets.get_child_optional(presetKey);
                                if (node) {
                                    ImGui::NewLine();
                                    ImGui::SameLine(offsetFromStart, 0);
                                    ImGui::AlignTextToFramePadding();
                                    ImGui::Text("Preset:");
                                    for (const auto&[presetName, presetData] : node.get()) {
                                        ImGui::SameLine();
                                        if (ImGui::Button(presetName.c_str())) {
                                            setServerConfig(key, presetData, &c);
                                            requestServerConfig(&c);
                                        }
                                    }
                                }
                            };

                            // server config editor
                            bool configChanged = false;
                            function<void(pt::ptree&, int, std::string)> editValue = [&](pt::ptree& tree, int depth,
                                                                                         const std::string& treeKey) {
                                float W = 20, O = 5;

                                auto localTreeKey = [&](auto& t) {
                                    return treeKey + std::string(treeKey.empty() ? "" : ".").append(t.first);
                                };

                                auto setConfig = [&](auto& t, const auto& data) {
                                    if (syncServerConfigRequest)
                                        setServerConfig(localTreeKey(t), data);
                                    configChanged = true;
                                };

                                // variables
                                for (auto& t : tree) {
                                    auto& data = t.second.data();
                                    if ((t.second.empty() || !data.empty()) && data != "true" && data != "false") {
                                        ImGui::NewLine();
                                        ImGui::SameLine(depth * W + O, 0);
                                        if (ImGui::InputText(t.first.c_str(), &data,
                                                             ImGuiInputTextFlags_EnterReturnsTrue))
                                            setConfig(t, data);
                                    }
                                }

                                // bool variables
                                for (auto& t : tree) {
                                    auto& data = t.second.data();
                                    if (data == "true" || data == "false") {
                                        bool value = data == "true";
                                        ImGui::NewLine();
                                        ImGui::SameLine(depth * W + O, 0);
                                        if (ImGui::Checkbox(t.first.c_str(), &value)) {
                                            data = value ? "true" : "false";
                                            setConfig(t, data);
                                        }
                                    }
                                }

                                // subtrees
                                for (auto& t : tree) {
                                    if (!t.second.empty()) {
                                        ImGui::NewLine();
                                        ImGui::SameLine(depth * W + O, 0);
                                        std::string localKey = localTreeKey(t);
                                        const string stateConfigKey = "OpenConfig." + localKey;
                                        auto stateNode = gui.state.get_optional<int>(stateConfigKey);
                                        bool defaultOpen = !stateNode || *stateNode != 0;
                                        if (ImGui::CollapsingHeader(t.first.c_str(),
                                                                    defaultOpen ? ImGuiTreeNodeFlags_DefaultOpen
                                                                                : ImGuiTreeNodeFlags_None)) {
                                            gui.state.put(stateConfigKey, 1);
                                            ImGui::PushID(t.first.c_str());
                                            processPresets(localKey, depth * W + O);
                                            editValue(t.second, depth + 1, localKey);
                                            ImGui::PopID();
                                        } else gui.state.put(stateConfigKey, 0);
                                    }
                                }
                            };

                            ImGui::PushItemWidth(150);
                            ImGui::PushID("ServerConfig");
                            editValue(c.config, 0, "");
                            ImGui::PopID();
                            ImGui::PopItemWidth();

                            if (configChanged)
                                requestServerConfig(&c);
                        }
                        ImGui::EndChild();
                        ImGui::EndTabItem();
                    }
                    if (ImGui::BeginTabItem("Scenes")) {
                        if (ImGui::Button("Reload"))
                            requestAll(RequestType::GET_SCENE_LIST);
                        for (auto& s : c.sceneList) {
                            if (ImGui::Button(s.c_str())) {
                                if (boost::algorithm::ends_with(s, ".lights"))
                                    setServerConfig("Lighthouse.lights", s, &c);
                                else {
                                    setServerConfig("Lighthouse.lights", "", &c);
                                    setServerConfig("Scene", s,  &c);
                                }
                                requestServerConfig(&c);
                            }
                        }
                        ImGui::EndTabItem();
                    }
                    if (ImGui::BeginTabItem("Data")) {
                        if (ImGui::BeginChild(c.name.c_str())) {
                            ImGui::AlignTextToFramePadding();
                            addAllNone("Render", [&](auto b) {
                                for (auto s : c.scenes) for (auto& v : s->views) v.render = b;
                            });
                            int j = 0;
                            for (auto s : c.scenes) {
                                ImGui::PushID(j);
                                ImGui::Spacing();
                                ImGui::Spacing();
                                string sceneName = "Scene " + to_string(j);
                                if (ImGui::CollapsingHeader(sceneName.c_str())) {
                                    ImGui::Text("ID: %d", s->viewpoint.id);
                                    ImGui::Text("Time: %f", s->viewpoint.time);
                                    ImGui::Text("Flags:");
                                    if (s->viewpoint.flags & Viewpoint::AUTOMATIC_UPDATE) {
                                        ImGui::SameLine();
                                        ImGui::Text("AUTOMATIC_UPDATE");
                                    }
                                    if (s->viewpoint.flags & Viewpoint::SYNCHRONIZED) {
                                        ImGui::SameLine();
                                        ImGui::Text("SYNCHRONIZED");
                                    }
                                    if (s->viewpoint.flags & Viewpoint::DONT_SEND_PVS) {
                                        ImGui::SameLine();
                                        ImGui::Text("DONT_SEND_PVS");
                                    }
                                    ImGui::Text("Vertex Count: %d", s->vertexCount);
                                    auto& b = s->background;
                                    ImGui::Text("Background: %.1f %.1f %.1f %s", b.color.x, b.color.y, b.color.z,
                                                b.textureName.c_str());

                                    addAllNone("Render", [&](auto b) { for (auto& v : s->views) v.render = b; });

                                    if (ImGui::CollapsingHeader("Debug")) {
                                        ImGui::PushID("Debug");
                                        addAllNone("Views", [&](auto b) { for (auto& v : s->views) v.debug.renderView = b; });
                                        ImGui::Checkbox("Request View", &s->debug.renderRequestView);
                                        ImGui::Checkbox("Server Views", &s->debug.renderViews);
                                        ImGui::Checkbox("Server AABB", &s->debug.renderAABB);
                                        ImGui::Checkbox("Server Spheres", &s->debug.renderSpheres);
                                        ImGui::Checkbox("Server Lines", &s->debug.renderLines);
                                        ImGui::Text("PVS Cache Samples: %d", s->debug.PVSCacheSamples.size());
                                        ImGui::PopID();
                                    }
                                    if (!s->materials.empty() && ImGui::CollapsingHeader("Materials")) {
                                        ImGui::PushID("Materials");
                                        for (auto& m : s->materials)
                                            ImGui::Text("%.1f %.1f %.1f %s %s", m.textureName.c_str(),
                                                        m.color.x, m.color.y, m.color.z,
                                                        m.opaque ? "OPAQUE" : "TRANSPARENT");
                                        ImGui::PopID();
                                    }

                                    int i = 0;
                                    for (auto& v : s->views) {
                                        auto name = "View " + v.name;
                                        ImGui::PushID(name.c_str());
                                        if (ImGui::CollapsingHeader(name.c_str())) {
                                            ImGui::Checkbox("Render", &v.render);
                                            ImGui::Checkbox("Debug View", &v.debug.renderView);
                                            ImGui::Text("Flags:");
                                            if (v.flags & ViewFlag::CUBEMAP) {
                                                ImGui::SameLine();
                                                ImGui::Text("CUBEMAP");
                                            }
                                            if (v.flags & ViewFlag::VIDEO_STREAM) {
                                                ImGui::SameLine();
                                                ImGui::Text("VIDEO_STREAM");
                                            }
                                            if (v.flags & ViewFlag::SUBDIVIDE_CHECKER) {
                                                ImGui::SameLine();
                                                ImGui::Text("SUBDIVIDE_CHECKER");
                                            }
                                            if (v.flags & ViewFlag::LAYER_FIRST_BLOCK_ORDER) {
                                                ImGui::SameLine();
                                                ImGui::Text("LAYER_FIRST_BLOCK_ORDER");
                                            }
                                            if (v.flags & ViewFlag::RELOCATED_PIXELS) {
                                                ImGui::SameLine();
                                                ImGui::Text("RELOCATED_PIXELS");
                                            }

                                            ImGui::Text("Triangles Count: %d", v.triangles.count);
                                            ImGui::Text("Blend Factor: %f", v.blendFactor);
                                            ImGui::Text("Extended Blocks %d", (int) v.extendedBlocks);

                                            ImGui::Text("Layer Count: %d", v.layerCount);
                                            ImGui::Text("Layer Size: %dx%d", v.layerSize.x, v.layerSize.y);

                                            ImGui::Text("Full Layers Count: %d", v.fullLayers.count);
                                            ImGui::Text("Full Layers Offset: %dx%d", v.fullLayers.offset.x, v.fullLayers.offset.y);
                                            ImGui::Text("Full Layers Texture: %s", v.fullLayers.textureID.c_str());

                                            ImGui::Text("Blocks Size: %dx%d %d", v.blocks.size.x, v.blocks.size.y, v.blocks.size.z);
                                            ImGui::Text("Blocks Layer Count: %d", v.blocks.layerCount);
                                            ImGui::Text("Blocks Offset: %d", v.blocks.offset);
                                            ImGui::Text("Blocks Tile Count: %d", v.blocks.tileCount);
                                            ImGui::Text("Blocks Texture: %s", v.blocks.textureID.c_str());

                                            if (!v.triangles.triangleLayerCount.empty() &&
                                                ImGui::CollapsingHeader("Triangles / Layer"))
                                                for (auto t : v.triangles.triangleLayerCount)
                                                    ImGui::Text(" %d", t / 3);
                                        }
                                        ImGui::PopID();
                                    }
                                    for (auto& t : s->textures) {
                                        auto name = "Texture " + t.name;
                                        ImGui::PushID(name.c_str());
                                        if (ImGui::CollapsingHeader(name.c_str())) {
                                            if (ImGui::Button("Save")) {
                                                vector<unsigned char> image(t.size.x * t.size.y * 3);
                                                glBindTexture(GL_TEXTURE_2D, t.texture);
                                                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
                                                glPixelStorei(GL_PACK_ALIGNMENT, 1);
                                                glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, image.data());
                                                glPixelStorei(GL_PACK_ALIGNMENT, 4);
                                                glBindTexture(GL_TEXTURE_2D, 0);

                                                std::ofstream f(t.name + ".ppm", ios::binary);
                                                f << "P6" << endl << t.size.x << ' ' << t.size.y << endl << "255"
                                                  << endl;
                                                f.write((char*) image.data(), t.size.x * t.size.y * 3);
                                            }
                                            ImGui::Text("Size: %dx%d", t.size.x, t.size.y);
                                            ImGui::Text("Stream: %s", t.stream.c_str());
                                            if (t.stream != "jpeg" && t.stream != "raw") {
                                                ImGui::Text("Video Codec: %s", t.videoConfig.codec.c_str());
                                                ImGui::Text("Video Format: %s", t.videoConfig.format.c_str());
                                            }

                                            // texture
                                            ImGui::Begin((sceneName + " " + t.name).c_str());
                                            ImGui::SliderFloat("Scale", &textureScale, 0.1f, 4);
                                            ImGui::BeginChild("TextureView", ImVec2(0, 0), false,
                                                              ImGuiWindowFlags_HorizontalScrollbar);
                                            ImGui::GetWindowDrawList()->AddCallback(ImGuiDisableBlend, nullptr);
                                            ImGui::Image((void*) (intptr_t) t.texture,
                                                         ImVec2((float) t.size.x * textureScale,
                                                                (float) t.size.y * textureScale));
                                            ImGui::GetWindowDrawList()->AddCallback(ImGuiEnableBlend, nullptr);
                                            ImGui::EndChild();
                                            ImGui::End();
                                        }
                                        ImGui::PopID();
                                    }
                                }
                                j++;
                                ImGui::PopID();
                            }
                        }
                        ImGui::EndChild();
                        ImGui::EndTabItem();
                    }
                    ImGui::EndTabBar();
                }
                ImGui::EndTabItem();
            }
        }
        ImGui::EndTabBar();
    }
    ImGui::End();

    ImGui::Begin("Stats");
    if (ImGui::BeginTabBar("Stats Tab Bar")) {
        if (ImGui::BeginTabItem("Stats")) {
            ImGui::Checkbox("Show Min/Max", &counters.minMax);
            if (ImGui::TreeNodeEx("GPU Tasks", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Indent();
                ImGui::Text("Elapsed %.3f", gpu.elapsed.average());
                ImGui::Text("Elapsed Left %.3f", gpu.elapsedLeft.average());
                ImGui::Text("Elapsed Max %.3f", gpu.stats.elapsedMax);
                ImGui::Text("Delta Time %.3f", gpu.deltaTime.average());
                ImGui::Text("Tasks %d", gpu.stats.tasks);
                ImGui::Text("Compute %.3f", gpu.stats.compute);
                ImGui::Text("Upload %.3f", gpu.stats.upload);
                ImGui::Text("Left Tasks %d", gpu.tasks.size());
                ImGui::Text("Left Compute Tasks %d", gpu.compute.size());
                ImGui::Text("Left Upload Tasks %d", gpu.upload.size());
                ImGui::Unindent();
                ImGui::TreePop();
            }
            viewTree(stats.frame, "Frame");
            viewTree(stats.update, "Update");
            if (ImGui::TreeNodeEx("Connections", ImGuiTreeNodeFlags_DefaultOpen)) {
                for (auto& c : connections) {
                    if (ImGui::TreeNodeEx(c.name.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
                        if (ImGui::TreeNodeEx("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
                            ImGui::Indent();
                            ImGui::Text("Count %d", (int) c.scenes.size());
                            ImGui::Text("Processing %d", (int) c.processingScenes);
                            ImGui::Text("Reusing %d", (int) c.reuseScenes.size());
                            ImGui::Text("Pending %d", (int) c.syncScenes.lock()->size());
                            ImGui::Unindent();
                            ImGui::TreePop();
                        }

                        if (ImGui::TreeNodeEx("Update Prediction", ImGuiTreeNodeFlags_DefaultOpen)) {
                            auto& u = c.updatePrediction;
                            ImGui::Indent();
                            ImGui::Text("Delta Time %.3f", u.deltaTime.average() / 1000);
                            ImGui::Text("Min Delta Time %.3f %.3f", c.minDeltaTime() / 1000, c.minDeltaTime(false) / 1000);
                            ImGui::Text("Server %.3f", u.server.average() / 1000);
                            ImGui::Text("Receive %.3f", u.receive.average() / 1000);
                            ImGui::Text("Client CPU %.3f (%.3f %.3f)", u.cpu.average() / 1000, u.started.cpu.average() / 1000, u.waiting.cpu.average() / 1000);
                            ImGui::Text("Client GPU %.3f (%.3f %.3f)", u.gpu.average() / 1000, u.started.gpu.average() / 1000, u.waiting.gpu.average() / 1000);
                            ImGui::Text("Latency %.3f", u.latency.average() / 1000);
                            ImGui::Unindent();
                            ImGui::TreePop();
                        }
                        viewTree(c.stats.update, "Client");
                        auto serverStats = c.stats.server.get_child_optional("Server");
                        if (serverStats) viewTree(*serverStats, "Server");
                        else viewTree(c.stats.server, "Server", 2);
                        ImGui::TreePop();
                    }
                }
                ImGui::TreePop();
            }
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Benchmark")) {
            if (ImGui::Button("Save"))
                benchmark.save();
            ImGui::SameLine();
            if (ImGui::Button("Benchmark Path") && !benchmark.running)
                benchmark.startPath();
            ImGui::SameLine();
            if (ImGui::Button("Simulated Benchmark") && !benchmark.running)
                benchmark.startSimulated();
            ImGui::SameLine();
            bool b = benchmark.running;
            if (ImGui::Checkbox("Running", &b)) {
                if (b) benchmark.start();
                else benchmark.stop();
            }
            if (benchmark.simulated.isRunning())
                ImGui::Text("Simulated benchmark: %d %d, %.1f / %.1f",
                            benchmark.simulated.frame, (int)benchmark.simulated.serverProcessed,
                            benchmark.simulated.frameTime(), camera.realStop() - camera.start);
            viewTree(benchmark.report, "Benchmark", 2);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}