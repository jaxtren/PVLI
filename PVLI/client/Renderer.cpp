#include "Renderer.h"
#include "imguiHelper.h"
#include "Scene.h"
#include "SceneView.h"
#include "Application.h"
#include "Scene.inl"

using namespace std;
using namespace glm;

Renderer::Renderer() {
    shaderFlags["LINEAR_FILTER"] = false;
    shaderFlags["DEPTH_RANGE"] = false;
    shaderFlags["CLOSEST_DEPTH"] = false;
    shaderFlags["CLOSEST_DEPTH_RANGE"] = false;
    shaderFlags["DISABLE_TEXTURE_GATHER"] = false;
    glGenVertexArrays(1, &vao);

    // fallback textures
    glActiveTexture(GL_TEXTURE0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glGenTextures(1, &fallbackColor);
    glBindTexture(GL_TEXTURE_2D, fallbackColor);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    u8vec4 pixel(255);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, &pixel);

    glGenTextures(1, &fallbackBlock);
    glBindTexture(GL_TEXTURE_2D_ARRAY, fallbackBlock);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    ivec2 block(0);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RG32I, 1, 1, 1, 0, GL_RG_INTEGER, GL_INT, &block);

    glGenTextures(1, &fallbackDepth);
    glBindTexture(GL_TEXTURE_2D_ARRAY, fallbackDepth);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    float depth = 1e24f;
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_R32F, 1, 1, 1, 0, GL_RED, GL_FLOAT, &depth);

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
    loadShaders();

    glGenSamplers(1, &nearestSampler);
    glSamplerParameteri(nearestSampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glSamplerParameteri(nearestSampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glSamplerParameteri(nearestSampler, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glSamplerParameteri(nearestSampler, GL_TEXTURE_WRAP_T, GL_REPEAT);
}

Renderer::~Renderer() {
    glDeleteVertexArrays(1, &vao);
    glDeleteVertexArrays(1, &debug.linesVAO);
    glDeleteTextures(1, &fallbackColor);
    glDeleteTextures(1, &fallbackBlock);
    glDeleteTextures(1, &fallbackDepth);
    glDeleteSamplers(1, &nearestSampler);
    glDeleteBuffers(1, &debug.linesBuffer);
    destroyFBO();
}

void Renderer::updateConfig(const Config &cfg) {
    cfg.get("Wireframe", wireframe);
    cfg.get("Cullface", cullface);
    cfg.get("DepthOffset", depthOffset);
    cfg.get("DepthRange", depthRange);
    cfg.get("Composite", composite);
    cfg.get("AlphaMask", alphaMask);
    cfg.get("Inpaint", inpaint);
    cfg.get("InpaintFlags", inpaintFlags);
    cfg.get("BlendEdges", blendEdges);
    cfg.get("BlendScenes", blendScenes);
    cfg.get("OptimizePOT", optimizePOT);
    cfg.get("OptimizeCubemapAA", optimizeCubemapAA);
    cfg.get("OptimizeSharedOrigin", optimizeSharedOrigin);
    cfg.get("UseOldShader", useOldShader);
    cfg.get("DeferredRemote", deferredRemote);
    cfg.get("CombineViews", combineViews);
    cfg.get("Debug.Render", debug.render);
    cfg.get("Debug.SphereAxis", debug.sphereAxis);
    cfg.get("Debug.ViewScale", debug.viewScale);

    bool reloadShaders = false;
    for(auto& f : shaderFlags)
        reloadShaders |= cfg.get(f.first, f.second);
    if(reloadShaders) shaders.clear();
}

bool Renderer::loadShaders() {
    shaders.clear();
    texPlot.load("texPlot", "shaders");
    debug.simpleShader.load("simple", "shaders/debug");
    debug.circle.load("circle", "shaders/debug");
    debug.cubeWireframe.load("cube_wireframe", "shaders/debug");
    inpainter.loadShaders();
    return true;
}

gl::Shader& Renderer::getShader(const string& name, const ShaderParams& params) {
    auto it = shaders.emplace(make_pair(name, params), gl::Shader());
    auto& shader = it.first->second;
    if (it.second) {
        vector<string> defs;
        for (auto& f : shaderFlags)
            if (f.second) defs.push_back(f.first);
        for (auto& p : params)
            if (p.second) defs.push_back(p.first);

        shader.load(name, "shaders", defs);
        if (shader) cout << "Loaded shader " << name << ':';
        else cout << "Not loaded shader " << name << ':';
        for (auto& p : defs) cout << ' ' << p;
        cout << endl;
    }
    return shader;
}

void Renderer::GUI() {
    ImGui::Checkbox("Wireframe", &wireframe);
    ImGui::Checkbox("Cullface", &cullface);
    ImGui::Checkbox("Composite", &composite);
    ImGui::Checkbox("Alpha Mask", &alphaMask);
    ImGui::Checkbox("Blend Edges", &blendEdges);
    ImGui::Checkbox("Blend Scenes", &blendScenes);
    ImGui::Checkbox("Deferred Remote", &deferredRemote);
    ImGui::Checkbox("Combine Views", &combineViews);
    ImGui::Checkbox("Use Old Shader", &useOldShader);
    ImGui::Checkbox("Optimize POT", &optimizePOT);
    ImGui::Checkbox("Optimize Cubemap AA", &optimizeCubemapAA);
    ImGui::Checkbox("Optimize Shared Origin", &optimizeSharedOrigin);
    ImGui::InputInt("Inpaint", &inpaint);
    ImGui::InputInt("Inpaint Flags", &inpaintFlags);
    ImGui::InputFloat2("Depth Offset", &depthOffset.x);
    ImGui::InputFloat2("Depth Range", &depthRange.x);
    ImGui::Checkbox("Debug", &debug.render);
    ImGui::Checkbox("Debug Sphere Axis", &debug.sphereAxis);
    ImGui::InputFloat("Debug View Scale", &debug.viewScale);

    // flags
    bool reloadShaders = false;
    for(auto& f : shaderFlags)
        reloadShaders = ImGui::Checkbox(f.first.c_str(), &f.second) || reloadShaders;
    if(reloadShaders) loadShaders();
}

void Renderer::initFBO(const glm::ivec2& s) {
    if (!composite || size == s) return;
    destroyFBO();
    size = s;

    localColor = inpainter.createTexture(size);
    normals = inpainter.createTexture(size);
    remoteColor[0] = inpainter.createTexture(size);
    remoteColor[1] = inpainter.createTexture(size);
    auto halfSize = size / 2 + size % 2;
    remoteColorLayer2[0] = inpainter.createTexture(halfSize);
    remoteColorLayer2[1] = inpainter.createTexture(halfSize);

    glGenTextures(1, &depth);
    glBindTexture(GL_TEXTURE_2D, depth);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT32F, size.x, size.y);
    glBindTexture(GL_TEXTURE_2D, 0);

    //fbo
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depth, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, localColor, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, remoteColor[0], 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, remoteColor[1], 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, normals, 0);

    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        cerr << "GL ERROR Renderer FBO: "  << glCheckFramebufferStatus(GL_FRAMEBUFFER) << endl;
}

void Renderer::destroyFBO() {
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &depth);
    glDeleteTextures(1, &localColor);
    glDeleteTextures(1, &normals);
    glDeleteTextures(2, remoteColor);
    glDeleteTextures(2, remoteColorLayer2);
    fbo = depth = localColor = normals = remoteColor[0] = remoteColor[1] = 0;
    remoteColorLayer2[0] = remoteColorLayer2[1] = 0;
    size = {0, 0};
    inpainter.clear();
}

static bool isPOT(int n) { return (n & (n - 1)) == 0; };
static ivec2 genPOTMaskShift(int n) {
    int mask = n - 1;
    int shift = -1;
    while (n != 0){
        shift++;
        n >>= 1;
    }
    return ivec2(mask, shift);
};

static float computePixelSize(const mat4& projection, const ivec2& layerSize) {
    auto p = glm::inverse(projection) * vec4(1.0f / (float) layerSize.x, 1.0f / (float) layerSize.y, -1, 1);
    return std::max(fabs(p.x / p.z / p.w), fabs(p.y / p.z / p.w));
}

void Renderer::render(const glm::ivec2& frameSize, const Viewpoint& viewpoint, const std::list<ServerConnection>& connections, bool synchronized) {
    list<Scene*> scenes;

    // collect scenes
    for (auto& c : connections) {
        for (auto& s : c.scenes)
            if (s->allReady())
                scenes.push_back(s);
        if (synchronized) break; // collect only from one connection for synchronized updates
    }

    // sort by id for asynchronous updates on multiple connections
    if (!synchronized && connections.size() > 1)
        scenes.sort([](Scene* a, Scene* b) {
            return a->viewpoint.id == b->viewpoint.id ?
                   a->connection->name < b->connection->name :
                   a->viewpoint.id > b->viewpoint.id;
        });

    if (scenes.empty()) {
        glViewport(0, 0, frameSize.x, frameSize.y);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        return;
    }

    auto scene = scenes.front();

    // scenes blend weight
    float blendScenesFactor = 0;
    if (scenes.size() > 1 && blendScenes)
        blendScenesFactor = 1.0f - std::min(1.0f, viewpoint.latency);
    bool blendingScenes = blendScenesFactor > 0;
    if (inpaint > 0 || blendingScenes) composite = true;

    // deferred remote color
    if (deferredRemote) {
        composite = combineViews = true;
        useOldShader = false;
    }

    timer.reset();
    auto proj = scene->projParams;
    mat4 projection = perspective<float>(radians(proj.x), (float) frameSize.x / frameSize.y, proj.y, proj.z);
    auto MVP = projection * viewpoint.view;
    bool useLocalColor = !composite;

    auto useCullFace = [](bool enable) {
        if (enable) glEnable(GL_CULL_FACE);
        else glDisable(GL_CULL_FACE);
    };

    auto renderView = [&](SceneView& v, SceneView* combinedCubemap = nullptr) {
        if (!v.render) return;

        // subdivided view
        SceneView* v2 = nullptr;
        if (v.subdivide.x > 1 || v.subdivide.y > 1) {
            if (v.subdivideOffset != ivec2(0, 0))
                return; // render one view with all other views connected together
            if (v.subdivide != ivec2(1, 2) && v.subdivide != ivec2(2, 1))
                return; // other subdivisions currently not supported

            // find second view
            if (!v.scene->sibling) return;
            auto secondOffset = v.subdivide - 1;
            for (auto s = v.scene->sibling; s != v.scene; s = s->sibling) {
                auto v3 = s->findView(v.name);
                if (v3 && v3->subdivideOffset == secondOffset) {
                    v2 = v3;
                    break;
                }
            }
            if (!v2 || (v.flags & ViewFlag::SUBDIVIDE_CHECKER) != (v2->flags & ViewFlag::SUBDIVIDE_CHECKER))
                return;
        }

        if (v2 && (v.flags & ViewFlag::VIDEO_STREAM) != (v2->flags & ViewFlag::VIDEO_STREAM))
            return;

        // video stream
        if (v.flags & ViewFlag::VIDEO_STREAM) {
            if (!v2) {
                auto& shader = getShader("texPlot", {});
                shader.use();
                shader.uniform("tex", 0);
            } else {
                auto& shader = getShader("texPlotSubdivided", {
                    {"REMOTE_TEXTURE_2X1",     v.subdivide == ivec2(2, 1)},
                    {"REMOTE_TEXTURE_1X2",     v.subdivide == ivec2(1, 2)},
                    {"REMOTE_TEXTURE_CHECKER", v.flags & ViewFlag::SUBDIVIDE_CHECKER}
                });
                shader.use();
                shader.uniform("layerSize", v.layerSize);
                glActiveTexture(GL_TEXTURE1);
                glBindTexture(GL_TEXTURE_2D, v2->fullLayers.texture && v2->fullLayers.texture->texture ? v2->fullLayers.texture->texture : fallbackColor);
            }
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, v.fullLayers.texture && v.fullLayers.texture->texture ? v.fullLayers.texture->texture : fallbackColor);
            glBindVertexArray(vao);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            return;
        }

        if (v.triangles.count == 0 || (combinedCubemap && !v.scene->vertexCount)) return;

        const float eps = 1e-5f;
        bool oldShader = useOldShader && !combinedCubemap;
        bool useEdgeBlending = blendEdges && v.blendFactor > 0;
        bool hasTexture = v.scene->findSourceForPVS()->hasUV && (alphaMask || useLocalColor);
        bool useColorMapping = hasTexture && useLocalColor && v.scene->colorMapping.x > 0;
        bool useGamma = hasTexture && useLocalColor && v.scene->gamma != 1.0f;
        bool hasPOTSizes = !optimizePOT && !combinedCubemap &&
            isPOT(v.blocks.size.z) && (!v.blocks.texture || isPOT(v.blocks.texture->size.x));
        bool sharedOrigin = optimizeSharedOrigin && combinedCubemap &&
            glm::distance(vec3(inverse(v.view)[3]), vec3(inverse(combinedCubemap->view)[3])) < eps;
        bool cubemapAA = false;
        if (optimizeCubemapAA && (v.flags & ViewFlag::CUBEMAP || combinedCubemap)) {
            auto rot = mat3(inverse(combinedCubemap ? combinedCubemap->view : v.view));
            cubemapAA = glm::distance(rot[0], vec3(1, 0, 0)) < eps &&
                        glm::distance(rot[1], vec3(0, 0, 1)) < eps &&
                        glm::distance(rot[2], vec3(0,-1, 0)) < eps;
        }
        auto& shader = getShader(oldShader ? "renderOld" : "render", {
            {"CUBEMAP",                      v.flags & ViewFlag::CUBEMAP},
            {"CUBEMAP_AA",                   cubemapAA},
            {"COMBINED_VIEWS",               combinedCubemap},
            {"COMBINED_VIEWS_SHARED_ORIGIN", sharedOrigin},
            {"BINARY_BLOCKS",                v.flags & ViewFlag::LAYER_FIRST_BLOCK_ORDER},
            {"EXTENDED_BLOCKS",              v.extendedBlocks},
            {"REMOTE_TEXTURE",               true},
            {"REMOTE_TEXTURE_DEFERRED",      deferredRemote},
            {"REMOTE_TEXTURE_COLOR_MAPPING", useColorMapping},
            {"REMOTE_TEXTURE_2X1",           v.subdivide == ivec2(2, 1)},
            {"REMOTE_TEXTURE_1X2",           v.subdivide == ivec2(1, 2)},
            {"REMOTE_TEXTURE_CHECKER",       v.flags & ViewFlag::SUBDIVIDE_CHECKER},
            {"LOCAL_TEXTURE",                hasTexture},
            {"LOCAL_TEXTURE_COLOR",          hasTexture && useLocalColor},
            {"LOCAL_TEXTURE_ALPHA_MASK",     hasTexture && alphaMask},
            {"LOCAL_TEXTURE_GAMMA",          useGamma},
            {"BLEND_EDGES",                  useEdgeBlending},
            {"POT_SIZES",                    hasPOTSizes},
        });

        if (!shader) return;
        shader.use();

        if (useGamma)
            shader.uniform("localGamma", 1.0f / v.scene->gamma);
        if (useColorMapping)
            shader.uniform("colorMapping", vec2(v.scene->colorMapping.x, 1.0f / v.scene->colorMapping.y));
        if (useEdgeBlending)
            shader.uniform("blendEdges", 1.0f / v.blendFactor);
        if (hasPOTSizes) {
            shader.uniform("blocksSizePOT", genPOTMaskShift(v.blocks.size.z));
            if (!v.blocks.texture) shader.uniform("blocksTexPOT", ivec2(0));
            else shader.uniform("blocksTexPOT", genPOTMaskShift(v.blocks.texture->size.x / v.blocks.size.z));
        }

        // textures
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D_ARRAY, v.depth ? v.depth : fallbackDepth);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, v.fullLayers.texture && v.fullLayers.texture->texture ? v.fullLayers.texture->texture : fallbackColor);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, v.blocks.texture && v.blocks.texture->texture ? v.blocks.texture->texture : fallbackColor);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D_ARRAY, v.blocks.blocksTexture ? v.blocks.blocksTexture : fallbackBlock);
        if (v2) {
            glActiveTexture(GL_TEXTURE4);
            glBindTexture(GL_TEXTURE_2D, v2->fullLayers.texture && v2->fullLayers.texture->texture ? v2->fullLayers.texture->texture : fallbackColor);
            glActiveTexture(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_2D, v2->blocks.texture && v2->blocks.texture->texture ? v2->blocks.texture->texture : fallbackColor);
            glActiveTexture(GL_TEXTURE6);
            glBindTexture(GL_TEXTURE_2D_ARRAY, v2->blocks.blocksTexture ? v2->blocks.blocksTexture : fallbackBlock);
        }
        if (combinedCubemap) {
            auto& c = *combinedCubemap;
            glActiveTexture(GL_TEXTURE8);
            glBindTexture(GL_TEXTURE_2D_ARRAY, c.depth ? c.depth : fallbackDepth);
            glActiveTexture(GL_TEXTURE9);
            glBindTexture(GL_TEXTURE_2D, c.fullLayers.texture && c.fullLayers.texture->texture ? c.fullLayers.texture->texture : fallbackColor);
            glActiveTexture(GL_TEXTURE10);
            glBindTexture(GL_TEXTURE_2D, c.blocks.texture && c.blocks.texture->texture ? c.blocks.texture->texture : fallbackColor);
            glActiveTexture(GL_TEXTURE11);
            glBindTexture(GL_TEXTURE_2D_ARRAY, c.blocks.blocksTexture ? c.blocks.blocksTexture : fallbackBlock);
        }

        // common uniforms
        shader.uniform("MVP", MVP);
        shader.uniform("blockSize", v.blocks.size.z);
        shader.uniform("depthOffset", depthOffset);
        shader.uniform("depthRange", depthRange);

        // view uniforms
        if (!(v.flags & ViewFlag::CUBEMAP))
            shader.uniform("texMVP", v.projection * v.view);
        shader.uniform("texMV", v.view);
        shader.uniform("texOrigin", vec3(inverse(v.view)[3]));
        shader.uniform("texNormalTransform", transpose(inverse(mat3(v.view))));
        shader.uniform("layerSize", v.layerSize);
        shader.uniform("fullLayersCount", v.fullLayers.count);
        shader.uniform("fullLayersOffset", v.fullLayers.offset);
        shader.uniform("pixelSize", computePixelSize(v.projection, v.layerSize));

        if (combinedCubemap) {
            auto& c = *combinedCubemap;
            shader.uniform("cubemapMV", c.view);
            shader.uniform("cubemapOrigin", vec3(inverse(c.view)[3]));
            shader.uniform("cubemapNormalTransform", transpose(inverse(mat3(c.view))));
            shader.uniform("cubemapLayerSize", c.layerSize);
            shader.uniform("cubemapFullLayersCount", c.fullLayers.count);
            shader.uniform("cubemapFullLayersOffset", c.fullLayers.offset);
            shader.uniform("cubemapPixelSize", computePixelSize(c.projection, c.layerSize));
        }

        if (combinedCubemap || v.flags & ViewFlag::CUBEMAP) {
            auto& c = combinedCubemap ? *combinedCubemap : v;
            shader.uniform("cubemapFaceTransform[0]", c.cubemapProjection(0));
            shader.uniform("cubemapFaceTransform[1]", c.cubemapProjection(1));
            shader.uniform("cubemapFaceTransform[2]", c.cubemapProjection(2));
            shader.uniform("cubemapFaceTransform[3]", c.cubemapProjection(3));
            shader.uniform("cubemapFaceTransform[4]", c.cubemapProjection(4));
            shader.uniform("cubemapFaceTransform[5]", c.cubemapProjection(5));
        }

        auto renderMaterialRanges = [&] (auto& materialRanges, auto& materials, bool indices) {
            int i = 0;
            glActiveTexture(GL_TEXTURE7);
            for (auto& m : materialRanges) {
                auto& mat = materials[m.first];
                useCullFace(cullface && mat.opaque);
                if (!useLocalColor && mat.opaque)
                    glBindTexture(GL_TEXTURE_2D, fallbackColor); // used only for alpha mask, that is always 1
                else {
                    shader.uniform("localColor", mat.color);
                    glBindTexture(GL_TEXTURE_2D, mat.texture && mat.texture->loaded ? mat.texture->id : fallbackColor);
                }
                if (indices) gl::drawElementsInt(GL_TRIANGLES, m.second, i);
                else glDrawArrays(GL_TRIANGLES, i, m.second);
                i += m.second;
            }
        };

        if (deferredRemote) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            shader.uniform("projInv", inverse(projection));
            shader.uniform("viewInv", inverse(viewpoint.view));
            glActiveTexture(GL_TEXTURE0 + 12);
            glBindTexture(GL_TEXTURE_2D, depth);
            glActiveTexture(GL_TEXTURE0 + 13);
            glBindTexture(GL_TEXTURE_2D, normals);
            glDisable(GL_BLEND);
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_CULL_FACE);
            glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
        } else {
            glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);
            if (combinedCubemap) { // use PVS of whole scene
                auto pvsSource = v.scene->findSourceForPVS();
                glBindVertexArray(pvsSource->vao);
                glDisable(GL_BLEND);
                if (hasTexture) renderMaterialRanges(pvsSource->materialRanges, pvsSource->materials, false);
                else glDrawArrays(GL_TRIANGLES, 0, pvsSource->vertexCount);
            } else { // use PVS of current view
                glBindVertexArray(v.vao);
                if (hasTexture) renderMaterialRanges(v.triangles.materialRanges, v.scene->materials, true);
                else gl::drawElementsInt(GL_TRIANGLES, v.triangles.count * 3, 0);
            }
        }
    };

    auto renderViews = [&](auto& s, int p = 0) {

        // render primary view and cubemap in one pass
        if (combineViews && p == 0 && !s->isSynchronized() && s->views.size() == 2) {
            SceneView* primary = nullptr, *cubemap = nullptr;
            for (auto& v : s->views) {
                if (v.flags & ViewFlag::CUBEMAP) cubemap = &v;
                else primary = &v;
            }
            if (primary && cubemap) {
                renderView(*primary, cubemap);
                return;
            }
        }

        for (auto& v : s->views)
            if (p == 0 || (v.priority >= 0) == (p > 0))
                renderView(v);
        if (s->isSynchronized())
            for (auto s2 = s->sibling; s != s2; s2 = s2->sibling)
                for (auto& v : s2->views)
                    if (p == 0 || (v.priority >= 0) == (p > 0))
                        renderView(v);
    };

    // video stream
    if (scene->isVideoStream()) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, frameSize.x, frameSize.y);
        glDisable(GL_CULL_FACE);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        renderViews(scene);
        glBindVertexArray(0);
        glUseProgram(0);
        return;
    }

    timer("Init");
    glCullFace(GL_BACK);
    useCullFace(cullface);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDisable(GL_BLEND);
    auto pvsSource = scene->findSourceForPVS();

    if (composite) {
        initFBO(frameSize);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        if (deferredRemote) {
            GLuint bufs[2] = { (GLuint)(pvsSource->hasUV ? GL_COLOR_ATTACHMENT0 : GL_NONE), GL_COLOR_ATTACHMENT3 };
            glDrawBuffers(2, bufs); // local color, normal
        } else glDrawBuffer(pvsSource->hasUV ? GL_COLOR_ATTACHMENT0 : GL_NONE); // local color
    } else {
        destroyFBO();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    glViewport(0, 0, frameSize.x, frameSize.y);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // depth + albedo pass
    if (composite || blendEdges) {
        glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);
        if (pvsSource->hasUV) {
            timer("Depth + Albedo Pass");

            // local texture pass
            bool useGamma = useLocalColor && scene->gamma != 1.0;
            auto& shader = getShader("render", {
                {"LOCAL_TEXTURE",            true},
                {"LOCAL_TEXTURE_COLOR",      true},
                {"LOCAL_TEXTURE_ALPHA_MASK", alphaMask},
                {"LOCAL_TEXTURE_GAMMA",      useGamma},
                {"COLOR_MAPPING",            scene->colorMapping.x > 0},
                {"PACK_NORMAL",              deferredRemote},
            });

            if (!shader) return;
            shader.use();
            shader.uniform("MVP", MVP);
            if (useGamma)
                shader.uniform("localGamma", 1.0f / scene->gamma);
            if (scene->colorMapping.x > 0)
                shader.uniform("colorMapping", vec2(scene->colorMapping.x, 1.0f / scene->colorMapping.y));

            glBindVertexArray(pvsSource->vao);

            int i = 0;
            glActiveTexture(GL_TEXTURE7);
            for (auto& m : pvsSource->materialRanges) {
                auto& mat = pvsSource->materials[m.first];
                useCullFace(cullface && mat.opaque);
                shader.uniform("localColor", mat.color);
                glBindTexture(GL_TEXTURE_2D, mat.texture && mat.texture->loaded ? mat.texture->id : fallbackColor);
                glDrawArrays(GL_TRIANGLES, i, m.second);
                i += m.second;
            }

        } else {
            timer("Depth Pass");

            // depth only pass
            auto& shader = getShader("render", {
                {"DEPTH_ONLY",  true},
                {"PACK_NORMAL", deferredRemote},
            });
            if (!shader) return;
            shader.use();
            shader.uniform("MVP", MVP);
            glBindVertexArray(pvsSource->vao);
            glDrawArrays(GL_TRIANGLES, 0, pvsSource->vertexCount);
        }

        // blend
        glEnable(GL_BLEND);
        glDepthMask(GL_FALSE);
        glBlendFunc(GL_ONE_MINUS_DST_ALPHA, GL_DST_ALPHA);
        glDepthFunc(GL_LEQUAL);
    }

    timer("Main Pass");
    if (composite) {
        glDrawBuffer(GL_COLOR_ATTACHMENT1);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    if (blendScenes || combineViews || scenes.size() == 1) {

        // current scene
        renderViews(scene);

        // previous scene
        if (blendingScenes) {
            timer("Second Pass");
            glDrawBuffer(GL_COLOR_ATTACHMENT2);
            glClear(GL_COLOR_BUFFER_BIT);
            renderViews(*(++scenes.begin()));
        }

    } else if (true /*TODO configurable*/) {
        // multi-server mode: separated async. prim. view and cubemap

        // find scene with primary view (priority >= 0)
        for (auto& s : scenes) {
            bool found = false;
            for (auto& v : s->views) {
                if (v.priority >= 0) {
                    renderView(v);
                    found = true;
                }
            }
            if (found) break;
        }

        // find scene with cubemap (priority < 0)
        for (auto& s : scenes) {
            bool found = false;
            for (auto& v : s->views) {
                if (v.priority < 0) {
                    renderView(v);
                    found = true;
                }
            }
            if (found) break;
        }

    } else { // use as cache

        // render views with priority >= 0
        for (auto& s : scenes)
            renderViews(s, 1);

        // render other views from current scene
        renderViews(scene, -1);
    }

    // clear state
    glEnable(GL_CULL_FACE);
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LESS);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    if (composite) {
        bool remoteTextureLayer2 = inpaintFlags & INPAINT_PULL_IN_COMPOSITE && inpaint > 0;
        bool remoteTexture2Layer2 = inpaintFlags & INPAINT_PULL_IN_COMPOSITE && inpaint > 1 && blendingScenes;

        if (inpaint > 0) {
            timer("Inpaint");
            inpainter.process(frameSize, remoteColor[0],
                              remoteTextureLayer2 ? remoteColorLayer2[0] : 0,
                              inpaintFlags & INPAINT_SKIP_INIT);
        }

        if (blendingScenes && inpaint > 1) {
            timer("Second Inpaint");
            inpainter.process(frameSize, remoteColor[1],
                              remoteTexture2Layer2 ? remoteColorLayer2[1] : 0,
                              inpaintFlags & INPAINT_SKIP_INIT);
        }

        timer("Composite");
        bool hasTexture = pvsSource->hasUV;
        bool hasBackgroundTexture = !debugLocal.render && scene->background.texture && scene->background.texture->loaded;
        bool useGamma = scene->gamma != 1.0;
        bool useColorMapping = hasTexture && scene->colorMapping.x > 0;
        auto& shader = getShader("composite", {
             {"LOCAL_TEXTURE", hasTexture},
             {"LOCAL_GAMMA", useGamma},
             {"BACKGROUND_TEXTURE", hasBackgroundTexture},
             {"REMOTE_TEXTURE_COLOR_MAPPING", useColorMapping},
             {"BLEND_SCENES", blendingScenes},
             {"REMOTE_TEXTURE_LAYER2", remoteTextureLayer2},
             {"REMOTE_TEXTURE2_LAYER2", remoteTexture2Layer2}
         });
        shader.use();
        shader.uniform("projection", projection);
        shader.uniform("view", viewpoint.view);

        glBindSampler(0, nearestSampler);
        glBindSampler(1, nearestSampler);
        glBindSampler(2, nearestSampler);
        glBindSampler(3, nearestSampler);
        glBindSampler(4, nearestSampler);

        // depth
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, depth);

        // local texture
        if (hasTexture) {
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, localColor);
        }
        if (useGamma)
            shader.uniform("localGamma", 1.0f / scene->gamma);

        // remote texture
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, remoteColor[0]);
        if (remoteTextureLayer2) {
            glActiveTexture(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_2D, remoteColorLayer2[0]);
        }
        if (useColorMapping)
            shader.uniform("colorMapping", vec2(scene->colorMapping.x, 1.0f / scene->colorMapping.y));

        // second remote texture
        if (blendingScenes) {
            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_2D, remoteColor[1]);
            if (remoteTexture2Layer2) {
                glActiveTexture(GL_TEXTURE6);
                glBindTexture(GL_TEXTURE_2D, remoteColorLayer2[1]);
            }
            shader.uniform("blendScenesFactor", blendScenesFactor);
        }

        // background
        shader.uniform("backgroundColor", debugLocal.render ? vec3(0) : scene->background.color);
        if (hasBackgroundTexture) {
            glActiveTexture(GL_TEXTURE4);
            glBindTexture(GL_TEXTURE_2D, scene->background.texture->id);
        }

        glDisable(GL_BLEND);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);

        glBindSampler(0, 0);
        glBindSampler(1, 0);
        glBindSampler(2, 0);
        glBindSampler(3, 0);
        glBindSampler(4, 0);
    }

    glBindVertexArray(0);
    glUseProgram(0);

    if (debug.render || debugLocal.render) {
        timer("Debug");
        if (composite) {
            glViewport(0, 0, frameSize.x, frameSize.y);
            glClear(GL_DEPTH_BUFFER_BIT);
            auto& shader = getShader("render", { {"DEPTH_ONLY",  true} });
            if (!shader) return;
            shader.use();
            shader.uniform("MVP", MVP);
            glBindVertexArray(pvsSource->vao);
            glDrawArrays(GL_TRIANGLES, 0, pvsSource->vertexCount);
        }

        glDepthFunc(GL_LEQUAL);
        if (debug.render)
            renderDebug(scenes, projection, viewpoint.view);
        if (debugLocal.render)
        {
            glBindVertexArray(vao);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            auto MVP = projection * viewpoint.view;
            auto viewCube = glm::scale(glm::translate(mat4(1), vec3(-1.0f)), vec3(2.0f, 2.0f, 1.0f + debug.viewScale));
            debug.cubeWireframe.use();
            debug.cubeWireframe.uniform("color", glm::vec3(1));
            debug.cubeWireframe.uniform("MVP", MVP * inverse(debugLocal.view) * inverse(projection) * viewCube);
            glDrawArrays(GL_LINES, 0, 24);
            glBindVertexArray(0);
            glUseProgram(0);
        }
        glDepthFunc(GL_LESS);
    }

    timer();
}

void Renderer::renderDebug(const std::list<Scene*>& scenes, const glm::mat4& projection, const glm::mat4& view) {
    auto MVP = projection * view;
    for (auto& scene : scenes) {
        int cubeWireframePoints = 24;

        // server lines
        if (scene->debug.renderLines && !scene->debug.lines.empty()) {
            debug.simpleShader.use();
            debug.simpleShader.uniform("MVP", MVP);
            auto& lines = scene->debug.lines;
            if (!debug.linesVAO)
                glGenVertexArrays(1, &debug.linesVAO);
            if (!debug.linesBuffer)
                glGenBuffers(1, &debug.linesBuffer);
            glBindVertexArray(debug.linesVAO);
            glBindBuffer(GL_ARRAY_BUFFER, debug.linesBuffer);
            glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(Debug::Vertex), lines.data(), GL_STREAM_DRAW);
            glEnableVertexAttribArray(0);
            glEnableVertexAttribArray(1);
            gl::vertexAttribPointer(0, 3, GL_FLOAT, sizeof(Debug::Vertex), 0);
            gl::vertexAttribPointer(1, 3, GL_FLOAT, sizeof(Debug::Vertex), sizeof(vec3));
            glDrawArrays(GL_LINES, 0, (unsigned)lines.size());
        }

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // server spheres
        if (scene->debug.renderSpheres) {
            int circlePoints = 32;
            debug.circle.use();
            debug.circle.uniform("step", glm::pi<float>() * 2 / circlePoints);
            for (auto& s : scene->debug.spheres) {
                debug.circle.uniform("color", s.color);
                debug.circle.uniform("MVP", projection * scale(translate(mat4(1), vec3(view * vec4(s.pos, 1))), vec3(s.radius)));
                glDrawArrays(GL_LINE_LOOP, 0, circlePoints);
                if (debug.sphereAxis) {
                    auto transform = MVP * scale(translate(mat4(1), s.pos), vec3(s.radius));
                    debug.circle.uniform("MVP", transform);
                    glDrawArrays(GL_LINE_LOOP, 0, circlePoints);
                    debug.circle.uniform("MVP", transform * rotate(mat4(1), pi<float>() / 2, { 1, 0, 0 }));
                    glDrawArrays(GL_LINE_LOOP, 0, circlePoints); // Y axis
                    debug.circle.uniform("MVP", transform * rotate(mat4(1), pi<float>() / 2, { 0, 1, 0 }));
                    glDrawArrays(GL_LINE_LOOP, 0, circlePoints);
                }
            }
        }

        debug.cubeWireframe.use();
        auto viewCube = glm::scale(glm::translate(mat4(1), vec3(-1.0f)), vec3(2.0f, 2.0f, 1.0f + debug.viewScale));

        // server AABB
        if (scene->debug.renderAABB)
            for (auto& b : scene->debug.aabb) {
                debug.cubeWireframe.uniform("MVP", MVP * glm::scale(glm::translate(mat4(1), b.min), b.max - b.min));
                debug.cubeWireframe.uniform("color", b.color);
                glDrawArrays(GL_LINES, 0, cubeWireframePoints);
            }

        // server views
        if (scene->debug.renderViews)
            for (auto& v : scene->debug.views) {
                debug.cubeWireframe.uniform("MVP", MVP * v.transform * inverse(v.projection) * viewCube);
                debug.cubeWireframe.uniform("color", v.color);
                glDrawArrays(GL_LINES, 0, cubeWireframePoints);
            }

        // views
        debug.cubeWireframe.uniform("color", vec3(1, 0.2, 0.2));
        for (auto& v : scene->views)
            if (v.debug.renderView) {
                debug.cubeWireframe.uniform("MVP", MVP * inverse(v.view) * inverse(v.projection) * viewCube);
                glDrawArrays(GL_LINES, 0, cubeWireframePoints);
            }

        // requested view
        if (scene->debug.renderRequestView) {
            auto& params = scene->projParams;
            auto proj = perspective<float>(radians(params.x), (float)scene->frameSize.x / scene->frameSize.y, params.y, params.z);
            debug.cubeWireframe.uniform("MVP", MVP * inverse(scene->viewpoint.view) * inverse(proj) * viewCube);
            debug.cubeWireframe.uniform("color", vec3(0.2, 1, 0.2));
            glDrawArrays(GL_LINES, 0, cubeWireframePoints);
        }
    }

    glBindVertexArray(0);
    glUseProgram(0);
}

void Renderer::renderTexture(const ivec2& frameSize, GLuint texture) {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, frameSize.x, frameSize.y);

    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    texPlot.use();

    glBindVertexArray(vao);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture ? texture : fallbackColor);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glBindVertexArray(0);
    glUseProgram(0);
}
