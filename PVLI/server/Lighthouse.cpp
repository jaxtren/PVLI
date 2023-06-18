#ifdef ENABLE_LIGHTHOUSE
#include "Lighthouse.h"
#include "common.h"
#include <array>
#include <set>

#define CUDABUILD
#include "helper_math.h"
#include "rendersystem.h"
using namespace lighthouse2;

#include <filesystem>
namespace fs = std::filesystem;

static std::map<std::string, float> configParams = {
    { "spp", 4 },
    { "minPathLength", 1 },
    { "maxPathLength", 100 },
    { "geometryEpsilon", 0.0001 },
    { "filter", 1 },
    { "filterPhases", 5 },
    { "clampValue", 10000 },
    { "clampDirect", 10000 },
    { "clampIndirect", 1000 },
    { "clampReflection", 1000 },
    { "clampRefraction", 1000 },
    { "reprojWeight", 0.8 },
    { "reprojWeightFallback", 0.4 },
    { "reprojSpatialCount", 4 },
    { "shadeKeepPhase", 1 },
    { "shadeMergePhase", 0 },
    { "closestOffset", 0 },
    { "closestOffsetMin", 0 },
    { "closestOffsetMax", 0 },
    { "normalFactor", 128 },
    { "distanceFactor", 0.5 },
    { "varianceFactor", 4 },
    { "varianceGauss", 1 },
    { "varianceReprojFactor", 0 },
    { "reprojLinearFilter", 0 },
    { "firstLayerOnly", 0 },
    { "reprojMaxDistFactor", 10 },
    { "depthMode", 2 },
    { "reorderFragments", 1 },
    { "storeBackground", 0 },
    { "disableAlphaMask", 0 },
    { "useAnyHit", 1 },
    { "emissiveFactor", 1 },
    { "pathRegularization", 0 },
    { "deterministicLight", -1 }
};

bool LighthouseConfig::updateConfig(const Config& cfg, bool fallbackToDefault) {
    bool ret = false;
    for (const auto& c : configParams) {
        auto found = values.find(c.first) != values.end();
        if (cfg.contains(c.first))
            ret = cfg.get(c.first, values[c.first]) || ret || !found;
        else if (fallbackToDefault && !found)
            values[c.first] = c.second;
    }
    return ret;
}

void LighthouseConfig::provideConfig(Config cfg) {
    for (auto& v : values) cfg.set(v.first, v.second);
}

Lighthouse::Lighthouse(const Config& cfg, const std::string& rendererName, const std::string& sceneName) {
    this->rendererName = rendererName;
    this->sceneName = sceneName;

    // renderer
    cfg.get("path", appPath);
    auto p = fs::current_path();
    fs::current_path(appPath); // temporary change path for correct backend initialization
    renderer = RenderAPI::CreateRenderAPI(rendererName.c_str());
    fs::current_path(p);

    // camera
    renderer->DeserializeCamera(cameraFile.c_str());

    // scene
    fs::path path = sceneName;
    if (!sceneName.empty()) {
        if (path.extension() == ".aao") loadSceneFromAAO(sceneName);
        else renderer->AddScene(path.filename().string().c_str(), path.parent_path().string().c_str());
        renderer->GetScene()->SetSkyDome(nullptr);
    }

    // config
    cfg.get("skyTexture", skyTexture);

	// lights
    cfg.get("lights", lightsName);
    if (lightsName.empty()) lightsName = path.replace_extension(".lights").string();
    auto scene = renderer->GetScene();
    pt::ptree tree;
    if (pt::read_info_ext(lightsName, tree)) {
        cout << "Lights: " << lightsName << endl;
        for(auto& t : tree) {

            // sky
            if (t.first == "skyTexture") {
                skyTexture = t.second.data();
                continue;
            }

            if (t.first == "skyColor") {
                skyColor = t.second.get_value<glm::vec3>();
                continue;
            }

            // type
            DynamicLight l;
            if (t.first == "point") l.type = POINT;
            else if (t.first == "spot") l.type = SPOT;
            else if (t.first == "directional") l.type = DIRECTIONAL;
            if (l.type == NONE) continue;

            // data
            float3 radiance = {0, 0, 0}, position = {0, 0, 0}, direction = {0, 0, 1};
            float inner = 0, outer = 0, radius = 0, angle = 0;

            Config c(t.second);
            c.get("radiance", *(glm::vec3*)&radiance); // nasty workaround
            c.get("position", *(glm::vec3*)&position);
            c.get("direction", *(glm::vec3*)&direction);
            c.get("inner", inner);
            c.get("outer", outer);
            c.get("radius", radius);
            c.get("angle", angle);
            c.get("timeScale", l.timeScale);
            c.get("timeOffset", l.timeOffset);

            switch(l.type) {
                case POINT: l.id = scene->AddPointLight(glmVec3<float3>(position), radiance); break;
                case SPOT: l.id = scene->AddSpotLight(position, direction, cos(inner) , cos(outer), radiance); break;
                case DIRECTIONAL: l.id = scene->AddDirectionalLight(direction, radiance); break;
            }

            switch(l.type) {
                case POINT: scene->pointLights[l.id]->radius = radius; break;
                case SPOT: scene->spotLights[l.id]->radius = radius; break;
                case DIRECTIONAL: scene->directionalLights[l.id]->sinAngle = sin(angle); break;
            }

            // path
            string p;
            c.get("path", p);
            if (!p.empty())
                l.path.loadFromFile((path.parent_path() / p).string());

            lights.push_back(move(l));
        }
    }

    updateSky();
}

Lighthouse::~Lighthouse() {
    renderer->Shutdown();
}

void Lighthouse::loadSceneFromAAO(const std::string& path) {
    auto scene = renderer->GetScene();

    // load UV scales
    vector<float2> scales;
    {
        ifstream input(path + ".scales", ios::binary);
        if (!input) return;
        int count = 0;
        input.read(reinterpret_cast<char*>(&count), sizeof(int));
        scales.resize(count / 2);
        input.read(reinterpret_cast<char*>(scales.data()), sizeof(float) * count);
    }

    // load textures and create materials
    vector<int> materials;
    {
        auto basePath = fs::path(path).parent_path();
        for (int i=0; i<scales.size(); i++) {
            auto texturePath = basePath / (to_string(i) + ".png");
            if (fs::exists(texturePath)) {
                auto material = new HostMaterial();
                material->color.textureID = scene->FindOrCreateTexture(texturePath.string());
                material->roughness.value = 0.8f;
                material->metallic.value = 0.2f;
                materials.push_back(scene->AddMaterial(material));
            } else materials.push_back(-1); // no material
        }
    }

    // load mesh
    {
        ifstream input(path, ios::binary);
        if (!input) return;

        #ifdef SAVE_OBJ
        ofstream out("out.obj");
        #endif

        auto mesh = new HostMesh();
        struct AAOVertex {
            float3 pos, normal;
            float u, v; // not using float2 because of alignment
            int texture;
        } vertex;

        struct Triangle {
            float3 a, b, c;
            bool operator < (const Triangle& t) const {
                if (a.x != t.a.x) return a.x < t.a.x;
                if (a.y != t.a.y) return a.y < t.a.y;
                if (a.z != t.a.z) return a.z < t.a.z;

                if (b.x != t.b.x) return b.x < t.b.x;
                if (b.y != t.b.y) return b.y < t.b.y;
                if (b.z != t.b.z) return b.z < t.b.z;

                if (c.x != t.c.x) return c.x < t.c.x;
                if (c.y != t.c.y) return c.y < t.c.y;
                if (c.z != t.c.z) return c.z < t.c.z;

                return false;
            }
        };

        std::set<Triangle> triangles;

        vector<int> indices;
        vector<float3> vertices, normals;
        vector<float2> uv;
        int prevMaterial = -1;
        int vertexCount = 0;
        int duplicatedTriangles = 0;
        int duplicatedTrianglesFlipped = 0;
        while (input.read(reinterpret_cast<char*>(&vertex), sizeof(AAOVertex))) {
            if (vertex.texture < 0) vertex.texture = 0;
            int material = materials[vertex.texture];
            if (material < 0) continue; // ignore vertices without material
            if (prevMaterial != material && !indices.empty()) {
                mesh->BuildFromIndexedData(indices, vertices, normals, uv, {}, {}, {}, {}, prevMaterial);
                indices.clear();
                vertices.clear();
                normals.clear();
                uv.clear();
            }

            indices.push_back((int)indices.size());
            vertices.push_back(vertex.pos);
            normals.push_back(vertex.normal);
            uv.push_back(make_float2(vertex.u, vertex.v) * scales[vertex.texture]);
            vertexCount++;

            // check for duplicated triangles
            if (!vertices.empty() && vertices.size() % 3 == 0) {
                bool removeLast = false;
                auto& V = vertices;
                auto S = vertices.size();
                if (!triangles.insert({V[S-3], V[S-2], V[S-1]}).second) {
                    removeLast = true;
                    duplicatedTriangles++;
                } else if (triangles.count({V[S-1], V[S-2], V[S-3]}) > 0 ||
                           triangles.count({V[S-3], V[S-1], V[S-2]}) > 0 ||
                           triangles.count({V[S-2], V[S-3], V[S-1]}) > 0) {

                    // offset duplicated in direction of normal
                    auto normal = normalize(cross(V[S-2] - V[S-3], V[S-1] - V[S-3]));
                    auto offset = normal * 1e-3f;
                    V[S-1] += offset;
                    V[S-2] += offset;
                    V[S-3] += offset;

                    // removeLast = true;
                    duplicatedTrianglesFlipped++;
                }

                if (removeLast) {
                    for (int i=0; i<3; i++) {
                        indices.pop_back();
                        vertices.pop_back();
                        normals.pop_back();
                        uv.pop_back();
                    }
                    continue;
                }
            }

            prevMaterial = material;

            #ifdef SAVE_OBJ
            out << "v " << vertex.pos.x << ' ' << vertex.pos.y << ' ' << vertex.pos.z << endl;
            out << "vn " << vertex.normal.x << ' ' << vertex.normal.y << ' ' << vertex.normal.z << endl;
            out << "vt " << vertex.u * scales[vertex.texture].x << ' ' << vertex.v * scales[vertex.texture].y << endl;
            #endif
        }
        #ifdef SAVE_OBJ
        for (int i=0; i<vertexCount; i+=3)
            out << "f "
                << i+1 << '/' << i+1 << '/' << i+1 << ' '
                << i+2 << '/' << i+2 << '/' << i+2 << ' '
                << i+3 << '/' << i+3 << '/' << i+3 << endl;
        #endif

        cout << "[AAO scene] all triangles: " << vertexCount / 3
            << " duplicated: " << duplicatedTriangles
            << " duplicated flipped: " << duplicatedTrianglesFlipped << endl;

        if (!indices.empty())
            mesh->BuildFromIndexedData(indices, vertices, normals, uv, {}, {}, {}, {}, prevMaterial);

        scene->AddInstance(scene->AddMesh(mesh), mat4::RotateX(PI/2)); // Y-up to Z-up
    }
}

void Lighthouse::updateSky() {
    auto& sky = renderer->GetScene()->sky;
    delete sky;
    sky = new HostSkyDome();
    if (!skyTexture.empty() && skyTexture != "none") { // texture
        sky->Load(skyTexture.c_str(), glmVec3<float3>(skyColor));
        sky->worldToLight = mat4::RotateX(PI);
    } else sky->SetColor(glmVec3<float3>(skyColor)); // color
}

void Lighthouse::update(float time) {

    // lights
    auto scene = renderer->GetScene();
    for (auto& l : lights) {
        if (l.path.duration() <= 0) continue;
        auto s = l.path.sample(time * l.timeScale + l.timeOffset, true);
        float3 pos = glmVec3<float3>(s.pos);
        float3 dir = glmVec3<float3>(glm::vec3(0,0,-1) * s.rot); // FIXME direction/rotation

        switch(l.type) {
            case POINT: scene->pointLights[l.id]->position = pos; break;
            case SPOT: scene->spotLights[l.id]->position = pos; break;
        }

        switch(l.type) {
            case SPOT: scene->spotLights[l.id]->direction = dir; break;
            case DIRECTIONAL: scene->directionalLights[l.id]->direction = dir; break;
        }
    }

    renderer->SynchronizeSceneData();
}

void Lighthouse::set(const string& name, float value) {
    if (renderer)
        renderer->Setting(name.c_str(), value);
}

void Lighthouse::set(const LighthouseConfig& cfg) {
    for (auto& v : cfg.values)
        set(v.first, v.second);
}

int Lighthouse::render(const glm::ivec2& size, const glm::mat4 &cam, const int* const* cuMask,
                       LinkedFragment* cuFragments, int maxCount, bool setSize) {

    // camera
    auto camera = renderer->GetCamera();
    camera->distortion = 0;
    camera->aperture = 0;
    camera->FOV = fovy;
    camera->aspectRatio = (float)size.x / (float)size.y;
    camera->pixelCount = {size.x, size.y};

    // transform
    auto C = cam;
    C[1] = -C[1];
    C[2] = -C[2];
    C = glm::transpose(C);
    mat4 transform;
    memcpy(&transform, &C, sizeof(float) * 16);
    camera->SetMatrix(transform);

    if (setSize) {
        set("width", size.x);
        set("height", size.y);
    }

    // lighthouse2::Fragment and LinkedFragment are binary compatible
    return renderer->Render(cuMask, (Fragment*) cuFragments, maxCount);
}

void Lighthouse::initScene(Scene& scene, bool loadTextures) {
    renderer->SynchronizeSceneData();
    scene.free();
    auto srcScene = renderer->GetScene();

    // materials
    auto path = fs::path(sceneName).parent_path().string() + '/';
    scene.materials.resize(srcScene->materials.size());
    for (int i = 0; i < scene.materials.size(); i++) {
        auto src = srcScene->materials[i];
        auto& dst = scene.materials[i];
        dst.id = i;
        dst.color = glmVec3(src->color.value);
        if (dst.color.x < 0 || dst.color.y < 0 || dst.color.z < 0) dst.color = glm::vec3(1);
        if (src->color.textureID >= 0) {
            auto tex = srcScene->textures[src->color.textureID];
            if (!tex->origin.empty()) {
                if (loadTextures)
                    dst.loadTexture(tex->origin);
                else {
                    dst.textureName = tex->origin;

                    // detect opaque
                    if (tex->flags & HostTexture::LDR) {
                        auto pixels = (uchar4*) tex->GetLDRPixels();
                        for (unsigned j = 0; j < tex->width * tex->height; j++)
                            if (pixels[j].w < 255) {
                                dst.opaque = false;
                                break;
                            }
                    } else if (tex->flags & HostTexture::HDR) {
                        auto pixels = tex->GetHDRPixels();
                        for (unsigned j = 0; j < tex->width * tex->height; j++)
                            if (pixels[j].w < 1) {
                                dst.opaque = false;
                                break;
                            }
                    }
                }
            }
        }
    }

    // meshes
    scene.meshes.resize(srcScene->meshPool.size());
    for (int i = 0; i < scene.meshes.size(); i++) {
        auto src = srcScene->meshPool[i];
        auto& dst = scene.meshes[i];

        vector<Mesh::VertexData> vertices;
        int mat = -1, primCount = 0;
        for (auto t : src->triangles) {
            if (t.material != mat) {
                if (primCount > 0)
                    dst.primitives.push_back({primCount, &scene.materials[mat]});
                mat = t.material;
                primCount = 0;
            }
            vertices.push_back({glmVec3(t.vertex0), glmVec3(t.vN0), {t.u0, t.v0}, (int) t.material});
            vertices.push_back({glmVec3(t.vertex1), glmVec3(t.vN1), {t.u1, t.v1}, (int) t.material});
            vertices.push_back({glmVec3(t.vertex2), glmVec3(t.vN2), {t.u2, t.v2}, (int) t.material});
            primCount += 3;
        }
        if (primCount > 0)
            dst.primitives.push_back({primCount, &scene.materials[mat]});
        dst.setVertices(vertices.data(), (int)vertices.size());
    }

    // nodes from renderer instances to keep IDs synchronized
    scene.root = new Scene::Node();
    for (int i : renderer->GetInstances()) {
        auto src = srcScene->nodePool[i];
        auto n = new Scene::Node();
        n->meshes.push_back(&scene.meshes[src->meshID]);
        memcpy(&n->transform, &src->combinedTransform, sizeof(float) * 16);
        n->transform = transpose(n->transform);
        scene.root->children.push_back(n);
    }
}

bool Lighthouse::updateConfig(const Config& cfg) {
    cfg.get("path", appPath);
    if (cfg.get("skyTexture", skyTexture) | cfg.get("skyColor", skyColor)) updateSky();
    if (cfg.get("lights", lightsName)) return false;
    return true;
}

void Lighthouse::provideConfig(Config cfg) {
    cfg.set("skyTexture", skyTexture);
    cfg.set("skyColor", skyColor);
    cfg.set("lights", lightsName);
}

void Lighthouse::stats(vector<pair<string, double>>& data, const string& name) {
    auto prefix = !name.empty() && name.back() != '.' ? name + "." : name;
    double m = 1000;
    auto s = renderer->GetCoreStats();

    data.push_back({prefix + "Time", s.renderTime * m});
    data.push_back({prefix + "Time.Shade", s.shadeTime * m});
    data.push_back({prefix + "Time.Filter", s.filterTime * m});
    data.push_back({prefix + "Time.Trace", (s.traceTime0 + s.traceTime1 + s.traceTimeX + s.shadowTraceTime) * m});
    data.push_back({prefix + "Time.Trace.0", s.traceTime0 * m});
    data.push_back({prefix + "Time.Trace.1", s.traceTime1 * m});
    data.push_back({prefix + "Time.Trace.X", s.traceTimeX * m});
    data.push_back({prefix + "Time.Trace.Shadow", s.shadowTraceTime * m});

    data.push_back({prefix + "Rays", s.totalRays});
    data.push_back({prefix + "Rays.Extension", s.totalExtensionRays});
    data.push_back({prefix + "Rays.Extension.Primary", s.primaryRayCount});
    data.push_back({prefix + "Rays.Extension.Secondary", s.bounce1RayCount});
    data.push_back({prefix + "Rays.Extension.Deep", s.deepRayCount});
    data.push_back({prefix + "Rays.Shadow", s.totalShadowRays});
}

#endif
