#pragma once

#include <GL/glew.h>
#include <string>
#include <map>
#include "cudaHelpers.h"
#include "glmHelpers.h"
#include "structs.h"
#include "Scene.h"
#include "Config.h"
#include "CameraPath.h"

namespace lighthouse2 {
    class RenderAPI;
    class GLTexture;
}

struct LighthouseConfig {
    std::map<std::string, float> values;
    bool updateConfig(const Config&, bool fallbackToDefault = false);
    void provideConfig(Config);
};

class Lighthouse {
    lighthouse2::RenderAPI* renderer = nullptr;

    enum LightType {
        NONE, POINT, DIRECTIONAL, SPOT
    };

    struct DynamicLight {
        CameraPath path;
        float timeOffset = 0;
        float timeScale = 1;
        int id = -1;
        LightType type = NONE;
    };

    std::vector<DynamicLight> lights;

    void loadSceneFromAAO(const std::string&);
    void updateSky();

public:
    // read-only settings
    std::string rendererName;
    std::string sceneName;
    std::string lightsName;
    std::string skyTexture;
    glm::vec3 skyColor = {1, 1, 1};

    // settings
    int cuBlock2D = 8;
    std::string cameraFile = "camera.xml";
    std::string appPath = "./";
    float fovy = 60;

    Lighthouse(const Config& cfg, const std::string& renderer, const std::string& scene);
    ~Lighthouse();

    bool updateConfig(const Config&);
    void provideConfig(Config);

    void update(float time);
    int render(const glm::ivec2& size, const glm::mat4 &cam, const int* const* cuMask,
               LinkedFragment* cuFragments, int maxCount, bool setSize = true);

    void set(const LighthouseConfig&);
    void set(const string& name, float value);
    inline void set(const string& name, int value) { set(name, (float)value); }
    inline void resetViews() { set("resetViews", 1); }

    void stats(std::vector<std::pair<std::string, double>>&, const std::string& name);
    void initScene(Scene& scene, bool loadTextures = false);
};