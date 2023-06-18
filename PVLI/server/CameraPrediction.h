#pragma once

#include <vector>
#include <limits>
#include "glmHelpers.h"
#include "StatCounter.h"
#include "Config.h"

/**
 * Simple camera prediction for PVS
 */
class CameraPrediction {
private:
    struct Sample {
        glm::mat4 camera;
        float time = std::numeric_limits<float>::max();
        bool isValid() { return time != std::numeric_limits<float>::max(); }
    };
    Sample prev, cur;
    bool isValid () { return cur.isValid() && prev.isValid() && cur.time > prev.time; }

public:

    // settings
    bool rotation = true;

    void add(float time, const glm::mat4& camera);
    void reset();
    float speed();

    glm::mat4 predict(float duration);
    std::vector<glm::mat4> predict(float durationStart, float durationEnd, int sampleCount);
    std::vector<glm::mat4> predictCorners(float duration, float scale);
};
