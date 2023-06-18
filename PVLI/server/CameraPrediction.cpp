#include "CameraPrediction.h"

using namespace std;
using namespace glm;

static inline vec3 position(const mat4& m) { return vec3(m[3]); }
static inline void setPosition(mat4& m, const vec3& p) { m[3] = vec4(p, 1); }

void CameraPrediction::add(float t, const glm::mat4& c) {
    prev = cur;
    cur = {c, t};
}

void CameraPrediction::reset() {
    prev = cur = Sample();
}

float CameraPrediction::speed() {
    if (!isValid()) return 0;
    return distance(position(cur.camera), position(prev.camera)) / (cur.time - prev.time);
}

vector<glm::mat4> CameraPrediction::predictCorners(float duration, float scale) {
    if (!isValid() || scale <= 0) return {};

    vector<vec3> offsets = {
        { 1,  1, 0},
        {-1,  1, 0},
        {-1, -1, 0},
        { 1, -1, 0},
    };

    vector<mat4> cameras;
    auto cam = duration > 0 ? predict(duration) : cur.camera;
    for (auto& o : offsets)
        cameras.push_back(cam * translate(mat4(1), o * scale));
    return cameras;
}

glm::mat4 CameraPrediction::predict(float duration) {
    if (!isValid() || duration == 0) return cur.camera;
    float scale = 1.0f + duration / (cur.time - prev.time);
    mat4 cam = cur.camera;

    // rotation
    if (rotation) {
        quat q1 = quat_cast(prev.camera), q2 = quat_cast(cur.camera);
        cam = mat4_cast(q1 != q2 ? slerp(q1, q2, scale) : q2);
    }

    // position
    vec3 p1 = position(prev.camera), p2 = position(cur.camera);
    setPosition(cam, distance(p1, p2) > 1e-12f ? p1 + (p2 - p1) * scale : p2);

    return cam;
}

vector<glm::mat4> CameraPrediction::predict(float durationStart, float durationEnd, int sampleCount) {
    if (!isValid()) return {cur.camera};
    vector<mat4> cameras = {};
    for (int i = 0; i < sampleCount; i++) {
        float v = (float) i / (float) (sampleCount - 1);
        cameras.push_back(predict(durationStart + v * (durationEnd - durationStart)));
    }
    return cameras;
}