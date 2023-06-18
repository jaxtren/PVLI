#include "PVScache.h"

using namespace std;
using namespace glm;

inline bool inRange(const vec4& area, const vec3& point) {
    return distance(vec3(area), point) < area.w;
}

static inline vec3 position(const mat4& m) { return vec3(m[3]); }
static inline void setPosition(mat4& m, const vec3& p) { m[3] = vec4(p, 1); }

bool PVScache::add(const mat4& transform, const mat4& projection, PVS::State& state, float mergeDist) {

    // find closest
    float closestDist = 1e18f;
    auto closest = data.end();
    auto pos = position(transform);
    for (auto it = data.begin(); it != data.end(); it++) {
        float d = distance(pos, position(it->transform));
        if (d < closestDist) {
            closest = it;
            closestDist = d;
        }
    }

    // merge with closest
    if (closest != data.end() && closestDist < mergeDist) {
        if (closest != data.begin())
            data.splice(data.begin(), data, closest);
        auto& s = data.front();
        auto prevPos = position(s.transform);
        s.projection = projection;
        s.transform = transform;
        setPosition(s.transform, (prevPos + position(s.transform)) * 0.5f);
        s.pvs.max(state);
        return false; // merged
    } else {  // add new state
        data.emplace_front();
        auto& s = data.front();
        s.transform = transform;
        s.projection = projection;
        s.pvs = state;
        return true; // new sample
    }
}

vector<PVScache::Sample*> PVScache::collect(int lastSamples, const vector<glm::vec4>& areas) {
    vector<PVScache::Sample*> ret;
    int i = 0;
    for (auto& s : data) {
        bool add = lastSamples < 0 || i++ < lastSamples;
        if (!add)
            for (auto& a : areas)
                if (inRange(a, position(s.transform))) {
                    add = true;
                    break;
                }

        if (add) ret.emplace_back(&s);
    }
    return ret;
}

void PVScache::clear(const vector<glm::vec4>& areas) {
    // remove all outside of areas
    for (auto it = data.begin(); it != data.end(); ) {
        bool remove = true;
        for (auto& a : areas)
            if (inRange(a, position(it->transform))) {
                remove = false;
                break;
            }
        if (remove) it = data.erase(it);
        else it++;
    }
}

void PVScache::clear(int maxSamples) {
    while (data.size() > maxSamples)
        data.pop_back();
}