#pragma once

#include <vector>
#include <list>
#include <GL/glew.h>
#include "cudaHelpers.h"
#include "glmHelpers.h"
#include "PVS.h"

class PVScache {
public:

    struct Sample {
        glm::mat4 transform;
        glm::mat4 projection;
        PVS::State pvs;
    };

    std::list<Sample> data;

    //area = sphere, x,y,z: center, w: radius
    bool add(const glm::mat4& transform, const glm::mat4& projection, PVS::State& state, float mergeDist = 0);
    std::vector<Sample*> collect(int lastSamples = 0, const std::vector<glm::vec4>& areas = {});
    void clear(const std::vector<glm::vec4>& areas);
    void clear(int maxSamples = 0);
    inline size_t size () { return data.size(); }
};
