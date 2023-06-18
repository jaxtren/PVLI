#pragma once

#include "glmHelpers.h"

struct LinkedFragment {
    glm::vec4 color;
    float depth;
    int instID;
    int primID;
    int next;
};