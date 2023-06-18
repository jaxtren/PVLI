#include "Culling.h"

using namespace std;
using namespace glm;

AABB AABB::transform(const mat4& transform) const {
    AABB box;
    box.extend(vec3(transform * vec4(min.x, min.y, min.z, 1)));
    box.extend(vec3(transform * vec4(min.x, min.y, max.z, 1)));
    box.extend(vec3(transform * vec4(min.x, max.y, min.z, 1)));
    box.extend(vec3(transform * vec4(min.x, max.y, max.z, 1)));
    box.extend(vec3(transform * vec4(max.x, min.y, min.z, 1)));
    box.extend(vec3(transform * vec4(max.x, min.y, max.z, 1)));
    box.extend(vec3(transform * vec4(max.x, max.y, min.z, 1)));
    box.extend(vec3(transform * vec4(max.x, max.y, max.z, 1)));
    return box;
}