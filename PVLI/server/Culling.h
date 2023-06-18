#pragma once

#include <GL/glew.h>
#include "cudaHelpers.h"
#include "glmHelpers.h"

__host__ __device__ inline glm::vec4 normalizePlane(const glm::vec4& p) {
    return p / length(glm::vec3(p));
}

__host__ __device__ inline float pointPlaneDistance(const glm::vec3& point, const glm::vec4& plane) {
    return dot(point, glm::vec3(plane)) + plane.w;
}

struct AABB {
    glm::vec3 min = {1, 1, 1}, max = {-1, -1, -1};

    inline bool isNull() const { return min.x > max.x || min.y > max.y || min.z > max.z; }
    inline void setNull() {min = {1, 1, 1}, max = {-1, -1, -1}; }

    inline void extend(const glm::vec3& p) {
        if (isNull()) min = max = p;
        else {
            min = glm::min(min, p);
            max = glm::max(max, p);
        }
    }

    AABB transform(const glm::mat4& transform) const;

    __host__ __device__ inline int testPlane(const glm::vec4& plane) const {
        glm::vec3 p = min, n = max;
        if (plane.x >= 0) {
            p.x = max.x;
            n.x = min.x;
        }
        if (plane.y >= 0) {
            p.y = max.y;
            n.y = min.y;
        }
        if (plane.z >= 0) {
            p.z = max.z;
            n.z = min.z;
        }
        if (pointPlaneDistance(p, plane) < 0) return 1;
        if (pointPlaneDistance(n, plane) > 0) return -1;
        return 0;
    }
};

struct CullingPlanes {
    glm::vec4 planes[6]; //left, right, bottom, top, near, far

    CullingPlanes() = default;
    __host__ __device__ inline CullingPlanes(const glm::mat4& mvp) { create(mvp);}

    __host__ __device__ inline glm::vec4& operator[](size_t i) {return planes[i] ;}

    __host__ __device__ inline void create(const glm::mat4& mvp) {
        auto mat = transpose(mvp);
        planes[0] = normalizePlane(mat[3] + mat[0]); // left
        planes[1] = normalizePlane(mat[3] - mat[0]); // right
        planes[2] = normalizePlane(mat[3] + mat[1]); // bottom
        planes[3] = normalizePlane(mat[3] - mat[1]); // top
        planes[4] = normalizePlane(mat[3] + mat[2]); // near
        planes[5] = normalizePlane(mat[3] - mat[2]); // far
    }

    // test function results: 0: fully outside, 1: partially inside (conservative), 2: full inside

    __host__ __device__ inline int testTriangle(const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3) {
        int inSum = 0;
        for (auto& p : planes) {
            int in = (pointPlaneDistance(v1, p) > 0) +
                     (pointPlaneDistance(v2, p) > 0) +
                     (pointPlaneDistance(v3, p) > 0);
            if (in == 0) return 0;
            inSum += in;
        }
        return 1 + (inSum == 18);
    }

    __host__ __device__ inline int testAABB(const AABB& box) {
        int inSum = 0;
        for (auto& p : planes) {
            int test = box.testPlane(p);
            if (test > 0) return 0;
            inSum += test < 0;
        }
        return 1 + (inSum == 6);
    }
};