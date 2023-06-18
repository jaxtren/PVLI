#pragma once

#include <GL/glew.h>
#include "glmHelpers.h"
#include "graphic.h"
#include "types.h"
#include "Shader.h"

class DepthPeeling {
    gl::Shader shader;
    GLuint fbo = 0;

public:
    DepthPeeling();
    ~DepthPeeling();

    bool loadShaders();

    static GLuint createLayers(glm::ivec2 size, int count);
    void bindLayer(GLuint layers, glm::ivec2 size, int layer, const glm::mat4& projection, float epsilon, const glm::vec3& skipLayers);
    void setProjection(const glm::mat4&);
    void unbind();
};