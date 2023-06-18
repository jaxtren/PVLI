#pragma once

#include <GL/glew.h>
#include <algorithm>
#include "glmHelpers.h"
#include "graphic.h"
#include "TimerGL.h"
#include "Shader.h"

class Inpainter {
    GLuint vao = 0;

    struct {
        gl::Shader init, pushMasked, push, pull;
    } shader;

public:
    struct Layer {
        glm::ivec2 size;
        GLuint texture = 0;
    };

    std::vector<Layer> layers;

    Inpainter();
    ~Inpainter();

    bool loadShaders();
    GLuint createTexture(const glm::ivec2& size);
    void process(const glm::ivec2& size, GLuint topLayer, GLuint secondLayer = 0, bool skipInit = false);
    void clear();
};