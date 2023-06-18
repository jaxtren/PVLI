#pragma once

#include <GL/glew.h>
#include "glmHelpers.h"
#include "graphic.h"
#include "types.h"
#include "Shader.h"

class BlockPixelRelocator {
    GLuint fbo = 0, vao = 0, tex = 0;
    glm::ivec2 size = {0, 0};
    gl::Shader shader;
    bool binBlocks = false;
    GLuint fallbackBlock = 0;

public:
    BlockPixelRelocator();
    ~BlockPixelRelocator();

    // settings
    bool skipFullLayers = false;

    bool loadShaders();
    void relocate(GLuint fullLayersTex, GLuint blocksTex, GLuint blocks,
                  const glm::ivec2& layerSize, int fullLayersCount, const glm::ivec2& fullLayersOffset,
                  const glm::ivec2& blocksTexSize, const glm::ivec3& blocksSize, bool binBlocks);
};