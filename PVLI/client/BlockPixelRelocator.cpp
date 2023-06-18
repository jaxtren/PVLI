#include "BlockPixelRelocator.h"
#include <vector>
#include <string>

using namespace std;
using namespace glm;

BlockPixelRelocator::BlockPixelRelocator() {
    loadShaders();
    glGenVertexArrays(1, &vao);
    glActiveTexture(GL_TEXTURE0);
    glGenTextures(1, &fallbackBlock);
    glBindTexture(GL_TEXTURE_2D_ARRAY, fallbackBlock);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    ivec2 block(0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RG32I, 1, 1, 1, 0, GL_RG_INTEGER, GL_INT, &block);
    glBindTexture(GL_TEXTURE_2D, 0);
}

BlockPixelRelocator::~BlockPixelRelocator() {
    glDeleteVertexArrays(1, &vao);
    if(tex) glDeleteTextures(1, &tex);
    if(fbo) glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &fallbackBlock);
}

bool BlockPixelRelocator::loadShaders() {
    vector<string> defs;
    if(binBlocks) defs.push_back("BINARY_BLOCKS");
    if(skipFullLayers) defs.push_back("SKIP_FULL_LAYERS");
    shader.load("pixelRelocator", "shaders", defs);
    if (shader) {
        shader.use();
        shader.uniform("blocks", 0);
        glUseProgram(0);
    }
    return shader;
}

void BlockPixelRelocator::relocate(GLuint fullLayersTex, GLuint blocksTex, GLuint blocks,
                                   const ivec2& layerSize, int fullLayersCount, const ivec2& fullLayersOffset,
                                   const ivec2& blocksTexSize, const ivec3& blocksSize, bool binBlocks) {
    if (layerSize.x > size.x || layerSize.y > size.y) {
        size = glm::max(layerSize, size);
        if(tex) glDeleteTextures(1, &tex);
        if(fbo) glDeleteFramebuffers(1, &fbo);

        //tex
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, size.x, size.y);
        glBindTexture(GL_TEXTURE_2D, 0);

        //fbo
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex, 0);
        glDrawBuffer(GL_COLOR_ATTACHMENT0);
        if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            cerr << "GL ERROR depth FBO: "  << glCheckFramebufferStatus(GL_FRAMEBUFFER) << endl;
    }

    if (this->binBlocks != binBlocks) {
        this->binBlocks = binBlocks;
        loadShaders();
    }

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, layerSize.x, layerSize.y);
    glDisable(GL_DEPTH_TEST);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, blocks ? blocks : fallbackBlock);
    glBindImageTexture(0, fullLayersTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8I);
    glBindImageTexture(1, blocksTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8I);

    glBindVertexArray(vao);
    shader.use();
    shader.uniform("layerSize", layerSize);
    shader.uniform("fullLayersCount", fullLayersCount);
    shader.uniform("fullLayersOffset", fullLayersOffset);
    shader.uniform("blocksSize", blocksSize);
    shader.uniform("blocksImageSize", blocksTexSize);

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glUseProgram(0);
    glBindVertexArray(0);
}