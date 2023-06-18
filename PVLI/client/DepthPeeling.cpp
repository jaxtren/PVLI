#include "DepthPeeling.h"

using namespace std;
using namespace glm;

DepthPeeling::DepthPeeling() {
    loadShaders();
    glGenFramebuffers(1, &fbo);
}

DepthPeeling::~DepthPeeling() {
    glDeleteFramebuffers(1, &fbo);
}

bool DepthPeeling::loadShaders() {
    shader.load("depthPeeling", "shaders");
    if (shader) {
        shader.use();
        shader.uniform("prevDepth", 0);
        glUseProgram(0);
    }
    return shader;
}

void DepthPeeling::unbind() {
    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

GLuint DepthPeeling::createLayers(ivec2 size, int count) {
    if (!size.x || !size.y || !count) return 0;
    GLuint l = 0;
    glDeleteTextures(1, &l);
    glGenTextures(1, &l);
    glBindTexture(GL_TEXTURE_2D_ARRAY, l);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_DEPTH_COMPONENT32F, size.x, size.y, count);
    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
    return l;
}

void DepthPeeling::bindLayer(GLuint layers, glm::ivec2 s, int layer, const mat4& projection, float epsilon, const vec3& skipLayers) {
    if (!shader || !layers || !s.x || !s.y) return;
    glActiveTexture(GL_TEXTURE0);

    // FBO
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, layers, 0, layer);
    glFramebufferParameteri(GL_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_WIDTH, s.x);
    glFramebufferParameteri(GL_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_HEIGHT, s.y);
    glDrawBuffer(GL_NONE);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        cerr << "GL ERROR depth FBO: " << glCheckFramebufferStatus(GL_FRAMEBUFFER) << endl;

    glViewport(0, 0, s.x, s.y);
    glClear(GL_DEPTH_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_2D_ARRAY, layers);

    shader.use();
    shader.uniform("prevLayer", layer - 1);
    shader.uniform("projection", projection);
    shader.uniform("epsilon", epsilon);
    shader.uniform("skipLayers", skipLayers);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDisable(GL_CULL_FACE);
}

void DepthPeeling::setProjection(const mat4& projection) {
    shader.uniform("projection", projection);
}
