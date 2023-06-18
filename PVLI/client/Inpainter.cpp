#include "Inpainter.h"

using namespace std;
using namespace glm;

Inpainter::Inpainter() {
    loadShaders();
}

Inpainter::~Inpainter() {
    clear();
}

bool Inpainter::loadShaders() {
    shader.init.load("init", "shaders/inpainter");
    shader.pushMasked.load("process", "shaders/inpainter", {"PUSH", "SRC_ALPHA_MASK"});
    shader.push.load("process", "shaders/inpainter", {"PUSH"});
    shader.pull.load("process", "shaders/inpainter", {"PULL"});
    return true;
}

GLuint Inpainter::createTexture(const ivec2& size) {
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, size.x, size.y);
    return tex;
}

void Inpainter::clear() {
    for (auto l : layers)
        glDeleteTextures(1, &l.texture);
    layers.clear();
}

void Inpainter::process(const ivec2& size, GLuint topLayer, GLuint secondLayer, bool skipInit) {

    // allocate layers
    if (layers.empty() || layers.front().size != size) {
        layers.clear();

        auto s = size;
        while (true) {
            layers.push_back({s, layers.empty() ? 0 : createTexture(s)});
            if (s.x <= 1 && s.y <= 1) break;
            s = glm::max(ivec2(1), s / 2 + s % 2);
        }
    }

    // init
    if (!skipInit) {
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        if (!shader.init) return;
        shader.init.use();
        shader.init.uniform("size", size);
        glBindImageTexture(0, topLayer, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);
        shader.init.dispatchCompute(size);
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    // borrow textures
    layers[0].texture = topLayer;
    GLuint backupLayer1 = layers[1].texture;
    if (secondLayer) layers[1].texture = secondLayer;

    // push
    if (!shader.push || (skipInit && !shader.pushMasked)) return;
    glActiveTexture(GL_TEXTURE0);
    for (int i=1; i<layers.size(); i++) {
        auto& push = skipInit && i == 1 ? shader.pushMasked : shader.push;
        push.use();
        push.uniform("srcSize", layers[i-1].size);
        glBindTexture(GL_TEXTURE_2D, layers[i-1].texture);
        push.uniform("dstSize", layers[i].size);
        glBindImageTexture(0, layers[i].texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);
        push.dispatchCompute(layers[i].size);
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    // pull
    if (!shader.pull) return;
    shader.pull.use();
    glActiveTexture(GL_TEXTURE0);
    for (int i=(int)layers.size() - 2, e = secondLayer ? 1 : 0; i >= e; i--) {
        // skip pushing to first layer, when second layer is provided
        shader.pull.uniform("srcSize", layers[i+1].size);
        glBindTexture(GL_TEXTURE_2D, layers[i+1].texture);
        shader.pull.uniform("dstSize", layers[i].size);
        glBindImageTexture(0, layers[i].texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);
        shader.pull.dispatchCompute(layers[i].size);
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    // restore textures
    layers[0].texture = 0;
    layers[1].texture = backupLayer1;

    glUseProgram(0);
}
