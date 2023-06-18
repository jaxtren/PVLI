#pragma once

#include <GL/glew.h>
#include <atomic>
#include <memory>
#include <map>
#include <future>
#include <functional>
#include "px_sched.h"
#include "glmHelpers.h"
#include "graphic.h"
#include "common.h"

class TextureManager;

struct Texture {
    using Ref = std::shared_ptr<Texture>;
    using WeakRef = std::weak_ptr<Texture>;

    ~Texture();

    TextureManager* manager = nullptr;
    unsigned char* data = nullptr;

    std::string name;
    GLuint id = 0;
    glm::ivec2 size;
    std::atomic_bool loaded = false, finished = false;
};

class TextureManager {
    using Scheduler = std::function<void(std::function<void()>)>;
    mutexed<std::map<std::string, std::pair<Texture::WeakRef, std::future<void>>>> textures;

public:
    std::string path = "./";
    std::thread::id gpuThreadId;
    Scheduler runOnGpuThread;

    Texture::Ref load(const std::string&);
    void GC();
    size_t getSize();
    bool loading();
};