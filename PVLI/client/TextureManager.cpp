#include "TextureManager.h"
#include "stb_image.h"
#include <future>

using namespace std;

Texture::~Texture() {
    if (data) stbi_image_free(data);
    if (!id) return;
    if (this_thread::get_id() == manager->gpuThreadId)
        glDeleteTextures(1, &id);
    else manager->runOnGpuThread([i = id, n = name]{
        glDeleteTextures(1, &i);
    });
}

Texture::Ref TextureManager::load(const std::string& name) {
    auto tex = textures.lock();
    auto it = tex->find(name);
    if (it != tex->end()) {
        auto t = it->second.first.lock();
        if (t) return t;
        tex->erase(it);
    }

    auto t = make_shared<Texture>();
    t->manager = this;
    t->name = name;
    auto it2 = tex->emplace(name, make_pair(Texture::WeakRef(t), future<void>()));

    // store future to prevent waiting for async operation on current thread
    it2.first->second.second = async(launch::async, [=, w = Texture::WeakRef(t)] {
        auto t = w.lock();
        if (!t) return;

        int channels;
        t->data = stbi_load((path + t->name).c_str(), &t->size.x, &t->size.y, &channels, 4);

        if (!t->data) {
            cout << "Not loaded texture " << t->name << endl;
            t->finished = true;
            return;
        }

        runOnGpuThread([=, w = Texture::WeakRef(t)] {
            auto t = w.lock();
            if (!t) return;

            glGenTextures(1, &t->id);
            glBindTexture(GL_TEXTURE_2D, t->id);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, t->size.x, t->size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, t->data);
            glGenerateMipmap(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, 0);

            stbi_image_free(t->data);
            t->data = nullptr;

            t->loaded = true;
            t->finished = true;

            cout << "Loaded texture " << t->name << endl;
        });
    });

    return t;
}

void TextureManager::GC() {
    auto tex = textures.lock();
    for (auto it = tex->begin(); it != tex->end(); )
        if (it->second.first.expired()) it = tex->erase(it);
        else it++;
}

size_t TextureManager::getSize() {
    size_t s = 0;
    for (auto& tex : *textures.lock() ) {
        auto t = tex.second.first.lock();
        if (t) s += t->size.x * t->size.y;
    }
    return s * 4;
}

bool TextureManager::loading() {
    for (auto& tex : *textures.lock())
        if (!tex.second.first.lock()->finished) return true;
    return false;
}