#pragma once

#include <initguid.h>
#include <d3d9.h>
#include <dxva2api.h>

#include <GL/glew.h>
#include <GL/wglew.h>
#include <GLFW/glfw3.h>

#include "Shader.h"
#include "VideoCodingUtil.h"

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavutil/hwcontext.h>
    #include <libavutil/hwcontext_dxva2.h>
};

namespace hw_accel
{
    class dxva2_opengl : public Interop
    {
        GLuint vbo = 0, vao = 0;
        gl::Shader yuv420toRGBAandMasking;
        std::map<GLuint, GLuint> FBOs;
        bool isInitialized = false;

        struct Textures
        {
            GLuint mask = 0, yuv = 0;

            ~Textures()
            {
                glDeleteTextures(1, &mask);
                glDeleteTextures(1, &yuv);
                mask = yuv = 0;
            }
        };

        class uid
        {
            uint16_t freeId = 0;
            std::queue<uint16_t> freeIds;
        public:
            uint16_t get()
            {
                uint16_t id;
                if (!freeIds.empty())
                {
                    id = freeIds.front();
                    freeIds.pop();
                }
                else
                    id = freeId++;
                return id;
            }
            void putBack(uint16_t id)
            {
                assert(id <= freeId);
                freeIds.emplace(id);
            }
        };

        uid uidGen;
        std::unordered_map<uint16_t, std::unique_ptr<Textures>> pool;
    public:
        dxva2_opengl() = default;
        ~dxva2_opengl() override
        {
            for (auto& [tex, fbo] : FBOs)
                glDeleteFramebuffers(1, &fbo);
            FBOs.clear();
            glDeleteVertexArrays(1, &vao);
            glDeleteBuffers(1, &vbo);
            vao = vbo = 0;
        }

        // realloc per texture for now
        uint16_t GetGPUResources() override
        {
            auto id = uidGen.get();
            pool[id] = std::make_unique<Textures>();
            return id;
        }
        void PutBackGPUResources(uint16_t id) override
        {
            pool.erase(id);
            uidGen.putBack(id);
        }

        void SubmitMask(uint16_t resId, int x, int y, unsigned char* rawMask) override
        {
            setTexture(pool[resId]->mask, GL_RED, glm::ivec2(x, y), rawMask);
        }

        void SubmitColor(uint16_t resId, AVFrame* avframe) override
        {
            if (!avframe)
                return;

            auto* surface = reinterpret_cast<IDirect3DSurface9*>(avframe->data[3]);
            D3DSURFACE_DESC descriptor = {};
            surface->GetDesc(&descriptor);

            D3DLOCKED_RECT lockedRect;
            surface->LockRect(&lockedRect, nullptr, D3DLOCK_READONLY);

            const glm::ivec2 yuvSize(lockedRect.Pitch, descriptor.Height * 1.5);
            void* yuvPtr = lockedRect.pBits;
            setTexture(pool[resId]->yuv, GL_RED, yuvSize, yuvPtr);

            surface->UnlockRect();
        }

        void ConversionAndMasking(uint16_t resId, unsigned target, int x, int y) override
        {
            if (!isInitialized)
                init();

            render(resId, target, glm::ivec2(x, y));

            for (auto& [tex, fbo] : FBOs)
                glDeleteFramebuffers(1, &fbo);
            FBOs.clear();
        }

    private:
        void init()
        {
            const float vertices[] = {
            -0.5f, -0.5f, 0.0f,
             0.5f, -0.5f, 0.0f,
             0.0f,  0.5f, 0.0f
            };
            glGenVertexArrays(1, &vao);
            glGenBuffers(1, &vbo);
            glBindVertexArray(vao);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
            glEnableVertexAttribArray(0);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindVertexArray(0);
            yuv420toRGBAandMasking.load("texProcess", "shaders");

            isInitialized = true;
        }

        static void setTexture(GLuint& texture, GLint iFormat, glm::ivec2 size, void* data)
        {
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

            GLenum format = 0;
            if (iFormat == GL_RED)
                format = GL_RED;
            else if (iFormat == GL_RG)
                format = GL_RG;
            if (texture == 0)
            {
                glGenTextures(1, &texture);
                glBindTexture(GL_TEXTURE_2D, texture);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            }
            else
                glBindTexture(GL_TEXTURE_2D, texture);

            glTexImage2D(GL_TEXTURE_2D, 0, iFormat, size.x, size.y, 0, format, GL_UNSIGNED_BYTE, data);
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        void render(uint16_t resId, GLuint target, glm::ivec2 size)
        {
            const auto fbo = getFBO(target);

            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glViewport(0, 0, size.x, size.y);

            glDisable(GL_CULL_FACE);
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            yuv420toRGBAandMasking.use();
            yuv420toRGBAandMasking.uniform("dstDim", glm::vec2(size.x, std::max(144, size.y)));

            glBindVertexArray(vao);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, pool[resId]->yuv);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, pool[resId]->mask);

            glDrawArrays(GL_TRIANGLES, 0, 3);

            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glBindVertexArray(0);
            glUseProgram(0);
        }

        GLuint getFBO(GLuint texture)
        {
            auto fbo = FBOs.find(texture);
            if (fbo != FBOs.end())
                return fbo->second;

            GLuint newFBO = 0;

            glGenFramebuffers(1, &newFBO);
            glBindFramebuffer(GL_FRAMEBUFFER, newFBO);
            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture, 0);

            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
                std::cerr << "GL ERROR Renderer FBO: " << glCheckFramebufferStatus(GL_FRAMEBUFFER) << "\n";

            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            FBOs[texture] = newFBO;
            return newFBO;
        }
    };
}

namespace notUsed
{
    typedef struct DXVA2DevicePriv
    {
        HMODULE d3dlib;
        HMODULE dxva2lib;

        HANDLE device_handle;

        IDirect3D9* d3d9;
        IDirect3DDevice9* d3d9device;
    } DXVA2DevicePriv;

    class Interop
    {
        IDirect3DDevice9* device = nullptr;
        IDirect3DDevice9Ex* deviceEx = nullptr;
        HANDLE deviceH = nullptr;

        IDirect3DDeviceManager9* deviceManager = nullptr;
        HRESULT res = 0;

        HANDLE interop;

    public:
        Interop(const AVHWDeviceContext* ctx)
        {
            auto* priv = static_cast<DXVA2DevicePriv*>(ctx->user_opaque);
            device = reinterpret_cast<IDirect3DDevice9*>(priv->d3d9device);
            deviceEx = reinterpret_cast<IDirect3DDevice9Ex*>(device);

            auto* dxva2dc = static_cast<AVDXVA2DeviceContext*>(ctx->hwctx);
            deviceManager = dxva2dc->devmgr;

            interop = wglDXOpenDeviceNV(deviceEx);
        }
        ~Interop()
        {
            wglDXCloseDeviceNV(interop);
        }

        void process(const AVFrame* src)
        {
            res = deviceManager->LockDevice(deviceH, &device, TRUE);
            auto* surface = reinterpret_cast<IDirect3DSurface9*>(src->data[3]);

            GLuint glTex;
            glGenTextures(1, &glTex);
            glBindTexture(GL_TEXTURE_2D, glTex);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glBindTexture(GL_TEXTURE_2D, 0);

            auto* texture_h = wglDXRegisterObjectNV(interop, surface, glTex, GL_TEXTURE_2D, WGL_ACCESS_READ_ONLY_NV);
            if (!wglDXLockObjectsNV(interop, 1, &texture_h))
                std::cout << "Failed locking texture for access by OpenGL %s\n";

            // OPEN GL WORK HERE

            if (!wglDXUnlockObjectsNV(interop, 1, &texture_h))
                std::cout << "Failed locking texture for access by OpenGL %s\n";
            wglDXUnregisterObjectNV(interop, texture_h);

            res = deviceManager->UnlockDevice(deviceH, TRUE);
        }

    };
}