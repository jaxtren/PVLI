#pragma once

#include <string>
#include <GL/glew.h>

class GLbackend {
public:
    virtual ~GLbackend() = default;
    virtual bool init(int w, int h, const std::string& name) = 0;
    virtual void swapBuffers() = 0;
    virtual void terminate() = 0;
};

#ifdef ENABLE_GLFW
#include <GLFW/glfw3.h>

class GLFWbackend : public GLbackend {
public:
    GLFWwindow* window;

    virtual bool init(int w, int h, const std::string& name);
    virtual void swapBuffers();
    virtual void terminate();
};
#endif

#ifdef ENABLE_EGL
#include <EGL/egl.h>

class EGLbackend : public GLbackend {
public:
    EGLDisplay display = 0;
    EGLContext context = 0;
    EGLSurface surface = 0;

    virtual bool init(int w, int h, const std::string& name);
    virtual void swapBuffers();
    virtual void terminate();
};
#endif
