#include "GLbackend.h"
#include <iostream>

#ifdef ENABLE_LIGHTHOUSE
//lighthouse API uses GLAD
extern "C" {
    typedef void* (* GLADloadproc)(const char* name);
    int gladLoadGLLoader(GLADloadproc load);
}
#endif

using namespace std;

#ifdef ENABLE_GLFW
static void glfw_error_callback(int error, const char* description){
    cerr << "Error: " << description << endl;
}

bool GLFWbackend::init(int w, int h, const std::string& name){
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) return false;

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    window = glfwCreateWindow(1024, 768, name.c_str(), NULL, NULL);

    if (!window) {
        cerr << "GLFW error: cannot init window" << endl;
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);

    //GLEW
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        terminate();
        cerr << "GLEW error: " << glewGetErrorString(err) << endl;
        return false;
    }

    //GLAD
    #ifdef ENABLE_LIGHTHOUSE
    if (!gladLoadGLLoader( (GLADloadproc)glfwGetProcAddress )) return false;
    #endif

    return true;
}

void GLFWbackend::swapBuffers(){
    glfwSwapBuffers(window);
    glfwPollEvents();
}

void GLFWbackend::terminate() {
    glfwDestroyWindow(window);
    glfwTerminate();
}
#endif

#ifdef ENABLE_EGL
bool EGLbackend::init(int w, int h, const std::string& name){

    display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if(display == EGL_FALSE || display == EGL_NO_DISPLAY){
      cerr << "EGL: no display" << endl;
      return false;
    }
    EGLint major, minor;
    if(!eglInitialize(display, &major, &minor)){
      cerr << "EGL: cannot initialize" << endl;
      return false;
    }

    EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_BLUE_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_RED_SIZE, 8,
        EGL_DEPTH_SIZE, 24,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
    };

    EGLint numConfigs;
    EGLConfig config;
    eglChooseConfig(display, configAttribs, &config, 1, &numConfigs);
    eglBindAPI(EGL_OPENGL_API);

    if(w > 0 && h > 0){
        EGLint pbufferAttribs[] = {
            EGL_WIDTH, w,
            EGL_HEIGHT, h,
            EGL_NONE,
        };
        surface = eglCreatePbufferSurface(display, config, pbufferAttribs);
        context = eglCreateContext(display, config, EGL_NO_CONTEXT, NULL);
        eglMakeCurrent(display, surface, surface, context);
    } else {
        context = eglCreateContext(display, config, EGL_NO_CONTEXT, NULL);
        eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context);
    }

    //GLEW
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        terminate();
        cerr << "GLEW error: " << glewGetErrorString(err) << endl;
        return false;
    }

    //GLAD
    #ifdef ENABLE_LIGHTHOUSE
    if (!gladLoadGLLoader( (GLADloadproc)eglGetProcAddress )) return false;
    #endif

    return true;
}

void EGLbackend::swapBuffers(){
    eglSwapBuffers(display, surface);
}

void EGLbackend::terminate() {
    eglTerminate(display);
}
#endif