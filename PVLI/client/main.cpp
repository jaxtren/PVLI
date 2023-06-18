#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <string>
#include "Application.h"

#define PX_SCHED_IMPLEMENTATION 1
#include "px_sched.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace std;

static void GLFWErrorCallback(int error, const char* description){
    cerr << "GLFW Error: " << description << endl;
}

int main(int argc, char **argv) {
    glfwSetErrorCallback(GLFWErrorCallback);
    if (!glfwInit()) return -1;

    int w = 400, h = 300;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow *window = glfwCreateWindow(w, h, "Client", NULL, NULL);

    if(!window){
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        window = glfwCreateWindow(w, h, "Client", NULL, NULL);
    }

    if(!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    GLenum err = glewInit();
    if (GLEW_OK != err) {
        glfwDestroyWindow(window);
        glfwTerminate();
        cerr << "GLEW error: " << glewGetErrorString(err) << endl;
        return 0;
    }

    {
        Application app(window, argc > 1 ? string(argv[1]) : "");
        app.run();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}