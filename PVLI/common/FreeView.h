#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "glmHelpers.h"
#include "Config.h"

class FreeView {
    glm::vec2 prevCursor = {0, 0};
    double prevTime = 0;

public:
    struct Location {
        glm::vec3 pos = {0, 0, 0};
        glm::vec2 rot = {0, 0};

        glm::mat4 getTransform();
        void setTransform(const glm::mat4&);
    };

    Location location;
    float speed = 10;
    float speedFast = 20;
    float mouseSensitivity = 0.005f;

    void updateConfig(const Config& cfg);
    void update(GLFWwindow* window, glm::vec2 cursorPos, bool allowMouse = true, bool allowKeyboard = true);
};

