#include "FreeView.h"
using namespace glm;

void FreeView::updateConfig(const Config &cfg) {
    cfg.get("Speed", speed);
    cfg.get("FastSpeed", speedFast);
    cfg.get("MouseSensitivity", mouseSensitivity);
    cfg.get("Position", location.pos);
    cfg.get("Rotation", location.rot);
}

void FreeView::update(GLFWwindow *window, glm::vec2 cursorPos, bool allowMouse, bool allowKeyboard) {

    //time
    double frameTime = glfwGetTime();
    float relTime = (float)(frameTime - prevTime);
    prevTime = frameTime;

    //cursor
    vec2 relCursor = cursorPos - prevCursor;
    prevCursor = cursorPos;

    //rotation
    if(allowMouse && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1)) {
        location.rot += relCursor * mouseSensitivity;
        const auto pi = glm::pi<float>(), pi2 = pi * 2;
        while (location.rot.x >  pi) location.rot.x -= pi2;
        while (location.rot.x < -pi) location.rot.x += pi2;
        while (location.rot.y >  pi) location.rot.y -= pi2;
        while (location.rot.y < -pi) location.rot.y += pi2;
    }

    //movement
    if(allowKeyboard) {
        vec3 add(0);
        if (glfwGetKey(window, GLFW_KEY_W) || glfwGetKey(window, GLFW_KEY_UP)) add.y += 1;
        if (glfwGetKey(window, GLFW_KEY_S) || glfwGetKey(window, GLFW_KEY_DOWN)) add.y -= 1;
        if (glfwGetKey(window, GLFW_KEY_A) || glfwGetKey(window, GLFW_KEY_LEFT)) add.x -= 1;
        if (glfwGetKey(window, GLFW_KEY_D) || glfwGetKey(window, GLFW_KEY_RIGHT)) add.x += 1;
        if (glfwGetKey(window, GLFW_KEY_SPACE)) add.z += 1;
        if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL)) add.z -= 1;
        if(add != vec3(0)) {
            bool shift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT);
            add = normalize(add) * speed * (shift ? speedFast : 1.0f) * relTime;
            location.pos -= vec3(rotate(mat4(1), -location.rot.x, {0, 0, 1}) * vec4(add, 1));
        }
    }
}

glm::mat4 FreeView::Location::getTransform() {
    mat4 t(1);
    t = glm::rotate(t, rot.y, {1, 0, 0});
    t = glm::rotate(t, rot.x, {0, 0, 1});
    t = translate(t, pos);
    return t;
}

void FreeView::Location::setTransform(const glm::mat4& t) {
    auto m = inverse(t);
    pos = -vec3(m[3]);
    rot.x = -atan2f(m[0].y, m[0].x);
    rot.y = atan2f(sqrtf(m[2].x*m[2].x + m[2].y*m[2].y) * (m[1].z > 0 ? -1 : 1), m[2].z);
}
