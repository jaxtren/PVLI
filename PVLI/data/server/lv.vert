#version 430

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform int maskOffset;

layout(std430, binding = 1) buffer Mask {
    int mask[];
};

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

flat out int primID;
out vec2 fuv;
out vec3 fnormal;

void main() {
    gl_Position = projection * view * model * vec4(pos, 1.0);
    fnormal = vec3(model * vec4(normal, 0));
    fuv = uv;
    primID = gl_VertexID / 3;

    if (mask[maskOffset + primID] <= 0) { //do not render triangle
        gl_Position = vec4(0);
        primID = -1;
    }
}
