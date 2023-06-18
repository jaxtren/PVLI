#version 330

uniform mat4 mvp;
uniform int firstID;

layout(location = 0) in vec3 pos;
flat out int id;

void main() {
    id = firstID + gl_VertexID / 3; // works only without element buffer
    gl_Position = mvp * vec4(pos, 1.0);
}