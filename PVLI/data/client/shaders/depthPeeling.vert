#version 330

uniform mat4 projection;
layout(location = 0) in vec3 pos;

void main() {
	gl_Position = projection * vec4(pos, 1.0);
}
