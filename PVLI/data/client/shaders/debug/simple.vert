#version 330

uniform mat4 MVP;
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;
out vec3 color;

void main() {
	gl_Position = MVP * vec4(inPos, 1);
	color = inColor;
}
