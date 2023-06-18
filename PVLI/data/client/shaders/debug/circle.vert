#version 330

uniform mat4 MVP;
uniform float step;

void main() {
	float a = float(gl_VertexID) * step;
	gl_Position = MVP * vec4(cos(a), sin(a), 0, 1);
}
