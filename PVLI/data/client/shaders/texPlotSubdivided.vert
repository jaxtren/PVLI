#version 420

uniform ivec2 layerSize;
out vec2 coord;

void main() {
	vec2 uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
	gl_Position = vec4(uv * 2 - 1, 0, 1);
	coord = uv * vec2(layerSize);
}