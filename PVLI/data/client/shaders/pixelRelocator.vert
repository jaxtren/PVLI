#version 330

uniform ivec2 layerSize;

const vec2 vertices[4] = vec2[] (
	vec2(0, 0),
	vec2(1, 0),
	vec2(1, 1),
	vec2(0, 1)
);

out vec2 coord;

void main() {
	vec2 v = vertices[gl_VertexID];
	gl_Position = vec4(v * 2 - 1, 0.0, 1.0);
	coord = v * vec2(layerSize);
}
