#version 330

uniform mat4 MVP;

const vec3 lines[24] = vec3[] (
	vec3(0, 0, 0), vec3(1, 0, 0),
	vec3(1, 0, 0), vec3(1, 1, 0),
	vec3(1, 1, 0), vec3(0, 1, 0),
	vec3(0, 1, 0), vec3(0, 0, 0),

	vec3(0, 0, 1), vec3(1, 0, 1),
	vec3(1, 0, 1), vec3(1, 1, 1),
	vec3(1, 1, 1), vec3(0, 1, 1),
	vec3(0, 1, 1), vec3(0, 0, 1),

	vec3(0, 0, 0), vec3(0, 0, 1),
	vec3(1, 0, 0), vec3(1, 0, 1),
	vec3(1, 1, 0), vec3(1, 1, 1),
	vec3(0, 1, 0), vec3(0, 1, 1)
);

void main() {
	gl_Position = MVP * vec4(lines[gl_VertexID], 1.0);
}
