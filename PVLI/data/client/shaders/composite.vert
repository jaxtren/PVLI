#version 420

uniform mat4 projection;
uniform mat4 view;

const vec2 vertices[4] = vec2[] (
	vec2(0, 0),
	vec2(1, 0),
	vec2(1, 1),
	vec2(0, 1)
);

out vec2 coord;
out vec3 dir;

void main() {
	coord = vertices[gl_VertexID];
	gl_Position = vec4(coord * 2 - 1, 0, 1);

	// ray direction for background
	vec4 d = inverse(projection) * vec4(gl_Position.xy, -1, 1);
	dir = vec3(inverse(view) * vec4(d.xyz, 0));
}
