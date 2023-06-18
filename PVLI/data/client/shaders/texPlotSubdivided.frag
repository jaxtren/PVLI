#version 420

layout (binding = 0) uniform sampler2D tex;
layout (binding = 1) uniform sampler2D tex2;

in vec2 coord;
layout (location = 0) out vec4 color;

// same as in render.frag
bool isFirstRemote (ivec2 coord) {
	#ifdef REMOTE_TEXTURE_CHECKER
	return (coord.x & 1) == (coord.y & 1);
	#elif defined(REMOTE_TEXTURE_2X1)
	return (coord.x & 1) == 0;
	#else
	return (coord.y & 1) == 0;
	#endif
}
ivec2 subCoord(ivec2 coord) {
	#ifdef REMOTE_TEXTURE_2X1
	return ivec2(coord.x >> 1, coord.y);
	#else
	return ivec2(coord.x, coord.y >> 1);
	#endif
}

void main() {
	ivec2 icoord = ivec2(coord);
	ivec2 loc = subCoord(icoord);
	if (isFirstRemote(icoord))
		color = texelFetch(tex, loc, 0);
	else color = texelFetch(tex2, loc, 0);
}