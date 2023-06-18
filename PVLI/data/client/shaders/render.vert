#version 330

#ifndef REMOTE_TEXTURE_DEFERRED
uniform mat4 MVP;
layout(location = 0) in vec3 pos;
#endif

#ifdef PACK_NORMAL
layout(location = 1) in vec4 packedFlatNormal;
flat out vec4 packedNormal;
#endif

#ifdef LOCAL_TEXTURE
layout(location = 2) in vec2 uv;
out vec2 localTexUV;
#endif

#ifdef REMOTE_TEXTURE
#ifdef REMOTE_TEXTURE_DEFERRED

const vec2 vertices[4] = vec2[] (
	vec2(0, 0),
	vec2(1, 0),
	vec2(1, 1),
	vec2(0, 1)
);
out vec2 coord;

#else // not REMOTE_TEXTURE_DEFERRED

layout(location = 1) in vec4 packedFlatNormal;
#ifndef CUBEMAP
uniform mat4 texMVP;
out vec4 texClipCoord;
#endif
uniform mat4 texMV;
uniform vec3 texOrigin;
uniform mat3 texNormalTransform;
#ifndef COMBINED_VIEWS_SHARED_ORIGIN
out vec3 texCoord;
flat out vec3 texNormal;
#endif
#ifdef COMBINED_VIEWS
uniform mat4 cubemapMV;
uniform vec3 cubemapOrigin;
uniform mat3 cubemapNormalTransform;
out vec3 cubemapCoord;
flat out vec3 cubemapNormal;
#endif

#endif // REMOTE_TEXTURE_DEFERRED
#endif // REMOTE_TEXTURE

vec3 rotX(vec3 v) { return vec3(v.x, v.z, -v.y); }

void main() {
	#ifndef REMOTE_TEXTURE_DEFERRED
	gl_Position = MVP * vec4(pos, 1.0);
	#endif

	#ifdef PACK_NORMAL
	packedNormal = packedFlatNormal;
	#endif

	#ifdef LOCAL_TEXTURE
	localTexUV = uv;
	#endif

	#ifdef REMOTE_TEXTURE
	#ifdef REMOTE_TEXTURE_DEFERRED

	// render one rectangle over whole screen
	coord = vertices[gl_VertexID];
	gl_Position = vec4(coord * 2 - 1, 0, 1);

	#else // not REMOTE_TEXTURE_DEFERRED
	vec3 flatNormal = packedFlatNormal.xyz * 2.0f - 1.0f;

	#ifdef CUBEMAP
		#ifdef CUBEMAP_AA
		texCoord = rotX(pos - texOrigin);
		texNormal = rotX(flatNormal);
		#else
		texCoord = vec3(texMV * vec4(pos, 1.0));
		texNormal = texNormalTransform * flatNormal;
		#endif
	#else // not CUBEMAP
		texClipCoord = texMVP * vec4(pos, 1.0);
		#ifndef COMBINED_VIEWS_SHARED_ORIGIN
		texCoord = pos - texOrigin; // no need to rotate as it is used only for depth offset
		texNormal = flatNormal;
		#endif
	#endif // CUBEMAP

	#ifdef COMBINED_VIEWS
		#ifdef CUBEMAP_AA
		cubemapCoord = rotX(pos - cubemapOrigin);
		cubemapNormal = rotX(flatNormal);
		#else
		cubemapCoord = vec3(cubemapMV * vec4(pos, 1.0));
		cubemapNormal = cubemapNormalTransform * flatNormal;
		#endif
	#endif // COMBINED_VIEWS

	#endif // REMOTE_TEXTURE_DEFERRED
	#endif // REMOTE_TEXTURE
}
