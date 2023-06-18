#version 330

uniform mat4 MVP;
layout(location = 0) in vec3 pos;

#ifdef LOCAL_TEXTURE
layout(location = 2) in vec2 uv;
out vec2 localTexUV;
#endif

#ifdef REMOTE_TEXTURE
layout(location = 1) in vec3 flatNormal;
#ifndef CUBEMAP
uniform mat4 texMVP;
out vec4 texCoord;
#endif
uniform mat4 texMV;
uniform mat3 texNormalTransform;
out vec3 texEyeCoord;
out vec3 texNormal;
#endif

void main() {
	gl_Position = MVP * vec4(pos, 1.0);

	#ifdef LOCAL_TEXTURE
	localTexUV = uv;
	#endif

	#ifdef REMOTE_TEXTURE
	#ifndef CUBEMAP
	texCoord = texMVP * vec4(pos, 1.0);
	#endif
	texEyeCoord = vec3(texMV * vec4(pos, 1.0));
	texNormal = texNormalTransform * flatNormal;
	#endif
}
