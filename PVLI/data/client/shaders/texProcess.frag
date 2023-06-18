#version 420

#define EPS 0.00001f
#define TO_FLOAT 0.0039215686f

// TODO platform specific, switch to uniform once more decode APIs are supported
#define TILE_ALIGNMENT 128
#define TILE_ALIGNMENT_INV 0.0078125f

layout(binding = 0)uniform sampler2D tex;
layout(binding = 1)uniform sampler2D pixelMask;
uniform vec2 dstDim;

in vec2 uv;
layout (location = 0) out vec4 color;

void yuv420_nv12_UVs(out vec2 uvY, out vec2 uvU, out vec2 uvV)
{
	vec2 yuvDim = textureSize(tex, 0);

	float hTilesCount = floor(dstDim.y * TILE_ALIGNMENT_INV - EPS) + 1;
	float chromaYOffset = hTilesCount*TILE_ALIGNMENT + floor(gl_FragCoord.y*0.5f) + 0.5f;
	vec2 uFrag = vec2(gl_FragCoord.x, chromaYOffset);
	vec2 vFrag = vec2(gl_FragCoord.x, chromaYOffset);

	bool isEvenColumn = abs(mod(gl_FragCoord.x, 2) - 0.5f) < EPS;
	if (isEvenColumn)
		vFrag.x += 1;
	else
		uFrag.x -= 1;		

	uvY = gl_FragCoord.xy / yuvDim;
	uvU = uFrag / yuvDim;
	uvV = vFrag / yuvDim;
}

vec3 yuv420_to_rgb()
{
	vec2 uvY, uvU, uvV;

	yuv420_nv12_UVs(uvY, uvU, uvV);

	int C = int(texture(tex, uvY).r * 255.0f) - 16;
	int D = int(texture(tex, uvU).r * 255.0f) - 128;
	int E = int(texture(tex, uvV).r * 255.0f) - 128;
	int R = clamp(( 298 * C           + 409 * E + 128) >> 8, 0, 255);
	int G = clamp(( 298 * C - 100 * D - 208 * E + 128) >> 8, 0, 255);
	int B = clamp(( 298 * C + 516 * D           + 128) >> 8, 0, 255);

	return vec3(R * TO_FLOAT, G * TO_FLOAT, B * TO_FLOAT);
}

void main()
{
	color.rgb = yuv420_to_rgb();
	color.a = texture(pixelMask, uv).r;
}