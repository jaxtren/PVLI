#version 420

#ifndef DEPTH_ONLY
layout(location = 0) out vec4 color;
#endif

#ifdef LOCAL_TEXTURE
in vec2 localTexUV;
#ifdef LOCAL_TEXTURE_GAMMA
uniform float localGamma;
#endif
uniform vec3 localColor;
layout(binding = 7) uniform sampler2D localTex;
#endif

#ifdef REMOTE_TEXTURE
#ifdef CUBEMAP
uniform mat4 cubemapFaceTransform[6];
#else
in vec4 texCoord;
#endif
in vec3 texEyeCoord;
in vec3 texNormal;
uniform ivec2 layerSize;
uniform float pixelSize;
uniform vec2 depthOffset;
uniform vec2 depthRange;
uniform vec2 colorMapping;

#ifdef BLEND_EDGES
uniform float blendEdges;
#endif

// ------------------- subdivide -------------------

#if defined(REMOTE_TEXTURE_2X1) || defined(REMOTE_TEXTURE_1X2)
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
    #endif

// ------------------- full layers -------------------

uniform int fullLayersCount;
uniform ivec2 fullLayersOffset;

layout(binding = 1) uniform sampler2D fullLayersTex;

#if defined(REMOTE_TEXTURE_2X1) || defined(REMOTE_TEXTURE_1X2)
layout(binding = 4) uniform sampler2D fullLayersTex2;
vec4 getColorFullLayers(ivec2 coord, int layer) {
    ivec2 loc = subCoord(fullLayersOffset + coord + ivec2(0, layer * layerSize.y));
    if (isFirstRemote(coord))
    return texelFetch(fullLayersTex, loc, 0);
    else return texelFetch(fullLayersTex2, loc, 0);
}
    #else
vec4 getColorFullLayers(ivec2 coord, int layer) {
    return texelFetch(fullLayersTex, fullLayersOffset + coord + ivec2(0, layer * layerSize.y), 0);
}
    #endif

// ------------------- blocks -------------------
uniform int blockSize;

#ifdef POT_SIZES
// optimisation where power of two (POT) modulo and division is replaced with bitwise AND and shift
// x: mask for bitwise AND, y: shift
uniform ivec2 blocksSizePOT;
uniform ivec2 blocksTexPOT;
#endif

vec4 getColorBlocks_(in isampler2DArray blocks, in sampler2D blocksTex, ivec2 coord, int i) {
    #ifdef POT_SIZES
    return texelFetch(blocksTex, ivec2(i & blocksTexPOT.x, i >> blocksTexPOT.y) * blockSize + (coord & blocksSizePOT.x), 0);
    #else
    int w = textureSize(blocksTex, 0).x / blockSize;
    return texelFetch(blocksTex, ivec2(i%w, i/w) * blockSize + coord % blockSize, 0);
    #endif
}
    #ifdef BINARY_BLOCKS
ivec2 getBlock_(in isampler2DArray blocks, ivec2 coord) {
    #ifdef POT_SIZES
    return ivec2(0, texelFetch(blocks, ivec3(coord >> blocksSizePOT.y, 0), 0).x);
    #else
    return ivec2(0, texelFetch(blocks, ivec3(coord / blockSize, 0), 0).x);
    #endif
}
int getBlockIndex_(in isampler2DArray blocks, ivec2 coord, int layer, ivec2 block) {
    #ifdef POT_SIZES
    return layer >= block.y ? -1 : texelFetch(blocks, ivec3(coord >> blocksSizePOT.y, layer + 1), 0).x;
    #else
    return layer >= block.y ? -1 : texelFetch(blocks, ivec3(coord / blockSize, layer + 1), 0).x;
    #endif
}
    #else
ivec2 getBlock_(in isampler2DArray blocks, ivec2 coord) {
    #ifdef POT_SIZES
    return texelFetch(blocks, ivec3(coord >> blocksSizePOT.y, 0), 0).xy;
    #else
    return texelFetch(blocks, ivec3(coord / blockSize, 0), 0).xy;
    #endif
}
int getBlockIndex_(in isampler2DArray blocks, ivec2 coord, int layer, ivec2 block) {
    return layer >= block.y ? -1 : block.x + layer;
}
#endif

layout(binding = 2) uniform sampler2D blocksTex;
layout(binding = 3) uniform isampler2DArray blocks;

#if defined(REMOTE_TEXTURE_2X1) || defined(REMOTE_TEXTURE_1X2)

layout(binding = 5) uniform sampler2D blocksTex2;
layout(binding = 6) uniform isampler2DArray blocks2;

// redirect functions
vec4 getColorBlocks(ivec2 coord, int i) {
    if (isFirstRemote(coord))
    return getColorBlocks_(blocks, blocksTex, subCoord(coord), i);
    else return getColorBlocks_(blocks2, blocksTex2, subCoord(coord), i);
}
ivec2 getBlock(ivec2 coord) {
    if (isFirstRemote(coord))
    return getBlock_(blocks, subCoord(coord));
    else return getBlock_(blocks2, subCoord(coord));
}
int getBlockIndex(ivec2 coord, int layer, ivec2 block) {
    if (isFirstRemote(coord))
    return getBlockIndex_(blocks, subCoord(coord), layer, block);
    else return getBlockIndex_(blocks2, subCoord(coord), layer, block);
}

    #else

// redirect functions
vec4 getColorBlocks(ivec2 coord, int i) { return getColorBlocks_(blocks, blocksTex, coord, i); }
ivec2 getBlock(ivec2 coord) { return getBlock_(blocks, coord); }
int getBlockIndex(ivec2 coord, int layer, ivec2 block) { return getBlockIndex_(blocks, coord, layer, block); }
    #endif

// ------------------- depth -------------------

layout(binding = 0) uniform sampler2DArray depth;

vec4 depthGather(vec2 uv, int l) {
    #ifdef DISABLE_TEXTURE_GATHER
    // workaround mainly for NVIDIA 2000 series, where textureGather doesn't always work correctly
    vec4 d;
    ivec2 iuv = ivec2(uv * vec2(layerSize) - 0.5f);
    d.w = texelFetch(depth, ivec3(iuv + ivec2(0, 0), l), 0).x;
    d.z = texelFetch(depth, ivec3(iuv + ivec2(1, 0), l), 0).x;
    d.x = texelFetch(depth, ivec3(iuv + ivec2(0, 1), l), 0).x;
    d.y = texelFetch(depth, ivec3(iuv + ivec2(1, 1), l), 0).x;
    return d;
    #else
    return textureGather(depth, vec3(uv, l), 0);
    #endif
}

    #ifdef CLOSEST_DEPTH_RANGE
// find layer with depth closest to 'd.x' within min (d.y) - max (d.z) range
// extended version of CLOSEST_DEPTH

int getLayer(ivec2 uv, vec3 d) {
    int layerCount = textureSize(depth, 0).z;
    float minDiff = 1e24f;
    int best = -1;
    for (int l=0; l<layerCount; l++) {
        float dist = texelFetch(depth, ivec3(uv, l), 0).x;
        float diff = abs(dist - d.x);
        if (dist > d.z || diff >= minDiff) break;
        if (dist >= d.y) {
            best = l;
            minDiff = diff;
        }
    }
    return best;
}

ivec4 getLayers(vec2 uv, vec3 d) {
    int layerCount = textureSize(depth, 0).z;
    vec4 minDiff = vec4(1e24f);
    vec4 best = vec4(-1);
    for (int l=0; l<layerCount; l++) {
        vec4 dist = depthGather(uv, l);
        vec4 diff = abs(dist - d.x);
        vec4 cmp = vec4(lessThanEqual(dist, vec4(d.z))) * vec4(lessThan(diff, minDiff));
        if (all(equal(cmp, vec4(0)))) break;
        cmp *= vec4(greaterThanEqual(dist, vec4(d.y)));
        best = mix(vec4(l), best, vec4(1) - vec4(cmp));
        minDiff = mix(diff, minDiff, vec4(1) - vec4(cmp));
    }
    return ivec4(best);
}

    #elif defined(CLOSEST_DEPTH)
// find layer with depth closest to 'd'
// more accurate, depthOffset can be {0, 0}

int getLayer(ivec2 uv, float d) {
    int layerCount = textureSize(depth, 0).z;
    float minDist = 1e24f;
    int best = -1;
    for (int l=0; l<layerCount; l++) {
        float dist = abs(texelFetch(depth, ivec3(uv, l), 0).x - d);
        if (dist >= minDist) break;
        best = l;
        minDist = dist;
    }
    return best;
}

ivec4 getLayers(vec2 uv, float d) {
    int layerCount = textureSize(depth, 0).z;
    vec4 minDist = vec4(1e24f);
    vec4 best = vec4(-1);
    for (int l=0; l<layerCount; l++) {
        vec4 dist = abs(depthGather(uv, l) - d);
        bvec4 cmp = lessThan(dist,minDist);
        if (!any(cmp)) break;

        // doesn't work on NVIDIA 2000 series:
        //   best = mix(best, vec4(l), vec4(cmp));
        //   minDist = mix(minDist, dist, vec4(cmp));
        // workaround:
        best = mix(vec4(l), best, vec4(1) - vec4(cmp));
        minDist = mix(dist, minDist, vec4(1) - vec4(cmp));
    }
    return ivec4(best);
}

    #elif defined(DEPTH_RANGE)
// find closest layer with depth larger than d.x but smaller than d.y
// doesn't works correctly for back-faced triangles and need to correctly set depthOffset

int getLayer(ivec2 uv, vec2 d) {
    int layerCount = textureSize(depth, 0).z;
    for (int l=0; l<layerCount; l++) {
        float D = texelFetch(depth, ivec3(uv, l), 0).x;
        if (d.x <= D) return D < d.y ? l : -1;
    }
    return -1;
}

ivec4 getLayers(vec2 uv, vec2 d) {
    int layerCount = textureSize(depth, 0).z;
    vec4 ret = vec4(-1);
    vec4 retDepth = vec4(0);
    for (int l=0; l<layerCount; l++) {
        bvec4 mask = equal(ret, vec4(-1));
        if (!any(mask)) break;
        vec4 D = depthGather(uv, l);
        vec4 mixMask = vec4(lessThanEqual(vec4(d.x), D)) * vec4(mask);
        ret = mix(ret, vec4(l), mixMask);
        retDepth = mix(retDepth, D, mixMask);
    }
    return ivec4(mix(ret, vec4(-1), vec4(greaterThan(retDepth, vec4(d.y)))));
}

    #else
// find closest layer with depth larger than 'd'
// doesn't works correctly for back-faced triangles and need to correctly set depthOffset

int getLayer(ivec2 uv, float d) {
    int layerCount = textureSize(depth, 0).z;
    for (int l=0; l<layerCount; l++)
    if (d <= texelFetch(depth, ivec3(uv, l), 0).x)
    return l;
    return -1;
}

ivec4 getLayers(vec2 uv, float d) {
    int layerCount = textureSize(depth, 0).z;
    vec4 ret = vec4(-1);
    for (int l=0; l<layerCount; l++) {
        bvec4 mask = equal(ret,vec4(-1));
        if (!any(mask)) break;
        bvec4 leq = lessThanEqual(vec4(d), depthGather(uv, l));
        ret = mix(ret, vec4(l), vec4(leq) * vec4(mask));
    }
    return ivec4(ret);
}

#endif

// ------------------- color -------------------

#ifdef EXTENDED_BLOCKS
vec4 computeColor(ivec2 uv, int l) {
    if (l < 0) return vec4(0);
    ivec2 block = getBlock(uv);
    vec4 color = vec4(0);
    for (int i=0, I=0; i<=l && I <= block.y + fullLayersCount; I++, i++) {
        if (I < fullLayersCount) color = getColorFullLayers(uv, I);
        else {
            int b = getBlockIndex(uv, I - fullLayersCount, block);
            if (b < 0) return vec4(0);
            color = getColorBlocks(uv, b);
        }
        if (color.w == 0) i--;
    }
    return vec4(color.rgb, 1);
}
#else
vec4 computeColor(ivec2 uv, int l) {
    if (l < 0) return vec4(0);
    if (l < fullLayersCount) return vec4(getColorFullLayers(uv, l).rgb, 1);
    int i = getBlockIndex(uv, l - fullLayersCount, getBlock(uv));
    if (i < 0) return vec4(0);
    return vec4(getColorBlocks(uv, i).rgb, 1);
}
#endif

vec4 computeColorSafe(ivec2 uv, int l) {
    return computeColor(max(ivec2(0), min(layerSize - 1, uv)), l);
}

#endif // REMOTE_TEXTURE

void main() {
    #ifdef LOCAL_TEXTURE
    vec4 localTexSample = texture(localTex, localTexUV);
    #ifdef LOCAL_TEXTURE_ALPHA_MASK
    if (localTexSample.w < 0.5) discard; // TODO configurable threshold
    #endif
    #endif

    #ifndef DEPTH_ONLY
    color = vec4(1);

    #ifdef LOCAL_TEXTURE_COLOR
    color = localTexSample * vec4(localColor, 1);
    #ifdef LOCAL_TEXTURE_GAMMA
    color = pow(color, vec4(localGamma));
    #endif
    color.w = 0;
    #endif

    #ifdef REMOTE_TEXTURE

    #ifdef CUBEMAP
    vec3 absTexEyeCoord = abs(texEyeCoord);
    float maxAxis = max(absTexEyeCoord.x, max(absTexEyeCoord.y, absTexEyeCoord.z));
    int face = 0;
    vec4 texCoord;
    if (texEyeCoord.x == maxAxis)       { face = 2; texCoord = cubemapFaceTransform[2] * vec4(texEyeCoord, 1); }
    else if (texEyeCoord.x == -maxAxis) { face = 0; texCoord = cubemapFaceTransform[0] * vec4(texEyeCoord, 1); }
    else if (texEyeCoord.y ==  maxAxis) { face = 4; texCoord = cubemapFaceTransform[4] * vec4(texEyeCoord, 1); }
    else if (texEyeCoord.y == -maxAxis) { face = 5; texCoord = cubemapFaceTransform[5] * vec4(texEyeCoord, 1); }
    else if (texEyeCoord.z ==  maxAxis) { face = 3; texCoord = cubemapFaceTransform[3] * vec4(texEyeCoord, 1); }
    else                                { face = 1; texCoord = cubemapFaceTransform[1] * vec4(texEyeCoord, 1); }
    #endif

    vec3 tc = texCoord.xyz / texCoord.w;
    vec3 nc = tc * 0.5f + 0.5f;
    #ifdef CUBEMAP
    nc.x = (float(face) + nc.x) * (1.0f / 6.0f);
    #endif
    vec2 uvf = nc.xy * vec2(layerSize);
    ivec2 uv = ivec2(uvf);

    if (texCoord.w < 0 || clamp(uv, ivec2(0), layerSize-1) != uv)
        discard;

    // depth
    float slope = dot(texNormal, normalize(texEyeCoord));
    float slopeWeight = 1.0f - abs(slope);
    //float depthWeight = (1.0f - nc.z) * (1.0f - nc.z) * pixelSize; // constant (independent on distance)
    float depthWeight = (1.0f - nc.z) * pixelSize; // linearly increases with distance
    float offset = depthWeight * (depthOffset.x + slopeWeight * depthOffset.y);

    #if defined(CLOSEST_DEPTH) || defined(CLOSEST_DEPTH_RANGE)
    float D = nc.z + offset * sign(slope);
    #else
    float D = nc.z - offset;
    #endif

    #if defined(DEPTH_RANGE) || defined(CLOSEST_DEPTH_RANGE)
    float rangeOffset = depthWeight * (depthRange.x + slopeWeight * depthRange.y);
    #endif

    #ifdef CLOSEST_DEPTH_RANGE
    vec3 d = vec3(D, nc.z - rangeOffset, nc.z + rangeOffset); // center, min, max
    #elif defined(DEPTH_RANGE) && !defined(CLOSEST_DEPTH)
    vec2 d = vec2(D - rangeOffset, D + rangeOffset); // min, max
    #else
    float d = D;
    #endif

    // color
    #ifdef LINEAR_FILTER
    uv = ivec2(uvf - 0.5f);
    ivec4 ls = getLayers(nc.xy, d);
    vec4 c1 = computeColorSafe(uv + ivec2(0, 0), ls.w);
    vec4 c2 = computeColorSafe(uv + ivec2(1, 0), ls.z);
    vec4 c3 = computeColorSafe(uv + ivec2(0, 1), ls.x);
    vec4 c4 = computeColorSafe(uv + ivec2(1, 1), ls.y);
    vec2 w = fract(uvf - 0.5f);
    vec4 c12 = mix(c1, c2, w.x);
    vec4 c34 = mix(c3, c4, w.x);
    vec4 c = mix(c12, c34, w.y);
    #else
    vec4 c = computeColor(uv, getLayer(uv, d));
    #endif
    if (c.w == 0) discard;
    #ifdef LINEAR_FILTER
    c /= c.w;
    #endif

    #ifdef REMOTE_TEXTURE_COLOR_MAPPING
    color *= (pow(vec4(colorMapping.x), c) - 1) * colorMapping.y;
    #else
    color *= c;
    #endif

    #ifdef BLEND_EDGES
    color.w = min(1.0f, (1.0f - max(abs(tc.x), abs(tc.y))) * blendEdges);
    #else
    color.w = 1;
    #endif

    #endif // REMOTE_TEXTURE

    #endif //DEPTH_ONLY
}
