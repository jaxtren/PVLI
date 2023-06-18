#version 420

#ifndef DEPTH_ONLY
layout(location = 0) out vec4 color;
#endif
#ifdef PACK_NORMAL
flat in vec4 packedNormal;
layout(location = 1) out vec4 outPackedNormal;
#endif
#ifdef LOCAL_TEXTURE
in vec2 localTexUV;
#ifdef LOCAL_TEXTURE_GAMMA
uniform float localGamma;
#endif
uniform vec3 localColor;
layout(binding = 7) uniform sampler2D localTex;
#endif
#ifdef BLEND_EDGES
uniform float blendEdges;
float computeBlendWeight(vec2 tc) {
    return clamp((1.0f - max(abs(tc.x), abs(tc.y))) * blendEdges, 0, 1);
}
#endif

#ifdef REMOTE_TEXTURE
uniform vec2 depthOffset;
uniform vec2 depthRange;
uniform vec2 colorMapping;
uniform int blockSize;

vec4 getColorFullLayers(in sampler2D fullLayersTex, ivec2 offset, ivec2 layerSize, ivec2 coord, int layer) {
    return texelFetch(fullLayersTex, offset + coord + ivec2(0, layer * layerSize.y), 0);
}

vec4 getColorBlocks(in isampler2DArray blocks, in sampler2D blocksTex, ivec2 coord, int i) {
    int w = textureSize(blocksTex, 0).x / blockSize;
    return texelFetch(blocksTex, ivec2(i%w, i/w) * blockSize + coord % blockSize, 0);
}

#ifdef BINARY_BLOCKS
ivec2 getBlock(in isampler2DArray blocks, ivec2 coord) {
    return ivec2(0, texelFetch(blocks, ivec3(coord / blockSize, 0), 0).x);
}
int getBlockIndex(in isampler2DArray blocks, ivec2 coord, int layer, ivec2 block) {
    return layer >= block.y ? -1 : texelFetch(blocks, ivec3(coord / blockSize, layer + 1), 0).x;
}
#else
ivec2 getBlock(in isampler2DArray blocks, ivec2 coord) {
    return texelFetch(blocks, ivec3(coord / blockSize, 0), 0).xy;
}
int getBlockIndex(in isampler2DArray blocks, ivec2 coord, int layer, ivec2 block) {
    return layer >= block.y ? -1 : block.x + layer;
}
#endif

vec4 depthGather(in sampler2DArray depthLayers, vec2 uv, int l) {
    #ifdef DISABLE_TEXTURE_GATHER
    // workaround mainly for NVIDIA 2000 series, where textureGather doesn't always work correctly
    vec4 d;
    ivec2 iuv = ivec2(uv * vec2(textureSize(depthLayers, 0)) - 0.5f);
    d.w = texelFetch(depthLayers, ivec3(iuv + ivec2(0, 0), l), 0).x;
    d.z = texelFetch(depthLayers, ivec3(iuv + ivec2(1, 0), l), 0).x;
    d.x = texelFetch(depthLayers, ivec3(iuv + ivec2(0, 1), l), 0).x;
    d.y = texelFetch(depthLayers, ivec3(iuv + ivec2(1, 1), l), 0).x;
    return d;
    #else
    return textureGather(depthLayers, vec3(uv, l), 0);
    #endif
}

#ifdef CLOSEST_DEPTH_RANGE
// find layer with depth closest to 'd.x' within min (d.y) - max (d.z) range
// extended version of CLOSEST_DEPTH

int getLayer(in sampler2DArray depthLayers, ivec2 uv, vec3 d) {
    int layerCount = textureSize(depthLayers, 0).z;
    float minDiff = 1e24f;
    int best = -1;
    for (int l=0; l<layerCount; l++) {
        float dist = texelFetch(depthLayers, ivec3(uv, l), 0).x;
        float diff = abs(dist - d.x);
        if (dist > d.z || diff >= minDiff) break;
        if (dist >= d.y) {
            best = l;
            minDiff = diff;
        }
    }
    return best;
}

ivec4 getLayers(in sampler2DArray depthLayers, vec2 uv, vec3 d) {
    int layerCount = textureSize(depthLayers, 0).z;
    vec4 minDiff = vec4(1e24f);
    vec4 best = vec4(-1);
    for (int l=0; l<layerCount; l++) {
        vec4 dist = depthGather(depthLayers, uv, l);
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

int getLayer(in sampler2DArray depthLayers, ivec2 uv, float d) {
    int layerCount = textureSize(depthLayers, 0).z;
    float minDist = 1e24f;
    int best = -1;
    for (int l=0; l<layerCount; l++) {
        float dist = abs(texelFetch(depthLayers, ivec3(uv, l), 0).x - d);
        if (dist >= minDist) break;
        best = l;
        minDist = dist;
    }
    return best;
}

ivec4 getLayers(in sampler2DArray depthLayers, vec2 uv, float d) {
    int layerCount = textureSize(depthLayers, 0).z;
    vec4 minDist = vec4(1e24f);
    vec4 best = vec4(-1);
    for (int l=0; l<layerCount; l++) {
        vec4 dist = abs(depthGather(depthLayers, uv, l) - d);
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

int getLayer(in sampler2DArray depthLayers, ivec2 uv, vec2 d) {
    int layerCount = textureSize(depthLayers, 0).z;
    for (int l=0; l<layerCount; l++) {
        float D = texelFetch(depthLayers, ivec3(uv, l), 0).x;
        if (d.x <= D) return D < d.y ? l : -1;
    }
    return -1;
}

ivec4 getLayers(in sampler2DArray depthLayers, vec2 uv, vec2 d) {
    int layerCount = textureSize(depthLayers, 0).z;
    vec4 ret = vec4(-1);
    vec4 retDepth = vec4(0);
    for (int l=0; l<layerCount; l++) {
        bvec4 mask = equal(ret, vec4(-1));
        if (!any(mask)) break;
        vec4 D = depthGather(depthLayers, uv, l);
        vec4 mixMask = vec4(lessThanEqual(vec4(d.x), D)) * vec4(mask);
        ret = mix(ret, vec4(l), mixMask);
        retDepth = mix(retDepth, D, mixMask);
    }
    return ivec4(mix(ret, vec4(-1), vec4(greaterThan(retDepth, vec4(d.y)))));
}

#else
// find closest layer with depth larger than 'd'
// doesn't works correctly for back-faced triangles and need to correctly set depthOffset

int getLayer(in sampler2DArray depthLayers, ivec2 uv, float d) {
    int layerCount = textureSize(depthLayers, 0).z;
    for (int l=0; l<layerCount; l++)
        if (d <= texelFetch(depthLayers, ivec3(uv, l), 0).x)
            return l;
    return -1;
}

ivec4 getLayers(in sampler2DArray depthLayers, vec2 uv, float d) {
    int layerCount = textureSize(depthLayers, 0).z;
    vec4 ret = vec4(-1);
    for (int l=0; l<layerCount; l++) {
        bvec4 mask = equal(ret,vec4(-1));
        if (!any(mask)) break;
        bvec4 leq = lessThanEqual(vec4(d), depthGather(depthLayers, uv, l));
        ret = mix(ret, vec4(l), vec4(leq) * vec4(mask));
    }
    return ivec4(ret);
}

#endif

#ifdef EXTENDED_BLOCKS
vec4 samplePixel(in sampler2D fullLayersTex, in sampler2D blocksTex, in isampler2DArray blocksIndices,
                 ivec2 layerSize, ivec3 fullLayers_offset_counts, ivec2 uv, int l) {
    if (l < 0) return vec4(0);
    ivec2 block = getBlock(blocksIndices, uv);
    vec4 color = vec4(0);
    for (int i=0, I=0; i<=l && I <= block.y + fullLayers_offset_counts.z; I++, i++) {
        if (I < fullLayers_offset_counts.z) color = getColorFullLayers(fullLayersTex, fullLayers_offset_counts.xy, layerSize, uv, I);
        else {
            int b = getBlockIndex(blocksIndices, uv, I - fullLayers_offset_counts.z, block);
            if (b < 0) return vec4(0);
            color = getColorBlocks(blocksIndices, blocksTex, uv, b);
        }
        if (color.w == 0) i--;
    }
    return vec4(color.rgb, 1);
}
#else
vec4 samplePixel(in sampler2D fullLayersTex, in sampler2D blocksTex, in isampler2DArray blocksIndices,
                 ivec2 layerSize, ivec3 fullLayers_offset_counts, ivec2 uv, int l) {
    if (l < 0) return vec4(0);
    if (l < fullLayers_offset_counts.z) return vec4(getColorFullLayers(fullLayersTex, fullLayers_offset_counts.xy, layerSize, uv, l).rgb, 1);
    int i = getBlockIndex(blocksIndices, uv, l - fullLayers_offset_counts.z, getBlock(blocksIndices, uv));
    if (i < 0) return vec4(0);
    return vec4(getColorBlocks(blocksIndices, blocksTex, uv, i).rgb, 1);
}
#endif

vec4 sampleLayered(in sampler2D fullLayersTex, in sampler2D blocksTex, in isampler2DArray blocksIndices, in sampler2DArray depthLayers,
                   ivec2 layerSize, ivec3 fullLayers, float pixelSize, vec3 coord, float slope) {

    vec2 uvf = coord.xy * vec2(layerSize);
    ivec2 uv = ivec2(uvf);
    if (clamp(uv, ivec2(0), layerSize-1) != uv)
        return vec4(0);

    // depth
    float slopeWeight = 1.0f - abs(slope);
    //float depthWeight = (1.0f - coord.z) * (1.0f - coord.z) * pixelSize; // constant (independent on distance)
    float depthWeight = (1.0f - coord.z) * pixelSize; // linearly increases with distance
    float offset = depthWeight * (depthOffset.x + slopeWeight * depthOffset.y);

    #if defined(CLOSEST_DEPTH) || defined(CLOSEST_DEPTH_RANGE)
    float D = coord.z + offset * sign(slope);
    #else
    float D = coord.z - offset;
    #endif

    #if defined(DEPTH_RANGE) || defined(CLOSEST_DEPTH_RANGE)
    float rangeOffset = depthWeight * (depthRange.x + slopeWeight * depthRange.y);
    #endif

    #ifdef CLOSEST_DEPTH_RANGE
    vec3 d = vec3(D, coord.z - rangeOffset, coord.z + rangeOffset); // center, min, max
    #elif defined(DEPTH_RANGE) && !defined(CLOSEST_DEPTH)
    vec2 d = vec2(D - rangeOffset, D + rangeOffset); // min, max
    #else
    float d = D;
    #endif

    #ifdef LINEAR_FILTER
    uv = ivec2(uvf - 0.5f);
    ivec4 ls = getLayers(depthLayers, coord.xy, d);
    vec4 c1 = samplePixel(fullLayersTex, blocksTex, blocksIndices, layerSize, fullLayers, clamp(uv + ivec2(0, 0), ivec2(0), layerSize-1), ls.w);
    vec4 c2 = samplePixel(fullLayersTex, blocksTex, blocksIndices, layerSize, fullLayers, clamp(uv + ivec2(1, 0), ivec2(0), layerSize-1), ls.z);
    vec4 c3 = samplePixel(fullLayersTex, blocksTex, blocksIndices, layerSize, fullLayers, clamp(uv + ivec2(0, 1), ivec2(0), layerSize-1), ls.x);
    vec4 c4 = samplePixel(fullLayersTex, blocksTex, blocksIndices, layerSize, fullLayers, clamp(uv + ivec2(1, 1), ivec2(0), layerSize-1), ls.y);
    vec2 w = fract(uvf - 0.5f);
    vec4 c12 = mix(c1, c2, w.x);
    vec4 c34 = mix(c3, c4, w.x);
    vec4 c = mix(c12, c34, w.y);
    #else
    vec4 c = samplePixel(fullLayersTex, blocksTex, blocksIndices, layerSize, fullLayers, uv, getLayer(depthLayers, uv, d));
    #endif
    if (c.w == 0) return vec4(0);
    #ifdef LINEAR_FILTER
    c /= c.w;
    #endif

    #ifdef REMOTE_TEXTURE_COLOR_MAPPING
    c = (pow(vec4(colorMapping.x), c) - 1) * colorMapping.y;
    #endif

    return vec4(c.xyz, 1);
}

#if defined(CUBEMAP) || defined(COMBINED_VIEWS)
uniform mat4 cubemapFaceTransform[6];
vec4 sampleLayeredCubemap(in sampler2D fullLayersTex, in sampler2D blocksTex, in isampler2DArray blocksIndices, in sampler2DArray depthLayers,
                          ivec2 layerSize, ivec3 fullLayers, float pixelSize, vec3 dir, vec3 normal) {

    vec3 absDir = abs(dir);
    float maxAxis = max(absDir.x, max(absDir.y, absDir.z));
    int face = 0;
    vec4 texCoord;
    if (dir.x == maxAxis)       { face = 2; texCoord = cubemapFaceTransform[2] * vec4(dir, 1); }
    else if (dir.x == -maxAxis) { face = 0; texCoord = cubemapFaceTransform[0] * vec4(dir, 1); }
    else if (dir.y ==  maxAxis) { face = 4; texCoord = cubemapFaceTransform[4] * vec4(dir, 1); }
    else if (dir.y == -maxAxis) { face = 5; texCoord = cubemapFaceTransform[5] * vec4(dir, 1); }
    else if (dir.z ==  maxAxis) { face = 3; texCoord = cubemapFaceTransform[3] * vec4(dir, 1); }
    else                        { face = 1; texCoord = cubemapFaceTransform[1] * vec4(dir, 1); }

    vec3 nc = (texCoord.xyz / texCoord.w) * 0.5f + 0.5f;
    nc.x = (float(face) + nc.x) * (1.0f / 6.0f);
    return sampleLayered(fullLayersTex, blocksTex, blocksIndices, depthLayers,
        layerSize, fullLayers, pixelSize, nc, dot(normal, normalize(dir)));
}
#endif

#ifdef REMOTE_TEXTURE_DEFERRED
in vec2 coord;
layout(binding = 12) uniform sampler2D depth;
layout(binding = 13) uniform sampler2D normals;
uniform mat4 projInv;
uniform mat4 viewInv;
#ifndef CUBEMAP
uniform mat4 texMVP;
#endif
uniform mat4 texMV;
uniform vec3 texOrigin;
uniform mat3 texNormalTransform;
#ifdef COMBINED_VIEWS
uniform mat4 cubemapMV;
uniform vec3 cubemapOrigin;
uniform mat3 cubemapNormalTransform;
#endif

#endif

#ifndef REMOTE_TEXTURE_DEFERRED
#ifndef CUBEMAP
in vec4 texClipCoord;
#endif
#ifndef COMBINED_VIEWS_SHARED_ORIGIN
in vec3 texCoord;
flat in vec3 texNormal;
#endif
#endif // REMOTE_TEXTURE_DEFERRED

uniform ivec2 layerSize;
uniform float pixelSize;
uniform int fullLayersCount;
uniform ivec2 fullLayersOffset;
layout(binding = 0) uniform sampler2DArray depthLayers;
layout(binding = 1) uniform sampler2D fullLayersTex;
layout(binding = 2) uniform sampler2D blocksTex;
layout(binding = 3) uniform isampler2DArray blocksIndices;

#ifdef COMBINED_VIEWS
#ifndef REMOTE_TEXTURE_DEFERRED
in vec3 cubemapCoord;
flat in vec3 cubemapNormal;
#endif
uniform ivec2 cubemapLayerSize;
uniform float cubemapPixelSize;
uniform int cubemapFullLayersCount;
uniform ivec2 cubemapFullLayersOffset;
layout(binding = 8) uniform sampler2DArray cubemapDepthLayers;
layout(binding = 9) uniform sampler2D cubemapFullLayersTex;
layout(binding = 10) uniform sampler2D cubemapBlocksTex;
layout(binding = 11) uniform isampler2DArray cubemapBlocksIndices;
#endif

#endif // REMOTE_TEXTURE

vec3 rotX(vec3 v) { return vec3(v.x, v.z, -v.y); }

void main() {
    #ifdef REMOTE_TEXTURE_DEFERRED

    // depth -> global position
    float z = texture(depth, coord).x;
    if (z == 1) {
        color = vec4(0);
        return;
    }
    vec4 clipSpacePos = vec4(vec3(coord, z) * 2.0 - 1.0, 1.0);
    vec4 viewSpacePos = projInv * clipSpacePos;
    viewSpacePos /= viewSpacePos.w;
    vec4 pos = viewInv * viewSpacePos;

    // create variables that with rasterisation would be interpolated from vertices
    vec3 flatNormal = texture(normals, coord).xyz * 2.0f - 1.0f;

    #ifdef CUBEMAP
        #ifdef CUBEMAP_AA
        vec3 texCoord = rotX(vec3(pos) - texOrigin);
        vec3 texNormal = rotX(flatNormal);
        #else
        vec3 texCoord = vec3(texMV * pos);
        vec3 texNormal = texNormalTransform * flatNormal;
        #endif
    #else // not CUBEMAP
        vec4 texClipCoord = texMVP * pos;
        #ifndef COMBINED_VIEWS_SHARED_ORIGIN
        vec3 texCoord = vec3(pos) - texOrigin; // no need to rotate as it is used only for depth offset
        vec3 texNormal = flatNormal;
        #endif
    #endif // CUBEMAP

    #ifdef COMBINED_VIEWS
        #ifdef CUBEMAP_AA
        vec3 cubemapCoord = rotX(vec3(pos) - cubemapOrigin);
        vec3 cubemapNormal = rotX(flatNormal);
        #else
        vec3 cubemapCoord = vec3(cubemapMV * pos);
        vec3 cubemapNormal = cubemapNormalTransform * flatNormal;
        #endif
    #endif // COMBINED_VIEWS

    #endif // REMOTE_TEXTURE_DEFERRED

    #ifdef PACK_NORMAL
    outPackedNormal = packedNormal;
    #endif

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

    #ifdef COMBINED_VIEWS

    #ifdef COMBINED_VIEWS_SHARED_ORIGIN
    float slope = dot(cubemapNormal, normalize(cubemapCoord));
    #else
    float slope = dot(texNormal, normalize(texCoord));
    #endif

    vec4 c = vec4(0);
    #ifdef BLEND_EDGES
    float blendWeight = 0;
    if (texClipCoord.w > 0) {
        vec3 nc = texClipCoord.xyz / texClipCoord.w;
        blendWeight = computeBlendWeight(nc.xy);
        c = sampleLayered(fullLayersTex, blocksTex, blocksIndices, depthLayers, layerSize,
           ivec3(fullLayersOffset, fullLayersCount), pixelSize, nc * 0.5f + 0.5f, slope);
    }
    if (blendWeight < 1) {
        vec4 c2 = sampleLayeredCubemap(cubemapFullLayersTex, cubemapBlocksTex, cubemapBlocksIndices, cubemapDepthLayers, cubemapLayerSize,
            ivec3(cubemapFullLayersOffset, cubemapFullLayersCount), cubemapPixelSize, cubemapCoord, cubemapNormal);
        if (c2.w > 0) c = mix(c2, c, blendWeight);
    }
    #else
    if (texClipCoord.w > 0)
        c = sampleLayered(fullLayersTex, blocksTex, blocksIndices, depthLayers, layerSize,
            ivec3(fullLayersOffset, fullLayersCount), pixelSize,
            texClipCoord.xyz / texClipCoord.w * 0.5f + 0.5f, slope);
    if (c.w == 0)
        c = sampleLayeredCubemap(cubemapFullLayersTex, cubemapBlocksTex, cubemapBlocksIndices, cubemapDepthLayers, cubemapLayerSize,
            ivec3(cubemapFullLayersOffset, cubemapFullLayersCount), cubemapPixelSize, cubemapCoord, cubemapNormal);
    #endif
    if (c.w == 0) discard;
    color *= c;
    color.w = 1;

    #else // not COMBINED_VIEWS

    #ifdef CUBEMAP
    vec4 c = sampleLayeredCubemap(fullLayersTex, blocksTex, blocksIndices, depthLayers, layerSize,
        ivec3(fullLayersOffset, fullLayersCount), pixelSize, texCoord, texNormal);
    #else
    vec3 nc = texClipCoord.xyz / texClipCoord.w;
    vec4 c = sampleLayered(fullLayersTex, blocksTex, blocksIndices, depthLayers, layerSize,
        ivec3(fullLayersOffset, fullLayersCount), pixelSize, nc * 0.5f + 0.5f, dot(texNormal, normalize(texCoord)));
    #endif
    if (c.w == 0) discard;
    color *= c;

    #if defined(BLEND_EDGES) && !defined(CUBEMAP)
    color.w = computeBlendWeight(nc.xy);
    #else
    color.w = 1;
    #endif

    #endif // COMBINED_VIEWS

    #endif // REMOTE_TEXTURE

    #endif //DEPTH_ONLY
}
