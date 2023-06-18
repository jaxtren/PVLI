#version 420

// base
uniform ivec2 layerSize;

// ------------------- full layers -------------------

layout (binding = 0, rgba8i) uniform iimage2D fullLayersImage;
uniform int fullLayersCount;
uniform ivec2 fullLayersOffset;

ivec2 getFullLayersLoc(ivec2 coord, int layer) {
    return fullLayersOffset + coord + ivec2(0, layer * layerSize.y);
}

// ------------------- blocks -------------------

layout (binding = 1, rgba8i) uniform iimage2D blocksImage;
uniform isampler2DArray blocks;
uniform ivec3 blocksSize;
uniform ivec2 blocksImageSize;

ivec2 getBlocksLoc(ivec2 coord, int i) {
    int w = blocksImageSize.x / blocksSize.z;
    return ivec2(i%w, i/w) * blocksSize.z + coord % blocksSize.z;
}

#ifdef BINARY_BLOCKS
ivec2 getBlock(ivec2 coord) { return ivec2(0, texelFetch(blocks, ivec3(coord / blocksSize.z, 0), 0).x); }
int getBlockIndex(ivec2 coord, int layer, ivec2 block) {
    return layer >= block.y ? -1 : texelFetch(blocks, ivec3(coord / blocksSize.z, layer + 1), 0).x;
}
#else
ivec2 getBlock(ivec2 coord) { return texelFetch(blocks, ivec3(coord / blocksSize.z, 0), 0).xy; }
int getBlockIndex(ivec2 coord, int layer, ivec2 block) {
    return layer >= block.y ? -1 : block.x + layer;
}
#endif

// ------------------- relocate -------------------

void process(ivec2 coord) {
    ivec2 block = getBlock(coord);
    int maxLayers = block.y + fullLayersCount;
    for(int dst=0, src=0; dst<maxLayers; dst++) {
        ivec4 c = ivec4(0);
        while (c.w == 0 && src<maxLayers) {
            if (src < fullLayersCount) c = imageLoad(fullLayersImage, getFullLayersLoc(coord, src));
            else c = imageLoad(blocksImage, getBlocksLoc(coord, getBlockIndex(coord, src - fullLayersCount, block)));
            src++;
        }
        if (c.w == 0) break;
        if (dst >= fullLayersCount)
            imageStore(blocksImage, getBlocksLoc(coord, getBlockIndex(coord, dst - fullLayersCount, block)), c);
        #ifndef SKIP_FULL_LAYERS
        else imageStore(fullLayersImage, getFullLayersLoc(coord, dst), c);
        #endif
    }
}

in vec2 coord;
layout(location = 0) out vec4 color;

void main() {
    process(ivec2(coord));
    //discard;
}
