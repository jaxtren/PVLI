#include "LinkedDepthLV.h"
#include "graphic.h"
#include "prefixsum.h"
#include <cmath>
#include "dilate.h"
#include "debug.h"

using namespace std;
using namespace glm;

using Fragment = LinkedDepthLV::Fragment;

bool PackConfig::updateConfig(const Config& cfg) {
    return  cfg.get("FullLayers", fullLayers) |
            cfg.get("MaxLayers", maxLayers) |
            cfg.get("FragmentCountThreshold", fragmentCountThreshold) |
            cfg.get("LayerFirstBlockOrder", layerFirstBlockOrder) |
            cfg.get("RelocatePixels", relocatePixels) |
            cfg.get("TriangleLayerMask", triangleLayerMask);
}

void PackConfig::provideConfig(Config cfg) {
    cfg.set("FullLayers", fullLayers);
    cfg.set("MaxLayers", maxLayers);
    cfg.set("FragmentCountThreshold", fragmentCountThreshold);
    cfg.set("LayerFirstBlockOrder", layerFirstBlockOrder);
    cfg.set("RelocatePixels", relocatePixels);
    cfg.set("TriangleLayerMask", triangleLayerMask);
}

LinkedDepthLV::LinkedDepthLV(const Config& cfg, Scene* s) {
    scene = s;
    updateConfig(cfg);
    glGenVertexArrays(1, &vao);
    loadShaders();
}

void LinkedDepthLV::updateConfig(const Config& cfg) {
    cfg.get("CudaBlock", cudaBlock);
    cfg.get("CudaBlock2D", cudaBlock2D);
}

void LinkedDepthLV::provideConfig(Config cfg) {
    cfg.set("CudaBlock", cudaBlock);
    cfg.set("CudaBlock2D", cudaBlock2D);
}

LinkedDepthLV::~LinkedDepthLV() {
    glDeleteVertexArrays(1, &vao);
    freeFrame();
}

void LinkedDepthLV::allocFrame(ivec2 size, int fragmentCount) {
    if (size.x > maxFrameSize.x || size.y > maxFrameSize.y) {
        size = glm::max(maxFrameSize, size);
        maxFrameSize = size;

        //free
        if (headTex) {
            cuUnmap();
            cuEC(cudaGraphicsUnregisterResource(cuHeadRes));
            glDeleteTextures(1, &headTex);
            glDeleteFramebuffers(1, &fbo);
        }

        //head
        glGenTextures(1, &headTex);
        glBindTexture(GL_TEXTURE_2D, headTex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32I, maxFrameSize.x, maxFrameSize.y);
        DEBUG_GPU_RAM << "glTexStorage2D LV " << maxFrameSize.x * maxFrameSize.y * 4 << endl;
        glBindTexture(GL_TEXTURE_2D, 0);
        cuEC(cudaGraphicsGLRegisterImage(&cuHeadRes, headTex, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));

        //framebuffer
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, headTex, 0);
        glDrawBuffer(GL_COLOR_ATTACHMENT0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            cerr << "GL ERROR main FBO: " << glCheckFramebufferStatus(GL_FRAMEBUFFER) << endl;

        //clear
        int ones[4] = { -1, -1, -1, -1 };
        glClearBufferiv(GL_COLOR, 0, ones);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    //fragments
    if (fragmentCount > maxFragmentCount) {
        maxFragmentCount = fragmentCount;

        //free
        if (dataBuf) {
            cuUnmap();
            cuEC(cudaGraphicsUnregisterResource(cuDataRes));
            glDeleteBuffers(1, &dataBuf);
        }

        //alloc
        glGenBuffers(1, &dataBuf);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, dataBuf);
        int dataBufSize = sizeof(int) + sizeof(Fragment) * fragmentCount;
        glBufferData(GL_SHADER_STORAGE_BUFFER, dataBufSize, nullptr, GL_STREAM_COPY);
        DEBUG_GPU_RAM << "glBufferData LV " << dataBufSize << endl;
        cuEC(cudaGraphicsGLRegisterBuffer(&cuDataRes, dataBuf, cudaGraphicsMapFlagsNone));
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }
}

void LinkedDepthLV::freeFrame() {
    cuUnmap();
    if (headTex) {
        cuEC(cudaGraphicsUnregisterResource(cuHeadRes));
        glDeleteTextures(1, &headTex);
        glDeleteFramebuffers(1, &fbo);
    }
    if (dataBuf) {
        cuEC(cudaGraphicsUnregisterResource(cuDataRes));
        glDeleteBuffers(1, &dataBuf);
    }

    fbo = headTex = dataBuf = 0;
    maxFrameSize = {0, 0};
    maxFragmentCount = 0;
}

bool LinkedDepthLV::loadShaders() {
    return program.load("lv");
}

static inline __device__ void limitLayerCount(LinkedDepthLV::PackOutput& output, int& layerCount, int& maxLayer, bool first) {
    layerCount = min(layerCount, output.config.maxLayers);
    atomicMax(&maxLayer, layerCount);
    __syncthreads();
    if (output.config.fragmentCountThreshold > 0) {
        __shared__ int fragmentCount;
        for (int i=output.config.fullLayers; i<maxLayer; i++) {
            if (first) fragmentCount = 0;
            __syncthreads();
            if (i < layerCount) atomicAdd(&fragmentCount, 1);
            __syncthreads();
            if (first && fragmentCount < output.config.fragmentCountThreshold)
                maxLayer = i;
            __syncthreads();
        }
    }
}
static __global__ void countBlocksKernel(cudaSurfaceObject_t headTex, Fragment* fragments,
                                         int* globalBlockCount, int* globalLayerCount,
                                         ivec2 offset, LinkedDepthLV::PackOutput output) {
    __shared__ int maxLayer;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= output.layerSize.x || iy >= output.layerSize.y) return;

    bool first = threadIdx.x == 0 && threadIdx.y == 0;
    if (first) maxLayer = output.config.fullLayers;
    __syncthreads();

    int layerCount = 0, it;
    surf2Dread(&it, headTex, (ix + offset.x) * sizeof(int), iy + offset.y);
    while (it >= 0) {
        it = fragments[it].next;
        layerCount++;
    }

    limitLayerCount(output, layerCount, maxLayer, first);

    if (first) {
        output.blocks[blockIdx.y * output.blocksSize.x + blockIdx.x] = maxLayer - output.config.fullLayers;
        atomicAdd(globalBlockCount, maxLayer - output.config.fullLayers);
        atomicMax(globalLayerCount, maxLayer);
    }
}

__device__ void sortFragments(Fragment* layers, int count) {
    for (int j = count - 1; j > 0; j--)
        for (int i = 0; i < j; i++) {
            auto& l1 = layers[i], &l2 = layers[i+1];
            if (l1.depth == l2.depth ? (l1.instID == l2.instID ? l1.primID < l2.primID : l1.instID < l2.instID) : l1.depth > l2.depth) {
                auto l = l1;
                l1 = l2;
                l2 = l;
            }
        }
}

__device__ int copyFragmentsToLocal(cudaSurfaceObject_t headTex, Fragment* fragments, Fragment* layers, int x, int y) {
    int it, layerCount = 0;
    surf2Dread(&it, headTex, x * sizeof(int), y);
    while (it >= 0) {
        auto& f = fragments[it];
        layers[layerCount] = f;
        it = f.next;
        layerCount++;
    }
    return layerCount;
}

__device__ void copyFragmentsToGlobal(cudaSurfaceObject_t headTex, Fragment* fragments, Fragment* layers, int x, int y) {
    int it; //TODO reorder indices instead of copy
    surf2Dread(&it, headTex, x * sizeof(int), y);
    for (int i = 0; it >= 0; i++) {
        auto& f = fragments[it];
        it = f.next;
        f = layers[i];
        f.next = it; //keep next
    }
}

static __global__ void relocateKernel(cudaSurfaceObject_t headTex, Fragment* fragments,
                                      int* globalBlockCount, int* globalLayerCount,
                                      ivec2 offset, bool sort, LinkedDepthLV::PackOutput output) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int x = threadIdx.x, y = threadIdx.y, blockSize = blockDim.x;
    bool first = threadIdx.x == 0 && threadIdx.y == 0;
    bool inside = ix < output.layerSize.x && iy < output.layerSize.y;
    ix += offset.x;
    iy += offset.y;

    //shared data
    __shared__ int blockCount; //for blocks
    extern __shared__ float buffer[];
    float* current = buffer;
    float* next = &buffer[(blockSize+2)*(blockSize+2)];
    float maxDepth = 1e20; //TODO configurable?

    //local data
    const int MAX_LAYERS = 30;
    Fragment layers[MAX_LAYERS];

    if(first) blockCount = output.config.fullLayers;
    __syncthreads();

    int layerCount = 0;
    if(inside) {
        layerCount = copyFragmentsToLocal(headTex, fragments, layers, ix, iy);
        if (sort) sortFragments(layers, layerCount);
    }
    limitLayerCount(output, layerCount, blockCount, first);
    for(int i=0; i<layerCount; i++)
        layers[i].skip = 0;

    #define LOC(X,Y) (((Y)+1) * (blockSize+2) + (X) + 1)

    //init borders
    if(x == 0) current[LOC(-1, y)] = next[LOC(-1, y)] = maxDepth;
    if(y == 0) current[LOC(x, -1)] = next[LOC(x, -1)] = maxDepth;
    if(x == blockSize-1) current[LOC(blockSize, y)] = next[LOC(blockSize, y)] = maxDepth;
    if(y == blockSize-1) current[LOC(x, blockSize)] = next[LOC(x, blockSize)] = maxDepth;
    __syncthreads();

    const int dX[4] = { 1,-1, 0, 0};
    const int dY[4] = { 0, 0, 1,-1};
    int leftMoves = blockCount - layerCount;
    for(int layer = 0, l = 0; layer < blockCount; layer++, l++) {
        u8vec4 c = l < layerCount ? layers[l].color : u8vec4(0);
        float d = current[LOC(x,y)] = l < layerCount ? layers[l].depth : maxDepth;
        next[LOC(x,y)] = l+1 < layerCount ? layers[l+1].depth : maxDepth;
        __syncthreads();

        bool moved = false;
        bool canMove = c != u8vec4(0) && leftMoves > 0;
        for(int I=0; I<blockSize*2; I++) {
            int n = 0;
            float D = 0;
            if (canMove) {
                for (int i = 0; i < 4; i++) {
                    float f1 = current[LOC(x + dX[i], y + dY[i])];
                    float f2 = next[LOC(x + dX[i], y + dY[i])];
                    float f = 0.5;
                    if (d > (f * f1 + (1.0 - f) * f2)) {
                        D += f1;
                        n++;
                    }
                }
            }
            __syncthreads();

            if (n > 0) {
                moved = true;
                canMove = false;
                layers[l].skip++;
                next[LOC(x, y)] = d;
                current[LOC(x, y)] = D / n;
            }
            __syncthreads();
        }

        if(moved) {
            l--;
            leftMoves--;
        }
        __syncthreads();
    }
    #undef LOC

    if(inside)
        copyFragmentsToGlobal(headTex, fragments, layers, ix, iy);

    __syncthreads();

    if (first) {
        output.blocks[blockIdx.y * output.blocksSize.x + blockIdx.x] = blockCount - output.config.fullLayers;
        atomicAdd(globalBlockCount, blockCount - output.config.fullLayers);
        atomicMax(globalLayerCount, blockCount);
    }
}

inline __device__ int blockPixel(const LinkedDepthLV::Blocks& blocks, int i) {
    if (blocks.occupied.size > 0 && (i < 0 || i >= blocks.occupied.size)) printf("ERROR blockPixel\n");
    int blocksWidth = blocks.size.x / blocks.blockSize;
    int x = (i % blocksWidth) * blockDim.x + threadIdx.x;
    int y = (i / blocksWidth) * blockDim.y + threadIdx.y;
    return y * blocksWidth * blockDim.x + x;
}

inline __device__ int blockIndex(const LinkedDepthLV::Blocks& blocks,
                                 const LinkedDepthLV::PackOutput& output,
                                 int* prefixes, int l) {
    auto bs = output.blocksSize;
    l -= output.config.fullLayers;
    if (output.config.layerFirstBlockOrder || blocks.track.use)
        return output.tileOffset + prefixes[blockIdx.y * bs.x + blockIdx.x + bs.x * bs.y * l];
    else return output.tileOffset + prefixes[blockIdx.y * bs.x + blockIdx.x] + l;
}

inline __device__ int blockPixel(const LinkedDepthLV::Blocks& blocks,
                                 const LinkedDepthLV::PackOutput& output,
                                 int* prefixes, int l) {
    return blockPixel(blocks, blockIndex(blocks, output, prefixes, l));
}

static __global__ void packKernel(cudaSurfaceObject_t headTex, Fragment* fragments, ivec2 srcOffset,
                                  LinkedDepthLV::FullLayers full, LinkedDepthLV::PackOutput output, LinkedDepthLV::Blocks blocks,
                                  int* blocksPrefixes, int triangleCount, int** triangleLayerMap, bool sort) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // clean
    int tileCount = output.blocks[blockIdx.y * output.blocksSize.x + blockIdx.x];
    for (int l=0; l<tileCount; l++) {
        int i = blockPixel(blocks, output, blocksPrefixes, output.config.fullLayers +  l);
        blocks.color[i] = {0, 0, 0};
        blocks.mask[i] = 2;
    }

    if (ix >= output.layerSize.x || iy >= output.layerSize.y) return;

    const int MAX_LAYERS = 30; // TODO set correctly
    Fragment layers[MAX_LAYERS];
    int fragmentCount = copyFragmentsToLocal(headTex, fragments, layers, ix + srcOffset.x, iy + srcOffset.y);
    fragmentCount = min(fragmentCount, tileCount + output.config.fullLayers);
    if (sort) sortFragments(layers, fragmentCount);
    __syncthreads();

    //full layers
    int fullBaseIndex = (output.layerOffset.y + iy) * full.size.x + (output.layerOffset.x + ix);
    int fullLayerStep = output.layerSize.y * full.size.x;
    for (int l = 0; l < min(fragmentCount, output.config.fullLayers); l++)
        full.color[fullBaseIndex + l * fullLayerStep] = layers[l].getColor();

    // pack
    if (output.config.relocatePixels) {
        int ol = 0;
        for (int l = 0; l < fragmentCount; l++, ol++) {

            // mask - 0
            for (int ol2 = ol + layers[l].skip; ol < ol2; ol++) {
                if (ol < output.config.fullLayers) full.mask[fullBaseIndex + ol * fullLayerStep] = 0;
                else blocks.mask[blockPixel(blocks, output, blocksPrefixes, ol)] = 0;
            }

            // color, mask - 1
            if (ol < output.config.fullLayers) {
                int idx = fullBaseIndex + ol * fullLayerStep;
                full.color[idx] = layers[l].getColor();
                full.mask[idx] = 1;
            } else {
                int idx = blockPixel(blocks, output, blocksPrefixes, ol);
                blocks.color[idx] = layers[l].getColor();
                blocks.mask[idx] = 1;
            }
        }

    } else {

        for (int l = 0; l < fragmentCount; l++) {
            if (l < output.config.fullLayers) full.color[fullBaseIndex + l * fullLayerStep] = layers[l].getColor();
            else blocks.color[blockPixel(blocks, output, blocksPrefixes, l)] = layers[l].getColor();
        }
    }

    // triangle-layer mask
    if (output.triangleLayerMask && triangleLayerMap)
        for (int l = 0; l < fragmentCount; l++) {
            auto& f = layers[l];
            if (f.instID >= 0 && f.primID >= 0) {
                int offset = triangleLayerMap[f.instID][f.primID];
                if (offset >= 0)
                    output.triangleLayerMask[l * triangleCount + offset] = 1;
            }
        }
}

static __global__ void markUsedBlocks(int* blocks, int* binBlocks, ivec2 blocksSize, int layerCount) {
    int x = blockIdx.x, y = blockIdx.y;
    int count = blocks[y * blocksSize.x + x];
    for(int l=0; l<layerCount; l++)
        binBlocks[l * blocksSize.x * blocksSize.y + y * blocksSize.x + x] = l < count;
}

static __global__ void copyExtFragmentsKernel(cudaSurfaceObject_t headTex, Fragment* fragments, LinkedFragment* extFragments,
                                              ivec2 size, int count, float toneMapping, float expToneMapping, float igamma,
                                              vec2 colorMapping, bool fullFirstLayer) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < count) {
        auto& src = extFragments[i];
        auto& dst = fragments[i];
        if(i < size.x * size.y)
            surf2Dwrite(fullFirstLayer || (src.instID >= 0 && src.primID >= 0) ? i : -1,
                        headTex, i%size.x * sizeof(int), i/size.x);

        auto color = src.color;
        if (toneMapping > 0)
            color = color / (color + toneMapping); // Reinhard tone mapping
        if (expToneMapping > 0)
            color = vec4(1) - exp(-color * expToneMapping);
        color = glm::pow(color, vec4(igamma));
        if (colorMapping.x > 0) color = glm::log(color * colorMapping.y + 1.0f) * colorMapping.x;
        dst.setColor(min(vec4(255), color * 255.0f));
        dst.next = src.next;
        dst.depth = src.depth;
        dst.skip = 0;
        dst.instID = src.instID;
        dst.primID = src.primID;
    }
}

void LinkedDepthLV::cuMap() {
    if(!headTex || cuCount) return; // not allocated or mapped
    timerCUDA("Map");

    // map head texture
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    cuEC(cudaGraphicsMapResources(1, &cuHeadRes));
    cuEC(cudaGraphicsSubResourceGetMappedArray(&resDesc.res.array.array, cuHeadRes, 0, 0));
    cuEC(cudaCreateSurfaceObject(&cuHeadTex, &resDesc));

    // map data buffer
    cuEC(cudaGraphicsMapResources(1, &cuDataRes));
    size_t tsize = 0;
    cuEC(cudaGraphicsResourceGetMappedPointer((void**)&cuCount, &tsize, cuDataRes));
    cuFragments = reinterpret_cast<Fragment*>(cuCount+1);
    timerCUDA();
}

void LinkedDepthLV::cuUnmap() {
    timerCUDA("Unmap");
    if (cuHeadTex) {
        cuEC(cudaDestroySurfaceObject(cuHeadTex));
        cuEC(cudaGraphicsUnmapResources(1, &cuHeadRes));
        cuHeadTex = 0;
    }
    if (cuCount) {
        cuEC(cudaGraphicsUnmapResources(1, &cuDataRes));
        cuCount = nullptr;
        cuFragments = nullptr;
    }
    timerCUDA();
}

void LinkedDepthLV::render(const ivec2& size, const mat4& projection, const mat4& view,
                           const std::vector<Scene::Instance>& instances, GLuint mask, int fragmentCount) {
    if(!program) return;
    allocFrame(size, fragmentCount);
    cuUnmap();

    //prepare
    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, size.x, size.y);
    program.use();
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, dataBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mask);
    glBindImageTexture(0, headTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32I);
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    program.uniform("view", inverse(view));
    program.uniform("projection", projection);

    //clear
    timerGL("Render.Clear");
    int count = 0;
    int ones[4] = { -1, -1, -1, -1 };
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glClearBufferiv(GL_COLOR, 0, ones);
    glDrawBuffers(0, nullptr);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, dataBuf);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(int), &count);

    timerGL("Render.Raster");
    int offset = 0;
    int instID = 0;
    for (auto& instance : instances) {
        if (!instance.mesh->vertexCount) continue;
        //TODO add frustum culling
        glBindBuffer(GL_ARRAY_BUFFER, instance.mesh->glVertices);
        gl::vertexAttribPointer(0, 3, GL_FLOAT, sizeof(Mesh::VertexData), 0);
        gl::vertexAttribPointer(1, 3, GL_FLOAT, sizeof(Mesh::VertexData), sizeof(vec3));
        gl::vertexAttribPointer(2, 2, GL_FLOAT, sizeof(Mesh::VertexData), sizeof(vec3) * 2);
        program.uniform("model", instance.transform);
        program.uniform("maskOffset", offset);
        program.uniform("instID", instID++);
        int primOffset = 0;
        for (auto& prim : instance.mesh->primitives) {
            glBindTexture(GL_TEXTURE_2D, prim.material->texture);
            glDrawArrays(GL_TRIANGLES, primOffset, prim.size);
            primOffset += prim.size;
        }
        offset += instance.mesh->vertexCount / 3;
    }
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    timerGL("Render.Finish");
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glUseProgram(0);
    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    timerGL();

    // stats
    output.fragmentCount = 0; // not supported
}

void LinkedDepthLV::setData(const ivec2& size, LinkedFragment* cuExtFragments,int extFragmentCount,
                            float toneMapping, float expToneMapping, float gamma,
                            glm::vec2 colorMapping, bool fullFirstLayer) {
    allocFrame(size, extFragmentCount);
    cuMap();
    timerCUDA("External Data");
    cuCheck;
    copyExtFragmentsKernel <<< extFragmentCount / cudaBlock + 1, cudaBlock >>>
            (cuHeadTex, cuFragments, cuExtFragments, size, extFragmentCount, toneMapping, expToneMapping,
             1.0f / gamma, {1.0f / log(colorMapping.x), colorMapping.y}, fullFirstLayer);
    cuCheck;
    timerCUDA();

    // stats
    output.fragmentCount = extFragmentCount;
}

int LinkedDepthLV::Blocks::extend(int count) {
    if (track.use) tileCount = tileCapacity();
    if (count == 0) return tileCount;
    int ret = tileCount;
    tileCount += count;
    int y = ((tileCount - 1) / (size.x / blockSize) + 1) * blockSize;
    if (y > size.y) {
        size.y = y;
        color.resize(size.x * size.y);
        mask.resize(size.x * size.y);
        if (track.use) occupied.resize(tileCapacity(), true);
    }
    return ret;
}

static __global__ void clearBlocksLinearKernel(LinkedDepthLV::Blocks blocks, int offset) {
    int i = blockPixel(blocks, offset + blockIdx.x);
    blocks.color[i] = {0, 0, 0};
    if (blocks.mask.size > 0)
        blocks.mask[i] = 2;
}

static __global__ void clearUnoccupiedBlocksKernel(LinkedDepthLV::Blocks blocks) {
    auto occupied = blocks.occupied[blockIdx.x + blockIdx.y * (blocks.size.x / blockDim.x)];
    if (occupied == 0) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int i = x + y * blocks.size.x;
        blocks.color[i] = {blocks.track.debug ? 200 : 0, 0, 0};
        if (blocks.mask.size > 0)
            blocks.mask[i] = 2;
    } else if (blocks.track.debug) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int i = x + y * blocks.size.x;
        if (occupied == 1)
            blocks.color[i].z = 200;
        //else blocks.color[i].y = min(occupied * 50, 255);
    }
}

void LinkedDepthLV::Blocks::clearEmptyBlocks() {
    if (track.use) {
        cuCheck;
        clearUnoccupiedBlocksKernel <<< dim3(size.x / blockSize, size.y / blockSize), dim3(blockSize, blockSize) >>> (*this);
        cuCheck;
    } else {
        int allTileCount = tileCapacity();
        if (allTileCount > tileCount) {
            cuCheck;
            clearBlocksLinearKernel <<< allTileCount - tileCount, dim3(blockSize, blockSize) >>> (*this, tileCount);
            cuCheck;
        }
    }
}

void LinkedDepthLV::Blocks::init(int width, int block, bool track) {
    this->track.use = track;
    blockSize = block;
    size = {((width - 1) / block + 1) * block, 0};
    tileCount = 0;
    color.free();
    mask.free();
    occupied.free();
    hasMask = false;
}

bool LinkedDepthLV::Blocks::Track::updateConfig(const Config& cfg) {
    return  cfg.get("ReservationAttempts", reservationAttempts) |
            cfg.get("Mode", mode) |
            cfg.get("Cycle", cycle) |
            cfg.get("ReprojMaxDistFactor", reprojMaxDistFactor) |
            cfg.get("Debug", debug);
}

void LinkedDepthLV::Blocks::Track::provideConfig(Config cfg) {
    cfg.set("ReservationAttempts", reservationAttempts);
    cfg.set("Mode", mode);
    cfg.set("Cycle", cycle);
    cfg.set("ReprojMaxDistFactor", reprojMaxDistFactor);
    cfg.set("Debug", debug);
}

void LinkedDepthLV::FullLayers::init(const ivec2& s) {
    size = s;
    color.alloc(size.x * size.y);
    mask.alloc(size.x * size.y);
    color.clear();
    mask.clear(2);
    hasMask = false;
}


// for ViewContext::indices
static const int BLOCK_UNUSED = -1; // block is unused at all and doesn't require a location in blocks texture
static const int BLOCK_NOT_ALLOCATED = -2; // block is used but doesn't have assigned location in blocks texture yet
static const unsigned long long NO_REZERVATION = ~0ull;

static __global__ void prepareBlocksForReprojectionKernel(cudaSurfaceObject_t headTex, Fragment* fragments,
                                                          ivec2 layerSize, int fullLayers, int mode,
                                                          LinkedDepthLV::ViewContext ctx) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= layerSize.x || iy >= layerSize.y) return;
    bool first = threadIdx.x == 0 && threadIdx.y == 0;

    // shared variables
    __shared__ float posX, posY, posZ;
    __shared__ int count;
    if (first) {
        posX = 0;
        posY = 0;
        posZ = 0;
        count = 0;
    }

    int it;
    surf2Dread(&it, headTex, ix * sizeof(int), iy);

    // skip full layers
    for (int i=0; i<fullLayers && it >= 0; i++) it = fragments[it].next;

    // ray direction
    ivec2 extendedLayerSize = ivec2(ctx.size) * ivec2(blockDim.x, blockDim.y);
    vec4 v = ctx.viewToGlobal * vec4(vec2(ix, iy) / vec2(extendedLayerSize) * 2.0f - 1.0f, 0, 1);
    vec3 rayDir = normalize(vec3(v) * (1.0f / v.w) - ctx.globalViewOrigin);

    // for mode > 0
    __shared__ unsigned int depth;
    vec3 centerRayDir;
    if (first && mode > 0) {
        depth = mode == 1 ? ~0u : 0u;
        vec2 centerPixel = vec2(ix, iy) + vec2(blockDim.x, blockDim.y) / 2.0f;
        vec4 v = ctx.viewToGlobal * vec4(centerPixel / vec2(extendedLayerSize) * 2.0f - 1.0f, 0, 1);
        centerRayDir = normalize(vec3(v) * (1.0f / v.w) - ctx.globalViewOrigin);
    }

    // average positions of all fragments, excluding skipped fragments
    int skip = it >= 0 ? fragments[it].skip : 0;
    for (int l=0; l<ctx.size.z; l++) {
        __syncthreads();

        if (skip > 0) skip--;
        else if (it >= 0) {
            atomicAdd(&count, 1);
            if (mode == 0) { // average position
                vec3 pos = ctx.globalViewOrigin + rayDir * fragments[it].depth; // depth is linear distance
                atomicAdd(&posX, pos.x);
                atomicAdd(&posY, pos.y);
                atomicAdd(&posZ, pos.z);
            } else if (mode == 1) // min depth
                atomicMin(&depth, __float_as_uint(fragments[it].depth));
            else if (mode == 2) // max depth
                atomicMax(&depth, __float_as_uint(fragments[it].depth));
            else atomicAdd(&posZ, fragments[it].depth); // average depth
            it = fragments[it].next;
            skip = it >= 0 ? fragments[it].skip : 0;
        }

        __syncthreads();
        if (first) {
            int i = blockIdx.x + blockIdx.y * ctx.size.x + l * ctx.size.x * ctx.size.y ;
            if (count > 0) {
                if (mode == 0) // average position
                    ctx.pos[i] = vec3(posX, posY, posZ) * (1.0f / (float) count);
                else { // min/max/average depth
                    float d = mode > 2 ? posZ / count : __uint_as_float(depth);
                    ctx.pos[i] = ctx.globalViewOrigin + centerRayDir * d;
                    depth = mode == 1 ? ~0u : 0u;
                }
                ctx.indices[i] = BLOCK_NOT_ALLOCATED;
                posX = 0;
                posY = 0;
                posZ = 0;
                count = 0;
            } else {
                ctx.pos[i] = vec3(0);
                ctx.indices[i] = BLOCK_UNUSED;
            }
            ctx.prevPos[i] = vec3(0); // debug
        }
    }
}

static __global__ void reprojectAndReserveBlocksKernel(LinkedDepthLV::ViewContext ctx, LinkedDepthLV::ViewContext prevCtx,
                                                       unsigned long long* reservations, float maxDistFactor) {
    int blockIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIndex >= ctx.indices.size || ctx.indices[blockIndex] != BLOCK_NOT_ALLOCATED) return;

    // reproject current position to previous blocks
    vec3 pos = ctx.pos[blockIndex];
    vec4 v = prevCtx.globalToView * vec4(pos, 1);
    ivec2 loc = ivec2((vec2(v) * (0.5f / v.w) + 0.5f) * vec2(prevCtx.size) - 0.5f);
    float maxDistance = distance(pos, ctx.globalViewOrigin) * maxDistFactor;

    // iterate over all layers around reprojected location to find best candidate
    float bestDistance = 1e20f;
    int bestPrevBlockIdx = -1;
    for (int oy = 0; oy < 2; oy++) {
        int y = loc.y + oy;
        if (y < 0 || y >= ctx.size.y) continue;
        for (int ox = 0; ox < 2; ox++) {
            int x = loc.x + ox;
            if (x < 0 || x >= ctx.size.x) continue;
            for (int l = 0; l < prevCtx.size.z; l++) {
                int prevBlockIdx = x + y * prevCtx.size.x + l * prevCtx.size.x * prevCtx.size.y;
                int texBlockIdx = prevCtx.indices[prevBlockIdx];
                if (texBlockIdx == BLOCK_UNUSED) break;
                if (texBlockIdx == BLOCK_NOT_ALLOCATED) continue;
                float dist = distance(pos, prevCtx.pos[prevBlockIdx]);
                if (dist < bestDistance && (maxDistance <= 0 || dist < maxDistance)) {
                    bestDistance = dist;
                    bestPrevBlockIdx = prevBlockIdx;
                }
            }
        }
    }

    if (bestPrevBlockIdx >= 0) {
        // store distance and blockIndex atomically to 64 bit value in a way, that a smaller distance is prioritized
        // high 32 bits: float distance reinterpreted to unsigned int, ordering should be preserved for positive floats
        // low 32 bits: block index
        auto reservation = (unsigned long long)blockIndex;
        *(reinterpret_cast<float*>(&reservation) + 1) = bestDistance;
        atomicMin(&reservations[bestPrevBlockIdx], reservation);
    }
}

static __global__ void assignReprojectedBlocksAndFreeUnusedKernel(
        LinkedDepthLV::ViewContext ctx, LinkedDepthLV::ViewContext prevCtx,
        unsigned long long* reservations, LinkedDepthLV::Blocks blocks, bool freeUnused) {
    int prevCtxIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (prevCtxIdx >= prevCtx.indices.size) return;
    int blockTexIdx = prevCtx.indices[prevCtxIdx];
    if (blockTexIdx >= (int)blocks.occupied.size)
        printf("ERROR assignReprojectedBlocksAndFreeUnusedKernel A %d, %d\n", blockTexIdx, (int)blocks.occupied.size);
    auto reservation = reservations ? reservations[prevCtxIdx] : NO_REZERVATION;
    if (reservation != NO_REZERVATION) {
        int dstCtxIdx = *reinterpret_cast<int*>(&reservation);
        if (blockTexIdx < 0)
            printf("ERROR assignReprojectedBlocksAndFreeUnusedKernel B %d\n", blockTexIdx);
        if (dstCtxIdx >= (int)ctx.indices.size)
            printf("ERROR assignReprojectedBlocksAndFreeUnusedKernel C %d\n", dstCtxIdx);
        ctx.indices[dstCtxIdx] = blockTexIdx;
        ctx.prevPos[dstCtxIdx] = prevCtx.pos[prevCtxIdx]; // debug
        reservations[prevCtxIdx] = NO_REZERVATION;
        if (blocks.track.debug)
            blocks.occupied[blockTexIdx] = min(blocks.occupied[blockTexIdx] + 1, 255);
        else blocks.occupied[blockTexIdx] = 1;
        prevCtx.indices[prevCtxIdx] = -1;
    } else if (freeUnused && blockTexIdx >= 0) {
        blocks.occupied[blockTexIdx] = 0;
        prevCtx.indices[prevCtxIdx] = -1;
    }
}

static __global__ void markUnallocatedBlocksKernel(LinkedDepthLV::ViewContext ctx, int* mark) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ctx.indices.size)
        mark[i] = ctx.indices[i] == BLOCK_NOT_ALLOCATED;
}

static __global__ void markFreeBlocksKernel(int count, const char* occupied, int* dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = occupied[i] == 0;
}

static __global__ void compactFreeBlocksKernel(int count, const char* occupied, const int* prefixes, int* compacted,
                                               int probe, int* probeResult) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (occupied[i] == 0) {
        if (prefixes[i] < 0 || prefixes[i] >= count)
            printf("ERROR compactFreeBlocksKernel\n");
        compacted[prefixes[i]] = i;
    }
    if (probeResult && probe == i)
        *probeResult = prefixes[i];
}

static __global__ void assignUnallocatedBlocksKernel( LinkedDepthLV::ViewContext ctx, LinkedDepthLV::Blocks blocks,
                                                      const int* unallocatedBlocksPrefixes,
                                                      const int* compactedFreeBlocks, int compactedFreeBlockCount,
                                                      int linearFreeBlocksStart, int allFreeBlockCount,
                                                      const int* localAllocationOffset,
                                                      int probe, int* probeResult) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ctx.indices.size || ctx.indices[i] != BLOCK_NOT_ALLOCATED) return;
    int localIndex = unallocatedBlocksPrefixes[i];
    bool storeProbe = probeResult && localIndex == probe;
    if (localAllocationOffset) localIndex = (localIndex + *localAllocationOffset) % allFreeBlockCount;
    int blockTexIndex = localIndex < compactedFreeBlockCount ? compactedFreeBlocks[localIndex] :
        localIndex - compactedFreeBlockCount + linearFreeBlocksStart;
    ctx.indices[i] = blockTexIndex;
    blocks.occupied[blockTexIndex] = 1;
    if (storeProbe) *probeResult = blockTexIndex;
}

void LinkedDepthLV::pack(const ivec2& srcOffset, const ivec2& dstOffset, const ivec2& size, const PackConfig& config,
                         int triangleCount, bool sort, int** cuTriangleLayerMap,
                         ViewContext* ctx, ViewContext* prevCtx) {
    auto& o = output;
    o.config = config;
    o.layerOffset = dstOffset;
    o.layerSize = size;
    cuMap();

    // block counts
    o.blocksSize = ((o.layerSize - 1) / blocks.blockSize) + 1;
    o.blocks.alloc(o.blocksSize.x * o.blocksSize.y);

    // reset data
    cuEC(cudaClear(cuCount));
    layerCount.alloc(1);
    layerCount.clear();

    // kernel launch parameters
    auto L1 = dim3(o.blocksSize.x, o.blocksSize.y);
    auto L2 = dim3(blocks.blockSize, blocks.blockSize);

    // compute o.layerCount, o.blocks and optionally relocate fragments
    if (o.config.relocatePixels) {
        timerCUDA(sort ? "Sort + Relocate + Count" : "Relocate + Count");
        int memSize = (blocks.blockSize+2) * (blocks.blockSize+2) * sizeof(float) * 2;
        cuCheck;
        relocateKernel <<< L1, L2, memSize >>> (cuHeadTex, cuFragments, cuCount, layerCount, srcOffset, sort, o);
        cuCheck;
    } else {
        timerCUDA("Count");
        cuCheck;
        countBlocksKernel <<< L1, L2 >>> (cuHeadTex, cuFragments, cuCount, layerCount, srcOffset, o);
        cuCheck;
    }
    timerCUDA();
    cuEC(cudaCopy(o.tileCount, cuCount));
    layerCount.get(&o.layerCount);

    if (blocks.track.use && ctx) {

        o.tileOffset = 0;
        ctx->size = {o.blocksSize.x, o.blocksSize.y, max(0, o.layerCount - o.config.fullLayers)};
        ctx->indices.alloc(ctx->size.x * ctx->size.y * ctx->size.z);
        ctx->pos.alloc(ctx->indices.size);
        ctx->prevPos.alloc(ctx->indices.size);
        if (ctx->indices.size > 0) {
            cuCheck;
            timerCUDA("Track.Prepare");
            prepareBlocksForReprojectionKernel <<< L1, L2 >>>
                (cuHeadTex, cuFragments, o.layerSize, config.fullLayers, blocks.track.mode, *ctx);
            cuCheck;

            // reproject
            timerCUDA("Track.Reproject");
            if (prevCtx && prevCtx->indices.size > 0) {
                CuBuffer<unsigned long long> reservations(prevCtx->indices.size);
                reservations.clear(255); // NO_REZERVATION
                for (int i = 0; i < max(1, blocks.track.reservationAttempts); i++) {
                    cuCheck;
                    reprojectAndReserveBlocksKernel <<< (unsigned)ctx->indices.size / cudaBlock + 1, cudaBlock >>>
                        (*ctx, *prevCtx, reservations, ctx->spreadDistance * blocks.track.reprojMaxDistFactor);
                    cuCheck;
                    assignReprojectedBlocksAndFreeUnusedKernel <<< (unsigned)prevCtx->indices.size / cudaBlock + 1, cudaBlock >>>
                        (*ctx, *prevCtx, reservations, blocks, i == blocks.track.reservationAttempts - 1);
                    cuCheck;
                }
            }

            // find unallocated blocks
            timerCUDA("Track.Collect Unallocated");
            CuBuffer<int> unallocatedBlocksPrefixes(ctx->indices.size);
            cuCheck;
            markUnallocatedBlocksKernel <<< (unsigned)ctx->indices.size / cudaBlock + 1, cudaBlock >>>
                (*ctx, unallocatedBlocksPrefixes);
            cuCheck;
            int unallocatedBlockCount = 0;
            prefixSum(unallocatedBlocksPrefixes.data, (int)unallocatedBlocksPrefixes.size, &unallocatedBlockCount, cudaBlock);
            cuCheck;
            o.reprojectedTileCount = o.tileCount - unallocatedBlockCount;

            // allocate remaining blocks
            if (unallocatedBlockCount > 0) {

                // collect free blocks
                timerCUDA("Track.Collect Free");
                int compactedFreeBlockCount = 0;
                CuBuffer<int> compactedFreeBlocks;
                CuBuffer<int> localAllocationOffset(1);
                if (blocks.occupied.size > 0) {
                    CuBuffer<int> freeBlocksPrefixes(blocks.occupied.size);
                    cuCheck;
                    markFreeBlocksKernel <<< (unsigned)blocks.occupied.size / cudaBlock + 1, cudaBlock >>>
                        ((int)blocks.occupied.size, blocks.occupied, freeBlocksPrefixes);
                    cuCheck;
                    prefixSum(freeBlocksPrefixes.data, (int)freeBlocksPrefixes.size, &compactedFreeBlockCount, cudaBlock);
                    cuCheck;
                    if (compactedFreeBlockCount > 0) {
                        compactedFreeBlocks.alloc(compactedFreeBlockCount);
                        cuCheck;
                        // compact free blocks and locate compacted location of last allocated block
                        compactFreeBlocksKernel <<< (unsigned)blocks.occupied.size / cudaBlock + 1, cudaBlock >>>
                                ((int)blocks.occupied.size, blocks.occupied, freeBlocksPrefixes, compactedFreeBlocks,
                                 blocks.track.lastAllocated, localAllocationOffset);
                        cuCheck;
                    }
                } else localAllocationOffset.clear();

                if (!blocks.track.cycle)
                    localAllocationOffset.clear();

                // enlarge block texture
                int linearFreeBlocksStart = blocks.tileCapacity();
                if (unallocatedBlockCount > compactedFreeBlockCount) {
                    cuCheck;
                    blocks.extend(unallocatedBlockCount - compactedFreeBlockCount);
                    cuCheck;
                }

                timerCUDA("Track.Allocate Rest");
                CuBuffer<int> probe(1);
                cuCheck;
                // allocate remaining blocks and locate texture block index of the last unallocated block
                assignUnallocatedBlocksKernel <<< (unsigned)ctx->indices.size / cudaBlock + 1, cudaBlock >>>
                    (*ctx, blocks, unallocatedBlocksPrefixes,
                     compactedFreeBlocks, compactedFreeBlockCount,
                     linearFreeBlocksStart,
                     compactedFreeBlockCount + blocks.tileCapacity() - linearFreeBlocksStart,
                     localAllocationOffset,
                     unallocatedBlockCount - 1, probe);
                cuCheck;

                if (blocks.track.cycle)
                    probe.get(&blocks.track.lastAllocated);
                else blocks.track.lastAllocated = 0;
            }

            timerCUDA();

        } else if (prevCtx && prevCtx->indices.size > 0) {
            // free all previous blocks
            assignReprojectedBlocksAndFreeUnusedKernel <<< (unsigned)prevCtx->indices.size / cudaBlock + 1, cudaBlock >>>
                (*ctx, *prevCtx, nullptr, blocks, true);
        }

    } else {

        // prefixes
        timerCUDA("Prefixes");
        if (o.layerCount > o.config.fullLayers) {
            if (o.config.layerFirstBlockOrder) {
                blocksPrefixes.alloc(o.blocksSize.x * o.blocksSize.y * (o.layerCount - o.config.fullLayers));
                cuCheck;
                markUsedBlocks <<< L1, 1 >>> (o.blocks, blocksPrefixes, o.blocksSize, o.layerCount - o.config.fullLayers);
                cuCheck;
            } else blocksPrefixes.set(o.blocks);
            cuCheck;
            prefixSum(blocksPrefixes.data, (int)blocksPrefixes.size, nullptr, cudaBlock);
            cuCheck;
        } else o.tileCount = 0;
        timerCUDA();
        o.tileOffset = blocks.extend(o.tileCount);
    }

    // mask
    if (o.config.relocatePixels) {
        if (o.config.fullLayers > 0) fullLayers.hasMask = true;
        if (o.layerCount > o.config.fullLayers) blocks.hasMask = true;
    }

    // triangle-layer mask
    if (config.triangleLayerMask && cuTriangleLayerMap && triangleCount * o.layerCount > 0) {
        o.triangleLayerMask.alloc(triangleCount * o.layerCount);
        o.triangleLayerMask.clear();
    } else o.triangleLayerMask.free();

    // pack
    timerCUDA(sort && !o.config.relocatePixels ? "Sort + Pack" : "Pack");
    cuCheck;
    packKernel <<< L1, L2 >>>
        (cuHeadTex, cuFragments, srcOffset, fullLayers, o, blocks, blocks.track.use && ctx ? ctx->indices : blocksPrefixes,
         triangleCount, cuTriangleLayerMap, sort && !o.config.relocatePixels);
    cuCheck;
    timerCUDA();
}

void LinkedDepthLV::finalizeBlocks(int minHeight, int dilateSize) {
    auto& b = blocks;
    if (b.track.use) b.tileCount = b.tileCapacity();
    if (!b.tileCount) return;

    // minimal texture height
    minHeight = ((minHeight - 1) / b.blockSize + 1) * b.blockSize; // round to blockSize
    if (b.size.y < minHeight) {
        b.size.y = minHeight;
        b.color.resize(b.size.x * b.size.y);
        b.mask.resize(b.size.x * b.size.y);
        if (b.track.use) b.occupied.resize(b.tileCapacity(), true);
    }

    b.clearEmptyBlocks();

    // postprocess
    if (dilateSize != 0) {
        timerCUDA("Dilate");
        cuCheck;
        dilate(b.color, b.size.x, b.size.y, b.blockSize, dilateSize);
        cuCheck;
        timerCUDA();
    }
}

static __global__ void packFirstLayerKernel(cudaSurfaceObject_t headTex, Fragment* fragments, LinkedDepthLV::FullLayers layer, bool flipY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= layer.size.x || y >= layer.size.y) return;

    int it;
    surf2Dread(&it, headTex, x * sizeof(int), y);
    if (it >= 0) {
        auto color = fragments[it].color;
        if (flipY) y = layer.size.y - y - 1;
        layer.color[y * layer.size.x + x] = {color.x, color.y, color.z};
    }
}

void LinkedDepthLV::packFirstLayer(bool flipY) {
    cuMap();
    packFirstLayerKernel <<< dim3(fullLayers.size.x / cudaBlock2D + 1, fullLayers.size.y / cudaBlock2D + 1),
                             dim3(cudaBlock2D, cudaBlock2D) >>> (cuHeadTex, cuFragments, fullLayers, flipY);
}

void LinkedDepthLV::ViewContext::setViewProjection(const glm::mat4& view, const glm::mat4& projection) {
    globalToView = projection * view;
    viewToGlobal = inverse(globalToView);
    globalViewOrigin = vec3(inverse(view)[3]);
}

void LinkedDepthLV::ViewContext::reset() {
    globalToView = viewToGlobal = mat4(1);
    globalViewOrigin = vec3(0);
    size = ivec3(0);
    pos.free();
    prevPos.free();
    indices.free();
}
