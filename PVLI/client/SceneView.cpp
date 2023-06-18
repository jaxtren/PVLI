#include "SceneView.h"
#include "Scene.h"
#include "Scene.inl"
#include "SceneData.inl"
#include "Application.h"
#include "imguiHelper.h"
#include "graphic.h"
#include "ffmpegDecoder.h"
#include "compression/Compression.h"
#include "compression/RLE.h"
#include "SceneTexture.h"

using namespace std;
using namespace glm;
using boost::asio::ip::tcp;

static inline bool hasRanges(ConstBuffer src) {
    auto head = Compression::header(src);
    return head.method == Compression::Method::RLE1 || head.method == Compression::Method::RLE8;
}

static inline vector<Compression::RLE::Range> decompressRanges(ConstBuffer src) {
    if(Compression::header(src).method == Compression::Method::RLE1)
        return Compression::RLE::decompressMaskRange(Compression::decompress(src, true));
    else return Compression::RLE::decompressRange(Compression::decompress(src, true));
}

/**
 * Generate indices for subsets of triangles referring to original set of vertices
 * source data can contains multiple subsets, sizes for dst and counts need to be known in advance
 * @param src subset of triangles in form of compressed binary mask of triangles (every 3 vertices has 1 value)
 * @param vertexCount size of original set (number of vertices)
 * @param dst indices per every vertex for every subset
 * @param counts stores number of vertices for every subset
 */
void decompressTriangleSubsets(const vector<unsigned char>& src, int vertexCount, int* dst, int* counts){
    counts[0] = 0;
    if (hasRanges(src)) {
        int o = 0, triangleCount = vertexCount / 3;
        for (auto r : decompressRanges(src)) {
            if (r.value == 1) {
                int t = r.start % triangleCount, l = r.start / triangleCount;
                for (int i = 0; i < r.count; i++, t++) {
                    if (t == triangleCount) {
                        t = 0;
                        counts[++l] = 0;
                    }
                    int v = t * 3;
                    counts[l] += 3;
                    dst[o++] = v;
                    dst[o++] = v + 1;
                    dst[o++] = v + 2;
                }
            }
        }
    } else {
        auto mask = Compression::decompress(src);
        for (int i = 0, l = 0, v = 0, o = 0; i < mask.size(); i++, v += 3) {
            if (v == vertexCount) {
                v = 0;
                counts[++l] = 0;
            }
            if (mask[i] == 1) {
                counts[l] += 3;
                dst[o++] = v;
                dst[o++] = v + 1;
                dst[o++] = v + 2;
            }
        }
    }
}

std::string SceneView::getStatsID() { return "View." + name; }

void SceneView::process(SocketReader& reader) {
    auto& sch = scene->app->scheduler;
    auto bufferSizeMultiple = scene->app->bufferSizeMultiple;

    reader.read(name);
    SceneView* reuse = scene->reuse ? scene->reuse->findView(name) : nullptr;

    timer("Base");
    reader.read(flags);
    reader.read(priority);
    reader.read(projection);
    reader.read(view);
    reader.read(blendFactor);
    reader.read(layerCount);
    reader.read(layerSize);
    reader.read(subdivide);
    reader.read(subdivideOffset);
    reader.read(skipLayers);

    timer("FullLayers");
    reader.read(fullLayers.count);
    reader.read(fullLayers.offset);
    reader.read(fullLayers.textureID);

    timer("Blocks");
    reader.read(blocks.size);
    reader.read(blocks.offset);
    reader.read(blocks.textureID);
    blocks.layerCount = std::max(0, layerCount - fullLayers.count);

    // blocks
    size_t blockSize = blocks.size.x * blocks.size.y * (flags & ViewFlag::LAYER_FIRST_BLOCK_ORDER ? (blocks.layerCount + 1) * sizeof(int) : sizeof(ivec2));
    if (blockSize > 0) {
        if (reuse) {
            if (blockSize <= reuse->blocks.blocksBuf.size)
                swap(blocks.blocksBuf, reuse->blocks.blocksBuf);
            swap(blocks.blocksTexture, reuse->blocks.blocksTexture);
        }
        if (!blocks.blocksBuf.size) {
            run(GPUTask::OTHER, [=]() {
                blocks.blocksBuf.allocate(GL_PIXEL_UNPACK_BUFFER, (size_t)(blockSize * bufferSizeMultiple));
                blocks.blocksBuf.map();
            }, &sync.blocksBuf);
        }
        readCompressed(reader, blocks.rawBlockCounts, "BlockCounts");
        readCompressed(reader, blocks.rawBlockIndices, "BlockIndices");
        runAfter(sync.blocksBuf, [=]() mutable {
            processBlocks();
            run(GPUTask::UPLOAD, [this]() { generateBlocksTexture(); }, &sync.blocks, "Blocks Upload", (double)blockSize / 1024);
        }, &sync.blocks, "Blocks");
    } else {
        // should be empty
        readCompressed(reader, blocks.rawBlockCounts, "BlockCounts");
        readCompressed(reader, blocks.rawBlockIndices, "BlockIndices");
    }

    timer("Triangles");
    reader.read(triangles.count);

    // triangle subset
    readCompressed(reader, triangles.rawSubset, "TriangleSubset");
    triangles.hasData = !triangles.rawSubset.empty();
    if (triangles.hasData) {
        size_t vertexIndicesSize = triangles.count * 3 * sizeof(int);
        if (reuse && scene->isBufferSuitable(reuse->triangles.vertexIndices, vertexIndicesSize))
            swap(triangles.vertexIndices, reuse->triangles.vertexIndices);
        else if (vertexIndicesSize > 0)
            run(GPUTask::OTHER, [=]() {
                scene->allocateBuffer(triangles.vertexIndices, GL_ELEMENT_ARRAY_BUFFER, vertexIndicesSize);
            }, &sync.vertexBuf);
        auto pvsSource = scene->findSourceForPVS();
        runAfter(pvsSource->sync.vertices, [=]() {
            runAfter(sync.vertexBuf, [=]() { processTriangles(pvsSource); }, &sync.vertices, "Triangles");
        }, &sync.vertices);
    }

    // triangle-layer mask
    readCompressed(reader, triangles.rawLayerMask, "TriangleLayerMask");
    if (!triangles.rawLayerMask.empty()) {
        size_t triangleLayerSize = scene->vertexCount * layerCount * sizeof(int);
        if (reuse && scene->isBufferSuitable(reuse->triangles.triangleLayer, triangleLayerSize))
            swap(triangles.triangleLayer, reuse->triangles.triangleLayer);
        else run(GPUTask::OTHER, [=]() {
                scene->allocateBuffer(triangles.triangleLayer, GL_ELEMENT_ARRAY_BUFFER, triangleLayerSize);
            }, &sync.triangleLayerBuf);
        runAfter(sync.triangleLayerBuf, [=]() {
            triangles.triangleLayerCount.resize(layerCount, 0);
            decompressTriangleSubsets(triangles.rawLayerMask,
                                      scene->vertexCount,
                                      (int*) triangles.triangleLayer.data,
                                      triangles.triangleLayerCount.data());
        }, &sync.vertices, "Triangle-Layer Subset");
    }

    // depth layers
    if (triangles.hasData)
        runAfter(sync.vertices, [this]() { generateDepthLayers(); });
}

void SceneView::received() {
    auto& f = fullLayers.texture;
    auto& b = blocks.texture;

    f = scene->findTexture(fullLayers.textureID);
    b = scene->findTexture(blocks.textureID);

    extendedBlocks = flags & ViewFlag::RELOCATED_PIXELS;
    if (scene->app->relocatePixels && extendedBlocks) {
        auto s1 = f ? f->sync.finish : px_sched::Sync();
        auto s2 = b ? b->sync.finish : px_sched::Sync();

        runAfter(s1, [=]() {
            runAfter(s2, [=]() {
                runAfter(sync.blocks, [=]() {
                    run(GPUTask::COMPUTE, [=]() {
                        scene->app->pixelRelocator.relocate(f ? f->texture : 0, b ? b->texture : 0, blocks.blocksTexture,
                                                            layerSize / subdivide, fullLayers.count, fullLayers.offset / subdivide,
                                                            b ? b->size : ivec2(0), blocks.size, flags & ViewFlag::LAYER_FIRST_BLOCK_ORDER);
                        extendedBlocks = false;
                    }, "Relocate Pixels", 3 /*TODO estimate*/);
                });
            });
        });
    }
}

void SceneView::beforeFirstRender() {
    if (triangles.hasData)
        scene->findSourceForPVS()->updateVAO(vao);
    triangles.vertexIndices.unmap();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangles.vertexIndices);
    glBindVertexArray(0);
}

void SceneView::beforeReuse() {
    glDeleteVertexArrays(1, &vao);
    vao = 0;
    if (!scene->app->reuseTextures) {
        glDeleteTextures(1, &blocks.blocksTexture);
        glDeleteTextures(1, &depth);
        blocks.blocksTexture = depth = 0;
    }
    blocks.blocksBuf.map();
    triangles.vertexIndices.map();
    triangles.triangleLayer.map();
}

void SceneView::free() {
    glDeleteVertexArrays(1, &vao);
    glDeleteTextures(1, &blocks.blocksTexture);
    glDeleteTextures(1, &depth);
    vao = blocks.blocksTexture = depth = 0;
}

void SceneView::processTriangles(Scene* pvsSource) {
    assert(scene->vertexCount == pvsSource->vertexCount);
    if (triangles.count == 0) return;

    // decompress indices
    vector<int> indices(triangles.count * 3);
    int vertexCount = 0;
    decompressTriangleSubsets(triangles.rawSubset, scene->vertexCount, indices.data(), &vertexCount);
    assert(indices.size() == vertexCount);

    // copy
    memcpy(triangles.vertexIndices.data, indices.data(), indices.size() * sizeof(int));

    // material ranges
    if (!pvsSource->materialPerTriangle.empty() && !pvsSource->materials.empty()) {
        int mat = 0, count = 0;
        for (auto i : indices) {
            int m = pvsSource->materialPerTriangle[i/3];
            if (m != mat) {
                if (count > 0) triangles.materialRanges.emplace_back(mat, count);
                mat = m;
                count = 1;
            } else count++;
        }
        if (count > 0) triangles.materialRanges.emplace_back(mat, count);
    }
}

void SceneView::processBlocks() {
    blocks.rawBlockCounts = Compression::decompress(blocks.rawBlockCounts);

    if (flags & ViewFlag::LAYER_FIRST_BLOCK_ORDER) {
        auto counts = (int*) blocks.blocksBuf.data;
        auto s = blocks.size;

        // counts
        for (int i = 0; i < s.x * s.y; i++)
            counts[i] = blocks.rawBlockCounts[i];

        // indices are sent
        if (!blocks.rawBlockIndices.empty()) {
            blocks.rawBlockIndices = Compression::decompress(blocks.rawBlockIndices);
            auto index = reinterpret_cast<int*>(blocks.rawBlockIndices.data());
            for (int l = 0; l < blocks.layerCount; l++)
                for (int y = 0; y < s.y; y++)
                    for (int x = 0; x < s.x; x++) {
                        bool used = l < blocks.rawBlockCounts[y * s.x + x];
                        counts[(l + 1) * s.x * s.y + y * s.x + x] = used ? *(index++) : -1;
                    }
            blocks.tileCount = (int)(blocks.rawBlockIndices.size() / sizeof(int));

        } else { // indices are computed
            int index = blocks.offset;
            for (int l = 0; l < blocks.layerCount; l++)
                for (int y = 0; y < s.y; y++)
                    for (int x = 0; x < s.x; x++) {
                        bool used = l < blocks.rawBlockCounts[y * s.x + x];
                        counts[(l + 1) * s.x * s.y + y * s.x + x] = used ? index++ : -1;
                    }
            blocks.tileCount = index - blocks.offset;
        }
    } else {
        int index = blocks.offset;
        auto counts = (ivec2*)blocks.blocksBuf.data;
        for (auto c : blocks.rawBlockCounts) {
            *(counts++) = {index, c};
            index += c;
        }
        blocks.tileCount = index - blocks.offset;
    }
}

void SceneView::generateBlocksTexture() {
    glActiveTexture(GL_TEXTURE0);
    if (!blocks.blocksTexture) {
        glGenTextures(1, &blocks.blocksTexture);
        glBindTexture(GL_TEXTURE_2D_ARRAY, blocks.blocksTexture);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    } else glBindTexture(GL_TEXTURE_2D_ARRAY, blocks.blocksTexture);

    blocks.blocksBuf.unmap();
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, blocks.blocksBuf);
    if (flags & ViewFlag::LAYER_FIRST_BLOCK_ORDER)
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_R32I, blocks.size.x, blocks.size.y, blocks.layerCount + 1, 0, GL_RED_INTEGER, GL_INT, 0);
    else glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RG32I, blocks.size.x, blocks.size.y, 1, 0, GL_RG_INTEGER, GL_INT, 0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void SceneView::generateDepthLayers() {
    bool useMask = scene->app->depth.useTriangleLayerMask;
    // generate additional layer when renderer requires it to correctly detect triangles without shading behind last layer
    int depthLayerCount = layerCount + (int)scene->app->renderer.requireAdditionalDepthLayer();
    for(int layer=0; layer<depthLayerCount; layer++) {
        run(GPUTask::COMPUTE, [=]() {

            // init
            auto& appDepth = scene->app->depth;
            if (layer == 0) {
                scene->findSourceForPVS()->updateVAO(vao);

                if (useMask && triangles.triangleLayer) {
                    triangles.triangleLayer.unmap();
                    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangles.triangleLayer);
                } else {
                    triangles.vertexIndices.unmap();
                    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangles.vertexIndices);
                }

                depth = DepthPeeling::createLayers(layerSize, depthLayerCount);
            } else glBindVertexArray(vao);

            auto drawLayer = [&](auto cmd) {
                if (flags & ViewFlag::CUBEMAP) {
                    int w = layerSize.x / 6;
                    for (int i=0; i<6; i++) {
                        glViewport(w * i, 0, w, layerSize.y);
                        // TODO subset of triangles per face
                        appDepth.peeling.setProjection(projection * cubemapView(i));
                        cmd();
                    }
                } else cmd();
            };

            // depth peeling
            appDepth.peeling.bindLayer(depth, layerSize, layer, projection * view, appDepth.epsilon, skipLayers);
            if (useMask && !triangles.triangleLayerCount.empty() && layer < layerCount) {
                if (layer == layerCount) { // additional layer doesn't have subset
                    triangles.vertexIndices.unmap();
                    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangles.vertexIndices);
                    drawLayer([&] { gl::drawElementsInt(GL_TRIANGLES, triangles.count * 3, 0); });
                } else {
                    int index = 0;
                    for (int l = 0; l < layer; l++) index += triangles.triangleLayerCount[l];
                    int count = triangles.triangleLayerCount[layer];
                    if (count > 0) drawLayer([&] { gl::drawElementsInt(GL_TRIANGLES, count, index); });
                }
            }
            else drawLayer([&] { gl::drawElementsInt(GL_TRIANGLES, triangles.count * 3, 0); });
            appDepth.peeling.unbind();

            glBindVertexArray(0);
        }, "Depth Peeling", 1 /*TODO estimate*/);
    }
}

static const vector<mat4> cubemapFaceTransform = {
    mat4( 0, 0, -1, 0, 0, 1,  0, 0,  1,  0,  0, 0, 0, 0, 0, 1), // left
    mat4(1), // front
    mat4( 0, 0,  1, 0, 0, 1,  0, 0, -1,  0,  0, 0, 0, 0, 0, 1), // right
    mat4(-1, 0,  0, 0, 0, 1,  0, 0,  0,  0, -1, 0, 0, 0, 0, 1), // back
    mat4( 1, 0,  0, 0, 0, 0,  1, 0,  0, -1,  0, 0, 0, 0, 0, 1), // top
    mat4( 1, 0,  0, 0, 0, 0, -1, 0,  0,  1,  0, 0, 0, 0, 0, 1), // bottom
};

glm::mat4 SceneView::cubemapView(int face) {
    return inverse(inverse(view) * cubemapFaceTransform[face]);
}

glm::mat4 SceneView::cubemapProjection(int face) {
    return projection * inverse(cubemapFaceTransform[face]);
}

