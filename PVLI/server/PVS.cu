#include "PVS.h"
#include <cstdio>
#include <iostream>
#include <limits>
#include "common.h"
#include "graphic.h"
#include "prefixsum.h"
#include "Culling.h"
#include "debug.h"

using namespace std;
using namespace glm;

PVS::PVS() {
    loadShaders();
    glGenVertexArrays(1, &vao);
}

PVS::~PVS() {
    glDeleteVertexArrays(1, &vao);
    freeFrame();
}

bool PVS::allocFrame(glm::ivec2 size) {
    if(fbo && size.x <= frameSize.x && size.y <= frameSize.y) return false;
    freeFrame();
    frameSize = size;

    // framebuffer
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    // texture
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32I, frameSize.x, frameSize.y);
    DEBUG_GPU_RAM << "glTexStorage2D PVS " << frameSize.x * frameSize.y * 4 << endl;
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    // depth
    glGenRenderbuffers(1, &depth);
    glBindRenderbuffer(GL_RENDERBUFFER, depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, frameSize.x, frameSize.y);
    DEBUG_GPU_RAM << "glRenderbufferStorage PVS " << frameSize.x * frameSize.y * 4 << endl;
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);

    GLenum drawBuf[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, drawBuf);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        throw init_error("PVS: fbo error");
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return true;
}

void PVS::freeFrame() {
    if (!fbo) return;
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    glDeleteRenderbuffers(1, &depth);
    fbo = tex = depth = 0;
}

void PVS::updateConfig(const Config& cfg){
    cfg.get("Block1D", cuBlock1D);
    cfg.get("Block2D", cuBlock2D);
    cfg.get("ConservativeRaster.Use", conservativeRaster.use);
    cfg.get("ConservativeRaster.Prepass", conservativeRaster.prepass);
    cfg.get("ConservativeRaster.Presnap", conservativeRaster.presnap);
    cfg.get("ConservativeRaster.Subpixel", conservativeRaster.subpixel);
    cfg.get("ConservativeRaster.PolygonOffset", conservativeRaster.polygonOffset);
}

void PVS::provideConfig(Config cfg){
    cfg.set("Block1D", cuBlock1D);
    cfg.set("Block2D", cuBlock2D);
    cfg.set("ConservativeRaster.Use", conservativeRaster.use);
    cfg.set("ConservativeRaster.Prepass", conservativeRaster.prepass);
    cfg.set("ConservativeRaster.Presnap", conservativeRaster.presnap);
    cfg.set("ConservativeRaster.Subpixel", conservativeRaster.subpixel);
    cfg.set("ConservativeRaster.PolygonOffset", conservativeRaster.polygonOffset);
}

bool PVS::loadShaders(){
    return rasterShader.load("pvs") &&
           markShader.load("mark") &&
           rasterMarkShader.load("pvs_mark");
}

static __global__ void setIntKernel(int count, int* dst, int value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = value;
}

void PVS::setInstances(const vector<Scene::Instance>& insts) {
    timerCUDA("SetInstances");
    instances = insts;

    // count triangles
    allTriangleCount = 0;
    for (auto& i : instances)
        allTriangleCount += i.mesh->vertexCount / 3;

    // create triangle to instance map
    triangleToInstance.alloc(allTriangleCount);
    auto dst = triangleToInstance.data;
    int index = 0;
    for (auto& i : instances) {
        int count = i.mesh->vertexCount / 3;
        cuCheck;
        setIntKernel <<< count / cuBlock1D + 1, cuBlock1D >>> (count, dst, index++);
        cuCheck;
        dst += count;
    }
    timerCUDA();

    // set instances
    vector<PVS::CuInstance> cuInsts;
    int offset = 0;
    for (auto& i : instances) {
        cuInsts.push_back({i.transform, i.mesh->cuVertices.data, offset});
        offset += i.mesh->vertexCount / 3;
    }
    cuInstances.set(cuInsts);
    timerCUDA();
}

void PVS::generate(const vector<View>& views, PVS::State& state) {
    if (!rasterShader || !rasterMarkShader || !markShader || !allTriangleCount) return;

    // init
    timerGL("Generate.Init");
    state.init(this, false);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, state.data.unmap());
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32I, GL_RED, GL_INT, nullptr);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    auto u_mvp = rasterShader.uniformLocation("mvp");
    auto u_firstID = rasterShader.uniformLocation("firstID");
    auto u2_mvp = rasterMarkShader.uniformLocation("mvp");
    auto u2_firstID = rasterMarkShader.uniformLocation("firstID");
    auto u3_size = markShader.uniformLocation("size");

    struct PendingRender {
        int instance, primOffset, primSize, firstID;
        bool cullface;
    };
    vector<PendingRender> pendingRenders;

    bool conservative = conservativeRaster.use && GLEW_NV_conservative_raster;
    if (conservative) {
        glSubpixelPrecisionBiasNV(conservativeRaster.subpixel, conservativeRaster.subpixel);
        if (GLEW_NV_conservative_raster_pre_snap_triangles)
            glConservativeRasterParameteriNV(GL_CONSERVATIVE_RASTER_MODE_NV,
                conservativeRaster.presnap ? GL_CONSERVATIVE_RASTER_MODE_PRE_SNAP_TRIANGLES_NV : GL_CONSERVATIVE_RASTER_MODE_POST_SNAP_NV);
    }

    // raster and mark for every camera
    for (auto& v : views) {
        timerGL("Generate.Raster");
        if (allocFrame(v.size))
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, v.size.x, v.size.y);
        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LESS);
        glClear(GL_DEPTH_BUFFER_BIT);
        if (conservative) {
            glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
            if (conservativeRaster.prepass)
                glEnable(GL_CONSERVATIVE_RASTERIZATION_NV);
        } else {
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            int ones[4] = { -1, -1, -1, -1 };
            glClearBufferiv(GL_COLOR, 0, ones);
        }

        auto cam = v.projection * inverse(v.camera);
        CullingPlanes culling(cam);
        pendingRenders.clear();

        // opaque | depth prepass
        rasterShader.use();
        for (int i = 0, firstID = 0; i < instances.size(); i++) {
            auto& ins = instances[i];
            if ((ins.aabb.isNull() || culling.testAABB(ins.aabb) > 0)) {
                bool inited = false;
                int primOffset = 0;
                for (auto& prim : ins.mesh->primitives) {
                    if (prim.material->opaque) {
                        if (!inited) {
                            gl::setUniform(u_mvp, cam * ins.transform);
                            gl::setUniform(u_firstID, firstID);
                            glBindBuffer(GL_ARRAY_BUFFER, ins.mesh->glVertices);
                            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Mesh::VertexData), 0);
                            inited = true;
                        }
                        glDrawArrays(GL_TRIANGLES, primOffset, prim.size);
                        if (conservative)
                            pendingRenders.push_back({i, primOffset, prim.size, firstID, true});
                    } else pendingRenders.push_back({i, primOffset, prim.size, firstID, false});
                    primOffset += prim.size;
                }
            }
            firstID += ins.mesh->vertexCount / 3;
        }

        // non opaque | conservative pass
        if (!pendingRenders.empty()) {
            if (conservative) {
                glDepthFunc(GL_LEQUAL);
                glEnable(GL_CONSERVATIVE_RASTERIZATION_NV);
                timerGL("Generate.Mark");
                glEnable(GL_POLYGON_OFFSET_FILL);
                glPolygonOffset(conservativeRaster.polygonOffset.x, conservativeRaster.polygonOffset.y);
            }
            rasterMarkShader.use();
            glDepthMask(GL_FALSE);
            glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, state.data.unmap());
            for (auto& i : pendingRenders) {
                if (i.cullface) glEnable(GL_CULL_FACE);
                else glDisable(GL_CULL_FACE);
                gl::setUniform(u2_mvp, cam * instances[i.instance].transform);
                gl::setUniform(u2_firstID, i.firstID);
                glBindBuffer(GL_ARRAY_BUFFER, instances[i.instance].mesh->glVertices);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Mesh::VertexData), 0);
                glDrawArrays(GL_TRIANGLES, i.primOffset, i.primSize);
            }
            glEnable(GL_CULL_FACE);
            if (conservative) {
                glDisable(GL_CONSERVATIVE_RASTERIZATION_NV);
                glDisable(GL_POLYGON_OFFSET_FILL);
                glPolygonOffset(0, 0);
            }
        }

        if (!conservative) {
            // mark
            timerGL("Generate.Mark");
            glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
            markShader.use();
            gl::setUniform(u3_size, v.size);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, state.data.unmap());
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, tex);
            markShader.dispatchCompute(v.size);
        }
    }

    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LESS);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glUseProgram(0);
    glBindVertexArray(0);
    timerGL();
}

static __global__ void geometryKernel(int count, PVS::CuInstance* instances, int* triangleToInstance, int* prefixes,
                                      vec3* vertices, vec2* uv, ivec2* triangles, int* material) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int loc = prefixes[i];
    if (loc >= 0) {
        int inst = triangleToInstance[i];
        auto& instance = instances[inst];
        int index = i - instance.offset;
        if (triangles) triangles[loc] = {inst, index};
        auto instanceVertices = instance.cuVertices;
        if (material) material[loc] = instanceVertices[index * 3].material;
        auto transform = instance.transform;
        for (int t = 0; t < 3; t++) {
            auto& v = instanceVertices[index * 3 + t];
            if (vertices) vertices[loc * 3 + t] = vec3(transform * vec4(v.vertex, 1));
            if (uv) uv[loc * 3 + t] = v.uv;
        }
    }
}

PVS::State::Geometry PVS::State::geometry(int count, bool vertices, bool uv, bool triangles, bool material) {
    pvs->timerCUDA("State.Memory.Map");
    data.map();
    pvs->timerCUDA("State.Memory.AllocTemp");
    auto& buffer = pvs->tempData;
    buffer.alloc(count * 3 * sizeof(vec3) * (int)vertices +
                 count * 3 * sizeof(vec2) * (int)uv +
                 count * sizeof(ivec2) * (int)triangles +
                 count * sizeof(int) * (int)material);
    size_t offset = 0;
    auto vertexBuf = buffer.stack<vec3>(offset, count * 3 * (int)vertices);
    auto uvBuf = buffer.stack<vec2>(offset, count * 3 * (int)uv);
    auto triangleBuf = buffer.stack<ivec2>(offset, count * (int)triangles);
    auto materialBuf = buffer.stack<int>(offset, count * (int)material);
    pvs->timerCUDA("State.Geometry");
    cuCheck;
    geometryKernel <<< data.size / pvs->cuBlock1D + 1, pvs->cuBlock1D >>>
        (data.size, pvs->cuInstances, pvs->triangleToInstance, data.map(),
         vertexBuf, uvBuf, triangleBuf, materialBuf);
    cuCheck;
    pvs->timerCUDA("State.Geometry.Copy");
    PVS::State::Geometry ret;
    if (vertices) ret.vertices = vertexBuf.get();
    if (uv) ret.uv = uvBuf.get();
    if (triangles) ret.triangles = triangleBuf.get();
    if (material) ret.material = materialBuf.get();
    pvs->timerCUDA();
    return ret;
}

void PVS::State::init(PVS* p, bool clear) {
    if (!p) return;
    pvs = p;
    pvs->timerCUDA("State.Memory.AllocGL");
    if (data.size != pvs->allTriangleCount) {
        data.free();
        data.init(GL_SHADER_STORAGE_BUFFER, pvs->allTriangleCount, nullptr, GL_STREAM_COPY);
    }
    if (clear) cuEC(cudaClear(data.map(), data.size));
    pvs->timerCUDA();
}

static __global__ void updateIndirectKernel(int n, PVS::CuInstance* instances, int* data, int** indirect) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) indirect[i] = data + instances[i].offset;
}

int** PVS::State::updateIndirect() {
    if (!pvs) return nullptr;
    pvs->timerCUDA("State.Memory.Map");
    data.map();
    pvs->timerCUDA("State.Memory.Alloc");
    int n = (int)pvs->instances.size();
    indirect.alloc(n);
    pvs->timerCUDA("State.UpdateIndirect");
    cuCheck;
    updateIndirectKernel <<< n / pvs->cuBlock1D + 1, pvs->cuBlock1D >>>(n, pvs->cuInstances, data.map(), indirect.data);
    cuCheck;
    pvs->timerCUDA();
    return indirect.data;
}


void PVS::State::set(int value) {
    if (pvs && data.size)
        setIntKernel <<< data.size / pvs->cuBlock1D + 1, pvs->cuBlock1D >>>(data.size, data.map(), value);
}

PVS::State& PVS::State::operator=(PVS::State& state) {
    if (!state.pvs) return *this;
    pvs = state.pvs;
    pvs->timerCUDA("State.Memory.AllocGL");
    data.init(GL_SHADER_STORAGE_BUFFER, state.data.size, nullptr, GL_STREAM_COPY);
    pvs->timerCUDA("State.Memory.Copy");
    cuEC(cudaCopy(data.map(), state.data.map(), data.size, cudaMemcpyKind::cudaMemcpyDeviceToDevice));
    pvs->timerCUDA();
    return *this;
}

static __global__ void maxKernel(int count, int* dst, int* src) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < count) dst[i] = max(dst[i], src[i]);
}

void PVS::State::max(PVS::State& state) {
    if (!pvs || !data.size || data.size != state.data.size) *this = state;
    else {
        pvs->timerCUDA("State.Memory.Map");
        data.map();
        state.data.map();
        pvs->timerCUDA("State.Max");
        maxKernel <<< data.size / pvs->cuBlock1D + 1, pvs->cuBlock1D >>> (data.size, data.map(), state.data.map());
        pvs->timerCUDA();
    }
}

static __device__ bool inRange (const ivec2& range, int v) {
    return range.x <= range.y ? (range.x <= v && v <= range.y) : (v <= range.y || range.x <= v);
}

static __global__ void replaceValueKernel(int count, int* dst, int* src, ivec2 range, int value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < count && inRange(range, src[i])) dst[i] = value;
}

void PVS::State::replace(PVS::State& state, ivec2 range, int value) {
    if (!state.pvs || !state.data.size) return;
    if (!pvs || data.size != state.data.size) init(state.pvs);
    pvs->timerCUDA("State.Memory.Map");
    data.map();
    state.data.map();
    pvs->timerCUDA("State.FilterValue");
    replaceValueKernel <<< data.size / pvs->cuBlock1D + 1, pvs->cuBlock1D >>> (data.size, data.map(), state.data.map(), range, value);
    pvs->timerCUDA();
}

static __device__ int countSum;
static __global__ void countKernel(int count, int* data, ivec2 range) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    __shared__ int sum;
    if (threadIdx.x == 0) sum = 0;
    __syncthreads();
    if (inRange(range, data[i])) atomicAdd(&sum, 1);
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(&countSum, sum);
}

int PVS::State::count(ivec2 range) {
    if (!pvs || !data.size) return 0;
    pvs->timerCUDA("State.Memory.Map");
    data.map();
    pvs->timerCUDA("State.Count");
    int count = 0;
    cuEC(cudaMemcpyToSymbol(countSum, &count, sizeof(int)));
    cuCheck;
    countKernel <<< data.size / pvs->cuBlock1D + 1, pvs->cuBlock1D >>> (data.size, data.map(), range);
    cuCheck;
    cuEC(cudaMemcpyFromSymbol(&count, countSum, sizeof(int)));
    pvs->timerCUDA();
    return count;
}

int PVS::State::compact(PVS::State& state, ivec2 range) {
    if (!state.pvs) return 0;
    init(state.pvs);
    pvs->timerCUDA("State.Memory.Map");
    data.map();
    state.data.map();
    pvs->timerCUDA("State.Compact");
    cuCheck;
    replaceValueKernel <<< data.size / pvs->cuBlock1D + 1, pvs->cuBlock1D >>>(data.size, data.map(), state.data.map(), range, 1);
    cuCheck;
    int count;
    prefixSum(data.cu, state.data.size, &count, pvs->cuBlock1D);
    cuCheck;
    replaceValueKernel <<< state.data.size / pvs->cuBlock1D + 1, pvs->cuBlock1D >>>(data.size, data.cu, state.data.cu, invertRange(range), -1);
    cuCheck;
    pvs->timerCUDA();
    return count;
}

static __global__ void filterCameraKernel(int count, int* dst, int* src, ivec2 range,
                                          PVS::CuInstance* instances, int* triangleToInstance,
                                          CullingPlanes planes, vec3 origin, float slopeLimit, int cullface) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count || !inRange(range, src[id])) return;
    int inst = triangleToInstance[id];
    int index = id - instances[inst].offset;
    auto vertices = instances[inst].cuVertices;
    mat4 transform = instances[inst].transform;

    vec3 points[3];
    for (int i = 0; i < 3; i++)
        points[i] = vec3(transform * vec4(vertices[index * 3 + i].vertex, 1));

    // frustum
    int state = planes.testTriangle(points[0], points[1], points[2]);

    // slope
    vec3 normal = normalize(cross(points[2] - points[0], points[1] - points[0]));
    vec3 center = (points[0] + points[1] + points[2]) * (1.0f / 3.0f);
    float slope = dot(normal, center - origin); // do not normalize center - origin to scale based on distance
    if ((!cullface || cullface == (slope < 0 ? 1 : -1)) && abs(slope) > slopeLimit)
        state += 4;

    dst[id] = state;
}

void PVS::State::filter(State& state, glm::ivec2 range, const glm::mat4& camera, const glm::mat4& projection, float slopeLimit, int cullface) {
    if (!pvs || !data.size) return;
    pvs->timerCUDA("State.Memory.Map");
    data.map();
    state.data.map();
    pvs->timerCUDA("State.FilterCamera");
    CullingPlanes planes(projection * inverse(camera));
    cuCheck;
    filterCameraKernel <<< data.size / pvs->cuBlock1D + 1, pvs->cuBlock1D >>>
        (data.size, data.map(), state.data.map(), range, pvs->cuInstances, pvs->triangleToInstance, planes, vec3(camera[3]), slopeLimit, cullface);
    cuCheck;
    pvs->timerCUDA();
}

static __global__ void maskKernel(int count, int maskSize, int* data, unsigned char* mask) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && 0 <= data[i] && data[i] < maskSize)
        mask[data[i]] = 1;
}

std::vector<unsigned char> PVS::State::mask(int count) {
    if (count <= 0 || !pvs) return std::vector<unsigned char>();
    pvs->timerCUDA("State.Memory.Map");
    data.map();
    pvs->timerCUDA("State.Memory.AllocTemp");
    auto& mask = pvs->tempData;
    mask.alloc(count);
    pvs->timerCUDA("State.Mask");
    mask.clear();
    cuCheck;
    maskKernel <<< data.size / pvs->cuBlock1D + 1, pvs->cuBlock1D >>> (data.size, count, data.map(), mask.data);
    cuCheck;
    pvs->timerCUDA("State.Mask.Copy");
    auto ret = mask.get();
    pvs->timerCUDA();
    return ret;
}
