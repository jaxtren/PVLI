#include <core_settings.h>

#ifdef __CLION_IDE__ //for clion preprocessor only, TODO remove
#define LH2_DEVFUNC
#define __launch_bounds__(x,y)
#define isfinite(x)
CoreData data;
#endif

__device__ const float maxHalfFloat = 3e4f;

__host__ __device__ void fixFloat(float& v) {
    if (!isfinite(v)) v = 0;
}

__global__ void updateFilterFragmentsKernel(int sample, bool accumulated) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= *data.count || data.positions[id].x < 0) return;
    int id1 = id, id2 = id + data.stride;
    auto& f = data.filter.current;
    auto& s = f.shading_var;

    float prescale = accumulated ? (1.0f / data.spp) : 1.0f;
    float scale = accumulated ? 1.0f : (1.0f / data.spp);
    unsigned int flags = __float_as_uint(f.normal_flags[id].w);

    // demodulate albedo
    float3 albedo = RGB32toHDRmin1(__float_as_uint(f.pos_albedo[id].w));
    float3 albedo2 = albedo;
    if (flags & FilterData::FLAG_SEPARATED_CHANNELS)
        albedo2 = RGB32toHDRmin1(__float_as_uint(f.pos_albedo[f.count + id].w));
    float3 ch1 = make_float3(data.accumulator[id1]) * prescale / albedo;
    float3 ch2 = make_float3(data.accumulator[id2]) * prescale / albedo2;

    // clamp
    if (flags & FilterData::FLAG_REFLECTION_REFRACTION) {
        ch1 = min3(ch1, data.filter.reflectionClamp);
        ch2 = min3(ch2, data.filter.refractionClamp);
    } else {
        ch1 = min3(ch1, data.filter.directClamp);
        ch2 = min3(ch2, data.filter.indirectClamp);
    }

    // clear accumulator
    data.accumulator[id] = data.accumulator[id2] = {0, 0, 0, 0};

    // luminance moments
    float4 moments = {Luminance(ch1), 0, Luminance(ch2), 0};
    moments.y = moments.x * moments.x;
    moments.w = moments.z * moments.z;

    // store data
    if (sample == 0) {
        f.moments[id] = moments * scale;
        s[id] = { make_half4(make_float4(min3(ch1 * scale, maxHalfFloat), 0)),
                  make_half4(make_float4(min3(ch2 * scale, maxHalfFloat), 0))};
    } else {
        f.moments[id] += moments * scale;
        half8 v = s[id];
        s[id] = { make_half4(min4(make_float4(v.x) + make_float4(ch1 * scale, 0), maxHalfFloat)),
                  make_half4(min4(make_float4(v.y) + make_float4(ch2 * scale, 0), maxHalfFloat))};
    }
}

__global__ void reorderFragmentsPrepareKernel() {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= data.scrsize.x * data.scrsize.y) return;
    data.temp.indices[id] = data.counts[id] - 1; // first layer is separated
}

__global__ void reorderFragmentsKernel() {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= data.scrsize.x * data.scrsize.y) return;
    int count = data.counts[id], first = data.scrsize.x * data.scrsize.y + data.temp.indices[id] - 1;
    for (int src = id, i = 0; src >= 0 && i < count; src = data.links[src], i++)
        data.temp.primary[i == 0 ? id : first + i] = data.primary[src]; // reorder primary rays
}

__global__ void reorderFragmentsUpdateKernel() {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= data.scrsize.x * data.scrsize.y) return;
    int count = data.counts[id], first = data.scrsize.x * data.scrsize.y + data.temp.indices[id];
    short2 pos = make_short2(id % data.scrsize.x, id / data.scrsize.x);
    for (int i = 0; i < count; i++) {
        int dst = i == 0 ? id : first + i - 1;
        data.links[dst] = i + 1 == count ? -1 : first + i; // generate links
        data.positions[dst] = pos; // save position
    }
}

__host__ __device__ void bilinearWeights(const float2& w, float* to) {
    to[0] = (1.0f - w.x) * (1.0f - w.y);
    to[1] = w.x * (1.0f - w.y);
    to[2] = (1.0f - w.x) * w.y;
    to[3] = w.x * w.y;
}

static __device__ __constant__ float gauss[] = {
    1.0f / 16.0f, 1.0f / 8.0f, 1.0f / 16.0f,
    1.0f / 8.0f, 1.0f / 4.0f, 1.0f / 8.0f,
    1.0f / 16.0f, 1.0f / 8.0f, 1.0f / 16.0f
};

static __device__ __constant__ float kernel[] = {
    1.0f/256.0f, 1.0f/64.0f, 3.0f/128.0f, 1.0f/64.0f, 1.0f/256.0f,
    1.0f/64.0f,  1.0f/16.0f, 3.0f/32.0f,  1.0f/16.0f, 1.0f/64.0f,
    3.0f/128.0f, 3.0f/32.0f, 9.0f/64.0f,  3.0f/32.0f, 3.0f/128.0f,
    1.0f/64.0f,  1.0f/16.0f, 3.0f/32.0f,  1.0f/16.0f, 1.0f/64.0f,
    1.0f/256.0f, 1.0f/64.0f, 3.0f/128.0f, 1.0f/64.0f, 1.0f/256.0f
};

static __device__ __constant__ float kernel7x7[] = {
    0.000036f, 0.000363f, 0.001446f, 0.002291f, 0.001446f, 0.000363f, 0.000036f,
    0.000363f, 0.003676f, 0.014662f, 0.023226f, 0.014662f, 0.003676f, 0.000363f,
    0.001446f, 0.014662f, 0.058488f, 0.092651f, 0.058488f, 0.014662f, 0.001446f,
    0.002291f, 0.023226f, 0.092651f, 0.146768f, 0.092651f, 0.023226f, 0.002291f,
    0.001446f, 0.014662f, 0.058488f, 0.092651f, 0.058488f, 0.014662f, 0.001446f,
    0.000363f, 0.003676f, 0.014662f, 0.023226f, 0.014662f, 0.003676f, 0.000363f,
    0.000036f, 0.000363f, 0.001446f, 0.002291f, 0.001446f, 0.000363f, 0.000036f
};

__device__ float4 spatialMomentsEstimation(short2 pixel, const float3& pos, const float3& normal, bool specular, int dataOffset = 0) {
    auto& f = data.filter;
    const float eps = 1e-5f;
    float dist = length(pos - data.view.pos);
    float distPlaneDot = dot(pos - data.view.pos, normal);
    float sumWeight = 0;
    float4 sumMoments = {0, 0, 0, 0};
    int face = data.face(pixel.x, pixel.y);
    const int r = 3;
    for (int vv = -r; vv <= r; vv++) {
        const int v = vv + pixel.y;
        for (int uu = -r; uu <= r; uu++) {
            const int u = uu + pixel.x;
            int id2 = data.index(u, v, face);
            if (id2 < 0) continue;
            const float h = kernel7x7[(vv + r) * (r+r+1) + (uu + r)];
            const float dist2D = sqrtf(uu * uu + vv * vv);

            // ray direction
            float3 rayDir;
            if (!specular) rayDir = data.viewDir(u, v, face);
            else rayDir = normalize(make_float3(f.current.pos_albedo[dataOffset + id2]) - data.view.pos); // --- load

            // find best fragment
            float distPlane = distPlaneDot / (dot(normal, rayDir) + eps), distTmp = 0;
            float offset = clamp(f.closestOffset * dist2D * dist, f.closestOffsetMin, f.closestOffsetMax);
            int bid = f.current.closestFragmentDepth(id2, distPlane - offset, distTmp, dataOffset);
            if (bid < 0) continue;

            // weights
            float w_depth = -fabs(distTmp - dist) / (fabs(distPlane - dist) + eps) * f.distanceFactor;
            float w_normal = powf(max(0.0f, dot(normal, make_float3(f.current.normal_flags[dataOffset + bid]))), f.normalFactor); // --- load

            // final weight
            float w = __expf(w_depth) * w_normal * h;
            fixFloat(w);

            // sum
            sumWeight += w;
            sumMoments += f.current.moments[bid] * w;
        }
    }
    return sumMoments * (1.0f / max(eps, sumWeight));
}

__global__ void prepareFilterFragmentsKernel(int phase) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    auto pixel = data.positions[id]; // --- load
    if (id >= *data.count || pixel.x < 0) return;
    auto& f = data.filter;
    auto pos = make_float3(f.current.pos_albedo[id]);
    auto normal_flags = f.current.normal_flags[id];
    auto normal = make_float3(normal_flags);
    unsigned int flags = __float_as_uint(normal_flags.w);
    bool separatedChannels = flags & FilterData::FLAG_SEPARATED_CHANNELS;
    bool fallback = false;
    unsigned int reprojCount = 0;
    float maxDist = f.reprojMaxDistFactor * length(pos - data.view.pos);
    float offsetOut = 0;

    // depth
    if (phase == 0) {
        if (f.current.depth) {
            f.current.depth[id] = length(make_float3(f.current.pos_albedo[id]) - data.view.pos);
            if (separatedChannels)
                f.current.depth[f.current.count + id] = length(make_float3(f.current.pos_albedo[f.current.count + id]) - data.view.pos);
        }
        return;
    }

    // reproject first channel
    float4 reprojectedMoments;
    if (!(flags & FilterData::FLAG_EMISSIVE) && !(flags & FilterData::FLAG_SPECULAR)) { // TODO specular fragment
        float3 posTmp;

        if (f.reprojLinearFilter) {
            int pid[4] = { -1, -1, -1, -1 };
            int nonEmpty = 0;
            float2 w2 = { 0, 0 };

            // primary
            if (f.reprojWeight > 0) {
                w2 = f.previous.reprojectLinear(pos, pid);
                for (int i=0; i<4; i++) {
                    auto& p = pid[i];
                    if (p >= 0) {
                        p = f.previous.closestFragmentPos(pos, normal, p, 0, posTmp, offsetOut);
                        if (length(posTmp - pos) > maxDist * data.view.spreadAngle) p = -1;
                        else nonEmpty++;
                    }
                }
            }

            // fallback
            if (nonEmpty == 0 && f.reprojWeightFallback > 0) {
                w2 = f.fallback.reprojectLinear(pos, pid);
                for (int i=0; i<4; i++) {
                    auto& p = pid[i];
                    if (p >= 0) {
                        p = f.fallback.closestFragmentPos(pos, normal, p, 0, posTmp, offsetOut);
                        if (length(posTmp - pos) > maxDist * f.fallback.view.spreadAngle) p = -1;
                        else nonEmpty++;
                    }
                }
                fallback = nonEmpty > 0;
            }

            if (nonEmpty > 0) {
                f.prevWeights[id] = make_half2(w2);

                auto& previous = fallback ? f.fallback : f.previous;
                float4 prevMoments = {0, 0, 0, 0};
                float4 prevShadingVarDir = {0, 0, 0, 0};
                float4 prevShadingVarInd = {0, 0, 0, 0};

                float W[4], w = 0;
                bilinearWeights(w2, W);
                for (int i = 0; i < 4; i++) {
                    int p = pid[i];
                    if (p >= 0) {
                        float cw = W[i];
                        w += cw;

                        // interpolate moments
                        prevMoments += previous.moments[p] * cw;

                        // interpolate shading
                        half8 prev = previous.shading_var[p];
                        prevShadingVarDir += make_float4(prev.x) * cw;
                        prevShadingVarInd += make_float4(prev.y) * cw;

                        // max reprojection count
                        reprojCount = max(reprojCount, (__float_as_uint(previous.normal_flags[p].w) >> FilterData::REPROJ_COUNT_SHIFT) + 1);

                        // link
                        f.prevLink[id + data.stride * i] = p | (fallback * CoreData::Filter::FALLBACK_BIT);
                    } else f.prevLink[id + data.stride * i] = CoreData::Filter::NO_REPROJECT | (fallback * CoreData::Filter::FALLBACK_BIT);
                }

                if (w > 0) {
                    w = 1.0f / w;
                    reprojectedMoments = prevMoments * w;

                    // mix shading
                    if (f.shadeMergePhase == 0) {
                        auto reprojWeight = fallback ? f.reprojWeightFallback : f.reprojWeight;
                        half8 cur = f.current.shading_var[id];
                        float4 l1 = lerp(make_float4(cur.x), prevShadingVarDir * w, reprojWeight);
                        float4 l2 = lerp(make_float4(cur.y), prevShadingVarInd * w, reprojWeight);
                        f.current.shading_var[id] = {make_half4(l1), make_half4(l2)};
                    }
                }
            }

        } else { // nearest
            int pid = -1;

            // primary
            if (f.reprojWeight > 0) {
                pid = f.previous.reprojectNearest(pos);
                if (pid >= 0) {
                    pid = f.previous.closestFragmentPos(pos, normal, pid, 0, posTmp, offsetOut);
                    if (length(posTmp - pos) > maxDist * data.view.spreadAngle) pid = -1;
                }
            }

            // fallback
            if (pid < 0 && f.reprojWeightFallback > 0) {
                pid = f.fallback.reprojectNearest(pos);
                if (pid >= 0) {
                    pid = f.fallback.closestFragmentPos(pos, normal, pid, 0, posTmp, offsetOut);
                    if (length(posTmp - pos) > maxDist * f.fallback.view.spreadAngle) pid = -1;
                    else fallback = true;
                }
            }

            if (pid >= 0) {
                auto& previous = fallback ? f.fallback : f.previous;
                reprojectedMoments = previous.moments[pid];

                // mix shading
                if (f.shadeMergePhase == 0) {
                    auto reprojWeight = fallback ? f.reprojWeightFallback : f.reprojWeight;
                    half8 cur = f.current.shading_var[id], prev = previous.shading_var[pid];
                    float4 l1 = lerp(make_float4(cur.x), make_float4(prev.x), reprojWeight);
                    float4 l2 = lerp(make_float4(cur.y), make_float4(prev.y), reprojWeight);
                    f.current.shading_var[id] = {make_half4(l1), make_half4(l2)};
                }

                // reprojection count
                reprojCount = (__float_as_uint(previous.normal_flags[pid].w) >> FilterData::REPROJ_COUNT_SHIFT) + 1;

                // link
                f.prevLink[id] = pid | (fallback * CoreData::Filter::FALLBACK_BIT);
            }
        }
    }

    // spatial estimation of moments
    float4 moments;
    if (flags & FilterData::FLAG_SPECULAR || (reprojCount < f.reprojSpatialCount && !(flags & FilterData::FLAG_EMISSIVE)))
        moments = spatialMomentsEstimation(pixel, pos, normal, flags & FilterData::FLAG_SPECULAR); // estimate both channels
    else moments = f.current.moments[id];
    if (separatedChannels) {
        float4 moments2 = spatialMomentsEstimation(pixel, make_float3(f.current.pos_albedo[f.current.count + id]),
                                                   make_float3(f.current.normal_flags[f.current.count + id]), true, f.current.count);
        moments.z = moments2.z, moments.w = moments2.w; // replace only moments of second channel
    }

    // mix with reprojected moments
    if (reprojCount != 0) {
        auto moments2 = lerp(moments, reprojectedMoments, fallback ? f.reprojWeightFallback : f.reprojWeight);
        moments.x = moments2.x, moments.y = moments2.y;
        if (!separatedChannels) // reprojection is not for second channel with separated channels
            moments.z = moments2.z, moments.w = moments2.w;
    }

    // store updated moments
    fixFloat(moments.x);
    fixFloat(moments.y);
    fixFloat(moments.z);
    fixFloat(moments.w);
    f.moments[id] = moments;

    // moments -> variance
    f.current.shading_var[id].x.w = make_half(max(0.0f, min(moments.y - moments.x * moments.x, maxHalfFloat)));
    f.current.shading_var[id].y.w = make_half(max(0.0f, min(moments.w - moments.z * moments.z, maxHalfFloat)));

    // store reprojection count
    f.current.normal_flags[id].w = __uint_as_float((flags & FilterData::FLAGS_MASK) |
        (min(reprojCount, FilterData::REPROJ_COUNT_MAX) << FilterData::REPROJ_COUNT_SHIFT));

    // clean previous links, when not reprojected
    if (reprojCount == 0) {
        f.prevLink[id] = CoreData::Filter::NO_REPROJECT;
        if (f.reprojLinearFilter) {
            f.prevLink[id + data.stride * 1] = CoreData::Filter::NO_REPROJECT;
            f.prevLink[id + data.stride * 2] = CoreData::Filter::NO_REPROJECT;
            f.prevLink[id + data.stride * 3] = CoreData::Filter::NO_REPROJECT;
        }
    }
}

__global__ void applyLayeredFilterKernel(int phase, int mode) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (mode == 1) { // only first layer (2D blocks)
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        if (id >= data.scrsize.x || y >= data.scrsize.y) return;
        id += y * data.scrsize.x;
    } else {
        if (mode == 2) // skip first layer (1D blocks)
            id += data.scrsize.x * data.scrsize.y;
        if (id >= *data.count) return;
    }

    auto p = data.positions[id]; // --- load
    if (p.x < 0) return;
    auto& f = data.filter;
    auto pixel = make_int2(p.x, p.y);
    int face = data.face(pixel.x, pixel.y);
    const float eps = 1e-5f;

    half8 sh = f.current.shading_var[id]; // --- load
    float4 dir_var = make_float4(sh.x), ind_var = make_float4(sh.y);
    float2 luminance = make_float2(Luminance(make_float3(dir_var)), Luminance(make_float3(ind_var)));

    // data for first channel
    float3 pos = make_float3(f.current.pos_albedo[id]); // --- load
    float4 normal_flags = f.current.normal_flags[id]; // --- load
    float3 normal = make_float3(normal_flags);
    unsigned int flags = __float_as_uint(normal_flags.w);
    float dist = length(pos - data.view.pos);

    // optional data for second channel
    bool separatedChannels = flags & FilterData::FLAG_SEPARATED_CHANNELS;
    float3 pos2, normal2;
    unsigned int flags2;
    float dist2;
    if (separatedChannels) {
        pos2 = make_float3(f.current.pos_albedo[f.current.count + id]); // --- load
        dist2 = length(pos2 - data.view.pos);
        auto normal_flags2 = f.current.normal_flags[f.current.count + id]; // --- load
        normal2 = make_float3(normal_flags2);
        flags2 = __float_as_uint(normal_flags2.w);
    }

    float3 channel1 = {0, 0, 0}, channel2 = {0, 0, 0};
    float2 weight = {0, 0}, variance = {0, 0};

    // base variance, optionally blurred using gaussian
    float2 baseVariance = {0, 0};
    if (phase <= f.varianceGauss) {
        float2 W = {0, 0};
        int R = 1;
        for (int y = -R; y <= R; y++)
            for (int x = -R; x <= R; x++) {
                int i = data.index(pixel.x + x, pixel.y + y, face);
                if (i < 0) continue;
                float dist2D = sqrtf(x * x + y * y);
                int i1 = -1, i2 = -1;

                // first channel
                float3 posTmp;
                float offsetOut = 0;
                float3 rayDir = normalize(make_float3(f.current.pos_albedo[i]) - data.view.pos); // --- load
                float offset = clamp(f.closestOffset * dist2D * dist * dot(normal, rayDir), f.closestOffsetMin, f.closestOffsetMax);
                i1 = i2 = f.current.closestFragmentPos(pos, normal, i, offset, posTmp, offsetOut);

                // second channel
                if (separatedChannels) {
                    rayDir = normalize(make_float3(f.current.pos_albedo[f.current.count + i]) - data.view.pos); // --- load
                    offset = clamp(f.closestOffset * dist2D * dist2 * dot(normal2, rayDir), f.closestOffsetMin, f.closestOffsetMax);
                    i2 = f.current.closestFragmentPos(pos2, normal2, i, offset, posTmp, offsetOut, f.current.count);
                }

                float w = gauss[(y + 1) * 3 + x + 1];
                if (i1 >= 0) { baseVariance.x += make_float(f.current.shading_var[i1].x.w) * w; W.x += w; }
                if (i2 >= 0) { baseVariance.y += make_float(f.current.shading_var[i2].y.w) * w; W.y += w; }
            }
        baseVariance.x /= W.x;
        baseVariance.y /= W.y;
    } else baseVariance = make_float2(make_float(sh.x.w), make_float(sh.y.w));

    // variance factor based on reprojection count
    unsigned int reprojCount = flags >> FilterData::REPROJ_COUNT_SHIFT;
    float localVarFactor = max(1.0f, f.varianceReprojFactor / (reprojCount + 1));

    // precomputed values
    float2 reci_sqrt_var;
    reci_sqrt_var.x = -1.0f / (sqrtf(baseVariance.x + eps) * f.varianceFactor * localVarFactor + eps);
    reci_sqrt_var.y = -1.0f / (sqrtf(baseVariance.y + eps) * f.varianceFactor * localVarFactor + eps);
    float distPlaneDot = dot(pos - data.view.pos, normal);
    float dist2PlaneDot = dot(pos2 - data.view.pos, normal2);

    // reconstruct illumination
    const int step = 1 << (phase - 1);
    for (int vv = -2; vv <= 2; vv++) {
        const int v = vv * step + pixel.y;
        const int r = abs(vv) == 2 ? 1 : 2;
        for (int uu = -r; uu <= r; uu++) {
            const int u = uu * step + pixel.x;
            int id2 = data.index(u, v, face);
            if (id2 < 0) continue;
            float dist2D = sqrtf(uu*uu + vv * vv) * step;

            auto getWeights = [=] (const float3& pos, const float3& normal, bool specular, float distPlaneDot, float dist, int dataOffset, float2& w) -> int {

                // ray direction
                float3 rayDir;
                if (!specular) rayDir = data.viewDir(u, v, face);
                else rayDir = normalize(make_float3(f.current.pos_albedo[dataOffset + id2]) - data.view.pos); // --- load

                // distance
                float distPlane = distPlaneDot / (dot(normal, rayDir) + eps);

                // find best fragment
                int i = id;
                float distTmp = dist;
                float offset = clamp(f.closestOffset * dist2D * dist * (f.depthMode != 2 ? dot(normal, rayDir) : 1.0f), f.closestOffsetMin, f.closestOffsetMax);
                if (f.depthMode == 2) // depth + normal
                    i = f.current.closestFragmentDepth(id2, distPlane - offset, distTmp, dataOffset);
                else { // pos + normal
                    float offsetOut = 0;
                    float3 posTmp;
                    i = f.current.closestFragmentPos(pos, normal, id2, offset, posTmp, offsetOut, dataOffset);
                    distTmp = length(posTmp - data.view.pos);
                }

                // depth weight
                w.x = -fabs(distTmp - dist) / (fabs(distPlane - dist) + eps) * f.distanceFactor;

                // normal weight
                w.y = powf(max(0.0f, dot(normal, make_float3(f.current.normal_flags[dataOffset + i]))), f.normalFactor); // --- load
                fixFloat(w.y);

                return i;
            };

            // weights for both channels
            float2 w1, w2; // weights: x: depth, y: normal
            int2 i;
            if (vv == 0 && uu == 0) {
                w1 = w2 = {0, 1};
                i = { id, id };
            } else {
                i.x = getWeights(pos, normal, flags & FilterData::FLAG_SPECULAR, distPlaneDot, dist, 0, w1);
                if (separatedChannels) // second channel is optional
                    i.y = getWeights(pos2, normal2, flags2 & FilterData::FLAG_SPECULAR, dist2PlaneDot, dist2, f.current.count, w2);
                else {
                    i.y = i.x;
                    w2 = w1;
                }
            }

            // channels
            float4 ch1 = make_float4(f.current.shading_var[i.x].x);
            float4 ch2 = make_float4(f.current.shading_var[i.y].y);

            // luminance + depth weight
            float2 w = make_float2(
                __expf(fabs(luminance.x - Luminance(make_float3(ch1))) * reci_sqrt_var.x + w1.x),
                __expf(fabs(luminance.y - Luminance(make_float3(ch2))) * reci_sqrt_var.y + w2.x)
            );

            // apply normal weights
            w.x *= w1.y;
            w.y *= w2.y;

            // apply filter kernel
            w *= kernel[(vv+2) * 5 + (uu+2)];
            fixFloat(w.x);
            fixFloat(w.y);

            // variance
            variance += make_float2(ch1.w, ch2.w) * w * w;

            // luminance
            channel1 += make_float3(ch1) * w.x;
            channel2 += make_float3(ch2) * w.y;

            weight += w;
        }
    }

    channel1 *= 1.0f / max(eps, weight.x);
    channel2 *= 1.0f / max(eps, weight.y);
    variance *= 1.0f / max2(eps, weight * weight);

    // mix shading
    if (f.shadeMergePhase == phase) {
        bool fallback = f.prevLink[id] & CoreData::Filter::FALLBACK_BIT;
        auto reprojWeight = fallback ? f.reprojWeightFallback : f.reprojWeight;
        auto& previous = fallback ? f.fallback : f.previous;
        if (f.reprojLinearFilter) {
            float2 w2 = make_float2(f.prevWeights[id]);
            float3 prevChannel1 = {0, 0, 0}, prevChannel2 = {0, 0, 0};

            float W[4], w = 0;
            bilinearWeights(w2, W);
            for (int i = 0; i < 4; i++) {
                int pid = f.prevLink[id + data.stride * i] & ~CoreData::Filter::FALLBACK_BIT;
                if (pid != CoreData::Filter::NO_REPROJECT) {
                    half8 prev = previous.shading_var[pid]; // --- load
                    prevChannel1 += make_float3(make_float4(prev.x)) * W[i];
                    prevChannel2 += make_float3(make_float4(prev.y)) * W[i];
                    w += W[i];
                }
            }

            if (w > 0) {
                w = 1.0f / w;
                channel1 = lerp(channel1, prevChannel1 * w, reprojWeight);
                channel2 = lerp(channel2, prevChannel2 * w, reprojWeight);
            }
        } else {
            int pid = f.prevLink[id] & ~CoreData::Filter::FALLBACK_BIT;
            if (pid != CoreData::Filter::NO_REPROJECT) {
                half8 prev = previous.shading_var[pid & ~CoreData::Filter::FALLBACK_BIT]; // --- load
                channel1 = lerp(channel1, make_float3(make_float4(prev.x)), reprojWeight);
                channel2 = lerp(channel2, make_float3(make_float4(prev.y)), reprojWeight);
            }
        }
    }

    f.shading_var[id] = {make_half4(min4(make_float4(channel1, max(0.0f, variance.x)), maxHalfFloat)),
                         make_half4(min4(make_float4(channel2, max(0.0f, variance.y)), maxHalfFloat))};
}

__global__ void finalizeAllLayersKernel(Fragment* fragments, int filter) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= *data.count || id >= data.maxFragments) return;

    // color
    float3 color;
    if (filter > 0 && data.positions[id].x >= 0) {
        auto& f = data.filter.current;

        // modulate channels with albedo(s)
        float3 albedo = RGB32toHDRmin1(__float_as_uint(f.pos_albedo[id].w));
        float3 albedo2 = albedo;
        if (__float_as_uint(f.normal_flags[id].w) & FilterData::FLAG_SEPARATED_CHANNELS)
            albedo2 = RGB32toHDRmin1(__float_as_uint(f.pos_albedo[f.count + id].w));
        half8 sh = f.shading_var[id];
        color = make_float3(make_float4(sh.x)) * albedo + make_float3(make_float4(sh.y)) * albedo2;
        if (data.filter.albedo)
            color /= RGB32toHDRmin1(data.filter.albedo[id]); // demodulate albedo of primary rays

    } else color = make_float3(data.accumulator[id] + data.accumulator[id + data.stride]) * (1.0f / data.spp);

    // hit data
    float4 h = data.primary[id];

    // store
    auto& o = fragments[id];
    o.color = make_float4(color, 0);
    o.instID = __float_as_int(h.y);
    o.primID = __float_as_int(h.z);
    o.depth = h.w;
    o.next = data.links[id];
}
