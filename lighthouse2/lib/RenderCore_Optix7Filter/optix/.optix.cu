/* .optix.cu - Copyright 2019/2020 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   This file contains a minimal set of Optix functions. From here we will
   dispatch program flow to our own functions that implement the path tracer.
*/

#include "../kernels/noerrors.h"
#include "helper_math.h"

// global include files
#include "../../RenderSystem/common_settings.h"
#include "../../RenderSystem/common_types.h"
#include "../../RenderSystem/common_classes.h"
#define OPTIX_CU // skip CUDAMaterial definition in core_settings.h; not needed here 
#include "../core_settings.h"

// error checking
//#define CHECK_ERRORS
#define PRINT_ERROR(str, ...) printf("ERROR:.optix.cu:%s():%d: " str "\n", __func__, __LINE__, __VA_ARGS__)

// global path tracing parameters
extern "C" { __constant__ CoreData data; }
const int SBTstride = 3;
const float tmax = 1e34f;

#if __CUDA_ARCH__ >= 700
#define THREADMASK	__activemask() // volta, turing
#else
#define THREADMASK	0xffffffff // pascal, kepler, fermi
#endif

static __device__ void tracePrimaryRays(int id) {
    auto pos = make_short2(id % data.scrsize.x, id / data.scrsize.x);

    // ray from eye
    float3 O = data.view.pos;
    float3 D = data.viewDir(pos.x, pos.y, data.face(pos.x, pos.y));

    // trace primary rays for all layers and store into linked lists
    float t = data.geometryEpsilon;
    int count = 0, fid = id;
    while (true) {
        uint u0 = 0, u1 = 0xffffffff, u2 = 0xffffffff, u3 = __float_as_uint(tmax);
        optixTrace(data.bvhRoot, O, D, t, tmax, 0.0f, OptixVisibilityMask(1),
                   data.primaryRayFlags, 0, SBTstride, 0, u0, u1, u2, u3);

        // nohit
        if (u1 == 0xffffffff) {
            data.links[id] = -1; // close linked list
            if (count == 0) {
                data.positions[id] = pos;
                data.primary[id] = make_float4(__uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2), __uint_as_float(u3));
            }
            break;
        }

        if (!data.primMask || data.primMask[u1][u2] > 0) {

            if (count > 0) {
                // allocate fragment
                int nid = atomicAdd(data.count.Ptr(), 1);
                if (nid >= data.maxFragments) {
                    data.links[id] = -1;
                    break;
                }
                data.links[id] = nid;
                id = nid;
            }

            data.positions[id] = pos;
            data.primary[id] = make_float4(__uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2), __uint_as_float(u3));
            count++;

            if (count >= data.maxLayers) {
                data.links[id] = -1;
                break;
            }

            t = __uint_as_float(u3) * (1.0f + data.skipLayers) + data.geometryEpsilon; // skip layers

        } else t = __uint_as_float(u3) + data.geometryEpsilon;
    }
    data.counts[fid] = max(1, count);
}

static __device__ void tracePrimaryRaysAnyhit(int id) {
    auto pos = make_short2(id % data.scrsize.x, id / data.scrsize.x);

    // ray from eye
    float3 O = data.view.pos;
    float3 D = data.viewDir(pos.x, pos.y, data.face(pos.x, pos.y));

    // trace primary rays for all layers and store into linked lists using any hit program
    data.counts[id] = 0;
    uint p0 = id;
    optixTrace(data.bvhRoot, O, D, data.geometryEpsilon, tmax, 0.0f, OptixVisibilityMask(1),
               data.primaryRayFlags | OPTIX_RAY_FLAG_ENFORCE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
               2, SBTstride, 0, p0);

    if (data.counts[id] == 0) {
        // no fragment in whole linked list
        data.primary[id] = make_float4(__uint_as_float(0), __uint_as_float(0xffffffff), __uint_as_float(0xffffffff), tmax);
        data.positions[id] = pos;
        data.links[id] = -1;
        data.counts[id] = 1;
    } else {
        // reverse order
        int prevID = -1, curID = data.links[id];
        while (curID >= 0) {
            int nextID = data.links[curID];
            data.links[curID] = prevID;
            prevID = curID;
            curID = nextID;
        }
        data.links[id] = prevID;

        // skip fragments
        if (data.skipLayers > 0) {
            prevID = id;
            curID = data.links[id];
            int count = 1;
            float minDist = data.primary[id].w * (1.0f + data.skipLayers);
            while (curID >= 0 && count < data.maxLayers) {
                float dist = data.primary[curID].w;
                if (dist >= minDist) { // keep
                    minDist = data.primary[curID].w * (1.0f + data.skipLayers);
                    prevID = curID;
                    curID = data.links[curID];
                    count++;
                } else { // skip
                    data.positions[curID] = make_short2(-1, -1);
                    curID = data.links[prevID] = data.links[curID];
                }
            }
            if (curID >= 0) data.links[curID] = -1;
            data.counts[id] = count;
        }
    }

    #ifdef CHECK_ERRORS
    {
        // check depth order of fragments
        int it = id, i = 0;
        while (it >= 0) {
            int it2 = data.links[it];
            if (it2 < 0) break;
            float d = data.primary[it].w;
            float d2 = data.primary[it2].w;
            if (d > d2) PRINT_ERROR("incorrect depth order id=%d i=%d d=%f/%f", id, i, d, d2);
            it = it2;
            i++;
        }
    }
    #endif
}

static __device__ void traceRay(int id) {
	auto buf = data.temp.paths[0].Ptr();
	float3 O = make_float3(buf[id]);
	float3 D = make_float3(buf[id + data.stride]);
	uint u0 = 0, u1 = 0xffffffff, u2 = 0xffffffff, u3 = __float_as_uint(tmax);
	optixTrace(data.bvhRoot, O, D, data.geometryEpsilon, tmax, 0.0f, OptixVisibilityMask(1), 0, 0, SBTstride, 0, u0, u1, u2, u3);
	buf[id + data.stride * 3] = make_float4(__uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2), __uint_as_float(u3));
}

static __device__ void traceShadowRay(int id) {
	auto O = data.temp.shadow[id];
	auto D = data.temp.shadow[id + data.stride];
	uint u0 = 1;
	optixTrace(data.bvhRoot, make_float3(O), make_float3(D), data.geometryEpsilon, D.w, 0.0f, OptixVisibilityMask( 1 ),
				OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 1, SBTstride, 1, u0 );
	if (u0) return;
	int flags =  __float_as_int(O.w);
	data.accumulator[(flags >> S_ID_BITSHIFT) + (flags & S_SECOND_CHANNEL ? data.stride : 0)] += data.temp.shadow[id + data.stride * 2];
}

extern "C" __global__ void __raygen__rg(){
	const uint3 idx = optixGetLaunchIndex();
	const uint id = idx.x + idx.y * data.scrsize.x;
	switch (data.phase)
	{
		case CoreData::SPAWN_PRIMARY: tracePrimaryRays(id); break;
        case CoreData::SPAWN_PRIMARY_ANYHIT: tracePrimaryRaysAnyhit(id); break;
		case CoreData::SPAWN_SECONDARY: traceRay(id); break;
		case CoreData::SPAWN_SHADOW: traceShadowRay(id); break;
	}
}

extern "C" __global__ void __miss__occlusion() {
	optixSetPayload_0( 0u ); // instead of any hit. suggested by WillUsher.io.
}

extern "C" __global__ void __closesthit__radiance() {
	const uint prim_idx = optixGetPrimitiveIndex();
	const uint inst_idx = optixGetInstanceIndex();
	const float2 bary = optixGetTriangleBarycentrics();
	const float tmin = optixGetRayTmax();
	optixSetPayload_0( (uint)(65535.0f * bary.x) + ((uint)(65535.0f * bary.y) << 16) );
	optixSetPayload_1( inst_idx );
	optixSetPayload_2( prim_idx );
	optixSetPayload_3( __float_as_uint( tmin ) );
}

extern "C" __global__ void __anyhit__primary() {
    const uint instID = optixGetInstanceIndex();
    const uint primID = optixGetPrimitiveIndex();
    if (data.primMask && data.primMask[instID][primID] <= 0) // ignore based on mask
        optixIgnoreIntersection();

    const int id = optixGetPayload_0();
    const float t = optixGetRayTmax();
    const float2 bary = optixGetTriangleBarycentrics();
    auto pos = make_short2(id % data.scrsize.x, id / data.scrsize.x);

    auto hitData = make_float4(
        __uint_as_float((uint)(65535.0f * bary.x) + ((uint)(65535.0f * bary.y) << 16)),
        __uint_as_float(instID), __uint_as_float(primID), t);

    auto& count = data.counts[id];
    if (count == 0) {
        // first fragment: store directly to pixel location
        data.primary[id] = hitData;
        data.positions[id] = pos;
        data.links[id] = -1;
        count = 1;
    } else {
        // find insert location to keep the list sorted by depth from furthest to closest
        // except the first fragment, which is always closest

        int pixelID = id, prevID = pixelID, nextID = data.links[pixelID];
        while (nextID >= 0) {
            float t2 = data.primary[nextID].w;
            if (t > t2) break;
            if (t == t2) {
                // check for duplicates, OptiX can call any hit program for same primitive multiple times
                // unless OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL is specified in build of accel. struct
                auto hitData2 = data.primary[nextID];
                if (__float_as_uint(hitData2.y) == instID && __float_as_uint(hitData2.z) == primID)
                    optixIgnoreIntersection();
            }
            prevID = nextID;
            nextID = data.links[nextID];
        }

        // compare with first
        bool isFirst = false;
        if (nextID == -1) {
            float t2 = data.primary[pixelID].w;
            if (t < t2) isFirst = true;
            else if (t == t2) { // check for duplicates
                auto hitData2 = data.primary[pixelID];
                if (__float_as_uint(hitData2.y) == instID && __float_as_uint(hitData2.z) == primID)
                    optixIgnoreIntersection();
            }
        }

        #ifdef CHECK_ERRORS
        {
            // check for duplicates in whole list
            int it = id, i = 0;
            while (it >= 0) {
                auto hitData2 = data.primary[it];
                if (__float_as_uint(hitData2.y) == instID && __float_as_uint(hitData2.z) == primID) {
                    PRINT_ERROR("duplicate fragment id=%d i=%d inst=%d prim=%d d=%f/%f", id, i, instID, primID, t, hitData2.w);
                    optixIgnoreIntersection();
                }
                it = data.links[it];
                i++;
            }
        }
        #endif

        // allocate fragment
        int newID;
        if (count == data.maxLayers && data.skipLayers <= 0) {
            int lastFragment = data.links[pixelID];

            // special cases
            if (isFirst && data.maxLayers <= 2) {
                if (data.maxLayers == 2)
                    data.primary[lastFragment] = data.primary[pixelID]; // move first to second
                    data.primary[pixelID] = hitData; // new fragment is first
                optixIgnoreIntersection();
            }

            if (prevID == pixelID) // new fragment is last
                optixIgnoreIntersection();

            if (prevID == lastFragment) { // replace last
                data.primary[lastFragment] = hitData;
                optixIgnoreIntersection();
            }

            // steal last fragment
            newID = lastFragment;
            data.links[pixelID] = data.links[newID];
        } else {
            // allocate new fragment
            newID = atomicAdd(data.count.Ptr(), 1);
            if (newID >= data.maxFragments) optixIgnoreIntersection();
            count++;
        }

        if (isFirst) {
            // new fragment need to be stored at pixelID: move current fragment from pixelID to new allocated memory
            data.primary[newID] = data.primary[pixelID];
            data.positions[newID] = pos;
            data.links[newID] = -1;
            data.links[prevID] = newID;
            data.primary[pixelID] = hitData; // save new fragment
        } else {
            // save new fragment
            data.primary[newID] = hitData;
            data.positions[newID] = pos;
            data.links[prevID] = newID;
            data.links[newID] = nextID;
        }
    }
    optixIgnoreIntersection();
}
