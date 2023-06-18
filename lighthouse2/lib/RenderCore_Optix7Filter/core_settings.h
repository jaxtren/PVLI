/* core_settings.h - Copyright 2019/2020 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   The settings and classes in this file are core-specific:
   - available in host and device code
   - specific to this particular core.
   Global settings can be configured shared.h.
*/

#pragma once

#define STRINGIZE(x) #x
#define ERROR "[ERROR on line " STRINGIZE(__LINE__) " of " __FILE__ "]"

// core-specific settings
// #define CONSISTENTNORMALS	// consistent normal interpolation

// low-level settings
#define BLUENOISE			// use blue noise instead of uniform random numbers
#define BILINEAR			// enable bilinear interpolation
// #define NOTEXTURES		// all texture reads will be white

#define APPLYSAFENORMALS	if (dot( N, wi ) <= 0) pdf = 0;
#define NOHIT				-1

#ifndef __CUDACC__

#define CUDABUILD			// signal system.h to include full CUDA headers
#include "helper_math.h"	// for vector types
#include "platform.h"
#undef APIENTRY				// get rid of an anoying warning
#include "cuda.h"
#include "cuda_gl_interop.h"
#include "nvrtc.h"
#include "FreeImage.h"		// for loading blue noise
#include "shared_host_code/cudatools.h"
#include "shared_host_code/interoptexture.h"

#include <optix.h>
#include <optix_stubs.h>

const char* ParseOptixError( OptixResult r );
#define CHK_OPTIX( stmt ) FATALERROR_IN_CALL( ( stmt ), ParseOptixError, "" )
#define CHK_OPTIX_LOG( stmt ) FATALERROR_IN_CALL( ( stmt ), ParseOptixError, "\n%s", log )

using namespace lighthouse2;

#include "core_mesh.h"

using namespace lh2core;

#else

#include <optix.h>
#include "core_buffer.h"

#endif

// internal material representation
struct CUDAMaterial
{
#ifndef OPTIX_CU
	void SetDiffuse( float3 d ) { diffuse_r = d.x, diffuse_g = d.y, diffuse_b = d.z; }
	void SetTransmittance( float3 t ) { transmittance_r = t.x, transmittance_g = t.y, transmittance_b = t.z; }
	struct Map { short width, height; half uscale, vscale, uoffs, voffs; uint addr; };
	// data to be read unconditionally
	half diffuse_r, diffuse_g, diffuse_b, transmittance_r, transmittance_g, transmittance_b; uint flags;
	uint4 parameters; // 16 Disney principled BRDF parameters, 0.8 fixed point
	// texture / normal map descriptors; exactly 128-bit each
	Map tex0, tex1, nmap0, nmap1, smap, rmap;
#endif
};

struct CUDAMaterial4
{
	uint4 baseData4;
	uint4 parameters;
	uint4 t0data4;
	uint4 t1data4;
	uint4 n0data4;
	uint4 n1data4;
	uint4 sdata4;
	uint4 rdata4;
	// flag query macros
#define ISDIELECTRIC				(1 << 0)
#define DIFFUSEMAPISHDR				(1 << 1)
#define HASDIFFUSEMAP				(1 << 2)
#define HASNORMALMAP				(1 << 3)
#define HASSPECULARITYMAP			(1 << 4)
#define HASROUGHNESSMAP				(1 << 5)
#define ISANISOTROPIC				(1 << 6)
#define HAS2NDNORMALMAP				(1 << 7)
#define HAS2NDDIFFUSEMAP			(1 << 9)
#define HASSMOOTHNORMALS			(1 << 11)
#define HASALPHA					(1 << 12)
#define HASMETALNESSMAP				(1 << 13)
#define MAT_ISDIELECTRIC			(flags & ISDIELECTRIC)
#define MAT_DIFFUSEMAPISHDR			(flags & DIFFUSEMAPISHDR)
#define MAT_HASDIFFUSEMAP			(flags & HASDIFFUSEMAP)
#define MAT_HASNORMALMAP			(flags & HASNORMALMAP)
#define MAT_HASSPECULARITYMAP		(flags & HASSPECULARITYMAP)
#define MAT_HASROUGHNESSMAP			(flags & HASROUGHNESSMAP)
#define MAT_ISANISOTROPIC			(flags & ISANISOTROPIC)
#define MAT_HAS2NDNORMALMAP			(flags & HAS2NDNORMALMAP)
#define MAT_HAS2NDDIFFUSEMAP		(flags & HAS2NDDIFFUSEMAP)
#define MAT_HASSMOOTHNORMALS		(flags & HASSMOOTHNORMALS)
#define MAT_HASALPHA				(flags & HASALPHA)
#define MAT_HASMETALNESSMAP			(flags & HASMETALNESSMAP)
};

// path state flags
#define S_SPECULAR              1   // previous path vertex was specular
#define S_VIASPECULAR           2   // path has seen at least one specular vertex
#define S_SECOND_CHANNEL        4   // store illumination and filter data to second channel
#define S_COMBINED_CHANNELS     8   // filter data are combined for both channels and stored only once in first channel
#define S_SEPARATED_CHANNELS    16  // both channels have separated filter data
#define S_REFLECTION_REFRACTION 32  // channels contain reflection and refraction instead of direct and indirect illumination
#define S_ALBEDO_PACKED         64  // albedo already packed
#define S_CHANNEL_PACKED        128 // first/second channel (determined by S_SECOND_CHANNEL) already packed
#define S_ID_BITSHIFT           8   // bit shift of fragment/path id

#ifdef __CUDACC__
#define __ALIGN__(n) __align__(n)
#else
#define __ALIGN__(n) alignas(n)
#endif

// helper for half, TODO move to common libraries or replace

#ifdef OPTIX_CU //FIXME cannot include cuda_fp16.h

using __half = unsigned short;
struct __ALIGN__(4) half2 {
    __half x, y;
};

#else
#include <cuda_fp16.h>
#endif

struct __ALIGN__(8) half4 {
    __half x, y, z, w;
};

#ifndef OPTIX_CU
__host__ __device__ inline half4 make_half4 (const float4& v) {
    return {__float2half(v.x), __float2half(v.y), __float2half(v.z), __float2half(v.w)};
}
__host__ __device__ inline float4 make_float4 (const half4& v) {
    return {__half2float(v.x), __half2float(v.y), __half2float(v.z), __half2float(v.w)};
}

__host__ __device__ inline float make_float (__half v) { return __half2float(v); }
__host__ __device__ inline __half make_half (float v) { return __float2half(v); }

__host__ __device__ inline half2 make_half2 (const float2& v) {
    return {__float2half(v.x), __float2half(v.y)};
}
__host__ __device__ inline float2 make_float2 (const half2& v) {
    return {__half2float(v.x), __half2float(v.y)};
}
#endif

struct __ALIGN__(16) half8 {
    half4 x, y;
};

struct ViewDirection {
    float3 p1, p2, p3;
};

/**
 * Computes location in cubemap buffer for specified face and position
 * position can have coordinates outside of face in range (-size, 2*size), then it returns location to adjacent faces,
 * expect for corners, where it returns {-1, -1},
 * position outsize of allowed range returns incorrect location
 *
 * @param position
 * @param face face index
 * @param size face size
 * @return location
 */
__host__ __device__ inline int2 cubemapLocation(int2 position, int face, int2 size) {
    size.x /= 6;

    // corners
    if ((position.x < 0 || position.x >= size.x) &&
        (position.y < 0 || position.y >= size.y)) return {-1, -1};

    const int ROT0 = 0;
    const int ROT90 = 1;
    const int ROT180 = 2;
    const int ROT270 = 3;

    int2 mode = make_int2(face, ROT0);
    if (position.y < 0) { // bellow
        switch(face) {
            case 0: mode = {5, ROT270}; break;
            case 1: mode = {5, ROT0}; break;
            case 2: mode = {5, ROT90}; break;
            case 3: mode = {5, ROT180}; break;
            case 4: mode = {1, ROT0}; break;
            case 5: mode = {3, ROT180}; break;
            default: return {-1, -1};
        }
    } else if(position.y >= size.y) { // above
        switch(face) {
            case 0: mode = {4, ROT90}; break;
            case 1: mode = {4, ROT0}; break;
            case 2: mode = {4, ROT270}; break;
            case 3: mode = {4, ROT180}; break;
            case 4: mode = {3, ROT180}; break;
            case 5: mode = {1, ROT0}; break;
            default: return {-1, -1};
        }
    }
    else if(face < 4) { // cycle left/right
        if (position.x < 0) mode.x = (face - 1 + 4) % 4;
        else if (position.x > size.x) mode.x = (face + 1) % 4;
    } else if (face == 4) { // top
        if (position.x < 0) mode = {0, ROT270}; // left
        else if (position.x >= size.x) mode = {2, ROT90}; // right
    } else if (face == 5) { // bottom
        if (position.x < 0) mode = {0, ROT90}; // left
        else if (position.x >= size.x) mode = {2, ROT270}; // right
    } else return {-1, -1};

    int x = (position.x + size.x) % size.x;
    int y = (position.y + size.y) % size.y;
    int f = mode.x * size.x;
    int X = size.x - 1 - x, Y = size.y - 1 - y;
    switch (mode.y) {
        case ROT0:   return {f + x, y};
        case ROT90:  return {f + y * size.x / size.y, X * size.y / size.x};
        case ROT180: return {f + X, Y};
        case ROT270: return {f + Y * size.x / size.y, x * size.y / size.x};
        default: return {-1, -1};
    }
}

/**
 * Sample cubemap using direction vector
 * @param direction
 * @param face output face index
 * @param size face size
 * @return position on returned face
 */
__host__ __device__ inline float2 sampleCubemap(const float3& direction, int& face, int2 size) {
    auto& v = direction;
    float m = max(fabs(v.x), max(fabs(v.y), fabs(v.z)));
    float2 p = {0, 0};
    face = -1;
    if (v.x == m)       { p = { v.z, v.y}; face = 2; }
    else if (v.x == -m) { p = {-v.z, v.y}; face = 0; }
    else if (v.y ==  m) { p = { v.x, v.z}; face = 4; }
    else if (v.y == -m) { p = { v.x,-v.z}; face = 5; }
    else if (v.z ==  m) { p = {-v.x, v.y}; face = 3; }
    else if (v.z == -m) { p = { v.x, v.y}; face = 1; }
    else return {-1, -1};
    return (p / m + 1.0f) * 0.5f * make_float2(size);
}

struct FilterData {
    CoreBuffer<half8> shading_var;
    CoreBuffer<float4> moments;
    CoreBuffer<float4> pos_albedo; // 2 * count
    CoreBuffer<float4> normal_flags; // 2 * count
    CoreBuffer<float> depth; // 2 * count
    CoreBuffer<int> links;
    CoreBuffer<unsigned char> counts;
    ViewPyramid view;
    ViewDirection views[6]; // for cubemap
    int2 scrsize;
    float2 subpixelOffset;
    float2 evenPixelsOffset;
    mat4 projection; // for reprojection
    int count;
    bool reordered;
    bool cubemap;
    bool firstLayerOnly;

    enum : unsigned int {
        FLAG_SPECULAR = 1,
        FLAG_EMISSIVE = 2,
        FLAG_SEPARATED_CHANNELS = 4, // pos_albedo, normal_Flags and depth are separated for both channels
        FLAG_REFLECTION_REFRACTION = 8, // channels contain reflection and refraction instead of direct and indirect illumination
        FLAGS_MASK = 16-1
    };

    static const unsigned int REPROJ_COUNT_SHIFT = 4;
    static const unsigned int REPROJ_COUNT_MAX = 0xffffffffu >> REPROJ_COUNT_SHIFT;

#ifndef __CUDACC__
    FilterData() {
        scrsize = {0, 0};
        count = 0;
        reordered = false;
        cubemap = false;
        firstLayerOnly = false;
	}
#endif

    __host__ __device__ inline int closestFragmentDepth(int dst, float dist, float& distOut, int dataOffset = 0) const {
        float de = distOut = depth[dataOffset + dst]; // --- load
        float D = fabs(de - dist);
        if (firstLayerOnly) return dst;
        if (reordered) {
            for (int id = links[dst], e = id + counts[dst] - 1; id < e; id++) { // -- one load
                de = depth[dataOffset + id]; // --- load
                float d = fabs(de - dist);
                if (d < D) {
                    distOut = de;
                    D = d;
                    dst = id;
                }
            }
        } else {
            for (int id = dst; id >= 0; id = links[id]) {
                de = depth[dataOffset + id]; // --- load
                float d = fabs(de - dist);
                if (d < D) {
                    distOut = de;
                    D = d;
                    dst = id;
                }
            }
        }
        return dst;
    }

    __host__ __device__ inline int closestFragmentPos(const float3& pos, const float3& normal, int dst, float offset,
                                                      float3& posOut, float& offsetOut, int dataOffset = 0) const {
        posOut = make_float3(pos_albedo[dataOffset + dst]); // --- load
        offsetOut = fabs(dot(posOut - pos, normal) + offset);
        if (firstLayerOnly) return dst;
        if (reordered) {
            for (int id = links[dst], e = id + counts[dst] - 1; id < e; id++) { // -- one load
                float3 pos2 = make_float3(pos_albedo[dataOffset + id]); // --- load
                float o = fabs(dot(pos2 - pos, normal) + offset);
                if (o < offsetOut) {
                    offsetOut = o;
                    posOut = pos2;
                    dst = id;
                }
            }
        } else {
            for (int id = dst; id >= 0; id = links[id]) {
                float3 pos2 = make_float3(pos_albedo[dataOffset + id]); // --- load
                float o = fabs(dot(pos2 - pos, normal) + offset);
                if (o < offsetOut) {
                    offsetOut = o;
                    posOut = pos2;
                    dst = id;
                }
            }
        }
        return dst;
    }

    // first argument of reproject functions is global position of point

    __host__ __device__ inline float2 applyPixelOffset(const float2& pixel) {
        int x = ((int)pixel.x) | 1, y = ((int)pixel.y) | 1; // find closest even pixel (odd integers)
        float2 distance = fabs(pixel - make_float2((float)x, (float)y));
        return pixel - evenPixelsOffset * (make_float2(1, 1) - make_float2(distance.y, distance.x)); // subtract offset with linear interpolation
    }

    __host__ __device__ inline float2 sampleCubemap(const float3& direction, int& face) {
        return applyPixelOffset(::sampleCubemap(direction, face, {scrsize.x / 6, scrsize.y}) - subpixelOffset) + 0.5f;
    }

    __host__ __device__ inline float2 reprojectPerspective(const float3& v) {
        auto point = projection * make_float4(v, 1);
        if (point.w < 0) return {-1, -1};
        return applyPixelOffset((make_float2(point.x, point.y) / point.w + 1.0f) * 0.5f * make_float2(scrsize)) + 0.5f;
    }

    __host__ __device__ inline int reprojectNearest(const float3& v) {
        if (count <= 0) return -1;
        if (cubemap) {
            int face, w = scrsize.x / 6;
            auto pixel = make_int2(sampleCubemap(projection.TransformPoint(v), face));
            if (face >= 0 && pixel.x >= 0 && pixel.y >= 0 && pixel.x < w && pixel.y < scrsize.y)
                return pixel.y * scrsize.x + pixel.x + face * w;
        } else {
            auto pixel = make_int2(reprojectPerspective(v));
            if (pixel.x >= 0 && pixel.y >= 0 && pixel.x < scrsize.x && pixel.y < scrsize.y)
                return pixel.y * scrsize.x + pixel.x;
        }
        return -1;
    }

    __host__ __device__ inline bool inside(int x, int y) {
        return x >= 0 && y >= 0 && x < scrsize.x && y < scrsize.y;
    }

    __host__ __device__ inline float2 reprojectLinear(const float3& v, int* ids) {
        ids[0] = ids[1] = ids[2] = ids[3] = -1;
        if (count <= 0) return {0, 0};
        if (cubemap) {
            int face;
            auto pos = sampleCubemap(projection.TransformPoint(v), face) - 0.5f;
            auto p = make_int2(pos);
            auto loc = cubemapLocation({p.x, p.y}, face, scrsize);
            if (loc.x >= 0) ids[0] = loc.y * scrsize.x + loc.x;
            loc = cubemapLocation({p.x + 1, p.y}, face, scrsize);
            if (loc.x >= 0) ids[1] = loc.y * scrsize.x + loc.x;
            loc = cubemapLocation({p.x, p.y + 1}, face, scrsize);
            if (loc.x >= 0) ids[2] = loc.y * scrsize.x + loc.x;
            loc = cubemapLocation({p.x + 1, p.y + 1}, face, scrsize);
            if (loc.x >= 0) ids[3] = loc.y * scrsize.x + loc.x;
            return {pos.x - floor(pos.x), pos.y - floor(pos.y)};
        } else {
            auto pos = reprojectPerspective(v) - 0.5f;
            auto p = make_int2(pos);
            if (inside(p.x, p.y)) ids[0] = p.y * scrsize.x + p.x;
            if (inside(p.x + 1, p.y)) ids[1] = p.y * scrsize.x + p.x + 1;
            if (inside(p.x, p.y + 1)) ids[2] = (p.y + 1) * scrsize.x + p.x;
            if (inside(p.x + 1, p.y + 1)) ids[3] = (p.y + 1) * scrsize.x + p.x + 1;
            return {pos.x - floor(pos.x), pos.y - floor(pos.y)};
        }
    }
};

struct ViewData {
    FilterData filter;
    uint RNGseed = 0x12345678;
    int blueNoiseSlot = 0;
};

struct CoreData {

	/* Path States:
	 *   origin, path/fragment ID | flags
	 *   direction, previous normal
	 *   throughput, bsdfPdf
	 *   hit data: bary, inst, prim, depth
	 *
	 * Shadow Rays:
	 *   origin, path/fragment ID | flags
	 *   direction, max depth
	 *   contribution
	 *
	 * Cubemap View:
	 *   six views of cubemap in row: left, front, right, back, top, bottom
	 *   expected:
	 *     output size: { scrsize.x * 6, scrsize.y }
	 */

	#ifndef __CUDACC__
	CoreData() {
		spp = 1;
		minPathLength = 1;
		maxPathLength = 100;
		maxLayers = 100;
		blueSlot = 0;
		primaryRayFlags = 0;
		pathRegularization = 0;
		geometryEpsilon = 0;
		clampValue = 10;
		emissiveFactor = 1;
		skipLayers = 0;
		subpixelOffset = make_float2(0.5f, 0.5f);
		evenPixelsOffset = make_float2(0, 0);
		filter.directClamp = filter.indirectClamp = filter.reflectionClamp = filter.refractionClamp = 1e24f;
		primMask = 0;
		deterministicLight = -1;
		storeBackground = false;
		demodulateAlbedo = false;
		disableAlphaMask = false;
		filter.previous.count = 0;
		filter.shadeKeepPhase = 1;
		filter.shadeMergePhase = 0;
		filter.reprojWeight = 0.8f;
		filter.reprojWeightFallback = 0.6f;
		filter.reprojSpatialCount = 4;
		filter.varianceFactor = 4;
		filter.varianceGauss = 1;
		filter.varianceReprojFactor = 10;
		filter.normalFactor = 128;
		filter.distanceFactor = 0.5f;
		filter.reprojMaxDistFactor = 10;
		filter.reprojLinearFilter = false;
		filter.closestOffset = 0;
		filter.closestOffsetMin = 0;
		filter.closestOffsetMax = 1e24f;
		filter.depthMode = 0;
		filter.reorderFragments = true;
		filter.firstLayerOnly = false;
	}
	#endif

	// data for all layers
    // first layer is stored deterministically at y*screen_width+x, other layers use linked lists
    CoreBuffer<float4> primary; // hit data of primary rays for all layers
	CoreBuffer<float4> accumulator; // direct, indirect
	CoreBuffer<short2> positions; // screen-space position of primary paths or (-1, -1) for empty pixel (only in first layer)
	CoreBuffer<int> links; // linked lists for primary rays
    CoreBuffer<unsigned char> counts; // sizes of linked lists for primary rays
	CoreBuffer<int> count;
	int stride;

	// temporary data
	struct {
        // for fragment reordering
        CoreBuffer<int> indices;
        CoreBuffer<float4> primary;

        // for secondary/shadow ray waves
		CoreBuffer<float4> paths[2]; //path states
		CoreBuffer<float4> shadow;
		CoreBuffer<int> count;
        __host__ __device__ inline int* shadowCount() { return count.Ptr() + 1; }
    } temp;

	// filter data
	struct Filter {
        FilterData current;
        FilterData previous;
        FilterData fallback;

        CoreBuffer<half8> shading_var;
        CoreBuffer<uint> albedo; // albedo of primary rays, for demodulateAlbedo
        CoreBuffer<float4> moments;
        CoreBuffer<int> prevLink; // indices (*4 when reprojLinearFilter = true) to reprojected fragment from previous frame (or -1) | FALLBACK_BIT
        CoreBuffer<half2> prevWeights; // bilinear weights to previous fragments (when reprojLinearFilter = true)

		// settings
		float directClamp, indirectClamp, reflectionClamp, refractionClamp;
        float reprojWeight, reprojWeightFallback;
		int reprojSpatialCount, shadeKeepPhase, shadeMergePhase, varianceGauss;
		float varianceFactor, normalFactor, distanceFactor, varianceReprojFactor;
		float reprojMaxDistFactor, closestOffset, closestOffsetMin, closestOffsetMax;
		int depthMode; // 1: pos + normal, 2: depth + normal (faster, but could not work with specular fragments)
        bool reorderFragments; // reorder fragments to improve memory access
        bool reprojLinearFilter; // linear filter reprojection
        bool firstLayerOnly; // search only in first layer to simulate filter without multi-layer extension

        static const int FALLBACK_BIT = 0x80000000; // use fallback instead of previous filter data
        static const int NO_REPROJECT = 0x80000000 - 1; // use fallback instead of previous filter data
	} filter;

	ViewPyramid view;
    ViewDirection views[6]; // for cubemap
	int2 scrsize;
    float2 subpixelOffset;
	float2 evenPixelsOffset;
	int spp, minPathLength, maxPathLength, maxLayers, maxFragments, blueSlot;
    int cubemap; // 0: normal perspective projected view, 1: cubemap
    int pathRegularization;
    unsigned int primaryRayFlags;
	float geometryEpsilon, clampValue, emissiveFactor;
	float skipLayers;
	float deterministicLight; // >0: enables non-PBR deterministic light sampling that can be used without filter
	                               // value determines ratio between indirect (ambient) and direct (using shadow rays) lighting
	bool storeBackground; // store background in first layer
	bool demodulateAlbedo;
	bool disableAlphaMask;

    __host__ __device__ inline float3 viewDir(int x, int y, int face) {
        if (cubemap) x -= (scrsize.x / 6) * face;
        float X = (float) x + (y & 1) * evenPixelsOffset.x;
        float Y = (float) y + (x & 1) * evenPixelsOffset.y;
        if (!cubemap) return normalize(view.p1 + X * view.p2 + Y * view.p3);
        auto& v = views[face];
        return normalize(v.p1 + X * v.p2 + Y * v.p3);
    }

	// blue noise table: contains the three tables distributed by Heitz.
	// Offset 0: an Owen-scrambled Sobol sequence of 256 samples of 256 dimensions.
	// Offset 65536: scrambling tile of 128x128 pixels; 128 * 128 * 8 values.
	// Offset 65536 * 3: ranking tile of 128x128 pixels; 128 * 128 * 8 values. Total: 320KB.
	CoreBuffer<uint> blueNoise;

	// data from client
	const int* const* primMask;

    // helper functions
    __host__ __device__ inline int face(int x, int y) const {
        return cubemap ? x / (scrsize.x / 6) : 0;
    }

	__host__ __device__ inline int index(int x, int y, int face) const {
	    if (cubemap) {
	        auto l = cubemapLocation({x - face * (scrsize.x / 6), y}, face, scrsize);
	        return l.x < 0 ? -1 : l.y * scrsize.x + l.x;
	    }
	    return x < 0 || y < 0 || x >= scrsize.x || y >= scrsize.y ? -1 : y * scrsize.x + x;
	}

	__host__ __device__ inline void clamp(float3& v) {
		float m = max(v.x, max(v.y, v.z));
		if (m > clampValue) v *= clampValue / m;
	}

	// OptiX
	enum
	{
		SPAWN_PRIMARY = 0,
        SPAWN_PRIMARY_ANYHIT,
		SPAWN_SECONDARY,
		SPAWN_SHADOW
	};
	int phase;
	OptixTraversableHandle bvhRoot;
};

#ifndef __CUDACC__
#define OPTIXU_MATH_DEFINE_IN_NAMESPACE
#define _USE_MATH_DEFINES
#include "core_api_base.h"
#include "rendercore.h"
#include <cstdint>

namespace lh2core
{

// setters / getters
void stageInstanceDescriptors( CoreInstanceDesc* p );
void stageMaterialList( CUDAMaterial* p );
void stageAreaLights( CoreLightTri* p );
void stagePointLights( CorePointLight* p );
void stageSpotLights( CoreSpotLight* p );
void stageDirectionalLights( CoreDirectionalLight* p );
void stageLightCounts( int area, int point, int spot, int directional );
void stageARGB32Pixels( uint* p );
void stageARGB128Pixels( float4* p );
void stageNRM32Pixels( uint* p );
void stageSkyPixels( float4* p );
void stageSkySize( int w, int h );
void stageWorldToSky( const mat4& worldToLight );
void stageDebugData( float4* p );
void stageMemcpy( void* d, void* s, int n );
void pushStagedCopies();

} // namespace lh2core

#include "../RenderSystem/common_bluenoise.h"

#endif

// EOF
