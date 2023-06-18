/* pathtracer.cu - Copyright 2019/2020 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   This file implements the shading stage of the wavefront algorithm.
   It takes a buffer of hit results and populates a new buffer with
   extension rays. Shadow rays are added with 'potential contributions'
   as fire-and-forget rays, to be traced later. Streams are compacted
   using simple atomics. The kernel is a 'persistent kernel': a fixed
   number of threads fights for food by atomically decreasing a counter.

   The implemented path tracer is deliberately simple.
   This file is as similar as possible to the one in OptixPrime_B.
*/

#include <core_settings.h>
#include "noerrors.h"

#ifdef __CLION_IDE__ //for clion preprocessor only, TODO remove
#define LH2_DEVFUNC
#define __launch_bounds__(x,y)
#define isfinite(x)
CoreData data;
#endif

LH2_DEVFUNC void PackFilterChannel(int sample, int id, uint& flags, const float3& pos, const float3& normal, float depth,
								   const float3& albedo, bool emissive = false) {
	if (flags & S_CHANNEL_PACKED) return;
	flags |= S_CHANNEL_PACKED;
	if (sample > 1 || (sample == 1 && !(flags & S_SEPARATED_CHANNELS))) return;
	if (sample > 1) return;
	int filterFlags = 0;
	if (flags & S_VIASPECULAR) filterFlags |= FilterData::FLAG_SPECULAR;
	if (flags & S_SEPARATED_CHANNELS) filterFlags |= FilterData::FLAG_SEPARATED_CHANNELS;
	if (flags & S_REFLECTION_REFRACTION) filterFlags |= FilterData::FLAG_REFLECTION_REFRACTION;
	if (emissive) filterFlags |= FilterData::FLAG_EMISSIVE;
	if (flags & S_SEPARATED_CHANNELS && flags & S_SECOND_CHANNEL) id += data.filter.current.count;
	data.filter.current.normal_flags[id] = make_float4(normal, __int_as_float(filterFlags));
    data.filter.current.pos_albedo[id] = make_float4(pos, __uint_as_float(HDRtoRGB32(albedo)));
}

LH2_DEVFUNC void PackFilterAlbedo(int sample, int id, uint& flags, const float3& albedo) {
    if (flags & S_ALBEDO_PACKED) return;
	flags |= S_ALBEDO_PACKED;
	if (sample == 0 && data.filter.albedo)
		data.filter.albedo[id] = HDRtoRGB32(albedo);
}

//  +-----------------------------------------------------------------------------+
//  |  shadeKernel                                                                |
//  |  Implements the shade phase of the wavefront path tracer.             LH2'19|
//  +-----------------------------------------------------------------------------+
__global__ void shadeKernel(int sample, int pathLength, int pathCount, int R0) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= pathCount) return;
	uint flags, fid, prevNormalPacked = 0;
	short2 pixel;
	float3 D, O, throughput = make_float3(1);
	float4 hitData;
	float bsdfPdf = 1;
	if (pathLength == 1) {
		fid = id;
		flags = (fid << S_ID_BITSHIFT);
		pixel = data.positions[fid];
		O = data.view.pos;
		D = data.viewDir(pixel.x, pixel.y, data.face(pixel.x, pixel.y));
		hitData = data.primary[id];
	} else {
		// gather data by reading sets of four floats for optimal throughput
		auto src = data.temp.paths[0].Ptr();

		float4 O4 = src[id];// ray origin, flags
		O = make_float3(O4);
		flags = __float_as_uint(O4.w);
		fid = flags >> S_ID_BITSHIFT;
		pixel = data.positions[fid];

		float4 D4 = src[id + data.stride]; // ray direction, packed normal of previous vertex
		D = make_float3(D4);
		prevNormalPacked = __float_as_uint(D4.w);

		float4 T4 =  src[id + data.stride * 2]; // path throughput, bsdfPdf
		throughput = make_float3(T4);
		bsdfPdf = T4.w;

		hitData = src[id + data.stride * 3];
	};
	if (pixel.x < 0 && !data.storeBackground) return;

	bool primaryRay = !(flags & S_ALBEDO_PACKED);
	auto dst = data.temp.paths[1].Ptr();
	bool hasLights = lightCounts.x + lightCounts.y + lightCounts.z + lightCounts.w > 0;
	float4& accumulator = data.accumulator[fid + (flags & S_SECOND_CHANNEL ? data.stride : 0)];

	float hitU  = (__float_as_uint(hitData.x) & 65535) * (1.0f / 65535.0f);
	float hitV  = (__float_as_uint(hitData.x) >> 16) * (1.0f / 65535.0f);
	int instID = __float_as_int(hitData.y);
	int primID = __float_as_int(hitData.z);
	float hitT = hitData.w;
	bool nornd = false; // for testing

	// use skydome if we didn't hit any geometry
	if (instID == NOHIT) {
		float3 contribution = {0, 0, 0};
		if (bsdfPdf > 0 && (data.storeBackground || !primaryRay)) { // do not store background in layers
			contribution = throughput * SampleSkydome(-worldToSky.TransformVector(D)) * (1.0f / bsdfPdf);
			data.clamp(contribution);
			FIXNAN_FLOAT3(contribution);
			accumulator += make_float4(contribution, 0);
		}
		PackFilterAlbedo(sample, fid, flags, contribution);
		PackFilterChannel(sample, fid, flags, O + 1e3f * D, D * -1.0f, hitT, contribution, true);
		return;
	}

	// get shadingData and normals
	CoreTri4* instanceTriangles = (CoreTri4*)instanceDescriptors[instID].triangles;
	ShadingData shadingData;
	float3 N, iN, fN, T;
	const float3 I = O + hitT * D;
	const float coneWidth = data.view.spreadAngle * hitT;
	GetShadingData(D, hitU, hitV, coneWidth, instanceTriangles[primID], instID, shadingData, N, iN, fN, T, -1, !data.disableAlphaMask);
	float faceDir = (dot(D, N) > 0) ? -1 : 1;
	if (faceDir == 1) shadingData.transmittance = make_float3(0);

	// reflect direction for backfaced triangles on primary rays to get better shading
	float3 fD = faceDir == -1 && primaryRay ? reflect(D, N) : D;

	// we need to detect alpha in the shading code.
	if (shadingData.flags & ALPHA) {
		if (pathLength == data.maxPathLength) {
			// it ends here, so store something sensible (otherwise alpha doesn't reproject)
			PackFilterAlbedo(sample, fid, flags, shadingData.color);
			PackFilterChannel(sample, fid, flags, I, fN, hitT, shadingData.color);
		} else {
			int eid = atomicAdd(data.temp.count.Ptr(), 1);
			dst[eid] = make_float4(I + D * data.geometryEpsilon, __uint_as_float(flags));
			dst[eid + data.stride] = make_float4(D, __uint_as_float(prevNormalPacked));
			FIXNAN_FLOAT3(throughput);
			dst[eid + data.stride * 2]  = make_float4(throughput, bsdfPdf);
		}
		return;
	}

    PackFilterAlbedo(sample, fid, flags, shadingData.color);

	// stop on light
	if (shadingData.IsEmissive()) {
		float3 contribution = make_float3(0);
		if (faceDir > 0 || primaryRay) { // lights are not double sided
			if (primaryRay || flags & S_SPECULAR || !hasLights)
				contribution = shadingData.color; // accept light contribution if previous vertex was specular
			else {
				// last vertex was not specular: apply MIS
				const float3 lastN = UnpackNormal(__float_as_uint(prevNormalPacked));
				const CoreTri& tri = (const CoreTri&)instanceTriangles[primID];
				const float lightPdf = CalculateLightPDF(fD, hitT, tri.area, N);
				const float pickProb = LightPickProb(tri.ltriIdx, O, lastN, I /* the N at the previous vertex */);
				if ((bsdfPdf + lightPdf * pickProb) > 0)
					contribution = throughput * shadingData.color * (1.0f / (bsdfPdf + lightPdf * pickProb));
			}
			data.clamp(contribution);
			if (flags & S_CHANNEL_PACKED)
				// ignore factor to get nicer direct look, otherwise it would be often fully white (normally should be post-processed using tone mapping)
				contribution *= data.emissiveFactor;
			FIXNAN_FLOAT3(contribution);
			accumulator += make_float4(contribution, 0);
		}
		PackFilterChannel(sample, fid, flags, I, fN, hitT, shadingData.color, true);
		return;
	}

	// apply postponed bsdf pdf
	if (bsdfPdf == 0) throughput = make_float3(0);
	else throughput *= 1.0f / bsdfPdf;

	// path regularization
	if (data.pathRegularization && flags & S_CHANNEL_PACKED)
		shadingData.parameters.x |= 255u << 24; // set roughness to 1 after a bounce.

	// switch to second channel for indirect illumination
	if (flags & S_COMBINED_CHANNELS && !(flags & S_REFLECTION_REFRACTION))
		flags |= S_SECOND_CHANNEL;

	// initialize randomness
	int blueSlot = data.blueSlot + sample;
	uint seed = WangHash((pixel.y * data.scrsize.x + pixel.x) * 17 + R0);
	if (nornd) seed = blueSlot = 1;

	// detect specular surfaces
	bool specular = ROUGHNESS <= 0.001f && METALLIC >= 0.999f;

	// non-PBR deterministic light source sampling
    if (data.deterministicLight > 0) {
        float pickProb, lightPdf = 0;
        uint seed2 = WangHash(17 + sample);
        const float r0 = RandomFloat(seed2);
        const float r1 = RandomFloat(seed2);
        float3 lightColor, L = RandomPointOnLight(r0, r1, I, fN, pickProb, lightPdf, lightColor) - I;
        const float dist = length(L);
        L *= 1.0f / dist;
        const float NdotL = dot(L, fN);
        float3 color = data.demodulateAlbedo ? make_float3(1) : shadingData.color;
        accumulator += make_float4(color * lightColor * data.emissiveFactor * (NdotL * 0.5f + 0.5f) * data.deterministicLight, 0);
        if (NdotL > 0) {
            // add fire-and-forget shadow ray to the connections buffer
            int sid = atomicAdd(data.temp.shadowCount(), 1);
            data.temp.shadow[sid] = make_float4(SafeOrigin(I, L, N, data.geometryEpsilon), __uint_as_float(flags));
            data.temp.shadow[sid + data.stride] = make_float4(L, dist - 2 * data.geometryEpsilon);
            data.temp.shadow[sid + data.stride * 2] = make_float4(color * lightColor * data.emissiveFactor * NdotL * (1.0f - data.deterministicLight), 0);
        }
        return;
    }

	// next event estimation: connect eye path to light
	if (!specular && hasLights) { // skip for specular vertices
		float pickProb, lightPdf = 0;
		float r0 = blueNoiseSampler(data.blueNoise.Ptr(), pixel.x, pixel.y, blueSlot, 1 + 4 * pathLength);
		float r1 = blueNoiseSampler(data.blueNoise.Ptr(), pixel.x, pixel.y, blueSlot, 2 + 4 * pathLength);
		if (nornd) r0 = r1 = 0.5;
		float3 lightColor, L = RandomPointOnLight(r0, r1, I, fN, pickProb, lightPdf, lightColor) - I;
		const float dist = length(L);
		L *= 1.0f / dist;
		const float NdotL = dot(L, fN);
		if (NdotL > 0 && lightPdf > 0) {
			float neeBsdfPdf;
			float3 sampledBSDF = EvaluateBSDF(shadingData, fN, T, fD * -1.0f, L, neeBsdfPdf);
			if (neeBsdfPdf > 0) {
				// calculate potential contribution
				#ifdef BSDF_HAS_PURE_SPECULARS // see note in lambert.h
				sampledBSDF *= ROUGHNESS;
				#endif
				float3 contribution = throughput * sampledBSDF * lightColor * (NdotL / (pickProb * lightPdf + neeBsdfPdf)) * data.emissiveFactor;
				data.clamp(contribution);
				FIXNAN_FLOAT3(contribution);
				// add fire-and-forget shadow ray to the connections buffer
                int sid = atomicAdd(data.temp.shadowCount(), 1);
				data.temp.shadow[sid] = make_float4(SafeOrigin(I, L, N, data.geometryEpsilon), __uint_as_float(flags));
				data.temp.shadow[sid + data.stride] = make_float4(L, dist - 2 * data.geometryEpsilon);
				data.temp.shadow[sid + data.stride * 2] = make_float4(contribution, 0);
			}
		}
	}

	// depth cap
	if (pathLength == data.maxPathLength) {
        PackFilterChannel(sample, fid, flags, I, fN, hitT, shadingData.color);
        return;
    }

	bool refraction = TRANSMISSION >= 0.001f;
	bool sharpRefraction = specular && refraction;
    bool setChannelFromSpecular = false;

	// found non-deterministic vertex (non-specular or specular refraction with reflection)
	if (!specular || refraction) {
	    if ((flags & (S_COMBINED_CHANNELS | S_SEPARATED_CHANNELS)) == 0) {
	    	// first non-deterministic vertex - determine mode of channels for the rest of path
	    	if (refraction) {
				setChannelFromSpecular = true;
				flags |= S_REFLECTION_REFRACTION | (sharpRefraction ? S_SEPARATED_CHANNELS : S_COMBINED_CHANNELS);
	    	}
            else flags |= S_COMBINED_CHANNELS;
        }
	}

	// for S_SEPARATED_CHANNELS: force refraction/reflection in part of path until the second non-deterministic vertex is found
	int forceRefraction = 0; // >0: force refraction, <0: force reflection (forbid refraction)
	float forceRefractionPdf = -1;
    bool allowSpecularFallback = false;
	if (refraction && !(flags & S_CHANNEL_PACKED)) {
		if (setChannelFromSpecular && sample < 2) {
			forceRefraction = sample == 1 ? 1 : -1;
			if (forceRefraction > 0) flags |= S_SECOND_CHANNEL;
			setChannelFromSpecular = false;
			forceRefractionPdf = 0.5; // we have exactly 1 forced refraction and 1 forced reflection
		} else if (sharpRefraction) {
			allowSpecularFallback = true;
			if (!setChannelFromSpecular)
				forceRefraction = flags & S_SECOND_CHANNEL ? 1 : -1; // keep path only reflected or refracted but not both
		}
	}

	// evaluate bsdf to obtain direction for next path segment
	bool reflection = false;
	float3 R = {0, 0, 0};
	float newBsdfPdf = 0;
	float r3 = blueNoiseSampler(data.blueNoise.Ptr(), pixel.x, pixel.y, blueSlot, 3 + 4 * pathLength);
	float r4 = blueNoiseSampler(data.blueNoise.Ptr(), pixel.x, pixel.y, blueSlot, 4 + 4 * pathLength);
	float r5 = RandomFloat(seed);
	if (nornd) r3 = r4 = r5 = 0.5;
    float3 bsdf = SampleBSDF(shadingData, fN, N, T, fD * -1.0f, hitT, r3, r4, r5, R, newBsdfPdf, reflection, false, forceRefraction, forceRefractionPdf);
    if (newBsdfPdf == 0 && allowSpecularFallback) // create reflection when refraction cannot be created and vice versa
		bsdf = SampleBSDF(shadingData, fN, N, T, fD * -1.0f, hitT, r3, r4, r5, R, newBsdfPdf, reflection, false, -forceRefraction, forceRefractionPdf);
	if (isnan(newBsdfPdf)) newBsdfPdf = 0;
    if (setChannelFromSpecular && !reflection) flags |= S_SECOND_CHANNEL;

    if (!specular) {
    	bool end = newBsdfPdf < EPSILON;

		// russian roulette
		if (!end && pathLength >= data.minPathLength) {
			const float p = SurvivalProbability(bsdf);
			if (p <= RandomFloat(seed)) end = true;
			else throughput /= p;
		}

		// pack filter data
		if (end || !sharpRefraction || flags & S_COMBINED_CHANNELS)
			PackFilterChannel(sample, fid, flags, I, fN, hitT, shadingData.color);

		if (end) return;
	}

	// write extension ray
	if (specular) flags |= S_SPECULAR | S_VIASPECULAR;
	else flags &= ~S_SPECULAR;
	int eid = atomicAdd(data.temp.count.Ptr(), 1);
	dst[eid] = make_float4(SafeOrigin(I, R, N, data.geometryEpsilon), __uint_as_float(flags));
	dst[eid + data.stride] = make_float4(R, __uint_as_float(PackNormal(fN)));
	FIXNAN_FLOAT3(throughput);
	dst[eid + data.stride * 2] = make_float4(throughput * bsdf * abs(dot(fN, R)), newBsdfPdf);
}
