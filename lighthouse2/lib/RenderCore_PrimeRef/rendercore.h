/* rendercore.h - Copyright 2019/2020 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#pragma once

namespace lh2core
{

#define CHK_PRIME( stmt )                                                                 \
	{                                                                                     \
		RTPresult r = ( stmt );                                                           \
		if ( r )                                                                          \
		{                                                                                 \
			const char* e;                                                                \
			rtpContextGetLastErrorString( RenderCore::context, &e );                      \
			FatalError( #stmt " returned error '%s' at %s:%d\n", e, __FILE__, __LINE__ ); \
		}                                                                                 \
	}                                                                                     \
	while ( 0 )

//  +-----------------------------------------------------------------------------+
//  |  DeviceVars                                                                 |
//  |  Copy of device-side variables, to detect changes.                    LH2'19|
//  +-----------------------------------------------------------------------------+
struct DeviceVars
{
	// impossible values to trigger an update in the first frame
	float clampValue = -1.0f;
	float geometryEpsilon = 1e34f;
};

//  +-----------------------------------------------------------------------------+
//  |  RenderCore                                                                 |
//  |  Encapsulates device code.                                            LH2'19|
//  +-----------------------------------------------------------------------------+
class RenderCore : public CoreAPI_Base
{
public:
	// methods
	void Init();
	void Render( const ViewPyramid& view, const Convergence converge, bool async );
	void WaitForRender() { /* this core does not support asynchronous rendering yet */ }
	void Setting( const char* name, const float value );
	void SetTarget( GLTexture* target, const uint spp );
	void Shutdown();
	// passing data. Note: RenderCore always copies what it needs; the passed data thus remains the
	// property of the caller, and can be safely deleted or modified as soon as these calls return.
	void SetTextures( const CoreTexDesc* tex, const int textureCount );
	void SetMaterials( CoreMaterial* mat, const int materialCount ); // textures must be in sync when calling this
	void SetLights( const CoreLightTri* areaLights, const int areaLightCount,
		const CorePointLight* pointLights, const int pointLightCount,
		const CoreSpotLight* spotLights, const int spotLightCount,
		const CoreDirectionalLight* directionalLights, const int directionalLightCount );
	void SetSkyData( const float3* pixels, const uint width, const uint height, const mat4& worldToLight );
	// geometry and instances:
	// a scene is setup by first passing a number of meshes (geometry), then a number of instances.
	// note that stored meshes can be used zero, one or multiple times in the scene.
	// also note that, when using alpha flags, materials must be in sync.
	void SetGeometry( const int meshIdx, const float4* vertexData, const int vertexCount, const int triangleCount, const CoreTri* triangles );
	void SetInstance( const int instanceIdx, const int modelIdx, const mat4& transform );
	void FinalizeInstances();
	int4 GetScreenParams();
	void SetProbePos( const int2 pos );
	CoreStats GetCoreStats() const override;
	// internal methods
private:
	void UpdateToplevel();
	void SyncStorageType( const TexelStorage storage );
	// helpers
	template <class T> CUDAMaterial::Map Map( T v )
	{
		CUDAMaterial::Map m;
		CoreTexDesc& t = texDescs[v.textureID];
		m.width = t.width, m.height = t.height, m.uscale = v.uvscale.x, m.vscale = v.uvscale.y;
		m.uoffs = v.uvoffset.x, m.voffs = v.uvoffset.y, m.addr = t.firstPixel;
		return m;
	}
	// data members
	int scrwidth = 0, scrheight = 0;				// current screen width and height
	int scrspp = 1;									// samples to be taken per screen pixel
	int skywidth = 0, skyheight = 0;				// size of the skydome texture
	int maxPixels = 0;								// max screen size buffers can accomodate without a realloc
	int currentSPP = 0;								// spp count which will be accomodated without a realloc
	int2 probePos = make_int2( 0 );					// triangle picking; primary ray for this pixel copies its triid to coreStats.probedTriid
	vector<CoreMesh*> meshes;						// list of meshes, to be referenced by the instances
	vector<CoreInstance*> instances;					// list of instances: model id plus transform
	bool instancesDirty = true;						// we need to sync the instance array to the device
	InteropTexture renderTarget;					// CUDA will render to this texture
	CoreBuffer<CUDAMaterial>* materialBuffer = 0;	// material array
	CUDAMaterial* hostMaterialBuffer = 0;			// core-managed copy of the materials
	CoreBuffer<CoreLightTri>* areaLightBuffer;		// area lights
	CoreBuffer<CorePointLight>* pointLightBuffer;	// point lights
	CoreBuffer<CoreSpotLight>* spotLightBuffer;		// spot lights
	CoreBuffer<CoreDirectionalLight>* directionalLightBuffer;	// directional lights
	CoreBuffer<float4>* texel128Buffer = 0;			// texel buffer 1: hdr ARGB128 texture data
	CoreBuffer<uint>* normal32Buffer = 0;			// texel buffer 2: integer-encoded normals
	CoreBuffer<float4>* skyPixelBuffer = 0;			// skydome texture data
	RTPmodel* topLevel = 0;							// the top-level node; combines all instances and is the entry point for ray queries
	CoreBuffer<float4>* accumulator = 0;			// accumulator buffer for the path tracer
	CoreBuffer<Counters>* counterBuffer = 0;		// counters for persistent threads
	CoreBuffer<CoreInstanceDesc>* instDescBuffer = 0; // instance descriptor array
	CoreBuffer<uint>* texel32Buffer = 0;			// texel buffer 0: regular ARGB32 texture data
	CoreBuffer<Ray4>* extensionRayBuffer[2] = { 0, 0 }; // buffer for OptiX extension ray data
	CoreBuffer<float4>* extensionRayExBuffer[2] = { 0, 0 };	// additional path state data
	CoreBuffer<Intersection>* extensionHitBuffer = 0; // buffer for OptiX intersection results
	CoreBuffer<Ray4>* shadowRayBuffer = 0;			// buffer for OptiX shadow ray data
	CoreBuffer<float4>* shadowRayPotential = 0;		// potential throughput for shadow rays
	CoreBuffer<uint>* shadowHitBuffer = 0;			// buffer for OptiX intersection results, 1 bit per shadow ray
	RTPbufferdesc extensionRaysDesc[2];				// buffer descriptor for extension rays
	RTPbufferdesc extensionHitsDesc;				// buffer descriptor for extension ray hits
	RTPbufferdesc shadowRaysDesc;					// buffer descriptor for shadow rays
	RTPbufferdesc shadowHitsDesc;					// buffer descriptor for shadow ray hits
	CoreTexDesc* texDescs = 0;						// array of texture descriptors
	int textureCount = 0;							// size of texture descriptor array
	int SMcount = 0;								// multiprocessor count, used for persistent threads
	int computeCapability;							// device compute capability
	int samplesTaken = 0;							// number of accumulated samples in accumulator
	int inBuffer = 0, outBuffer = 1;				// extension ray buffers are double buffered
	uint camRNGseed = 0x12345678;					// seed for the RNG that feeds the renderer
	uint seed = 0x23456789;							// generic seed
	DeviceVars vars;								// copy of device-side variables, to detect changes
	bool firstConvergingFrame = false;				// to reset accumulator for first converging frame
	// timing
	cudaEvent_t shadeStart[MAXPATHLENGTH], shadeEnd[MAXPATHLENGTH];	// events for timing CUDA code
public:
	static RTPcontext context;						// the OptiX prime context
	CoreStats coreStats;							// rendering statistics
};

} // namespace lh2core

// EOF