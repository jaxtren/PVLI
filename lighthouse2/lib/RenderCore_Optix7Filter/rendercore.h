/* rendercore.h - Copyright 2019/2020 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under thbvhBuildTimee License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#pragma once
#include <map>

namespace lh2core
{

class RenderThread;

struct SBTRecord { __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };

//  +-----------------------------------------------------------------------------+
//  |  RenderCore                                                                 |
//  |  Encapsulates device code.                                            LH2'19|
//  +-----------------------------------------------------------------------------+
class RenderCore : public CoreAPI_Base
{
	friend class RenderThread;
public:
	// methods
	void Init();
	void Render( const ViewPyramid& view, const Convergence converge, bool async );
	int Render(const ViewPyramid& view, const int* const* mask, Fragment* output, int outputSize);
	void WaitForRender();
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
	void SetProbePos( const int2 pos );
	CoreStats GetCoreStats() const override;
	// internal methods
private:
	int RenderImpl( const ViewPyramid& view, const int* const* mask = nullptr, Fragment* output = nullptr, int outputSize = 0);
	template <class T> T* StagedBufferResize( CoreBuffer<T>*& lightBuffer, const int newCount, const T* sourceData );
	void UpdateToplevel();
	void SyncStorageType( const TexelStorage storage );
	void CreateOptixContext( int cc );
	// helpers
	template <class T> CUDAMaterial::Map Map( T v )
	{
		CUDAMaterial::Map m;
		CoreTexDesc& t = texDescs[v.textureID];
		m.width = t.width, m.height = t.height, m.uscale = v.uvscale.x, m.vscale = v.uvscale.y;
		m.uoffs = v.uvoffset.x, m.voffs = v.uvoffset.y, m.addr = t.firstPixel;
		return m;
	}
	int filter = 1;
	int filterPhases = 5;
	int RNGseed = 0;
	bool useAnyHit = true;
    int viewID = 0;                                 // ID of view (>= 0), used to store filter state
    int fallbackID = -1;                            // ID of view for fallback filter reprojection (<0 - disabled fallback)
	int skywidth = 0, skyheight = 0;				// size of the skydome texture
	int2 probePos = make_int2( 0 );                 // triangle picking; primary ray for this pixel copies its triid to coreStats.probedTriid
	vector<CoreMesh*> meshes;						// list of meshes, to be referenced by the instances
	vector<CoreInstance*> instances;				// list of instances: model id plus transform
	bool instancesDirty = true;						// we need to sync the instance array to the device
	CoreBuffer<CUDAMaterial>* materialBuffer = 0;	// material array
	CUDAMaterial* hostMaterialBuffer = 0;			// core-managed copy of the materials
	CoreBuffer<CoreLightTri>* areaLightBuffer;		// area lights
	CoreBuffer<CorePointLight>* pointLightBuffer;	// point lights
	CoreBuffer<CoreSpotLight>* spotLightBuffer;		// spot lights
	CoreBuffer<CoreDirectionalLight>* directionalLightBuffer;	// directional lights
	CoreBuffer<float4>* texel128Buffer = 0;			// texel buffer 1: hdr ARGB128 texture data
	CoreBuffer<uint>* normal32Buffer = 0;			// texel buffer 2: integer-encoded normals
	CoreBuffer<float4>* skyPixelBuffer = 0;			// skydome texture data
	CoreBuffer<CoreInstanceDesc>* instDescBuffer = 0; // instance descriptor array
	CoreBuffer<uint>* texel32Buffer = 0;			// texel buffer 0: regular ARGB32 texture data
	CoreBuffer<OptixInstance>* instanceArray = 0;	// instance descriptors for Optix
	CoreTexDesc* texDescs = 0;						// array of texture descriptors
	int textureCount = 0;							// size of texture descriptor array
	CoreData data;
	std::map<int, ViewData> views;
	bool gpuHasSceneData = false;					// to block renders before first SynchronizeSceneData
	Timer renderTimer, frameTimer;					// timers for asynchronous rendering
public:
	CoreStats coreStats;							// rendering statistics
	OptixDeviceContext optixContext;
	static const int progGroupCount = 6;
	enum { RAYGEN = 0, RAD_MISS, OCC_MISS, RAD_HIT, OCC_HIT, PRIM_HIT };
	CoreBuffer<SBTRecord> sbtRecords;
	OptixShaderBindingTable sbt;
	OptixModule ptxModule;
	OptixPipeline pipeline;
	OptixProgramGroup progGroup[progGroupCount];
	OptixTraversableHandle bvhRoot;
};

} // namespace lh2core
