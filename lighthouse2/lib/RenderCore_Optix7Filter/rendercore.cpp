/* rendercore.cpp - Copyright 2019/2020 Utrecht University

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

#include "core_settings.h"
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>

static cudaError_t cudaHandleError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        cerr << "CUDA ERROR: " << cudaGetErrorString(error) << " in " << file << " at line " << line << endl;
		#ifdef NDEBUG
		exit(EXIT_FAILURE);
        #else
		assert(false);
        #endif
    }
    return error;
}
#define cuEC( error ) ( cudaHandleError( error, __FILE__, __LINE__ ) )
#define cuCheck { cuEC(cudaDeviceSynchronize()); cuEC(cudaGetLastError()); } //TODO ignore for release build

namespace lh2core
{
#include "kernels/host.h"
}
using namespace lh2core;

namespace CUDA {

	cudaEvent_t Record() {
		cudaEvent_t event;
		cudaEventCreate(&event);
		cudaEventRecord(event);
		return event;
	}

	float Elapsed(cudaEvent_t start, cudaEvent_t end) {
		float elapsed;
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&elapsed, start, end);
		return elapsed * 0.001f; // report in seconds
	}

    float Elapsed(cudaEvent_t start) {
        float elapsed;
        auto end = Record();
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed, start, end);
        cudaEventDestroy(start);
        cudaEventDestroy(end);
        return elapsed * 0.001f; // report in seconds
    }
}

const char* ParseOptixError( OptixResult r )
{
	switch (r)
	{
	case OPTIX_SUCCESS: return "NO ERROR";
	case OPTIX_ERROR_INVALID_VALUE: return "OPTIX_ERROR_INVALID_VALUE";
	case OPTIX_ERROR_HOST_OUT_OF_MEMORY: return "OPTIX_ERROR_HOST_OUT_OF_MEMORY";
	case OPTIX_ERROR_INVALID_OPERATION: return "OPTIX_ERROR_INVALID_OPERATION";
	case OPTIX_ERROR_FILE_IO_ERROR: return "OPTIX_ERROR_FILE_IO_ERROR";
	case OPTIX_ERROR_INVALID_FILE_FORMAT: return "OPTIX_ERROR_INVALID_FILE_FORMAT";
	case OPTIX_ERROR_DISK_CACHE_INVALID_PATH: return "OPTIX_ERROR_DISK_CACHE_INVALID_PATH";
	case OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR: return "OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR";
	case OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR: return "OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR";
	case OPTIX_ERROR_DISK_CACHE_INVALID_DATA: return "OPTIX_ERROR_DISK_CACHE_INVALID_DATA";
	case OPTIX_ERROR_LAUNCH_FAILURE: return "OPTIX_ERROR_LAUNCH_FAILURE";
	case OPTIX_ERROR_INVALID_DEVICE_CONTEXT: return "OPTIX_ERROR_INVALID_DEVICE_CONTEXT";
	case OPTIX_ERROR_CUDA_NOT_INITIALIZED: return "OPTIX_ERROR_CUDA_NOT_INITIALIZED";
	case OPTIX_ERROR_INVALID_PTX: return "OPTIX_ERROR_INVALID_PTX";
	case OPTIX_ERROR_INVALID_LAUNCH_PARAMETER: return "OPTIX_ERROR_INVALID_LAUNCH_PARAMETER";
	case OPTIX_ERROR_INVALID_PAYLOAD_ACCESS: return "OPTIX_ERROR_INVALID_PAYLOAD_ACCESS";
	case OPTIX_ERROR_INVALID_ATTRIBUTE_ACCESS: return "OPTIX_ERROR_INVALID_ATTRIBUTE_ACCESS";
	case OPTIX_ERROR_INVALID_FUNCTION_USE: return "OPTIX_ERROR_INVALID_FUNCTION_USE";
	case OPTIX_ERROR_INVALID_FUNCTION_ARGUMENTS: return "OPTIX_ERROR_INVALID_FUNCTION_ARGUMENTS";
	case OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY: return "OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY";
	case OPTIX_ERROR_PIPELINE_LINK_ERROR: return "OPTIX_ERROR_PIPELINE_LINK_ERROR";
	case OPTIX_ERROR_INTERNAL_COMPILER_ERROR: return "OPTIX_ERROR_INTERNAL_COMPILER_ERROR";
	case OPTIX_ERROR_DENOISER_MODEL_NOT_SET: return "OPTIX_ERROR_DENOISER_MODEL_NOT_SET";
	case OPTIX_ERROR_DENOISER_NOT_INITIALIZED: return "OPTIX_ERROR_DENOISER_NOT_INITIALIZED";
	case OPTIX_ERROR_ACCEL_NOT_COMPATIBLE: return "OPTIX_ERROR_ACCEL_NOT_COMPATIBLE";
	case OPTIX_ERROR_NOT_SUPPORTED: return "OPTIX_ERROR_NOT_SUPPORTED";
	case OPTIX_ERROR_UNSUPPORTED_ABI_VERSION: return "OPTIX_ERROR_UNSUPPORTED_ABI_VERSION";
	case OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH: return "OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH";
	case OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS: return "OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS";
	case OPTIX_ERROR_LIBRARY_NOT_FOUND: return "OPTIX_ERROR_LIBRARY_NOT_FOUND";
	case OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND: return "OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND";
	case OPTIX_ERROR_CUDA_ERROR: return "OPTIX_ERROR_CUDA_ERROR";
	case OPTIX_ERROR_INTERNAL_ERROR: return "OPTIX_ERROR_INTERNAL_ERROR";
	case OPTIX_ERROR_UNKNOWN: return "OPTIX_ERROR_UNKNOWN";
	default: return "UNKNOWN ERROR";
	};
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetProbePos                                                    |
//  |  Set the pixel for which the triid will be captured.                  LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetProbePos( int2 pos )
{
	probePos = pos; // triangle id for this pixel will be stored in coreStats
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::CreateOptixContext                                             |
//  |  Optix 7 initialization.                                              LH2'19|
//  +-----------------------------------------------------------------------------+
static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */ )
{
	printf( "[%i][%s]: %s\n", level, tag, message );
}
void RenderCore::CreateOptixContext( int cc )
{
    cuCheck;
	// prepare the optix context
	cudaFree( 0 );
	CUcontext cu_ctx = 0; // zero means take the current context
	CHK_OPTIX( optixInit() );
	OptixDeviceContextOptions contextOptions = {};
	contextOptions.logCallbackFunction = &context_log_cb;
	contextOptions.logCallbackLevel = 4;
	CHK_OPTIX( optixDeviceContextCreate( cu_ctx, &contextOptions, &optixContext ) );

	// load and compile PTX
	string ptx, ptxFile, ptxDir = "../../lib/RenderCore_Optix7Filter/optix/";
	if (cc / 10 == 7) ptxFile = ".optix.turing.cu.ptx";
	else if (cc / 10 == 6) ptxFile = ".optix.pascal.cu.ptx";
	else if (cc / 10 == 5) ptxFile = ".optix.maxwell.cu.ptx";

	if (NeedsRecompile(ptxDir.c_str(), ptxFile.c_str(), ".optix.cu", "../../RenderSystem/common_settings.h", "../core_settings.h" ))
	{
		CUDATools::compileToPTX( ptx, TextFileRead( ptxDir + ".optix.cu" ).c_str(), ptxDir.c_str(), cc, 7 );
		TextFileWrite( ptx, ptxDir + ptxFile );
		printf( "recompiled .optix.cu.\n" );
	}
	else
	{
		FILE* f;
	#ifdef _MSC_VER
		fopen_s( &f, (ptxDir + ptxFile).c_str(), "rb" );
	#else
		f = fopen( (ptxDir + ptxFile).c_str(), "rb" );
	#endif
		int len;
		fread( &len, 1, 4, f );
		char* t = new char[len];
		fread( t, 1, len, f );
		fclose( f );
		ptx = string( t );
		delete t;
	}

	// create the optix module
	OptixModuleCompileOptions module_compile_options = {};
	module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	#ifdef NDEBUG
	module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
	module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
    #else
	module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
	module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    #endif
	OptixPipelineCompileOptions pipeCompileOptions = {};
	pipeCompileOptions.usesMotionBlur = false;
	pipeCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
	pipeCompileOptions.numPayloadValues = 4;
	pipeCompileOptions.numAttributeValues = 2;
	pipeCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipeCompileOptions.pipelineLaunchParamsVariableName = "data";
	char log[2048];
	size_t logSize = sizeof( log );
	CHK_OPTIX_LOG( optixModuleCreateFromPTX( optixContext, &module_compile_options, &pipeCompileOptions,
		ptx.c_str(), ptx.size(), log, &logSize, &ptxModule ) );

	// create program groups
	OptixProgramGroupOptions groupOptions = {};
	OptixProgramGroupDesc group = {};

	// ray gen
	group.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	group.raygen.module = ptxModule;
	group.raygen.entryFunctionName = "__raygen__rg";
	logSize = sizeof( log );
	CHK_OPTIX_LOG( optixProgramGroupCreate( optixContext, &group, 1, &groupOptions, log, &logSize, &progGroup[RAYGEN] ) );

	// radiance miss (null program)
	group = {};
	group.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	group.miss.module = nullptr;
	group.miss.entryFunctionName = nullptr;
	logSize = sizeof( log );
	CHK_OPTIX_LOG( optixProgramGroupCreate( optixContext, &group, 1, &groupOptions, log, &logSize, &progGroup[RAD_MISS] ) );

	// occlusion miss
	group = {};
	group.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	group.miss.module = ptxModule;
	group.miss.entryFunctionName = "__miss__occlusion";
	logSize = sizeof( log );
	CHK_OPTIX_LOG( optixProgramGroupCreate( optixContext, &group, 1, &groupOptions, log, &logSize, &progGroup[OCC_MISS] ) );

	// radiance hit
	group = {};
	group.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	group.hitgroup.moduleCH = ptxModule;
	group.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
	logSize = sizeof( log );
	CHK_OPTIX_LOG( optixProgramGroupCreate( optixContext, &group, 1, &groupOptions, log, &logSize, &progGroup[RAD_HIT] ) );

	// occlusion hit (null program)
	group = {};
	group.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	logSize = sizeof( log );
	CHK_OPTIX_LOG( optixProgramGroupCreate( optixContext, &group, 1, &groupOptions, log, &logSize, &progGroup[OCC_HIT] ) );

	// primary any hit
	group = {};
	group.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	group.hitgroup.moduleAH = ptxModule;
	group.hitgroup.entryFunctionNameAH = "__anyhit__primary";
	logSize = sizeof( log );
	CHK_OPTIX_LOG( optixProgramGroupCreate( optixContext, &group, 1, &groupOptions, log, &logSize, &progGroup[PRIM_HIT] ) );

	// create the pipeline
	OptixPipelineLinkOptions linkOptions = {};
	linkOptions.maxTraceDepth = 1;
	linkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
	linkOptions.overrideUsesMotionBlur = false;
	logSize = sizeof( log );
	CHK_OPTIX_LOG( optixPipelineCreate( optixContext, &pipeCompileOptions, &linkOptions, progGroup, progGroupCount, log, &logSize, &pipeline ) );

	// calculate the stack sizes, so we can specify all parameters to optixPipelineSetStackSize
	OptixStackSizes stack_sizes = {};
	for (auto& p : progGroup) optixUtilAccumulateStackSizes( p, &stack_sizes );
	uint32_t ss0, ss1, ss2;
	CHK_OPTIX( optixUtilComputeStackSizes( &stack_sizes, 1, 0, 0, &ss0, &ss1, &ss2 ) );
	CHK_OPTIX( optixPipelineSetStackSize( pipeline, ss0, ss1, ss2, 2 ) );

	// create the shader binding table
	sbtRecords.Allocate(progGroupCount, ON_HOST | ON_DEVICE);
	for (int i = 0; i < progGroupCount; i++) optixSbtRecordPackHeader( progGroup[i], &sbtRecords[i] );
	sbtRecords.CopyToDevice();
	sbt.raygenRecord = (CUdeviceptr) sbtRecords.DevPtr();
	sbt.missRecordBase = (CUdeviceptr) (sbtRecords.DevPtr() + 1);
	sbt.hitgroupRecordBase = (CUdeviceptr) (sbtRecords.DevPtr() + 3);
	sbt.missRecordCount = 2;
	sbt.hitgroupRecordCount = 3;
	sbt.missRecordStrideInBytes = sbt.hitgroupRecordStrideInBytes = sizeof(SBTRecord);
    cuCheck;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Init                                                           |
//  |  Initialization.                                                      LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Init()
{
    cuCheck;
#ifdef _DEBUG
	printf( "Initializing Optix7Filter core - DEBUG build.\n" );
#else
	printf( "Initializing Optix7Filter core - RELEASE build.\n" );
#endif
	// select the fastest device
	uint device = CUDATools::FastestDevice();
	cudaSetDevice( device );
	cudaDeviceProp properties;
	cudaGetDeviceProperties( &properties, device );
	coreStats.SMcount = properties.multiProcessorCount;
	coreStats.ccMajor = properties.major;
	coreStats.ccMinor = properties.minor;
	int computeCapability = coreStats.ccMajor * 10 + coreStats.ccMinor;
	coreStats.VRAM = (uint)(properties.totalGlobalMem >> 20);
	coreStats.deviceName = new char[strlen( properties.name ) + 1];
	memcpy( coreStats.deviceName, properties.name, strlen( properties.name ) + 1 );
	printf( "running on GPU: %s (%i SMs, %iGB VRAM)\n", coreStats.deviceName, coreStats.SMcount, (int)(coreStats.VRAM >> 10) );
	// initialize Optix7
	CreateOptixContext( computeCapability );
	// allocate buffers
	data.count.Allocate(1, ON_HOST | ON_DEVICE);
    data.temp.count.Allocate(2, ON_HOST | ON_DEVICE);
	// prepare the bluenoise data
	const uchar* data8 = (const uchar*)sob256_64; // tables are 8 bit per entry
	uint* data32 = new uint[65536 * 5]; // we want a full uint per entry
	for (int i = 0; i < 65536; i++) data32[i] = data8[i]; // convert
	data8 = (uchar*)scr256_64;
	for (int i = 0; i < (128 * 128 * 8); i++) data32[i + 65536] = data8[i];
	data8 = (uchar*)rnk256_64;
	for (int i = 0; i < (128 * 128 * 8); i++) data32[i + 3 * 65536] = data8[i];
	data.blueNoise.Allocate(65536 * 5, ON_DEVICE, data32);
	delete data32;
	// preallocate optix instance descriptor array
	instanceArray = new CoreBuffer<OptixInstance>( 16 /* will grow if needed */, ON_HOST | ON_DEVICE );
	// allow CoreMeshes to access the core
	//CoreMesh::renderCore = this;
    cuCheck;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetTarget                                                      |
//  |  Set the OpenGL texture that serves as the render target.             LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetTarget( GLTexture* target, const uint spp )
{ /* empty, use setting parameters: width, height, spp */ }

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetGeometry                                                    |
//  |  Set the geometry data for a model.                                   LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetGeometry( const int meshIdx, const float4* vertexData, const int vertexCount, const int triangleCount, const CoreTri* triangles )
{
    cuCheck;
	// Note: for first-time setup, meshes are expected to be passed in sequential order.
	// This will result in new CoreMesh pointers being pushed into the meshes vector.
	// Subsequent mesh changes will be applied to existing CoreMeshes. This is deliberately
	// minimalistic; RenderSystem is responsible for a proper (fault-tolerant) interface.
	if (meshIdx >= meshes.size()) meshes.push_back( new CoreMesh() );
	meshes[meshIdx]->SetGeometry( vertexData, vertexCount, triangleCount, triangles );
    cuCheck;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetInstance                                                    |
//  |  Set instance details.                                                LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetInstance( const int instanceIdx, const int meshIdx, const mat4& matrix )
{
    cuCheck;
	// A '-1' mesh denotes the end of the instance stream;
	// adjust the instances vector if we have more.
	if (meshIdx == -1)
	{
		if (instances.size() > instanceIdx) instances.resize( instanceIdx );
		return;
	}
	// For the first frame, instances are added to the instances vector.
	// For subsequent frames existing slots are overwritten / updated.
	if (instanceIdx >= instances.size())
	{
		// create a geometry instance
		CoreInstance* newInstance = new CoreInstance();
		memset( &newInstance->instance, 0, sizeof( OptixInstance ) );
		newInstance->instance.flags = OPTIX_INSTANCE_FLAG_NONE;
		newInstance->instance.instanceId = instanceIdx;
		newInstance->instance.sbtOffset = 0;
		newInstance->instance.visibilityMask = 255;
		newInstance->instance.traversableHandle = meshes[meshIdx]->gasHandle;
		memcpy( newInstance->transform, &matrix, 12 * sizeof( float ) );
		memcpy( newInstance->instance.transform, &matrix, 12 * sizeof( float ) );
		instances.push_back( newInstance );
	}
	// update the matrices for the transform
	memcpy( instances[instanceIdx]->transform, &matrix, 12 * sizeof( float ) );
	memcpy( instances[instanceIdx]->instance.transform, &matrix, 12 * sizeof( float ) );
	// set/update the mesh for this instance
	instances[instanceIdx]->mesh = meshIdx;
    cuCheck;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::FinalizeInstances                                              |
//  |  Update instance descriptor array on device.                          LH2'20|
//  +-----------------------------------------------------------------------------+
void RenderCore::FinalizeInstances()
{
    cuCheck;
	// resize instance array if more space is needed
	if (instances.size() > (size_t)instanceArray->GetSize())
	{
		delete instanceArray;
		instanceArray = new CoreBuffer<OptixInstance>( (int)instances.size() + 4, ON_HOST | ON_DEVICE | STAGED );
	}
	// copy instance descriptors to the array, sync with device
	for (int s = (int)instances.size(), i = 0; i < s; i++)
	{
		instances[i]->instance.traversableHandle = meshes[instances[i]->mesh]->gasHandle;
		instanceArray->HostPtr()[i] = instances[i]->instance;
	}
	instanceArray->StageCopyToDevice();
	// pass instance descriptors to the device; will be used during shading.
	if (instancesDirty)
	{
		// prepare CoreInstanceDesc array. For any sane number of instances this should
		// be efficient while yielding supreme flexibility.
		vector<CoreInstanceDesc> instDescArray;
		for (auto instance : instances)
		{
			CoreInstanceDesc id;
			id.triangles = meshes[instance->mesh]->triangles->DevPtr();
			mat4 T, invT;
			if (instance->transform)
			{
				T = mat4::Identity();
				memcpy( &T, instance->transform, 12 * sizeof( float ) );
				invT = T.Inverted();
			}
			else T = mat4::Identity(), invT = mat4::Identity();
			id.invTransform = *(float4x4*)&invT;

			// instance LOD
            auto x = make_float3(T.cell[0], T.cell[4], T.cell[8]);
			auto y = make_float3(T.cell[1], T.cell[5], T.cell[9]);
            auto z = make_float3(T.cell[2], T.cell[6], T.cell[10]);
            float f = max(length(x), max(length(y), length(z)));
            id.LOD = log2(1.0f / f);

			instDescArray.push_back( id );
		}
		if (instDescBuffer == 0 || instDescBuffer->GetSize() < (int)instances.size())
		{
			delete instDescBuffer;
			// size of instance list changed beyond capacity.
			// Allocate a new buffer, with some slack, to prevent excessive reallocs.
			instDescBuffer = new CoreBuffer<CoreInstanceDesc>((int)instances.size() * 2, ON_HOST | ON_DEVICE );
			stageInstanceDescriptors( instDescBuffer->DevPtr() );
		}
		memcpy( instDescBuffer->HostPtr(), instDescArray.data(), instDescArray.size() * sizeof( CoreInstanceDesc ) );
		instDescBuffer->StageCopyToDevice();
        UpdateToplevel();
		// instancesDirty = false; // TODO: for now we do this every frame.
	}
	// rendering is allowed from now on
	gpuHasSceneData = true;
    cuCheck;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetTextures                                                    |
//  |  Set the texture data.                                                LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetTextures( const CoreTexDesc* tex, const int textures )
{
    cuCheck;
	// copy the supplied array of texture descriptors
	delete texDescs; texDescs = 0;
	textureCount = textures;
	if (textureCount == 0) return; // scene has no textures
	texDescs = new CoreTexDesc[textureCount];
	memcpy( texDescs, tex, textureCount * sizeof( CoreTexDesc ) );
	// copy texels for each type to the device
	SyncStorageType( TexelStorage::ARGB32 );
	SyncStorageType( TexelStorage::ARGB128 );
	SyncStorageType( TexelStorage::NRM32 );
    cuCheck;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SyncStorageType                                                |
//  |  Copies texel data for one storage type (argb32, argb128 or nrm32) to the   |
//  |  device. Note that this data is obtained from the original HostTexture      |
//  |  texel arrays.                                                        LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SyncStorageType( const TexelStorage storage )
{
    cuCheck;
	uint texelTotal = 0;
	for (int i = 0; i < textureCount; i++) if (texDescs[i].storage == storage) texelTotal += texDescs[i].pixelCount;
	texelTotal = max( 16, texelTotal ); // OptiX does not tolerate empty buffers...
	// construct the continuous arrays
	switch (storage)
	{
	case TexelStorage::ARGB32:
		delete texel32Buffer;
		texel32Buffer = new CoreBuffer<uint>( texelTotal, ON_HOST | ON_DEVICE );
		stageARGB32Pixels( texel32Buffer->DevPtr() );
		coreStats.argb32TexelCount = texelTotal;
		break;
	case TexelStorage::ARGB128:
		delete texel128Buffer;
		stageARGB128Pixels( (texel128Buffer = new CoreBuffer<float4>( texelTotal, ON_HOST | ON_DEVICE ))->DevPtr() );
		coreStats.argb128TexelCount = texelTotal;
		break;
	case TexelStorage::NRM32:
		delete normal32Buffer;
		stageNRM32Pixels( (normal32Buffer = new CoreBuffer<uint>( texelTotal, ON_HOST | ON_DEVICE ))->DevPtr() );
		coreStats.nrm32TexelCount = texelTotal;
		break;
	}
	// copy texel data to arrays
	texelTotal = 0;
	for (int i = 0; i < textureCount; i++) if (texDescs[i].storage == storage)
	{
		void* destination = 0;
		switch (storage)
		{
		case TexelStorage::ARGB32:  destination = texel32Buffer->HostPtr() + texelTotal; break;
		case TexelStorage::ARGB128: destination = texel128Buffer->HostPtr() + texelTotal; break;
		case TexelStorage::NRM32:   destination = normal32Buffer->HostPtr() + texelTotal; break;
		}
		memcpy( destination, texDescs[i].idata, texDescs[i].pixelCount * sizeof( uint ) );
		texDescs[i].firstPixel = texelTotal;
		texelTotal += texDescs[i].pixelCount;
	}
	// move to device
	if (storage == TexelStorage::ARGB32) if (texel32Buffer) texel32Buffer->StageCopyToDevice();
	if (storage == TexelStorage::ARGB128) if (texel128Buffer) texel128Buffer->StageCopyToDevice();
	if (storage == TexelStorage::NRM32) if (normal32Buffer) normal32Buffer->StageCopyToDevice();
    cuCheck;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetMaterials                                                   |
//  |  Set the material data.                                               LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetMaterials( CoreMaterial* mat, const int materialCount )
{
    cuCheck;
#define TOCHAR(a) ((uint)((a)*255.0f))
#define TOUINT4(a,b,c,d) (TOCHAR(a)+(TOCHAR(b)<<8)+(TOCHAR(c)<<16)+(TOCHAR(d)<<24))
	// Notes:
	// Call this after the textures have been set; CoreMaterials store the offset of each texture
	// in the continuous arrays; this data is valid only when textures are in sync.
	delete materialBuffer;
	delete hostMaterialBuffer;
	hostMaterialBuffer = new CUDAMaterial[materialCount];
	for (int i = 0; i < materialCount; i++)
	{
		// perform conversion to internal material format
		CoreMaterial& m = mat[i];
		CUDAMaterial& gpuMat = hostMaterialBuffer[i];
		memset( &gpuMat, 0, sizeof( CUDAMaterial ) );
		gpuMat.SetDiffuse( m.color.value );
		gpuMat.SetTransmittance( make_float3( 1 ) - m.absorption.value );
		gpuMat.parameters.x = TOUINT4( m.metallic.value, m.subsurface.value, m.specular.value, m.roughness.value );
		gpuMat.parameters.y = TOUINT4( m.specularTint.value, m.anisotropic.value, m.sheen.value, m.sheenTint.value );
		gpuMat.parameters.z = TOUINT4( m.clearcoat.value, m.clearcoatGloss.value, m.transmission.value, 0 );
		gpuMat.parameters.w = *((uint*)&m.eta);
		if (m.color.textureID != -1) gpuMat.tex0 = Map<CoreMaterial::Vec3Value>( m.color );
		if (m.detailColor.textureID != -1) gpuMat.tex1 = Map<CoreMaterial::Vec3Value>( m.detailColor );
		if (m.normals.textureID != -1) gpuMat.nmap0 = Map<CoreMaterial::Vec3Value>( m.normals );
		if (m.detailNormals.textureID != -1) gpuMat.nmap1 = Map<CoreMaterial::Vec3Value>( m.detailNormals );
		if (m.roughness.textureID != -1) gpuMat.rmap = Map<CoreMaterial::ScalarValue>( m.roughness );
		if (m.specular.textureID != -1) gpuMat.smap = Map<CoreMaterial::ScalarValue>( m.specular );
		bool hdr = false;
		if (m.color.textureID != -1) if (texDescs[m.color.textureID].flags & 8 /* HostTexture::HDR */) hdr = true;
		gpuMat.flags =
			(m.eta.value < 1 ? ISDIELECTRIC : 0) + (hdr ? DIFFUSEMAPISHDR : 0) +
			(m.color.textureID != -1 ? HASDIFFUSEMAP : 0) +
			(m.normals.textureID != -1 ? HASNORMALMAP : 0) +
			(m.specular.textureID != -1 ? HASSPECULARITYMAP : 0) +
			(m.roughness.textureID != -1 ? HASROUGHNESSMAP : 0) +
			(m.metallic.textureID != -1 ? HASMETALNESSMAP : 0) +
			(m.detailNormals.textureID != -1 ? HAS2NDNORMALMAP : 0) +
			(m.detailColor.textureID != -1 ? HAS2NDDIFFUSEMAP : 0) +
			((m.flags & 1) ? HASSMOOTHNORMALS : 0) + ((m.flags & 2) ? HASALPHA : 0);
	}
	materialBuffer = new CoreBuffer<CUDAMaterial>( materialCount, ON_HOST | ON_DEVICE | STAGED, hostMaterialBuffer );
	materialBuffer->StageCopyToDevice();
	stageMaterialList( materialBuffer->DevPtr() );
    cuCheck;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetLights                                                      |
//  |  Set the light data.                                                  LH2'20|
//  +-----------------------------------------------------------------------------+
template <class T> T* RenderCore::StagedBufferResize( CoreBuffer<T>*& lightBuffer, const int newCount, const T* sourceData )
{
    cuCheck;
	// helper function for (re)allocating light buffers with staged buffer and pointer update.
	if (lightBuffer == 0 || newCount > lightBuffer->GetSize())
	{
		delete lightBuffer;
		lightBuffer = new CoreBuffer<T>( newCount, ON_HOST | ON_DEVICE );
	}
	memcpy( lightBuffer->HostPtr(), sourceData, newCount * sizeof( T ) );
	lightBuffer->StageCopyToDevice();
    cuCheck;
	return lightBuffer->DevPtr();
}
void RenderCore::SetLights( const CoreLightTri* areaLights, const int areaLightCount,
	const CorePointLight* pointLights, const int pointLightCount,
	const CoreSpotLight* spotLights, const int spotLightCount,
	const CoreDirectionalLight* directionalLights, const int directionalLightCount )
{
    cuCheck;
	stageAreaLights( StagedBufferResize<CoreLightTri>( areaLightBuffer, areaLightCount, areaLights ) );
	stagePointLights( StagedBufferResize<CorePointLight>( pointLightBuffer, pointLightCount, pointLights ) );
	stageSpotLights( StagedBufferResize<CoreSpotLight>( spotLightBuffer, spotLightCount, spotLights ) );
	stageDirectionalLights( StagedBufferResize<CoreDirectionalLight>( directionalLightBuffer, directionalLightCount, directionalLights ) );
	stageLightCounts( areaLightCount, pointLightCount, spotLightCount, directionalLightCount );
    cuCheck;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetSkyData                                                     |
//  |  Set the sky dome data.                                               LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetSkyData( const float3* pixels, const uint width, const uint height, const mat4& worldToLight )
{
    cuCheck;
	delete skyPixelBuffer;
	if(pixels)
	{
		skyPixelBuffer = new CoreBuffer<float4>(width * height + (width >> 6) * (height >> 6), ON_HOST | ON_DEVICE, 0);
		for (uint i = 0; i < width * height; i++) skyPixelBuffer->HostPtr()[i] = make_float4(pixels[i], 0);
		stageSkyPixels( skyPixelBuffer->DevPtr() );
		skyPixelBuffer->CopyToDevice();
	}
	else
	{
		skyPixelBuffer = 0;
		stageSkyPixels( 0 );
	}
	stageSkySize( width, height );
	stageWorldToSky( worldToLight );
	skywidth = width;
	skyheight = height;
    cuCheck;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Setting                                                        |.
//  |  Modify a render setting.                                             LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Setting( const char* name, const float value )
{
	if (!strcmp( name, "geometryEpsilon" )) data.geometryEpsilon = value;
	else if (!strcmp( name, "emissiveFactor" )) data.emissiveFactor = value;
	else if (!strcmp( name, "skipLayers" )) data.skipLayers = std::max(0.0f, value);
    else if (!strcmp( name, "width" )) data.scrsize.x = value;
    else if (!strcmp( name, "height" )) data.scrsize.y = value;
    else if (!strcmp( name, "spp" )) data.spp = value;
    else if (!strcmp( name, "cubemap" )) data.cubemap = value;
	else if (!strcmp( name, "filter" )) filter = value;
	else if (!strcmp( name, "filterPhases" )) filterPhases = value;
    else if (!strcmp( name, "subpixelOffsetX" )) data.subpixelOffset.x = value;
    else if (!strcmp( name, "subpixelOffsetY" )) data.subpixelOffset.y = value;
	else if (!strcmp( name, "evenPixelsOffsetX" )) data.evenPixelsOffset.x = value;
	else if (!strcmp( name, "evenPixelsOffsetY" )) data.evenPixelsOffset.y = value;
    else if (!strcmp( name, "demodulateAlbedo" )) data.demodulateAlbedo = value > 0;
	else if (!strcmp( name, "useAnyHit" )) useAnyHit = value > 0;
    else if (!strcmp( name, "viewID" )) viewID = value;
    else if (!strcmp( name, "fallbackID" )) fallbackID = value;
	else if (!strcmp( name, "clampValue" )) data.clampValue = value;
	else if (!strcmp( name, "clampDirect" )) data.filter.directClamp = value;
	else if (!strcmp( name, "clampIndirect" )) data.filter.indirectClamp = value;
	else if (!strcmp( name, "clampReflection" )) data.filter.reflectionClamp = value;
	else if (!strcmp( name, "clampRefraction" )) data.filter.refractionClamp = value;
	else if (!strcmp( name, "reprojSpatialCount" )) data.filter.reprojSpatialCount = value;
	else if (!strcmp( name, "reprojWeight" )) data.filter.reprojWeight = value;
	else if (!strcmp( name, "reprojWeightFallback" )) data.filter.reprojWeightFallback = value;
	else if (!strcmp( name, "reprojMaxDistFactor" )) data.filter.reprojMaxDistFactor = value;
	else if (!strcmp( name, "shadeKeepPhase" )) data.filter.shadeKeepPhase = value;
	else if (!strcmp( name, "shadeMergePhase" )) data.filter.shadeMergePhase = value;
	else if (!strcmp( name, "closestOffset" )) data.filter.closestOffset = value;
	else if (!strcmp( name, "closestOffsetMin" )) data.filter.closestOffsetMin = value;
    else if (!strcmp( name, "closestOffsetMax" )) data.filter.closestOffsetMax = value;
    else if (!strcmp( name, "varianceFactor" )) data.filter.varianceFactor = value;
    else if (!strcmp( name, "varianceReprojFactor" )) data.filter.varianceReprojFactor = value;
    else if (!strcmp( name, "varianceGauss" )) data.filter.varianceGauss = value;
    else if (!strcmp( name, "reprojLinearFilter" )) data.filter.reprojLinearFilter = value > 0;
	else if (!strcmp( name, "firstLayerOnly" )) data.filter.firstLayerOnly = value > 0;
    else if (!strcmp( name, "normalFactor" )) data.filter.normalFactor = value;
    else if (!strcmp( name, "depthMode" )) data.filter.depthMode = value;
    else if (!strcmp( name, "distanceFactor" )) data.filter.distanceFactor = value;
    else if (!strcmp( name, "reorderFragments" )) data.filter.reorderFragments = value > 0;
	else if (!strcmp( name, "maxPathLength" )) data.maxPathLength = max(1, (int)value);
    else if (!strcmp( name, "minPathLength" )) data.minPathLength = max(1, (int)value);
    else if (!strcmp( name, "maxLayers" )) data.maxLayers = max(1, (int)value);
    else if (!strcmp( name, "storeBackground" )) data.storeBackground = value > 0;
	else if (!strcmp( name, "disableAlphaMask" )) data.disableAlphaMask = value > 0;
	else if (!strcmp( name, "pathRegularization" )) data.pathRegularization = value;
    else if (!strcmp( name, "deterministicLight" )) data.deterministicLight = value;
    else if (!strcmp( name, "resetView" )) views.erase((int)value);
    else if (!strcmp( name, "resetViews" )) views.clear();
	else if (!strcmp( name, "RNGseed" )) RNGseed = value;
    else if (!strcmp( name, "cullface" )) {
        data.primaryRayFlags &= ~(OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES);
        if (value > 0) data.primaryRayFlags |= OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES;
        else if (value < 0) data.primaryRayFlags |= OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    }
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::UpdateToplevel                                                 |
//  |  After changing meshes, instances or instance transforms, we need to        |
//  |  rebuild the top-level structure.                                     LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::UpdateToplevel()
{
    cuCheck;
	// build accstructs for modified meshes
	for (CoreMesh* m : meshes) if (m->accstrucNeedsUpdate) m->UpdateAccstruc(optixContext);
	// build the top-level tree
	OptixBuildInput buildInput = {};
	buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	buildInput.instanceArray.instances = (CUdeviceptr)instanceArray->DevPtr();
	buildInput.instanceArray.numInstances = (uint)instances.size();
	OptixAccelBuildOptions options = {};
	options.buildFlags = OPTIX_BUILD_FLAG_NONE;
	options.operation = OPTIX_BUILD_OPERATION_BUILD;
	static size_t reservedTemp = 0, reservedTop = 0;
	static CoreBuffer<uchar>* temp, * topBuffer = 0;
	OptixAccelBufferSizes sizes;
	CHK_OPTIX( optixAccelComputeMemoryUsage( optixContext, &options, &buildInput, 1, &sizes ) );
	if (sizes.tempSizeInBytes > reservedTemp)
	{
		reservedTemp = sizes.tempSizeInBytes + 1024;
		delete temp;
		temp = new CoreBuffer<uchar>( (int)reservedTemp, ON_DEVICE );
	}
	if (sizes.outputSizeInBytes > reservedTop)
	{
		reservedTop = sizes.outputSizeInBytes + 1024;
		delete topBuffer;
		topBuffer = new CoreBuffer<uchar>( (int)reservedTop, ON_DEVICE );
	}
	CHK_OPTIX( optixAccelBuild( optixContext, 0, &options, &buildInput, 1, (CUdeviceptr)temp->DevPtr(),
		reservedTemp, (CUdeviceptr)topBuffer->DevPtr(), reservedTop, &bvhRoot, 0, 0 ) );
	cudaStreamSynchronize( 0 );
    cuCheck;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Render                                                         |
//  |  Produce one image.                                                   LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Render( const ViewPyramid& view, const Convergence converge, bool async )
{
	RenderImpl( view );
}

int RenderCore::Render(const ViewPyramid& view, const int* const* mask, Fragment* output, int outputSize)
{
	return RenderImpl( view, mask, output, outputSize );
}

int RenderCore::RenderImpl(const ViewPyramid& view, const int* const* mask, Fragment* output, int outputSize)
{
    cuCheck;
	if (!gpuHasSceneData) return 0;

	auto renderStart = CUDA::Record();
	pushStagedCopies();
	renderTimer.reset();

	// view data
	bool initView = views.find(viewID) == views.end();
	auto& viewData = views[viewID];
	if (initView) {
	    viewData.RNGseed = 0x12345678 + (RNGseed * 7549);
	    viewData.blueNoiseSlot = (RNGseed * 17) & 255;
	}
	data.blueSlot = viewData.blueNoiseSlot;
	bool filterHasFallback = false;
	if (filter) {
		swap(viewData.filter, data.filter.previous);
		if (fallbackID >= 0 && fallbackID != viewID && views.find(fallbackID) != views.end()) {
			swap(views[fallbackID].filter, data.filter.fallback);
			filterHasFallback = true;
		}
		else data.filter.fallback = FilterData();
	} else {
		data.filter.previous = FilterData();
		data.filter.fallback = FilterData();
	}

    // testing variables
    //#define DEBUG_TIMES(x) x
    #define DEBUG_TIMES(x)

	// OptiX helper
	auto dataAddr = (CUdeviceptr)getCoreDataAddress();
	auto trace = [this, dataAddr](int phase, int w, int h = 1, int d = 1) {
		data.phase = phase;
		setCoreData(data);
		CHK_OPTIX(optixLaunch(pipeline, 0, dataAddr, sizeof(CoreData), &sbt, w, h, d));
	};

	// clear stats
	auto& stats = coreStats;
	stats.totalRays = stats.totalExtensionRays = stats.totalShadowRays  = 0;
	stats.primaryRayCount = stats.bounce1RayCount = stats.deepRayCount = 0;
	stats.traceTime0 = stats.traceTime1 = stats.traceTimeX = stats.shadowTraceTime = 0;
	stats.shadeTime = stats.filterTime = stats.renderTime = 0;

	// params
	data.bvhRoot = bvhRoot;
	data.primMask = mask;
	data.maxFragments = outputSize;
	data.filter.current.firstLayerOnly = data.filter.firstLayerOnly;
	data.filter.previous.firstLayerOnly = data.filter.firstLayerOnly;
	data.filter.fallback.firstLayerOnly = data.filter.firstLayerOnly;

	// view
	data.view = view;
	data.view.p2 = (data.view.p2 - data.view.p1) / data.scrsize.x;
    data.view.p3 = (data.view.p3 - data.view.p1) / data.scrsize.y;
    data.view.p1 += data.view.p2 * data.subpixelOffset.x + data.view.p3 * data.subpixelOffset.y - data.view.pos;
	data.view.spreadAngle = sin(data.view.spreadAngle);
    data.bvhRoot = bvhRoot;

    // view matrix
    mat4 viewMat, viewMatInv, projection;
    {
        float3 x = normalize(data.view.p2);
        float3 y = normalize(data.view.p3);
        float3 z = normalize(cross(x, y));
        float3 p = data.view.pos;
        viewMatInv = mat4({x.x, y.x, z.x, 0,
                           x.y, y.y, z.y, 0,
                           x.z, y.z, z.z, 0,
                           0, 0, 0, 1});
        viewMatInv = mat4::Translate(p) * viewMatInv;
        viewMat = viewMatInv.Inverted();

        // back projected view
        float3 p1 = viewMat.TransformVector(data.view.p1);
        float3 p2 = viewMat.TransformVector(data.view.p2);
        float3 p3 = viewMat.TransformVector(data.view.p3);

        if (data.cubemap) { // cubemap views
            projection = viewMat;
            data.scrsize.x *= 6;

            // rotations for cubemap faces/directions
            mat4 faces[6] = {
                    mat4({ 0, 0,-1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1}), // left
                    mat4({ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}), // front
                    mat4({ 0, 0, 1, 0, 0, 1, 0, 0,-1, 0, 0, 0, 0, 0, 0, 1}), // right
                    mat4({-1, 0, 0, 0, 0, 1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 1}), // back
                    mat4({ 1, 0, 0, 0, 0, 0, 1, 0, 0,-1, 0, 0, 0, 0, 0, 1}), // top
                    mat4({ 1, 0, 0, 0, 0, 0,-1, 0, 0, 1, 0, 0, 0, 0, 0, 1}), // bottom
            };

            // final cubemap views
            for (int i = 0; i < 6; i++) {
                mat4 rot = viewMatInv * faces[i].Transposed();
                data.views[i].p1 = rot.TransformVector(p1);
                data.views[i].p2 = rot.TransformVector(p2);
                data.views[i].p3 = rot.TransformVector(p3);
            }
        } else { // perspective projection
            float left = p1.x;
            float bottom = p1.y;
            float nearPlane = -p1.z;
            float right = (p1.x + p2.x * data.scrsize.x);
            float top = (p1.y + p3.y * data.scrsize.y);
            float farPlane = nearPlane * 2;
            mat4 frustum = mat4::ZeroMatrix();
            frustum[0 * 4 + 0] = 2 * nearPlane / (right - left);
            frustum[1 * 4 + 1] = 2 * nearPlane / (top - bottom);
            frustum[2 * 4 + 0] = (right + left) / (right - left);
            frustum[2 * 4 + 1] = (top + bottom) / (top - bottom);
            frustum[2 * 4 + 2] = - (farPlane + nearPlane) / (farPlane - nearPlane);
            frustum[2 * 4 + 3] = -1;
            frustum[3 * 4 + 2] = - (2 * farPlane * nearPlane) / (farPlane - nearPlane);
            projection = frustum.Transposed() * viewMat;
        }
    }

    // init buffers
    data.primary.Allocate(outputSize, ON_DEVICE);
    data.positions.Allocate(outputSize, ON_DEVICE);
    data.links.Allocate(outputSize, ON_DEVICE);
    data.counts.Allocate(data.scrsize.x * data.scrsize.y, ON_DEVICE);
    setCoreData(data);

    // generate and trace primary rays
    data.count.CopyToDevice(data.scrsize.x * data.scrsize.y);
    auto tracePrimaryStart = CUDA::Record();
	trace(useAnyHit && data.maxLayers > 1 ? CoreData::SPAWN_PRIMARY_ANYHIT : CoreData::SPAWN_PRIMARY, data.scrsize.x, data.scrsize.y);
    stats.traceTime0 = CUDA::Elapsed(tracePrimaryStart);
    stats.primaryRayCount = *data.count = min(outputSize, *data.count.CopyToHost());
    data.count.CopyToDevice();

	cudaEvent_t reorderStart, reorderEnd;
	cudaEventCreate(&reorderStart);
	cudaEventCreate(&reorderEnd);
	if (filter && data.filter.reorderFragments) {
		cudaEventRecord(reorderStart);
		data.temp.indices.Allocate(data.scrsize.x * data.scrsize.y, ON_DEVICE);
		data.temp.primary.Allocate(*data.count, ON_DEVICE);
		setCoreData(data);

		reorderFragmentsPrepare();
		prefixSum((unsigned int*)data.temp.indices.DevPtr(), data.scrsize.x * data.scrsize.y, 256);
		reorderFragments();
		reorderFragmentsUpdate();

		swap(data.primary, data.temp.primary);
		data.filter.current.reordered = true;
		cudaEventRecord(reorderEnd);
		DEBUG_TIMES(cout << "Filter - reorder " << CUDA::Elapsed(reorderStart, reorderEnd) * 1000 << endl;)
	} else if (filter) data.filter.current.reordered = false;

    // init rest buffers
    data.stride = *data.count;
	data.filter.current.count = *data.count;
    data.accumulator.Allocate(*data.count * 2, ON_DEVICE);
    data.temp.paths[0].Allocate(*data.count * 4 * 2, ON_DEVICE);
    data.temp.paths[1].Allocate(*data.count * 4 * 2, ON_DEVICE);
    data.temp.shadow.Allocate(*data.count * 3, ON_DEVICE);
    data.filter.shading_var.Allocate(*data.count, ON_DEVICE);
	data.filter.moments.Allocate(*data.count, ON_DEVICE);
    data.filter.current.shading_var.Allocate(*data.count, ON_DEVICE);
    data.filter.current.moments.Allocate(*data.count, ON_DEVICE);
    data.filter.current.pos_albedo.Allocate(*data.count * 2, ON_DEVICE);
    data.filter.current.normal_flags.Allocate(*data.count * 2, ON_DEVICE);
    data.filter.prevLink.Allocate(*data.count * (data.filter.reprojLinearFilter ? 4 : 1), ON_DEVICE);
    if (data.filter.reprojLinearFilter)
    	data.filter.prevWeights.Allocate(*data.count, ON_DEVICE);
    else data.filter.prevWeights.Free();
    if (data.filter.depthMode == 2)
    	data.filter.current.depth.Allocate(*data.count * 2, ON_DEVICE);
    else data.filter.current.depth.Free();
	if (data.demodulateAlbedo)
		data.filter.albedo.Allocate(*data.count, ON_DEVICE);
	else data.filter.albedo.Free();
    setCoreData(data);

    // create events
    cudaEvent_t shadeStart, shadeEnd, traceStart, traceEnd, traceShadowStart, traceShadowEnd, filterStart, filterEnd;
    cudaEventCreate(&shadeStart);
    cudaEventCreate(&shadeEnd);
    cudaEventCreate(&traceStart);
    cudaEventCreate(&traceEnd);
    cudaEventCreate(&traceShadowStart);
    cudaEventCreate(&traceShadowEnd);
    cudaEventCreate(&filterStart);
    cudaEventCreate(&filterEnd);

    // shade and trace another rays
    data.accumulator.Clear(ON_DEVICE);
    for (int sample = 0; sample < data.spp; sample++) {

        int pathCount = *data.count;
        bool hasShadowTraceTime = false;
        for (int pathLength = 1; pathLength <= data.maxPathLength && pathCount > 0; pathLength++) {

            // trace
            if (pathLength > 1) {
                cudaEventRecord(traceStart);
                trace(CoreData::SPAWN_SECONDARY, pathCount);
                cudaEventRecord(traceEnd);
            }

            // shade and generate next rays
            cudaEventRecord(shadeStart);
			data.temp.count.Clear(ON_DEVICE);
            shade(sample, pathLength, pathCount, RandomUInt(viewData.RNGseed));
            cudaEventRecord(shadeEnd);

            /// wait for result
			pathCount = *data.temp.count.CopyToHost();
			swap(data.temp.paths[0], data.temp.paths[1]);
			setCoreData(data);

            // stats
            if (pathLength == 1) stats.bounce1RayCount += pathCount;
            else stats.deepRayCount += pathCount;
            if (pathLength == 2) stats.traceTime1 += CUDA::Elapsed(traceStart, traceEnd);
            else if (pathLength > 2) stats.traceTimeX += CUDA::Elapsed(traceStart, traceEnd);
            stats.shadeTime += CUDA::Elapsed(shadeStart, shadeEnd);
            if (hasShadowTraceTime)
                stats.shadowTraceTime += CUDA::Elapsed(traceShadowStart, traceShadowEnd);

			// shadow
            int shadowCount = *data.temp.shadowCount();
            if (shadowCount > 0) {
                cudaEventRecord(traceShadowStart);
                trace(CoreData::SPAWN_SHADOW, shadowCount);
                cudaEventRecord(traceShadowEnd);
                stats.totalShadowRays += shadowCount;
                hasShadowTraceTime = true;
            } else hasShadowTraceTime = false;
        }

        // residual stats
        if (hasShadowTraceTime)
            stats.shadowTraceTime += CUDA::Elapsed(traceShadowStart, traceShadowEnd);

        if (filter) {
            if (sample > 0) // previous stats
                stats.filterTime += CUDA::Elapsed(filterStart, filterEnd);
            cudaEventRecord(filterStart);
        	updateFilterFragments(sample, false);
            cudaEventRecord(filterEnd);
        }
    }

    // residual stats
    if (filter)
        stats.filterTime += CUDA::Elapsed(filterStart, filterEnd);

	if (filter) {
        cudaEventRecord(filterStart);

	    // prepare
        DEBUG_TIMES(auto prepareStart = CUDA::Record();)
        data.filter.current.links = data.links;
        data.filter.current.counts = data.counts;
        setCoreData(data);
		prepareFilterFragments(0);
        prepareFilterFragments(1);
		swap(data.filter.moments, data.filter.current.moments);
		setCoreData(data);
		DEBUG_TIMES(cout << "Filter - prepare " << CUDA::Elapsed(prepareStart) * 1000 << endl;)

		// phases
		for (int p = 1; p <= filterPhases; p++) {
            DEBUG_TIMES(auto filterPhaseStart = CUDA::Record();)
            if (filter == 1)
				applyLayeredFilter(p, 0);
            else if (filter == 2) {
				applyLayeredFilter(p, 1);
				applyLayeredFilter(p, 2);
            }

			// keep shade for next frame
			if (p - 1 == data.filter.shadeKeepPhase) {
				swap(data.filter.current.shading_var, data.filter.previous.shading_var);
                data.filter.current.shading_var.Allocate(data.filter.previous.shading_var.GetSize(), ON_DEVICE);
			}
			swap(data.filter.current.shading_var, data.filter.shading_var);
            setCoreData(data);
            DEBUG_TIMES(cout << "Filter - apply " << CUDA::Elapsed(filterPhaseStart) * 1000 << endl);
		}

		// finalize
		if (output) finalizeAllLayers(output, 1);

		// keep data for next frame
		if(data.filter.shadeKeepPhase >= filterPhases)
			swap(data.filter.current.shading_var, data.filter.previous.shading_var);
		swap(data.filter.current.moments, data.filter.previous.moments);
		swap(data.filter.current.pos_albedo, data.filter.previous.pos_albedo);
        swap(data.filter.current.normal_flags, data.filter.previous.normal_flags);
		// currently not using:
        //swap(data.filter.current.depth, data.filter.previous.depth);
		swap(data.links, data.filter.previous.links);
        swap(data.counts, data.filter.previous.counts);
		data.filter.previous.count = data.filter.current.count;
		data.filter.previous.scrsize = data.scrsize;
		data.filter.previous.subpixelOffset = data.subpixelOffset;
		data.filter.previous.evenPixelsOffset = data.evenPixelsOffset;
        data.filter.previous.reordered = data.filter.current.reordered;
        data.filter.previous.cubemap = data.cubemap;
        data.filter.previous.view = data.view;
        data.filter.previous.projection = projection;
        for(int i=0; i<6; i++)
            data.filter.previous.views[i] = data.views[i];

        // stats
        cudaEventRecord(filterEnd);
        stats.filterTime += CUDA::Elapsed(filterStart, filterEnd);
        if (data.filter.reorderFragments)
			stats.filterTime += CUDA::Elapsed(reorderStart, reorderEnd);

	} else if(output)
        finalizeAllLayers(output, 0);

	// finalize stats
	stats.totalExtensionRays = stats.primaryRayCount + stats.bounce1RayCount + stats.deepRayCount;
	stats.totalRays = stats.totalExtensionRays + stats.totalShadowRays;
	stats.renderTime = CUDA::Elapsed(renderStart, CUDA::Record());
	stats.frameOverhead = max(0.0f, frameTimer.elapsed() - coreStats.renderTime);
	frameTimer.reset();

    // destroy events
    cudaEventDestroy(shadeStart);
    cudaEventDestroy(shadeEnd);
    cudaEventDestroy(traceStart);
    cudaEventDestroy(traceEnd);
    cudaEventDestroy(traceShadowStart);
    cudaEventDestroy(traceShadowEnd);
    cudaEventDestroy(filterStart);
    cudaEventDestroy(filterEnd);
	cudaEventDestroy(reorderStart);
	cudaEventDestroy(reorderEnd);

	// finalize
    data.primMask = nullptr;

    // restore width
	if (data.cubemap) data.scrsize.x /= 6;

    // store view data
	data.blueSlot = viewData.blueNoiseSlot = (viewData.blueNoiseSlot + data.spp) & 255;
    if (filter) {
		swap(viewData.filter, data.filter.previous);
		if (filterHasFallback)
			swap(views[fallbackID].filter, data.filter.fallback);
		else data.filter.fallback = FilterData();
	} else {
        data.filter.previous = FilterData();
        data.filter.fallback = FilterData();
    }

    cuCheck;
    return *data.count;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::WaitForRender                                                  |
//  |  Note: asynchronous rendering currently not supported                 LH2'20|
//  +-----------------------------------------------------------------------------+
void RenderCore::WaitForRender() {}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Shutdown                                                       |
//  |  Free all resources.                                                  LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Shutdown()
{
	optixPipelineDestroy( pipeline );
	for (int i = 0; i < 5; i++) optixProgramGroupDestroy( progGroup[i] );
	optixModuleDestroy( ptxModule );
	optixDeviceContextDestroy( optixContext );
	cudaFree( (void*)sbt.raygenRecord );
	cudaFree( (void*)sbt.missRecordBase );
	cudaFree( (void*)sbt.hitgroupRecordBase );
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::GetCoreStats                                                   |
//  |  Get a copy of the counters.                                          LH2'19|
//  +-----------------------------------------------------------------------------+
CoreStats RenderCore::GetCoreStats() const
{
	return coreStats;
}
