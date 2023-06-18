﻿/* rendersystem.cpp - Copyright 2019/2020 Utrecht University

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

#include "rendersystem.h"

//  +-----------------------------------------------------------------------------+
//  |  RenderSystem::Init                                                         |
//  |  Initialize the rendering system.                                     LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderSystem::Init( const char* dllName )
{
	// create core
	core = CoreAPI_Base::CreateCoreAPI( dllName );
	// create scene - load a scene using tinyobjloader
	scene = new HostScene();
	scene->Init();
}

//  +-----------------------------------------------------------------------------+
//  |  RenderSystem::SetTarget                                                    |
//  |  Use the specified render target.                                     LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderSystem::SetTarget( GLTexture* target, const uint spp )
{
	// forward to core
	core->SetTarget( target, spp );
	// update camera aspect ratio
	scene->camera->aspectRatio = (float)target->width / (float)target->height;
	scene->camera->pixelCount = make_int2( target->width, target->height );
}

//  +-----------------------------------------------------------------------------+
//  |  RenderSystem::SynchronizeSky                                               |
//  |  Detect changes to the skydome. If a change is found, send the new data to  |
//  |  the core. Note: does not detect changes to pixel data. When this data is   |
//  |  modified, 'MarkAsDirty' should be called on the sky dome object.     LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderSystem::SynchronizeSky()
{
	if (scene->sky && scene->sky->Changed())
	{
		// send sky data to core
		HostSkyDome* sky = scene->sky;
		core->SetSkyData( sky->pixels, sky->width, sky->height, sky->worldToLight );
	}
}

//  +-----------------------------------------------------------------------------+
//  |  RenderSystem::SynchronizeTextures                                          |
//  |  Detect changes to the textures. TODO: currently, the system always sends   |
//  |  all textures to the core whenever any of them changes.               LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderSystem::SynchronizeTextures()
{
	bool texturesDirty = false;
	for (auto texture : scene->textures) if (texture->Changed()) texturesDirty = true;
	if (texturesDirty)
	{
		// send texture data to core
		vector<CoreTexDesc> gpuTex;
		for (auto texture : scene->textures) gpuTex.push_back( texture->ConvertToCoreTexDesc() );
		core->SetTextures( gpuTex.data(), (int)gpuTex.size() );
	}
}

//  +-----------------------------------------------------------------------------+
//  |  RenderSystem::SynchronizeMaterials                                         |
//  |  Detect changes to the materials. Note: material data is small, so it is    |
//  |  probably fine to send all materials whenever a single one changes.   LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderSystem::SynchronizeMaterials()
{
	bool materialsDirty = false;
	for (auto material : scene->materials) if (material->Changed())
	{
		// send all material data to core
		vector<CoreMaterial> gpuMaterial;
		for (auto sceneMat : scene->materials)
		{
			CoreMaterial m;
			memcpy( &m, sceneMat, sizeof( CoreMaterial ) );
			gpuMaterial.push_back( m );
		}
		core->SetMaterials( gpuMaterial.data(), (int)gpuMaterial.size() );
		break;
	}
}

//  +-----------------------------------------------------------------------------+
//  |  RenderSystem::SynchronizeMeshes                                            |
//  |  Detect changes to scene models. Note: right now, if a single model was     |
//  |  modified, all geometry is sent to the core. This works fine when all       |
//  |  geometry is initialized at once, but for more dynamic scenes we will need  |
//  |  a better system.                                                     LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderSystem::SynchronizeMeshes()
{
	for (int s = (int)scene->meshPool.size(), modelIdx = 0; modelIdx < s; modelIdx++)
	{
		HostMesh* mesh = scene->meshPool[modelIdx];
		if (mesh->Changed())
		{
			mesh->MarkAsNotDirty();
			core->SetGeometry( modelIdx, mesh->vertices.data(), (int)mesh->vertices.size(), (int)mesh->triangles.size(), (CoreTri*)mesh->triangles.data() );
			meshesChanged = true; // trigger scene graph update
		}
	}
}

//  +-----------------------------------------------------------------------------+
//  |  RenderSystem::UpdateSceneGraph                                             |
//  |  Walk the scene graph:                                                      |
//  |  - update all node matrices                                                 |
//  |  - update the instance array (where an 'instance' is a node with            |
//  |    a mesh)                                                            LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderSystem::UpdateSceneGraph()
{
	// walk the scene graph to update matrices
	Timer timer;
	int instanceCount = 0;
	bool instancesChanged = false;
	for (int nodeIdx : HostScene::rootNodes)
	{
		HostNode* node = HostScene::nodePool[nodeIdx];
		mat4 T;
		instancesChanged |= node->Update( T /* start with an identity matrix */, instances, instanceCount );
	}
	stats.sceneUpdateTime = timer.elapsed();
	// synchronize instances to device if anything changed
	if (instancesChanged || meshesChanged || instances.size() != instanceCount)
	{
		// resize vector (this is free if the size didn't change)
		instances.resize( instanceCount );
		// send instances to core
		for (int instanceIdx = 0; instanceIdx < instanceCount; instanceIdx++)
		{
			HostNode* node = HostScene::nodePool[instances[instanceIdx]];
			node->instanceID = instanceIdx;
			int dummy = node->Changed(); // prevent superfluous update in the next frame
			core->SetInstance( instanceIdx, node->meshID, node->combinedTransform );
		}
		core->SetInstance( instanceCount, -1 );
		meshesChanged = false;
	}
	// allow the core to finalize after receiving all instances
	core->FinalizeInstances();
}

//  +-----------------------------------------------------------------------------+
//  |  RenderSystem::SynchronizeLights                                            |
//  |  Detect changes to the lights. Note: light data is small, so we can safely  |
//  |  send all data when something changes.                                LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderSystem::SynchronizeLights()
{
	bool lightsDirty = false;
	for (auto light : scene->areaLights) if (light->Changed()) lightsDirty = true;
	for (auto light : scene->pointLights) if (light->Changed()) lightsDirty = true;
	for (auto light : scene->spotLights) if (light->Changed()) lightsDirty = true;
	for (auto light : scene->directionalLights) if (light->Changed()) lightsDirty = true;
	if (lightsDirty)
	{
		// send delta lights to core
		vector<CoreLightTri> gpuAreaLights;
		vector<CorePointLight> gpuPointLights;
		vector<CoreSpotLight> gpuSpotLights;
		vector<CoreDirectionalLight> gpuDirectionalLights;
		for (auto light : scene->areaLights) if (light->enabled) gpuAreaLights.push_back( light->ConvertToCoreLightTri() );
		for (auto light : scene->pointLights) if (light->enabled) gpuPointLights.push_back( light->ConvertToCorePointLight() );
		for (auto light : scene->spotLights) if (light->enabled) gpuSpotLights.push_back( light->ConvertToCoreSpotLight() );
		for (auto light : scene->directionalLights) if (light->enabled) gpuDirectionalLights.push_back( light->ConvertToCoreDirectionalLight() );
		core->SetLights( gpuAreaLights.data(), (int)gpuAreaLights.size(),
			gpuPointLights.data(), (int)gpuPointLights.size(),
			gpuSpotLights.data(), (int)gpuSpotLights.size(),
			gpuDirectionalLights.data(), (int)gpuDirectionalLights.size() );
	}
}

//  +-----------------------------------------------------------------------------+
//  |  RenderSystem::Synchronize                                                  |
//  |  Send modified data to the RenderCore layer.                                |
//  |  Modifications are detected using the Changed() method implemented for most |
//  |  scene-related objects. These rely on a crc64 checksum; theoretically it is |
//  |  possible that a change goes undetected. Ideally, the system should at      |
//  |  least not crash when this happens.                                   LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderSystem::SynchronizeSceneData()
{
	SynchronizeSky();
	SynchronizeTextures();
	SynchronizeMaterials();
	SynchronizeMeshes();
	UpdateSceneGraph();
	SynchronizeLights();
}

//  +-----------------------------------------------------------------------------+
//  |  RenderSystem::Setting                                                      |
//  |  Set settings.                                                        LH2'20|
//  +-----------------------------------------------------------------------------+
void RenderSystem::Setting( const char* name, const float value ) {
    /* TODO remove
    auto& s = scene->camera->pixelCount;
    if(!strcmp(name, "width")) {
        s.x = value;
        if(s.y != 0) scene->camera->aspectRatio = (float)s.x / (float)s.y;
    }
    else if(!strcmp(name, "height")) {
        s.y = value;
        if(s.y != 0) scene->camera->aspectRatio = (float)s.x / (float)s.y;
    }*/
    if (core) core->Setting( name, value );
}

//  +-----------------------------------------------------------------------------+
//  |  RenderSystem::Render                                                       |
//  |  Produce one image.                                                   LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderSystem::Render( const ViewPyramid& view, Convergence converge, bool async )
{
	// forward to core; core may ignore or accept a setting
	core->Setting( "epsilon", settings.geometryEpsilon );
	core->Setting( "clampValue", scene->camera->clampValue );
	core->Setting( "clampDirect", settings.filterDirectClamp );
	core->Setting( "clampIndirect", settings.filterIndirectClamp );
	core->Setting( "filter", settings.filterEnabled );
	core->Setting( "TAA", settings.TAAEnabled );
	core->Render( view, converge, async );
}

//  +-----------------------------------------------------------------------------+
//  |  RenderSystem::WaitForRender                                                |
//  |  Wait for the asynchronous renderer to complete.                      LH2'20|
//  +-----------------------------------------------------------------------------+
void RenderSystem::WaitForRender()
{
	core->WaitForRender();
}

//  +-----------------------------------------------------------------------------+
//  |  RenderSystem::GetTriangleMaterial                                          |
//  |  Retrieve the material ID for the specified triangle. Input is the          |
//  |  'instance id' and 'core tri id' reported by the core for a mouse click.    |
//  |                                                                       LH2'19|
//  +-----------------------------------------------------------------------------+
int RenderSystem::GetTriangleMaterial( const int coreInstId, const int coreTriId )
{
	// see the notes at the top of host_scene.h for the relation between host nodes and core instances.
	if (coreTriId == -1) return -1; // probed the skydome
	if (coreInstId > instances.size()) return -1; // should not happen
	int nodeId = instances[coreInstId]; // lookup the node id for the core instance
	if (nodeId > scene->nodePool.size()) return -1; // should not happen
	int meshId = scene->nodePool[nodeId]->meshID; // get the id of the mesh referenced by the node
	if (meshId == -1) return -1; // should not happen
	if (coreTriId > scene->meshPool[meshId]->triangles.size()) return -1; // should not happen
	return scene->meshPool[meshId]->triangles[coreTriId].material;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderSystem::GetTriangleMaterial                                          |
//  |  Retrieve the id of the host-side mesh that the specified triangle belongs  |
//  |  to. Input is the 'instance id' and 'core tri id' reported by the core for  |
//  |  a mouse click.                                                       LH2'19|
//  +-----------------------------------------------------------------------------+
int RenderSystem::GetTriangleMesh( const int coreInstId, const int coreTriId )
{
	// see the notes at the top of host_scene.h for the relation between host nodes and core instances.
	if (coreTriId == -1) return -1; // probed the skydome
	if (coreInstId > instances.size()) return -1; // should not happen
	int nodeId = instances[coreInstId]; // lookup the node id for the core instance
	if (nodeId > scene->nodePool.size()) return -1; // should not happen
	return scene->nodePool[nodeId]->meshID; // return the id of the mesh referenced by the node
}

//  +-----------------------------------------------------------------------------+
//  |  RenderSystem::GetTriangleNode                                              |
//  |  Retrieve the id of the host-side node that the specified triangle belongs  |
//  |  to. Input is the 'instance id' and 'core tri id' reported by the core for  |
//  |  a mouse click.                                                       LH2'19|
//  +-----------------------------------------------------------------------------+
int RenderSystem::GetTriangleNode( const int coreInstId, const int coreTriId )
{
	// see the notes at the top of host_scene.h for the relation between host nodes and core instances.
	if (coreTriId == -1) return -1; // probed the skydome
	if (coreInstId > instances.size()) return -1; // should not happen
	int nodeId = instances[coreInstId]; // lookup the node id for the core instance
	if (nodeId > scene->nodePool.size()) return -1; // should not happen
	return nodeId; // return the id
}

//  +-----------------------------------------------------------------------------+
//  |  RenderSystem::Shutdown                                                     |
//  |  Free all resources.                                                  LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderSystem::Shutdown()
{
	// delete scene
	delete scene;
	// shutdown core
	core->Shutdown();
}

// EOF