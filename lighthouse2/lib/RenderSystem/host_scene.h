/* host_scene.h - Copyright 2019/2020 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   the Lighthouse 2 scene database, ingredients:
   1. vector<int> rootNodes
	  This is a list of indices of nodes that should be rendered. Each node
	  may be the top of a hierarchy, or it may be a node that directly
	  references a mesh.
   2. vector<HostNode*> nodes
	  This is a collection of all the nodes in the scene. The nodes may be
	  visible or not, and the collection may include nullptrs, in case nodes
	  have been deleted. In that case, 'nodeListHoles' is greater than 0.
   3. vector<HostMesh*> meshes
	  The collection of meshes, i.e. the actual geometry. Each mesh may be
	  referenced by 0 or more nodes.
   4. vector<HostSkin*> skins
	  Some node hierarchies use a deformable mesh, for which the HostSkin
	  contains the relevant data.

   Note that the concept 'instance' does not appear in the above overview.
   An instance is simply a node that references a mesh.

   In rendersystem.cpp the above data is converted to a simpler structure
   for the cores. Cores receive:
   - A collection of meshes, which mirrors HostScene::meshes;
   - A collection of instances. Here, an 'instance' is a core-specific
	 structure, that references a mesh, and stores the flattened transform
	 of the host node. All other nodes are irrelevant to the cores and
	 merely serve to produce the final matrices for the instances.
*/

#pragma once

#include "rendersystem.h"

namespace lighthouse2
{

//  +-----------------------------------------------------------------------------+
//  |  HostScene                                                                  |
//  |  Module for scene I/O and host-side management.                             |
//  |  This is a pure static class; we will not have more than one scene.   LH2'19|
//  +-----------------------------------------------------------------------------+
class HostNode;
class HostScene
{
public:
	// constructor / destructor
	HostScene();
	~HostScene();
	// serialization / deserialization
	static void SerializeMaterials( const char* xmlFile );
	static void DeserializeMaterials( const char* xmlFile );
	// methods
	static void Init();
	static void SetSkyDome( HostSkyDome* );
	static int FindOrCreateTexture( const string& origin, const uint modFlags = 0 );
	static int FindTextureID( const char* name );
	static int CreateTexture( const string& origin, const uint modFlags = 0 );
	static int FindOrCreateMaterial( const string& name );
	static int FindOrCreateMaterialCopy( const int matID, const uint color );
	static int FindMaterialID( const char* name );
	static int FindMaterialIDByOrigin( const char* name );
	static int FindNextMaterialID( const char* name, const int matID );
	static int FindNode( const char* name );
	static void SetNodeTransform( const int nodeId, const mat4& transform );
	static const mat4& GetNodeTransform( const int nodeId );
	static void ResetAnimation( const int animId );
	static void UpdateAnimation( const int animId, const float dt );
	static int AnimationCount() { return (int)animations.size(); }
	// scene construction / maintenance
	static int AddMesh( HostMesh* mesh );
	static int AddMesh( const char* objFile, const char* dir, const float scale = 1.0f, const bool flatShaded = false );
	static int AddMesh( const char* objFile, const float scale = 1.0f, const bool flatShaded = false );
	static int AddScene( const char* sceneFile, const mat4& transform = mat4::Identity() );
	static int AddScene( const char* sceneFile, const char* dir, const mat4& transform );
	static int AddMesh( const int triCount );
	static void AddTriToMesh( const int meshId, const float3& v0, const float3& v1, const float3& v2, const int matId );
	static int AddQuad( const float3 N, const float3 pos, const float width, const float height, const int matId, const int meshID = -1 );
	static int AddInstance( HostNode* node );
	static int AddInstance( const int meshId, const mat4& transform );
	static void RemoveNode( const int instId );
	static int AddMaterial( HostMaterial* material );
	static int AddMaterial( const float3 color, const char* name = 0 );
	static int AddPointLight( const float3 pos, const float3 radiance, bool enabled = true );
	static int AddSpotLight( const float3 pos, const float3 direction, const float inner, const float outer, const float3 radiance, bool enabled = true );
	static int AddDirectionalLight( const float3 direction, const float3 radiance, bool enabled = true );
	// data members
	static inline vector<int> rootNodes;
	static inline vector<HostNode*> nodePool;
	static inline vector<HostMesh*> meshPool;
	static inline vector<HostSkin*> skins;
	static inline vector<HostAnimation*> animations;
	static inline vector<HostMaterial*> materials;
	static inline vector<HostTexture*> textures;
	static inline vector<HostAreaLight*> areaLights;
	static inline vector<HostPointLight*> pointLights;
	static inline vector<HostSpotLight*> spotLights;
	static inline vector<HostDirectionalLight*> directionalLights;
	static inline HostSkyDome* sky;
	static inline Camera* camera;
private:
	static inline int nodeListHoles;	// zero if no instance deletions occurred; adding instances will be faster.
};

} // namespace lighthouse2

// EOF