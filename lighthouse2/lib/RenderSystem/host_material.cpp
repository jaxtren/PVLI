/* host_material.cpp - Copyright 2019/2020 Utrecht University

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

// Lighthouse2 data model design principles:
// 
// - If a data member can be given an obvious name, access is public, saving on LOC and repetition.
// - (Almost) all data members are initialized to promote deterministic behavior in debug and release.
// - Textures, materials, meshes etc. have IDs; use of pointers is minimized.
// - Textures, materials, meshes etc. also have an optional name and origin string (not a char*).
// - A struct is either host-oriented or GPU-oriented; the name (as well as the file name) indicates 
//   this. If both are needed, both will be created, even if they are the same.
// - The RenderCore is the only system communicating with the GPU. It is thus also the only owner of
//   device-side data. The RenderSystem is responsible for setting this data before rendering starts.
//   Data in RenderCore reached its final destination; it can thus not be queried, and internal
//   operations are minimal.

//  +-----------------------------------------------------------------------------+
//  |  HostMaterial::ConvertFrom                                                  |
//  |  Converts a tinyobjloader material to a HostMaterial.                 LH2'19|
//  +-----------------------------------------------------------------------------+
void HostMaterial::ConvertFrom( const tinyobjMaterial& original )
{
	// properties
	name = original.name;
	color.value = make_float3( original.diffuse[0], original.diffuse[1], original.diffuse[2] ); // Kd
	absorption.value = make_float3( original.transmittance[0], original.transmittance[1], original.transmittance[2] ); // Kt
	roughness = max( 0.0f, min( 1 - original.shininess, 1.0f ) );
	// maps
	if (original.diffuse_texname != "")
	{
		int diffuseTextureID = color.textureID = HostScene::FindOrCreateTexture( original.diffuse_texname, HostTexture::LINEARIZED | HostTexture::FLIPPED );
		color.value = make_float3( 1 ); // we have a texture now; default modulation to white
	}
	if (original.normal_texname != "")
	{
		normals.textureID = HostScene::FindOrCreateTexture( original.normal_texname, HostTexture::FLIPPED );
		HostScene::textures[normals.textureID]->flags |= HostTexture::NORMALMAP; // TODO: what if it's also used as regular texture?
	}
	else if (original.bump_texname != "")
	{
		int bumpMapID = normals.textureID = HostScene::CreateTexture( original.bump_texname, HostTexture::FLIPPED ); // cannot reuse, height scale may differ
		float heightScaler = 1.0f;
		auto heightScalerIt = original.unknown_parameter.find( "bump_height" );
		if (heightScalerIt != original.unknown_parameter.end()) heightScaler = static_cast<float>(atof( (*heightScalerIt).second.c_str() ));
		HostScene::textures[bumpMapID]->BumpToNormalMap( heightScaler );
		HostScene::textures[bumpMapID]->flags |= HostTexture::NORMALMAP; // TODO: what if it's also used as regular texture?
	}
	if (original.specular_texname != "")
	{
		roughness.textureID = HostScene::FindOrCreateTexture( original.specular_texname.c_str(), HostTexture::FLIPPED );
		roughness() = 1.0f;
	}
	// finalize
	auto shadingIt = original.unknown_parameter.find( "shading" );
	if (shadingIt != original.unknown_parameter.end() && shadingIt->second == "flat") flags &= ~SMOOTH; else flags |= SMOOTH;
}

//  +-----------------------------------------------------------------------------+
//  |  HostMaterial::ConvertFrom                                                  |
//  |  Converts a tinygltf material to a HostMaterial.                      LH2'19|
//  +-----------------------------------------------------------------------------+
void HostMaterial::ConvertFrom( const tinygltfMaterial& original, const tinygltfModel& model, const vector<int>& texIdx )
{
	name = original.name;
	flags |= HostMaterial::FROM_MTL; // this material will be serialized on exit.
	// default values
    roughness.value = 1;
	metallic.value = 1;
	eta.value = 1.001f; // value 1 doesn't work
	// set normal map, if any
	if (original.normalTexture.index > -1)
	{
		// note: may be overwritten by the "normalTexture" field in additionalValues.
		normals.textureID = texIdx[original.normalTexture.index];
		normals.scale = original.normalTexture.scale;
		HostScene::textures[normals.textureID]->flags |= HostTexture::NORMALMAP;
	}
	// process values list
	for (const auto& value : original.values)
	{
		if (value.first == "baseColorFactor")
		{
			tinygltf::Parameter p = value.second;
			color.value = make_float3( p.number_array[0], p.number_array[1], p.number_array[2] );
			if (p.number_array.size() > 3)
                transmission.value = 1.0 - p.number_array[3];
		}
		if (value.first == "metallicFactor") if (value.second.has_number_value)
		{
			metallic.value = (float)value.second.number_value;
		}
		if (value.first == "roughnessFactor") if (value.second.has_number_value)
		{
			roughness.value = (float)value.second.number_value;
		}
		if (value.first == "baseColorTexture") for (auto& item : value.second.json_double_value)
		{
			if (item.first == "index") color.textureID = texIdx[(int)item.second];
		}
		if (value.first == "metallicRoughnessTexture") for (auto& item : value.second.json_double_value)
		{
			if (item.first == "index") 
			{
				roughness.textureID = texIdx[(int)item.second];	// green channel contains roughness
				metallic.textureID = texIdx[(int)item.second];	// blue channel contains metalness
			}
		}
	}
	// process additionalValues list
	for (const auto& value : original.additionalValues)
	{
		if (value.first == "doubleSided")
		{
			// ignored; all faces are double sided in LH2.
		}
		if (value.first == "normalTexture")
		{
			tinygltf::Parameter p = value.second;
			for (auto& item : value.second.json_double_value)
			{
				if (item.first == "index") normals.textureID = texIdx[(int)item.second];
				if (item.first == "scale") normals.scale = item.second;
				if (item.first == "texCoord") { /* TODO */ };
			}
		}
		if (value.first == "occlusionTexture")
		{
			// ignored; the occlusion map stores baked AO, but LH2 is a path tracer.
		}
	}
	// process extensions
	// NOTE: LH2 currently does not properly support PBR materials. Below code is merely
	// here to ease a future proper implementation.
	for (const auto& extension : original.extensions)
	{
		if (extension.first == "KHR_materials_pbrSpecularGlossiness" )
		{
			tinygltf::Value value = extension.second;
			if (value.IsObject())
			{
				for (const auto& key : value.Keys())
				{
					if (key == "diffuseFactor")
					{
						tinygltf::Value v = value.Get( key );
						int w = 0; // TODO
					}
					if (key == "diffuseTexture" )
					{
						tinygltf::Value v = value.Get( key );
						color.textureID = texIdx[v.GetNumberAsInt()];

					}
					if (key == "glossinessFactor" )
					{
						tinygltf::Value v = value.Get( key );
						float glossyness = (float)v.GetNumberAsDouble();
						roughness = 1 - glossyness;
					}
					if (key == "specularFactor" )
					{
						tinygltf::Value v = value.Get( key );
						int w = 0; // TODO
					}
					if (key == "specularGlossinessTexture" )
					{
						tinygltf::Value v = value.Get( key );
						int w = 0; // TODO
					}
				}
			}
		}
	}

	if (original.emissiveFactor.size() == 3)
	{
		auto& f  = original.emissiveFactor;
		float s = 1.001f;
		if (original.extras.Has("emissionStrength"))
			s = original.extras.Get("emissionStrength").GetNumberAsDouble();
		auto e = make_float3( f[0], f[1], f[2] ) * s;
		if (e.x > 1 || e.y > 1 || e.z > 1) color.value = e;
	}

	if (original.extras.Has("ior"))
		eta.value = 1.0 / original.extras.Get("ior").GetNumberAsDouble();

	if (original.extras.Has("eta"))
		eta.value = original.extras.Get("eta").GetNumberAsDouble();

	if (original.extras.Has("transmission"))
		transmission.value = original.extras.Get("transmission").GetNumberAsDouble();

    if (original.extras.Has("absorption"))
		absorption.value = make_float3(original.extras.Get("absorption").GetNumberAsDouble());
}

// EOF