/* material_shared.h - Copyright 2019/2020 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   THIS IS A SHARED FILE:
   used in RenderCore_OptixPrime and RenderCore_OptixRTX.
*/

LH2_DEVFUNC float3 linear_rgb_to_ciexyz( const float3 rgb )
{
	return make_float3(
		max( 0.0f, 0.412453f * rgb.x + 0.357580f * rgb.y + 0.180423f * rgb.z ),
		max( 0.0f, 0.212671f * rgb.x + 0.715160f * rgb.y + 0.072169f * rgb.z ),
		max( 0.0f, 0.019334f * rgb.x + 0.119193f * rgb.y + 0.950227f * rgb.z ) );
}

LH2_DEVFUNC float3 ciexyz_to_linear_rgb( const float3 xyz )
{
	return make_float3(
		max( 0.0f, 3.240479f * xyz.x - 1.537150f * xyz.y - 0.498535f * xyz.z ),
		max( 0.0f, -0.969256f * xyz.x + 1.875992f * xyz.y + 0.041556f * xyz.z ),
		max( 0.0f, 0.055648f * xyz.x - 0.204043f * xyz.y + 1.057311f * xyz.z ) );
}

// extract information from the triangle instance, not (directly) related to any material (type)
LH2_DEVFUNC void SetupFrame(
	const float3 D,						// IN:	incoming ray direction, used for consistent normals
	const float u, const float v,		//		barycentric coordinates of intersection point
	const CoreTri4& tri,				//		triangle data
	const int instIdx,					//		instance index, for normal transform
	const bool hasSmoothNormals,		//		model has a normal per vertex (to interpolate)
	float3& N, float3& iN, float3& fN,	// OUT: geometric normal, interpolated normal, final normal (normal mapped)
	float3& T,							//		tangent vector
	float& w
)
{
	const float4 tdata2 = tri.vN0;
	const float4 tdata3 = tri.vN1;
	const float4 tdata4 = tri.vN2;
	const float4 tdata5 = tri.T4;
	// initialize normals
	N = iN = fN = TRI_N;
	T = TRI_T;
	w = 1 - (u + v);
	// calculate interpolated normal
#ifdef OPTIXPRIMEBUILD
	if (hasSmoothNormals) iN = normalize( u * TRI_N0 + v * TRI_N1 + w * TRI_N2 );
#else
	if (hasSmoothNormals) iN = normalize( w * TRI_N0 + u * TRI_N1 + v * TRI_N2 );
#endif
	// transform the normals and tangent for the current instance
	const float3 A = make_float3( instanceDescriptors[instIdx].invTransform.A );
	const float3 B = make_float3( instanceDescriptors[instIdx].invTransform.B );
	const float3 C = make_float3( instanceDescriptors[instIdx].invTransform.C );
	N = normalize( N.x * A + N.y * B + N.z * C );
	iN = normalize( iN.x * A + iN.y * B + iN.z * C );
	T = normalize( T.x * A + T.y * B + T.z * C );
	// "Consistent Normal Interpolation", Reshetov et al., 2010
	const bool backSide = dot( D, N ) > 0;
#ifdef CONSISTENTNORMALS
	const float4 vertexAlpha = tri.alpha4;
#ifdef OPTIXPRIMEBUILD
	const float alpha = u * vertexAlpha.x + v * vertexAlpha.y + w * vertexAlpha.z;
#else
	const float alpha = w * vertexAlpha.x + u * vertexAlpha.y + v * vertexAlpha.z;
#endif
	iN = (backSide ? -1.0f : 1.0f) * ConsistentNormal( D * -1.0f, backSide ? (iN * -1.0f) : iN, alpha );
#endif
	fN = iN;
}

LH2_DEVFUNC void GetShadingData(
	const float3 D,							// IN:	incoming ray direction, used for consistent normals
	const float u, const float v,			//		barycentric coordinates of intersection point
	const float coneWidth,					//		ray cone width, for texture LOD
	const CoreTri4& tri,					//		triangle data
	const int instIdx,						//		instance index, for normal transform
	ShadingData& retVal,					// OUT:	material properties of the intersection point
	float3& N, float3& iN, float3& fN,		//		geometric normal, interpolated normal, final normal (normal mapped)
	float3& T,								//		tangent vector
	const float waveLength = -1.0f,			// IN:	wavelength (optional)
	bool alphaMask = true
)
{
	// Note: GetShadingData is called from the 'shade' code, which is in turn
	// only called for intersections. We thus can assume that we have a valid
	// triangle reference.
	const float4 tdata1 = tri.v4;
	// fetch initial set of data from material
	const CUDAMaterial4& mat = (const CUDAMaterial4&)materials[TRI_MATERIAL];
	const uint4 baseData = mat.baseData4;
	// process common data (unconditional)
	const uint part0 = baseData.x; // diffuse_r, diffuse_g
	const uint part1 = baseData.y; // diffuse_b, medium_r
	const uint part2 = baseData.z; // medium_g, medium_b
	const uint flags = baseData.w;
	const float2 base_rg = __half22float2( __halves2half2( __ushort_as_half( part0 & 0xffff ), __ushort_as_half( part0 >> 16 ) ) );
	const float2 base_b_medium_r = __half22float2( __halves2half2( __ushort_as_half( part1 & 0xffff ), __ushort_as_half( part1 >> 16 ) ) );
	const float2 medium_gb = __half22float2( __halves2half2( __ushort_as_half( part2 & 0xffff ), __ushort_as_half( part2 >> 16 ) ) );
	ShadingData4& retVal4 = (ShadingData4&)retVal;
	retVal4.data0 = make_float4( base_rg.x, base_rg.y, base_b_medium_r.x, __uint_as_float( 0 ) );
	retVal4.data1 = make_float4( base_b_medium_r.y, medium_gb.x, medium_gb.y, __uint_as_float( 0 /* matid? */ ) );
	retVal4.data2 = mat.parameters;
	const float3 tint_xyz = linear_rgb_to_ciexyz( make_float3( base_rg.x, base_rg.y, base_b_medium_r.x ) );
	retVal4.tint4 = make_float4( tint_xyz.y > 0 ? ciexyz_to_linear_rgb( tint_xyz * (1.0f / tint_xyz.y) ) : make_float3( 1 ), tint_xyz.y );
	// initialize normals
	float w;
	SetupFrame( /* Input: */ D, u, v, tri, instIdx, MAT_HASSMOOTHNORMALS, /* Output: */ N, iN, fN, T, w );
	const float4 vertexAlpha = tri.alpha4;
	if (retVal.IsEmissive()) retVal.flags |= EMISSIVE; // set EMISSIVE flag as color can get below 1 after applying texture
#ifdef MAT_EMISSIVE_TWOSIDED
	if (MAT_EMISSIVE_TWOSIDED) retVal.flags |= EMISSIVE_TWOSIDED;
#endif
	// texturing
	float tu, tv;
	if (MAT_HASDIFFUSEMAP || MAT_HAS2NDDIFFUSEMAP || MAT_HASSPECULARITYMAP || MAT_HASNORMALMAP || MAT_HAS2NDNORMALMAP || MAT_HASROUGHNESSMAP)
	{
		const float4 tdata0 = tri.u4;
		const float w = 1 - (u + v);
	#ifdef OPTIXPRIMEBUILD
		tu = u * TRI_U0 + v * TRI_U1 + w * TRI_U2;
		tv = u * TRI_V0 + v * TRI_V1 + w * TRI_V2;
	#else
		tu = w * TRI_U0 + u * TRI_U1 + v * TRI_U2;
		tv = w * TRI_V0 + u * TRI_V1 + v * TRI_V2;
	#endif
	}
	if (MAT_HASDIFFUSEMAP)
	{
		// determine LOD
		const float lambda = instanceDescriptors[instIdx].LOD + TRI_LOD + log2f( coneWidth * (1.0f / fabs( dot( D, N ) )) ); // eq. 26
		const uint4 t0data = mat.t0data4; // layout: struct Map { short width, height; half uscale, vscale, uoffs, voffs; uint addr; }
		// fetch texels
		const float2 uvscale = __half22float2( __halves2half2( __ushort_as_half( t0data.y & 0xffff ), __ushort_as_half( t0data.y >> 16 ) ) );
		const float2 uvoffs = __half22float2( __halves2half2( __ushort_as_half( t0data.z & 0xffff ), __ushort_as_half( t0data.z >> 16 ) ) );
		const float4 texel = FetchTexelTrilinear( lambda, uvscale * (uvoffs + make_float2( tu, tv )), t0data.w, t0data.x & 0xffff, t0data.x >> 16 );
		if (alphaMask && texel.w < 0.5f)
		{
			retVal.flags |= ALPHA;
			return;
		}
		retVal.color = retVal.color * make_float3( texel );
		if (MAT_HAS2NDDIFFUSEMAP) // must have base texture; second and third layers are additive
		{
			const uint4 t1data = mat.t1data4; // layout: struct Map { short width, height; half uscale, vscale, uoffs, voffs; uint addr; }
			const float2 uvscale = __half22float2( __halves2half2( __ushort_as_half( t1data.y & 0xffff ), __ushort_as_half( t1data.y >> 16 ) ) );
			const float2 uvoffs = __half22float2( __halves2half2( __ushort_as_half( t1data.z & 0xffff ), __ushort_as_half( t1data.z >> 16 ) ) );
			retVal.color += make_float3( FetchTexel( uvscale * (uvoffs + make_float2( tu, tv )), t1data.w, t1data.x & 0xffff, t1data.x >> 16 ) ) - make_float3( 0.5f );
		}
	}
	// normal mapping
	if (MAT_HASNORMALMAP)
	{
		// fetch bitangent for applying normal map vector to geometric normal
		float4 tdata6 = tri.B4;
		float3 B = TRI_B;
		const uint4 n0data = mat.n0data4; // layout: struct Map { short width, height; half uscale, vscale, uoffs, voffs; uint addr; }
		const uint part3 = baseData.z;
		const float n0scale = copysignf( -0.0001f + 0.0001f * __expf( 0.1f * fabsf( (float)((part3 >> 8) & 255) - 128.0f ) ), (float)((part3 >> 8) & 255) - 128.0f );
		const float2 uvscale = __half22float2( __halves2half2( __ushort_as_half( n0data.y & 0xffff ), __ushort_as_half( n0data.y >> 16 ) ) );
		const float2 uvoffs = __half22float2( __halves2half2( __ushort_as_half( n0data.z & 0xffff ), __ushort_as_half( n0data.z >> 16 ) ) );
		float3 shadingNormal = (make_float3( FetchTexel( uvscale * (uvoffs + make_float2( tu, tv )), n0data.w, n0data.x & 0xffff, n0data.x >> 16, NRM32 ) ) - make_float3( 0.5f )) * 2.0f;
		shadingNormal.x *= n0scale, shadingNormal.y *= n0scale;
		if (MAT_HAS2NDNORMALMAP)
		{
			const uint4 n1data = mat.n1data4; // layout: struct Map { short width, height; half uscale, vscale, uoffs, voffs; uint addr; }
			const float n1scale = copysignf( -0.0001f + 0.0001f * __expf( 0.1f * ((float)((part3 >> 16) & 255) - 128.0f) ), (float)((part3 >> 16) & 255) - 128.0f );
			const float2 uvscale = __half22float2( __halves2half2( __ushort_as_half( n1data.y & 0xffff ), __ushort_as_half( n1data.y >> 16 ) ) );
			const float2 uvoffs = __half22float2( __halves2half2( __ushort_as_half( n1data.z & 0xffff ), __ushort_as_half( n1data.z >> 16 ) ) );
			float3 normalLayer1 = (make_float3( FetchTexel( uvscale * (uvoffs + make_float2( tu, tv )), n1data.w, n1data.x & 0xffff, n1data.x >> 16, NRM32 ) ) - make_float3( 0.5f )) * 2.0f;
			normalLayer1.x *= n1scale, normalLayer1.y *= n1scale;
			shadingNormal += normalLayer1;
		}
		shadingNormal = normalize( shadingNormal );
		fN = normalize( shadingNormal.x * T + shadingNormal.y * B + shadingNormal.z * iN );
	}
	// roughness map. Note: gltf stores roughness and metalness in a single map, so we'll assume we have metalness as well.
	if (MAT_HASROUGHNESSMAP)
	{
		const uint4 rmdata = mat.rdata4; // layout: struct Map { short width, height; half uscale, vscale, uoffs, voffs; uint addr; }
		const float2 uvscale = __half22float2( __halves2half2( __ushort_as_half( rmdata.y & 0xffff ), __ushort_as_half( rmdata.y >> 16 ) ) );
		const float2 uvoffs = __half22float2( __halves2half2( __ushort_as_half( rmdata.z & 0xffff ), __ushort_as_half( rmdata.z >> 16 ) ) );
		const float4 texel = FetchTexel( uvscale * (uvoffs + make_float2( tu, tv )), rmdata.w, rmdata.x & 0xffff, rmdata.x >> 16 );
		retVal.parameters.x = (retVal.parameters.x & 0x00ffffff) + ((int)(texel.y * 255.0f) << 24);
		retVal.parameters.x = (retVal.parameters.x & 0xffffff00) + (int)(texel.x * 255.0f);
	}
#ifdef FILTERINGCORE
	// prevent r, g and b from becoming zero, for albedo separation
	retVal.color.x = max( 0.05f, retVal.color.x );
	retVal.color.y = max( 0.05f, retVal.color.y );
	retVal.color.z = max( 0.05f, retVal.color.z );
#endif
}

// EOF