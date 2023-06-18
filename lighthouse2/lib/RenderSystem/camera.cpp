/* camera.cpp - Copyright 2019/2020 Utrecht University

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
//  |  Camera::Camera                                                             |
//  |  Constructor.                                                         LH2'19|
//  +-----------------------------------------------------------------------------+
Camera::Camera( const char* xmlFile )
{
	Deserialize( xmlFile );
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::~Camera                                                            |
//  |  Destructor.                                                          LH2'19|
//  +-----------------------------------------------------------------------------+
Camera::~Camera()
{
	Serialize( xmlFile.c_str() );
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::CalculateMatrix                                                    |
//  |  Helper function; constructs camera matrix.                           LH2'19|
//  +-----------------------------------------------------------------------------+
void Camera::CalculateMatrix( float3& x, float3& y, float3& z ) const
{
	x = make_float3( transform.cell[0], transform.cell[4], transform.cell[8] );
	y = make_float3( transform.cell[1], transform.cell[5], transform.cell[9] );
	z = make_float3( transform.cell[2], transform.cell[6], transform.cell[10] );
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::GetMatrix                                                          |
//  |  Return the current camera view in the form of a matrix.              LH2'20|
//  +-----------------------------------------------------------------------------+
mat4 Camera::GetMatrix() const
{
	return transform;
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::SetMatrix                                                          |
//  |  Set the camera view using a matrix.                                  LH2'20|
//  +-----------------------------------------------------------------------------+
void Camera::SetMatrix( mat4& T )
{
	transform = T;
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::LookAt                                                             |
//  |  Position and aim the camera.                                         LH2'19|
//  +-----------------------------------------------------------------------------+
void Camera::LookAt( const float3 O, const float3 T )
{
	transform = mat4::LookAt( O, T, make_float3( 0, 1, 0 ) );
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::TranslateRelative                                                  |
//  |  Move the camera with respect to the current orientation.             LH2'19|
//  +-----------------------------------------------------------------------------+
void Camera::TranslateRelative( float3 T )
{
	float3 x, y, z;
	CalculateMatrix( x, y, z );
	float3 delta = T.x * x + T.y * y + T.z * z;
	transform.SetTranslation( transform.GetTranslation() + delta );
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::TranslateTarget                                                    |
//  |  Move the camera target with respect to the current orientation.      LH2'19|
//  +-----------------------------------------------------------------------------+
void Camera::TranslateTarget( float3 T )
{
	float3 x, y, z;
	CalculateMatrix( x, y, z );
	float3 delta = T.x * x + T.y * y + T.z * z;
	z = normalize( z + delta );
	x = normalize( cross( z, normalize( make_float3( 0, 1, 0 ) ) ) );
	y = cross( x, z );
	transform[0] = x.x, transform[4] = x.y, transform[8] = x.z;
	transform[1] = y.x, transform[5] = y.y, transform[9] = y.z;
	transform[2] = z.x, transform[6] = z.y, transform[10] = z.z;
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::GetView                                                            |
//  |  Create a ViewPyramid for rendering in the RenderCore layer.          LH2'19|
//  +-----------------------------------------------------------------------------+
ViewPyramid Camera::GetView() const
{
	ViewPyramid view;
	float3 right, up, forward;
	CalculateMatrix( right, up, forward );
	view.pos = transform.GetTranslation();
	view.spreadAngle = (FOV * PI / 180) / (float)pixelCount.y;
	const float screenSize = tanf( FOV / 2 / (180 / PI) );
	const float3 C = view.pos + focalDistance * forward;
	view.p1 = C - screenSize * right * focalDistance * aspectRatio + screenSize * focalDistance * up;
	view.p2 = C + screenSize * right * focalDistance * aspectRatio + screenSize * focalDistance * up;
	view.p3 = C - screenSize * right * focalDistance * aspectRatio - screenSize * focalDistance * up;
	view.aperture = aperture;
	view.focalDistance = focalDistance;
	view.distortion = distortion;
	// BDPT
	float3 unitP1 = C - screenSize * right * aspectRatio + screenSize * up;
	float3 unitP2 = C + screenSize * right * aspectRatio + screenSize * up;
	float3 unitP3 = C - screenSize * right * aspectRatio - screenSize * up;
	view.imagePlane = length( unitP1 - unitP2 ) * length( unitP1 - unitP3 );
	return view;
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::WorldToScreenPos                                                   |
//  |  Converts an array of world positions to screen positions.                  |
//  |  Helper function for Joram's navmesh code.                            LH2'19|
//  +-----------------------------------------------------------------------------+
void Camera::WorldToScreenPos( const float3* W, float2* S, int count ) const
{
	// calculate camera axis
	ViewPyramid p = GetView();
	float3 p1p2 = p.p2 - p.p1, p3p1 = p.p1 - p.p3;		// screen edges
	float3 f = ((p.p3 - p.pos) + (p.p2 - p.pos)) / 2;	// focal point
	float3 x = normalize( p1p2 );						// camera unit axis
	float3 y = normalize( p3p1 );						// camera unit axis
	float3 z = normalize( f );							// camera unit axis
	float invflen = 1 / length( f );					// the inversed focal distance
	float invxscrlen = 1 / (length( p1p2 ) * .5f);		// half the screen width inversed
	float invyscrlen = 1 / (length( p3p1 ) * .5f);		// half the screen height inversed
	// transform coordinates
	float3 dir;
	for (int i = 0; i < count; i++)
	{
		dir = W[i] - p.pos;								// vector from camera to pos
		dir = { dot( dir, x ), dot( dir, y ), dot( dir, z ) }; // make dir relative to camera
		if (dir.z < 0) dir *= {1000.0f, 1000.0f, 0.0f};	// prevent looking backwards (TODO: improve)
		dir *= (1 / (dir.z * invflen));					// trim dir to hit the screen
		dir.x *= invxscrlen; dir.y *= invyscrlen;		// convert to screen scale
		S[i] = make_float2( dir );
	}
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::WorldToScreenPos                                                   |
//  |  Calculates the 2D screen position of a 3D world coordinate. Note: if the   |
//  |  screen distortion is 0, an analytic solution is trivial. When distortion   |
//  |  is active we will search the correct screen location. WARNING: search is   |
//  |  expensive; only suitable for incidental invocation.                  LH2'20|
//  +-----------------------------------------------------------------------------+
float EvaluatePos( const float3& toP, const float tx, const float ty, const float distortion, const ViewPyramid& p )
{
	const float rr = tx * tx + ty * ty, rq = sqrtf( rr ) * (1.0f + p.distortion * rr + p.distortion * rr * rr);
	const float theta = atan2f( tx, ty ), px = sinf( theta ) * rq + 0.5f, py = cosf( theta ) * rq + 0.5f;
	return dot( normalize( (p.p1 + px * (p.p2 - p.p1) + py * (p.p3 - p.p1)) - p.pos ), toP );
}
int2 Camera::WorldToScreenPos( const float3& P )
{
	// calculate camera axis
	const ViewPyramid p = GetView();
	const float3 p1p2 = p.p2 - p.p1, p3p1 = p.p1 - p.p3;
	const float3 f = ((p.p3 - p.pos) + (p.p2 - p.pos)) * 0.5f;
	const float rl12 = 1.0f / length( p1p2 ), rl31 = 1.0f / length( p3p1 ), rlf = 1.0f / length( f );
	const float3 x = p1p2 * rl12, y = p3p1 * rl31, z = f * rlf;
	float3 dir = P - p.pos;
	dir = make_float3( dot( dir, x ), dot( dir, y ), dot( dir, z ) );
	if (dir.z < 0) dir *= {1000.0f, 1000.0f, 0.0f};
	dir /= dir.z * rlf;
	float tx = dir.x * rl12, ty = -dir.y * rl31;
	if (p.distortion == 0) return make_int2( (tx + 0.5f) * pixelCount.x, (ty + 0.5f) * pixelCount.y );
	// distorted view; refine using diamond search
	const float3 toP = normalize( P - p.pos );
	float bestScore = EvaluatePos( toP, tx, ty, distortion, p ), origBest = bestScore;
	float bestTx = tx, bestTy = ty, stepSize = 64.0f / pixelCount.x, e;
	int evals = 1;
	for (int i = 0; i < 32; i++)
	{
		e = EvaluatePos( toP, tx - stepSize, ty, distortion, p ), evals++;
		if (e > bestScore) bestScore = e, tx -= stepSize, bestTx = tx, bestTy = ty; else
		{
			e = EvaluatePos( toP, tx + stepSize, ty, distortion, p ), evals++;
			if (e > bestScore) bestScore = e, tx += stepSize, stepSize = -stepSize, bestTx = tx, bestTy = ty; else
			{
				e = EvaluatePos( toP, tx, ty - stepSize, distortion, p ), evals++;
				if (e > bestScore) bestScore = e, ty -= stepSize, bestTx = tx, bestTy = ty; else
				{
					e = EvaluatePos( toP, tx, ty + stepSize, distortion, p ), evals++;
					if (e > bestScore) bestScore = e, ty += stepSize, stepSize = -stepSize, bestTx = tx, bestTy = ty; else stepSize *= 0.55f;
				}
			}
		}
		if (bestScore > 0.999995f) break; // actually needs to be this precise
	}
#if 0
	printf( "best: %7.5f (from %7.5f), with %i evals, at tx=%i,ty=%i\n",
		bestScore, origBest, evals, (int)((tx + 0.5f) * pixelCount.x), (int)((ty + 0.5f) * pixelCount.y) );
#endif
	return make_int2( (tx + 0.5f) * pixelCount.x, (ty + 0.5f) * pixelCount.y );
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::PrimaryHitPos                                                      |
//  |  Determine the world pos of a hit, at distance, through a pixel.      LH2'20|
//  +-----------------------------------------------------------------------------+
float3 Camera::PrimaryHitPos( int2 pos, float dist )
{
	ViewPyramid view = GetView();
	const float3 right = view.p2 - view.p1;
	const float3 up = view.p3 - view.p1;
	float3 pixelPlanePos = RayTarget( pos.x, pos.y, 0.5f, 0.5f, pixelCount, distortion, view.p1, right, up );
	return normalize( pixelPlanePos - transform.GetTranslation() );
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::Serialize                                                          |
//  |  Save the camera data to the specified xml file.                      LH2'19|
//  +-----------------------------------------------------------------------------+
void Camera::Serialize( const char* xmlFileName )
{
	XMLDocument doc;
	XMLNode* root = doc.NewElement( "camera" );
	doc.InsertFirstChild( root );
	XMLElement* t = doc.NewElement( "transform" );
	t->SetAttribute( "m00", transform.cell[0] );
	t->SetAttribute( "m01", transform.cell[1] );
	t->SetAttribute( "m02", transform.cell[2] );
	t->SetAttribute( "m03", transform.cell[3] );
	t->SetAttribute( "m10", transform.cell[4] );
	t->SetAttribute( "m11", transform.cell[5] );
	t->SetAttribute( "m12", transform.cell[6] );
	t->SetAttribute( "m13", transform.cell[7] );
	t->SetAttribute( "m20", transform.cell[8] );
	t->SetAttribute( "m21", transform.cell[9] );
	t->SetAttribute( "m22", transform.cell[10] );
	t->SetAttribute( "m23", transform.cell[11] );
	t->SetAttribute( "m30", transform.cell[12] );
	t->SetAttribute( "m31", transform.cell[13] );
	t->SetAttribute( "m32", transform.cell[14] );
	t->SetAttribute( "m33", transform.cell[15] );
	root->InsertEndChild( t );
	((XMLElement*)root->InsertEndChild( doc.NewElement( "FOV" ) ))->SetText( FOV );
	((XMLElement*)root->InsertEndChild( doc.NewElement( "brightness" ) ))->SetText( brightness );
	((XMLElement*)root->InsertEndChild( doc.NewElement( "contrast" ) ))->SetText( contrast );
	((XMLElement*)root->InsertEndChild( doc.NewElement( "gamma" ) ))->SetText( gamma );
	((XMLElement*)root->InsertEndChild( doc.NewElement( "aperture" ) ))->SetText( aperture );
	((XMLElement*)root->InsertEndChild( doc.NewElement( "distortion" ) ))->SetText( distortion );
	((XMLElement*)root->InsertEndChild( doc.NewElement( "focalDistance" ) ))->SetText( focalDistance );
	((XMLElement*)root->InsertEndChild( doc.NewElement( "clampValue" ) ))->SetText( clampValue );
	((XMLElement*)root->InsertEndChild( doc.NewElement( "tonemapper" ) ))->SetText( tonemapper );
	doc.SaveFile( xmlFileName ? xmlFileName : xmlFile.c_str() );
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::Deserialize                                                        |
//  |  Load the camera data from the specified xml file.                    LH2'19|
//  +-----------------------------------------------------------------------------+
void Camera::Deserialize( const char* xmlFileName )
{
	xmlFile = xmlFileName;
	XMLDocument doc;
	XMLError result = doc.LoadFile( "camera.xml" );
	if (result != XML_SUCCESS) return;
	XMLNode* root = doc.FirstChild();
	if (root == nullptr) return;
	XMLElement* element = root->FirstChildElement( "transform" );
	if (!element) return;
	element->QueryFloatAttribute( "m00", &transform.cell[0] );
	element->QueryFloatAttribute( "m01", &transform.cell[1] );
	element->QueryFloatAttribute( "m02", &transform.cell[2] );
	element->QueryFloatAttribute( "m03", &transform.cell[3] );
	element->QueryFloatAttribute( "m10", &transform.cell[4] );
	element->QueryFloatAttribute( "m11", &transform.cell[5] );
	element->QueryFloatAttribute( "m12", &transform.cell[6] );
	element->QueryFloatAttribute( "m13", &transform.cell[7] );
	element->QueryFloatAttribute( "m20", &transform.cell[8] );
	element->QueryFloatAttribute( "m21", &transform.cell[9] );
	element->QueryFloatAttribute( "m22", &transform.cell[10] );
	element->QueryFloatAttribute( "m23", &transform.cell[11] );
	element->QueryFloatAttribute( "m30", &transform.cell[12] );
	element->QueryFloatAttribute( "m31", &transform.cell[13] );
	element->QueryFloatAttribute( "m32", &transform.cell[14] );
	element->QueryFloatAttribute( "m33", &transform.cell[15] );
	if (element = root->FirstChildElement( "FOV" )) element->QueryFloatText( &FOV );
	if (element = root->FirstChildElement( "brightness" )) element->QueryFloatText( &brightness );
	if (element = root->FirstChildElement( "contrast" )) element->QueryFloatText( &contrast );
	if (element = root->FirstChildElement( "gamma" )) element->QueryFloatText( &gamma );
	if (element = root->FirstChildElement( "aperture" )) element->QueryFloatText( &aperture );
	if (element = root->FirstChildElement( "distortion" )) element->QueryFloatText( &distortion );
	if (element = root->FirstChildElement( "focalDistance" )) element->QueryFloatText( &focalDistance );
	if (element = root->FirstChildElement( "clampValue" )) element->QueryFloatText( &clampValue );
	if (element = root->FirstChildElement( "tonemapper" )) element->QueryIntText( &tonemapper );
}

// EOF