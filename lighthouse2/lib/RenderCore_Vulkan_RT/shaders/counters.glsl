/* counters.glsl - Copyright 2019/2020 Utrecht University

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

#ifndef COUNTERS_H
#define COUNTERS_H

#include "../bindings.h"

layout( set = 0, binding = cCOUNTERS ) buffer Counters
{
	uint pathLength;
	uint scrWidth;
	uint scrHeight;
	uint bufferSize;
	uint activePaths;
	uint shaded;
	uint generated;
	uint connected;
	uint extended;
	uint extensionRays;
	uint shadowRays;
	uint totalExtensionRays;
	uint totalShadowRays;
	int probedInstid;
	int probedTriid;
	float probedDist;
	float clampValue;
	float geometryEpsilon;
	uvec4 lightCounts;
};

#endif