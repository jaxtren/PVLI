/* tools.h - Copyright 2019 Utrecht University

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

struct HasPlacementNewOperator
{
	// When not compiling to PTX, nvcc fails to call this operator entirely (at least on Linux)
	// Defining an override (with the same void*) fixes this.
	__device__ static void* operator new( size_t, void* ptr )
	{
		return ptr;
	}

	// TODO: This does not silence the (useless) warning
	// __device__ static void operator delete(  void* ptr )
	// {
	// }
};
