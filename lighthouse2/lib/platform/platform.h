/* platform.h - Copyright 2019/2020 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   This file:

   Platform-specific header-, class- and function declarations.
*/

#pragma once

// GLFW
#define GLFW_USE_CHDIR 0 // Do not change cwd
#define GLFW_EXPOSE_NATIVE_WGL

// system includes
// clang-format off
#ifdef WIN32
//#define GLFW_DLL		 // Use DLL to let render cores be able to use GLFW as well
#define NOMINMAX
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#ifdef WIN32
#include <GLFW/glfw3native.h>
#endif
#include <zlib.h>
#include "emmintrin.h"
#include <FreeImage.h>
 #include <taskflow.hpp>

// clang-format on

// namespaces
using namespace std;

// include system-wide functionality
#include "system.h"

namespace lighthouse2
{

class Shader
{
public:
	// constructor / destructor
	Shader( const char* vfile, const char* pfile );
	~Shader();
	// methods
	void Init( const char* vfile, const char* pfile );
	void Compile( const char* vtext, const char* ftext );
	void Bind();
	void SetInputTexture( uint slot, const char* name, GLTexture* texture );
	void SetInputMatrix( const char* name, const mat4& matrix );
	void SetFloat( const char* name, const float v );
	void SetFloat3( const char* name, const float3 v );
	void SetFloat4( const char* name, const float4 v );
	void SetInt( const char* name, const int v );
	void SetUInt( const char* name, const uint v );
	void Unbind();
private:
	// data members
	uint vertex = 0;	// vertex shader identifier
	uint pixel = 0;		// fragment shader identifier
public:
	// public data members
	uint ID = 0;		// shader program identifier
};

// Low-level thread class
class StdThread
{
	std::thread t;
	static unsigned int static_proc( void* param );

public:
	StdThread() { }
	decltype(t)::native_handle_type handle() { return t.native_handle(); }
	void start();
	virtual void run() {};
	void setPriority( int p );
};

// std thread with start/stop synchronisation
// and (too) scoped locking.
class LoopThread : public StdThread
{
	// events
	std::mutex mut;
	std::condition_variable startEvent, doneEvent;
	bool shouldBeRunning = false;

public:
	inline void WaitForCompletion()
	{
		std::unique_lock<std::mutex> lock( mut );
		doneEvent.wait( lock, [this] { return !shouldBeRunning; } );
	}
	inline void SignalStart()
	{
		std::lock_guard<std::mutex> lock( mut );
		shouldBeRunning = true;
		startEvent.notify_one();
	}

	virtual void step() = 0;
	inline void run() final override
	{
		std::unique_lock<std::mutex> lock( mut );
		for ( ;; )
		{
			startEvent.wait( lock, [this] { return shouldBeRunning; } );
			step();
			shouldBeRunning = false;
			doneEvent.notify_one();
		}
	}
};

// Nils's jobmanager
class Job
{
public:
	virtual void Main() = 0;
protected:
	friend class JobThread;
	void RunCodeWrapper();
};
class JobThread : public StdThread
{
public:
	void CreateAndStartThread( unsigned int threadId );
	void WaitForThreadToStop();
	void Go();
	void run() override;
	std::mutex m_GoMutex;
	std::condition_variable m_GoSignal;
	int m_ThreadID;
};
class JobManager	// singleton class!
{
protected:
	JobManager( unsigned int numThreads );
public:
	static void CreateJobManager( unsigned int numThreads );
	static JobManager* GetJobManager();
	static void GetProcessorCount( uint& cores, uint& logical );
	void AddJob2( Job* a_Job );
	unsigned int GetNumThreads() { return m_NumThreads; }
	void RunJobs();
	void ThreadDone( unsigned int n );
	int MaxConcurrent() { return m_NumThreads; }
protected:
	friend class JobThread;
	Job* GetNextJob();
	Job* FindNextJob();
	static JobManager* m_JobManager;
	Job* m_JobList[256];
	std::mutex m_CS;
	std::condition_variable m_Done;
	std::atomic<int> m_JobCount, m_JobsToComplete;
	unsigned int m_NumThreads;
	JobThread* m_JobThreadList;
};

} // namespace lighthouse2

// forward declarations of platform-specific helpers
void _CheckGL( const char* f, int l );
#define CheckGL() { _CheckGL( __FILE__, __LINE__ ); }
GLuint CreateVBO( const GLfloat* data, const uint size );
void BindVBO( const uint idx, const uint N, const GLuint id );
void CheckShader( GLuint shader, const char* vshader, const char* fshader );
void CheckProgram( GLuint id, const char* vshader, const char* fshader );
void DrawQuad();
void DrawShapeOnScreen( std::vector<float2> verts, std::vector<float4> colors, uint GLshape, float width = 1.0f );

// EOF
