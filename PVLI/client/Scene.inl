#pragma once

#include "Scene.h"
#include "Application.h"

template<typename Job>
inline void Scene::runAfter(px_sched::Sync s, Job&& job, px_sched::Sync *os, const std::string& n, bool sf) {
    if (sf) app->scheduler.incrementSync(&sync.finish);
    app->scheduler.runAfter(s,[=, job = std::forward<Job>(job)]() mutable {
        stopwatch.cpu.start();
        auto start = TimerCPU::now();
        job();
        if(!n.empty()) stats.addTime(n, start);
        stopwatch.cpu.stop();
        if (sf) app->scheduler.decrementSync(&sync.finish);
    }, os);
}

template<typename Job>
inline void Scene::run(GPUTask type, Job&& job, px_sched::Sync *s, const std::string& n, double t, bool sf) {
    if (sf) app->scheduler.incrementSync(&sync.finish);
    app->run(type, [=, job = std::forward<Job>(job)]() mutable {
        if (type != GPUTask::OTHER) stopwatch.gpu.start();
        if (!n.empty()) timer.gpu.tasks(n);
        job();
        if (!n.empty()) timer.gpu.tasks();
        if (type != GPUTask::OTHER) stopwatch.gpu.stop();
        if (type != GPUTask::OTHER && timer.gpu.lastFrameID != app->frameID) {
            timer.gpu.frames += app->frameGPUtimeWithoutTasks;
            timer.gpu.usedFrames++;
            if (timer.gpu.lastFrameID >= 0)
                timer.gpu.skippedFrames += app->frameID - timer.gpu.lastFrameID;
            timer.gpu.lastFrameID = app->frameID;
        }
        if (sf) app->scheduler.decrementSync(&sync.finish);
    }, s, t);
}

bool Scene::isBufferSuitable(gl::Buffer& buffer, size_t size) {
    return size <= buffer.size &&
           buffer.usage == app->bufferUsage &&
           buffer.hasStagingBuffer() == app->useStagingBuffers &&
           (!app->useStagingBuffers || buffer.stagingUsage == app->stagingBufferUsage);
}

void Scene::allocateBuffer(gl::Buffer& buffer, GLenum type, size_t size) {
    buffer.allocate(type, (size_t)(size * app->bufferSizeMultiple),
                    app->bufferUsage, app->useStagingBuffers ? app->stagingBufferUsage : GL_NONE);
    buffer.map();
}