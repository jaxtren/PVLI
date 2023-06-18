#pragma once

#include <GL/glew.h>
#include <string>
#include <vector>
#include "px_sched.h"
#include "common.h"
#include "glmHelpers.h"
#include "asioHelpers.h"
#include "Timer.h"

class Scene;
enum GPUTask : int;

class SceneData {
public:
    Scene *scene = nullptr;
    std::string name;

    template<typename Job>
    inline void run(Job&& job, px_sched::Sync *out_sync_obj = nullptr, const std::string& n = "");

    template<typename Job>
    inline void runAfter(px_sched::Sync sync, Job&& job, px_sched::Sync *out_sync_obj = nullptr, const std::string& n = "");

    template<typename Job>
    inline void run(GPUTask type, Job&& job, px_sched::Sync *s = nullptr, const std::string& n = "", double t = 1);

    template<typename Job>
    inline void run(GPUTask type, Job&& job, const std::string& n, double t = 1) {
        run(type, std::forward<Job>(job), nullptr, n, t);
    }

    //stats
    inline void timer(const std::string& name = "");
    inline void readCompressed(SocketReader&, std::vector<unsigned char>& dst, const std::string& n, int uncompressed = -1);
    inline void readCompressed(SocketReader&, std::vector<std::vector<unsigned char>>& dst, const std::string& n, int uncompressed = -1);

    virtual std::string getStatsID() { return name; };
    virtual void process(SocketReader&) {}; // receive data and schedule processing
    virtual void received() {}; // called on communication thread after whole scene is received
    virtual void beforeFirstRender() {}; // called on main (GL) thread after whole scene is processed (all tasks), before first rendering
    virtual void beforeReuse() {}; // called on main (GL) thread after last rendering to prepare data for reusing (see Scene::reuse)
    virtual void free() {}; // free all data
};