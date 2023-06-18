#pragma once

#include "SceneData.h"
#include "Scene.h"
#include "Scene.inl"

template<typename Job>
inline void SceneData::run(Job&& job, px_sched::Sync* s, const std::string& n) {
    scene->run(std::forward<Job>(job), s, n.empty() ? "" : getStatsID() + "." + n);
}

template<typename Job>
inline void SceneData::runAfter(px_sched::Sync sync, Job&& job, px_sched::Sync* s, const std::string& n) {
    scene->runAfter(sync, std::forward<Job>(job), s, n.empty() ? "" : getStatsID() + "." + n);
}

template<typename Job>
inline void SceneData::run(GPUTask type, Job&& job, px_sched::Sync *s, const std::string& n, double t) {
    scene->run(type, std::forward<Job>(job), s, n.empty() ? "" : getStatsID() + "." + n, t);
}


inline void SceneData::timer(const std::string& n) {
    if (n.empty()) scene->timer.cpu();
    else scene->timer.cpu(getStatsID() + '.' + n);
}

inline void SceneData::readCompressed(SocketReader& reader, std::vector<unsigned char>& dst, const std::string& n, int uncompressed) {
    timer(n);
    reader.read(dst);
    timer();
    scene->stats.addSize(getStatsID() + '.' + n, dst, uncompressed);
}

inline void SceneData::readCompressed(SocketReader& reader, std::vector<std::vector<unsigned char>>& dst, const std::string& n, int uncompressed) {
    timer(n);
    reader.read(dst);
    timer();
    scene->stats.addSize(getStatsID() + '.' + n, dst, uncompressed);
};
