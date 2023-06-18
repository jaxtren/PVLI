#include "TimerCUDA.h"

using namespace std;
using namespace std::chrono;

TimerCUDA::~TimerCUDA() {
    TimerCUDA::reset();
}

void TimerCUDA::start(const std::string& what) {
    cudaEvent_t event;
    cuEC(cudaEventCreate(&event));
    cuEC(cudaEventRecord(event));
    entries.push_back({what, 0});
    events.push_back({event, 0});
    running = true;
}

void TimerCUDA::end() {
    if (!running) return;
    cudaEvent_t event;
    cuEC(cudaEventCreate(&event));
    cuEC(cudaEventRecord(event));
    events.back().second = event;
    running = false;
}

void TimerCUDA::finish() {
    end();
    for (int i = 0; i < events.size(); i++) {
        float elapsed = 0;
        auto& e = events[i];
        if (!e.first) continue;

        auto e1 = e.first;
        auto e2 = e.second ? e.second : events[i + 1].first;
        cuEC(cudaEventSynchronize(e1));
        cuEC(cudaEventSynchronize(e2));
        cudaEventElapsedTime(&elapsed, e1, e2);
        entries[i].elapsed = elapsed * 1000;

        if (e.first) cuEC(cudaEventDestroy(e.first));
        if (e.second) cuEC(cudaEventDestroy(e.second));
        e.first = e.second = 0;
    }
    events.clear();
}

void TimerCUDA::reset() {
    finish();
    entries.clear();
}

Timer::Entries TimerCUDA::getEntries() {
    finish();
    return entries;
}