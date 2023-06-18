#pragma once

#include "Timer.h"
#include "cudaHelpers.h"

class TimerCUDA : public Timer {
    bool running = false;
    Entries entries;
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> events;

public:
    virtual ~TimerCUDA();

    virtual void start(const std::string& what);
    virtual void end();

    virtual void finish();
    virtual void reset();

    virtual Entries getEntries();
};