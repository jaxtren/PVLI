#pragma once

#include <GL/glew.h>
#include "Timer.h"

class TimerGL : public Timer {
    bool running = false;
    Entries entries;
    std::vector<std::pair<GLuint,GLuint>> queries;

public:
    virtual ~TimerGL();

    virtual void start(const std::string& what);
    virtual void end();

    virtual void finish();
    virtual void reset();

    virtual Entries getEntries();
};