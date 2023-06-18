#pragma once

#include <utility>
#include <chrono>
#include <map>
#include <vector>
#include <mutex>

class Timer {
public:

    struct Entry {
        std::string what;
        double elapsed; //in us
    };
    using Entries = std::vector<Entry>;

    Timer() = default;
    inline virtual ~Timer() = default;

    inline void operator() (const std::string& what) { start(what); };
    inline void operator() () { end(); };

    virtual void start(const std::string& what) = 0;
    virtual void end() = 0;

    virtual void finish() = 0;
    virtual void reset() = 0;

    virtual Entries getEntries() = 0;
};

Timer::Entries compact(const Timer::Entries& entries, bool sort = false);
double sum(const Timer::Entries& entries);

std::string toString(const Timer::Entries& entries, const std::string& sum);

std::string toStringRecursive(
        const std::vector<std::string>& roots,
        const std::map<std::string, Timer::Entries>& entries);


class TimerCPU : public Timer {
public:
    using TimePoint = std::chrono::high_resolution_clock::time_point;

private:
    bool running = false;
    TimePoint startTimePoint;
    Entries entries;

public:
    static double diff(TimePoint, TimePoint);
    static TimePoint now();
    static std::chrono::duration<int64_t, std::nano> dur(double us);

    //high_resolution_clock::now();
    virtual void start(const std::string& what);
    virtual void end();

    virtual void finish();
    virtual void reset();

    virtual Entries getEntries();
};

class ParallelStopwatch {
    std::mutex m;
    int running = -1;

public:
    TimerCPU::TimePoint started;
    TimerCPU::TimePoint stopped;
    double elapsed = 0;
    double paused = 0;

    ParallelStopwatch () = default;
    inline ParallelStopwatch(const ParallelStopwatch& w) {
        running = w.running;
        stopped = w.stopped;
        started = w.started;
        elapsed = w.elapsed;
        paused = w.paused;
    }

    inline ParallelStopwatch& operator=(const ParallelStopwatch& w) {
        running = w.running;
        stopped = w.stopped;
        started = w.started;
        elapsed = w.elapsed;
        paused = w.paused;
        return *this;
    }

    inline ParallelStopwatch& operator=(ParallelStopwatch&& w) {
        running = w.running;
        stopped = w.stopped;
        started = w.started;
        elapsed = w.elapsed;
        paused = w.paused;
        return *this;
    }

    inline void start() {
        m.lock();
        auto now = TimerCPU::now();
        if (running < 0) {
            started = stopped = now;
            running = 0;
        } else if (running == 0) {
            paused += TimerCPU::diff(stopped, now);
            stopped = now;
        }
        running++;
        m.unlock();
    }

    inline void stop() {
        m.lock();
        auto now = TimerCPU::now();
        running--;
        if(running == 0) {
            elapsed += TimerCPU::diff(stopped, now);
            stopped = now;
        }
        m.unlock();
    }
};