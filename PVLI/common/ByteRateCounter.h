#pragma once

#include <queue>
#include "Timer.h"

class ByteRateCounter {
private:
    double sum = 0, softMaxDur = 1, hardMaxDur = 3;
    int softMinSam = 5;
    std::queue<std::pair<double, TimerCPU::TimePoint>> samples;

    inline void shrink() {
        while ((!samples.empty() && duration() > hardMaxDur) ||
               (samples.size() > softMinSam && duration() > softMaxDur)) {
            sum -= samples.front().first;
            samples.pop();
        }
    }

public:

    // duration is in seconds

    inline double duration() const {
        return samples.size() < 2 ? 0 : TimerCPU::diff(samples.front().second, samples.back().second) * 0.000001;
    }

    inline double hardMaxDuration() const {
        return hardMaxDur;
    }

    inline void hardMaxDuration(double d) {
        hardMaxDur = d;
        shrink();
    }

    inline double softMaxDuration() const {
        return softMaxDur;
    }

    inline void softMaxDuration(double d) {
        softMaxDur = d;
        shrink();
    }

    inline double softMinSamples() const {
        return softMinSam;
    }

    inline void softMinSamples(int s) {
        softMinSam = s;
        shrink();
    }

    inline void add(double v, TimerCPU::TimePoint t = TimerCPU::now()) {
        samples.emplace(v, t);
        sum += v;
        shrink();
    }

    inline double average() {
        shrink();
        double d = duration();
        // duration gives time for last N samples, which is only N-1 intervals
        // enlarge duration to N intervals by scaling with N/(N-1) - adding estimated duration of missing interval
        return d > 0 ? sum / (d * (double)samples.size() / (double)(samples.size() - 1)) : 0;
    }
};