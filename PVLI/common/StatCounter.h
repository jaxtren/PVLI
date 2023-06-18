#pragma once

#include <vector>
#include <map>
#include <limits>
#include <algorithm>
#include "Config.h"
#include "glmHelpers.h"

template<typename T>
class StatCounter {
    std::vector<T> rawSamples;
    std::vector<T> sortedSamples;
    T sum = 0;
    int lastSample = -1;

    T ema = 0;
    T emaFactor = 0;

    void sort() {
        if(!sortedSamples.empty()) return;
        sortedSamples = rawSamples;
        std::sort(sortedSamples.begin(), sortedSamples.end());
    }

    int percentileIndex(float p){
        int s = sortedSamples.size() - 1;
        return std::max(0, std::min(s, (int)(p * s)));
    }

public:

    explicit StatCounter(size_t samples = 0, T emaf = 0) {
        emaFactor = emaf;
        reset((int)samples);
    };

    void reset(int samples = -1) {
        if(samples < 0)
            samples = lastSample < 0 ? 0 : (int)rawSamples.size();
        sum = 0;
        sortedSamples.clear();
        if (samples > 0) {
            rawSamples.resize(samples);
            lastSample = 0;
        } else {
            rawSamples.clear();
            lastSample = -1;
        }
    }

    void resize(int samples) {
        T avg = average();
        reset(samples);
        for(auto& v : rawSamples) v = avg;
        sum = avg * rawSamples.size();
    }

    void add(T s) {
        if(emaFactor > 0) ema = s * emaFactor + ema * ((T)1 - emaFactor);

        sortedSamples.clear();
        if (lastSample >= 0) {
            if(rawSamples.size() == 1)
                sum = rawSamples[0] = s;
            else {
                lastSample = (lastSample + 1) % rawSamples.size();
                sum += s - rawSamples[lastSample];
                rawSamples[lastSample] = s;
            }
        } else {
            rawSamples.push_back(s);
            sum += s;
        }
    }

    inline const std::vector<T>& samples() { return rawSamples; }
    inline const T last() { return rawSamples.empty() ? 0 : rawSamples.back(); }

    inline void setEMAFactor(T f) { emaFactor = f; }
    inline T getEMAFactor() { return emaFactor; }
    inline T EMA() { return ema; }
    inline int getSampleCount() { return (int)rawSamples.size(); }

    inline T average() {
        if (emaFactor > 0) return ema;
        if (rawSamples.empty()) return 0;
        return sum / (T) rawSamples.size();
    }

    T min() {
        if (rawSamples.empty()) return 0;
        T ret = rawSamples[0];
        for (auto &s : rawSamples)
            ret = std::min(ret, s);
        return ret;
    }

    T max() {
        if (rawSamples.empty()) return 0;
        T ret = rawSamples[0];
        for (auto &s : rawSamples)
            ret = std::max(ret, s);
        return ret;
    }

    T percentile(float p){
        sort();
        return sortedSamples[percentileIndex(p)];
    }

    T average(float pmin, float pmax){
        sort();
        T s = 0;
        int i1 = percentileIndex(pmin);
        int i2 = percentileIndex(pmax) + 1;
        for(int i=i1; i<i2; i++)
            s += sortedSamples[i];
        return s / (T)(i2 - i1);
    }
};

template<typename T>
class StatCounters {
    int samples = 0;
    T ema = 0;

    struct Counter {
        int emptyCount = 0;
        bool used = false;
        double value = 0;
        StatCounter<T> counter;
    };

    std::map<std::string, Counter> counters;

public:
    using Entry = std::pair<std::string, T>;
    using Entries = std::vector<Entry>;

    inline void reset() { counters.clear(); }

    inline int maxSamples() { return samples; }
    inline void maxSamples(int ms) { samples  = ms; reset(); }

    T EMA() { return ema; }
    void EMA(T f) { ema = f; reset(); }

    inline int sampleCount() { return counters.empty() ? 0 : counters.begin()->second.counter.getSampleCount(); }

    void add(const Entries& entries){
        bool changed = false;
        int curSamplesCount = sampleCount();

        for (auto e : entries) {
            auto it = counters.insert({e.first, { 0, false, 0, StatCounter<T>(samples, ema)}});

            //prepend empty data for new counter if samples are not limited
            if (it.second && samples <= 0)
                for (int i=0; i<curSamplesCount; i++)
                    it.first->second.counter.add(0);

            it.first->second.emptyCount = 0;
            it.first->second.used = true;
            it.first->second.value += e.second;
        }

        // add empty data for unused entries and remove if it has only empty data
        for (auto it = counters.begin(); it != counters.end(); ) {
            it->second.counter.add(it->second.value);
            it->second.value = 0;
            if (!it->second.used && ++(it->second.emptyCount) >= samples && samples > 0)
                it = counters.erase(it);
            else (it++)->second.used = false;
        }
    }

    pt::ptree stats(bool addMinMax, std::string prefix = "") {
        if(!prefix.empty() && prefix.back() != '.') prefix += '.';
        pt::ptree tree;
        tree.put("Sample Count", sampleCount());
        if(addMinMax)
            for(auto c : counters)
                tree.put(prefix + c.first, glm::vec3(c.second.counter.average(), c.second.counter.min(), c.second.counter.max()));
        else
            for(auto c : counters)
                tree.put(prefix + c.first, c.second.counter.average());
        return tree;
    }

    pt::ptree reportSamples(std::string prefix = "") {
        if(!prefix.empty() && prefix.back() != '.') prefix += '.';
        pt::ptree tree;
        int count = sampleCount();
        tree.put("Sample Count", count);
        for (int i = 0; i < count; i++)
            for (auto c : counters)
                tree.put(std::to_string(i) + "." + prefix + c.first, c.second.counter.samples()[i]);
        return tree;
    }

    pt::ptree report(bool addMinMax, const std::string& prefix = "") {
        pt::ptree tree;
        tree.put_child("Stats", stats(addMinMax, prefix));
        tree.put_child("Samples", reportSamples(prefix));
        return tree;
    }
};