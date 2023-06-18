#include "Timer.h"
#include <sstream>
#include <ostream>

using namespace std;
using namespace std::chrono;

Timer::Entries compact(const Timer::Entries& entries, bool sort) {
    Timer::Entries ret;
    if (sort) {
        map<string, double> mapped;
        for (auto& e : entries)
            mapped[e.what] += e.elapsed;
        for (auto& e : mapped)
            ret.push_back({e.first, e.second});
    } else {
        map<string, int> used;
        for (auto& e : entries) {
            auto it = used.find(e.what);
            if (it == used.end()) {
                used[e.what] = (int)ret.size();
                ret.push_back(e);
            } else ret[it->second].elapsed += e.elapsed;
        }
    }
    return ret;
}

double sum(const Timer::Entries& entries) {
    double s = 0;
    for (auto& e : entries)
        s += e.elapsed;
    return s;
}

std::string toString(const Timer::Entries& entries, const string& label) {
    stringstream ss;
    double sum = 0;
    for(auto& e : entries){
        ss << e.what << ' ' << e.elapsed/1000 << endl;
        sum += e.elapsed;
    }
    if(!label.empty())
        ss << label << ' ' << sum/1000 << endl;
    return ss.str();
}

static void toStringRecursive(const std::map<std::string, Timer::Entries>& entries,
        const std::string& root, const string& space, stringstream& out){
    for(auto& e : entries.find(root)->second){
        out << space << e.what << ' ' << e.elapsed/1000 << endl;
        auto child = root + "." + e.what;
        auto cit = entries.find(child);
        if(cit != entries.end())
            toStringRecursive(entries, child, space + "  ", out);
    }
}

std::string toStringRecursive(
        const std::vector<std::string>& roots,
        const std::map<std::string, Timer::Entries>& entries){
    stringstream ss;
    for(auto& root : roots) {
        double sum = 0;
        for (auto& e : entries.find(root)->second)
            sum += e.elapsed;
        ss << root << ' ' << sum / 1000 << endl;
        toStringRecursive(entries, root, "  ", ss);
    }
    return ss.str();
}

double TimerCPU::diff(TimePoint a, TimePoint b) {
   return (double) duration_cast<nanoseconds>(b - a).count() / 1000;
}

TimerCPU::TimePoint TimerCPU::now() {
    return high_resolution_clock::now();
}

std::chrono::duration<int64_t, std::nano> TimerCPU::dur(double us) {
    return std::chrono::nanoseconds((int64_t)(us * 1000));
}

void TimerCPU::start(const std::string& what) {
    auto t = now();
    if(running) entries.back().elapsed = diff(startTimePoint, t);
    startTimePoint = t;
    entries.push_back({what, 0});
    running = true;
}

void TimerCPU::end() {
    if(!running) return;
    entries.back().elapsed = diff(startTimePoint, now());
    running = false;
}

void TimerCPU::finish() {
    end();
}

void TimerCPU::reset() {
    finish();
    entries.clear();
}

Timer::Entries TimerCPU::getEntries() {
    finish();
    return entries;
}