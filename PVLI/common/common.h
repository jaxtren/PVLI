#pragma once

#include <string>
#include <chrono>
#include <sstream>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

template<typename T>
class locked_ptr {
    T* v = nullptr;
    std::mutex* m = nullptr;
    std::function<void()> c;
public:
    inline locked_ptr(T* v, std::mutex* m)
        : locked_ptr(v,m, nullptr) {};

    inline locked_ptr(T* v, std::mutex* m, std::function<void()> c)
        : v(v), m(m), c(c) { if(m) m->lock(); }

    inline ~locked_ptr() { unlock(); }

    locked_ptr (locked_ptr &&) = default;
    locked_ptr(const locked_ptr&) = delete;
    locked_ptr & operator=(const locked_ptr&) = delete;

    void unlock() {
        if(m) {
            if(c) c();
            m->unlock();
        }
        v = nullptr;
        m = nullptr;
    }

    T& operator* () { return *v; }
    T* operator-> () { return v; }
};

template<typename T>
class mutexed {
    std::mutex m;
    T var;

public:
    inline mutexed<T>& operator = (const T& v){
        m.lock();
        var = v;
        m.unlock();
        return *this;
    }

    inline mutexed<T>& operator = (T&& v){
        m.lock();
        var = v;
        m.unlock();
        return *this;
    }

    inline T get(){
        m.lock();
        T   v = var;
        m.unlock();
        return v;
    }

    inline void swap(T& v){
        m.lock();
        std::swap(var, v);
        m.unlock();
    }

    inline locked_ptr<T> lock(){ return locked_ptr<T>(&var, &m); }
};

struct string_exception : public std::exception {
    std::string s;
    string_exception(const std::string& ss) : s(ss) {}
    ~string_exception() {}
    const char* what() const noexcept { return s.c_str(); }
};

struct init_error : public string_exception {
    init_error(const std::string& s) : string_exception(s) {}
};

template<typename T> void swap(mutexed<T>& mv, T& v){ mv.swap(v); }
template<typename T> void swap(T& v, mutexed<T>& mv){ mv.swap(v); }


std::string readFile(std::string name);
bool startsWith(const std::string& str, const std::string& start);
bool endsWith(const std::string& str, const std::string& end);
std::vector<std::string> split(std::string str, char delimiter);
std::string to_string(double, int num);