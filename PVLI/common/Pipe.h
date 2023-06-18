#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include "common.h"

template<typename T>
class Pipe {

public:
    enum State {
        PAUSED, RUNNING, CLOSED
    };

    enum Mode {
        WAIT, CHECK, REPLACE
    };

private:
    std::mutex mu;
    std::condition_variable empty;
    std::condition_variable full;
    std::queue<T> data;
    int buffer_size = 0;
    State state = RUNNING;
    using Lock = std::unique_lock<std::mutex>;

public:

    explicit inline Pipe(int size = 0) {
        setBufferSize(size);
    }

    inline void setBufferSize(int s) {
        Lock l(mu);
        buffer_size = s > 0 ? s : std::numeric_limits<int>::max();
    }

    inline void start() {
        Lock l(mu);
        state = RUNNING;
        empty.notify_all();
        full.notify_all();
    }

    inline void pause() {
        Lock l(mu);
        state = PAUSED;
        empty.notify_all();;
        full.notify_all();
    }

    inline void close() {
        Lock l(mu);
        while(!data.empty()) data.pop();
        state = CLOSED;
        empty.notify_all();;
        full.notify_all();
    }

    inline bool closed() {
        Lock l(mu);
        return state == CLOSED;
    }

    inline size_t size() {
        Lock l(mu);
        return data.size();
    }

    inline void clear() {
        Lock l(mu);
        while(!data.empty()) data.pop();
        empty.notify_all();;
        full.notify_all();
    }

    bool send(const T &v, Mode mode = WAIT) {
        Lock l(mu);
        if(state == CLOSED) return false;
        if (data.size() >= buffer_size) {
            if (mode == CHECK) return false;
            if (mode == REPLACE) {
                data.back() = v;
                return true;
            }
        }
        while (data.size() >= buffer_size && state != CLOSED) full.wait(l);
        if(state == CLOSED) return false;
        data.push(v);
        empty.notify_all();
        return true;
    }

    bool send(T &&v, Mode mode = WAIT) {
        Lock l(mu);
        if(state == CLOSED) return false;
        if (data.size() >= buffer_size) {
            if (mode == CHECK) return false;
            if (mode == REPLACE) {
                data.back() = std::forward<T>(v);
                return true;
            }
        }
        while (data.size() >= buffer_size && state != CLOSED) full.wait(l);
        if(state == CLOSED) return false;
        data.push(std::forward<T>(v));
        empty.notify_all();
        return true;
    }

    void wait(int left = 0) {
        Lock l(mu);
        while (data.size() > left) full.wait(l);
    }

    bool receive(T &v, bool wait = true) {
        Lock l(mu);
        if (data.empty() && !wait) return false;
        while ((data.empty() && state == RUNNING) || state == PAUSED) empty.wait(l);
        if (!data.empty()) {
            v = std::move(data.front());
            data.pop();
            full.notify_all();
            return true;
        }
        return false;
    }

    inline locked_ptr<std::queue<T>> editQueue() {
        return locked_ptr<std::queue<T>>(&data, &mu, [this](){
            empty.notify_all();
            full.notify_all();
        });
    }
};