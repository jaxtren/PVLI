#pragma once

#include <thread>
#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <future>
#include "asioHelpers.h"
#include "Pipe.h"

//#define DEBUG_ASYNC_WRITER

class AsyncWriter {
    boost::asio::ip::tcp::socket *socket = nullptr;
    using Function = std::function<size_t(boost::asio::ip::tcp::socket &)>;
    Pipe<Function> pipe;
    std::thread senderThread;
    size_t sentBytes = 0; //TODO thread safe

public:
    inline AsyncWriter() = default;
    inline AsyncWriter(boost::asio::ip::tcp::socket &s) : AsyncWriter() { socket = &s; }
    inline ~AsyncWriter() { close(); }

    inline void setSocket(boost::asio::ip::tcp::socket &s) {
        close();
        socket = &s;
    }

    void run();
    void close(bool closeSocket = false);
    inline void pause() { pipe.pause(); }

    inline size_t resetSentBytes() {
        auto ret = sentBytes;
        sentBytes = 0;
        return ret;
    }

    template<typename C>
    inline void call(C c) {
        pipe.send([=](auto&) { c(*socket); return 0; });
    }

#ifdef DEBUG_ASYNC_WRITER
    template<typename T>
    inline void write(const T &v) {
        pipe.send([=](auto &s) {
            auto ret = ::write(s, v);
            std::cout << "AsyncWriter: const " << ret << std::endl;
            return ret;
        });
    }

    template<typename T>
    inline void write(T &v) {
        pipe.send([=](auto &s) {
            auto ret = ::write(s, v);
            std::cout << "AsyncWriter: copy " << ret << std::endl;
            return ret;
        });
    }

    template<typename T, class = typename std::enable_if<!std::is_lvalue_reference<T>::value>::type>
    inline void write(T &&v) {
        pipe.send([v = std::move(v)](auto &s) {
            auto ret = ::write(s, v);
            std::cout << "AsyncWriter: move " << ret << std::endl;
            return ret;
        });
    }
#else
    template<typename T>
    inline void write(const T &v) {
        pipe.send([=](auto &s) {
            return ::write(s, v);
        });
    }

    template<typename T>
    inline void write(T &&v) {
        pipe.send([v = std::forward<T>(v)](auto &s) {
            return ::write(s, v);
        });
    }
#endif

    template<typename T>
    inline std::shared_ptr<std::promise<T>> writeLater() {
        auto promise = std::make_shared<std::promise<T>>();
        pipe.send([=](auto &s) mutable {
            auto ret = promise->get_future().get();
            return ::write(s, std::move(ret));
        });
        return promise;
    }

    inline void wait() {
        pipe.send([](auto&) { return 0; });
        pipe.wait();
    }
};