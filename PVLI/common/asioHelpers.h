#pragma once

#include <vector>
#include <string>

#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include <boost/asio.hpp>

#if BOOST_VERSION < 106600
//fix for older version
namespace boost{ namespace asio {
    using io_context = io_service;
}}
#endif

//helper asio functions
template<typename T, typename S>
inline std::size_t write(S& s, const T& v){
    return boost::asio::write(s, boost::asio::buffer(&v, sizeof(T)));
}

template<typename T, typename S>
inline std::size_t write(S& s, const T* v, size_t n){
    return boost::asio::write(s, boost::asio::buffer(v, n * sizeof(T)));
}

template<typename T, typename S>
inline std::size_t write(S& s, const std::vector<T>& v){
    const auto sSize = write(s, (int)(v.size() * sizeof(T)));
    const auto vSize = write(s, v.data(), v.size());
    return sSize + vSize;
}

template<typename S>
inline std::size_t write(S& s, const std::string& v){
    const auto sSize = write(s, (int)v.size());
    const auto vSize = write(s, v.data(), v.size());
    return sSize + vSize;
}

template<typename S>
inline std::size_t write(S& s, const std::vector<std::string>& v){
    auto ret = write(s, (int)v.size());
    for(auto& d : v)
        ret += write(s, d);
    return ret;
}

template<typename T, typename S>
inline std::size_t read(S& s, T& v){
    return boost::asio::read(s, boost::asio::buffer(&v, sizeof(T)));
}

template<typename T, typename S>
inline std::size_t read(S& s, T* v, int n){
    return boost::asio::read(s, boost::asio::buffer(v, n * sizeof(T)));
}

template<typename T, typename S>
inline std::size_t read(S& s, std::vector<T>& v){
    int size;
    auto ret = read(s, size);
    if (size < 0) { // multiple parts
        for (int part = 0, parts = -size; part < parts; part++) {
            ret += read(s, size);
            auto start = v.size();
            v.resize(start + size / sizeof(T));
            ret += boost::asio::read(s, boost::asio::buffer(v.data() + start, size));
        }
    } else {
        v.resize(size / sizeof(T));
        ret += boost::asio::read(s, boost::asio::buffer(v.data(), size));
    }
    return ret;
}

template<typename T, typename S>
inline std::size_t read(S& s, std::vector<std::vector<T>>& v){
    int size;
    auto ret = read(s, size);
    if (size < 0) { // multiple parts
        v.resize(-size);
        for (int part = 0, parts = -size; part < parts; part++) {
            ret += read(s, size);
            v[part].resize(size / sizeof(T));
            ret += boost::asio::read(s, boost::asio::buffer(v[part].data(), size));
        }
    } else {
        v.resize(1);
        v[0].resize(size / sizeof(T));
        ret += boost::asio::read(s, boost::asio::buffer(v[0].data(), size));
    }
    return ret;
}

template<typename S>
inline std::size_t read(S& s, std::string& str){
    std::vector<char> v;
    auto ret = read(s, v);
    str = std::string(v.data(), v.size());
    return ret;
}

template<typename S>
inline std::size_t read(S& s, std::vector<std::string>& v){
    int c;
    auto ret = read(s, c);
    v.resize(c);
    for(int i=0; i<c; i++)
        ret += read(s, v[i]);
    return ret;
}

class SocketReader {
public:
    boost::asio::ip::tcp::socket *socket = nullptr;
    size_t receivedBytes = 0;

    inline SocketReader() = default;
    inline SocketReader(boost::asio::ip::tcp::socket &s) : SocketReader() { socket = &s; }

    template<typename T>
    inline std::size_t read(T& v) {
        auto ret = ::read(*socket, v);
        receivedBytes += ret;
        return ret;
    }

    template<typename T>
    inline T read() {
        T value;
        receivedBytes += ::read(*socket, value);
        return std::move(value);
    }
};