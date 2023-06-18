#pragma once

#include <iostream>
#include <sstream>
#include <glm/glm.hpp>
#include <glm/gtc/type_precision.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/constants.hpp>

//conversions

template<typename D = glm::vec2, typename T>
inline D glmVec2(const T& v){
    return {v.x, v.y};
}

template<typename D = glm::vec3, typename T>
inline D glmVec3(const T& v){
    return {v.x, v.y, v.z};
}

template<typename D = glm::vec4, typename T>
inline D glmVec4(const T& v){
    return {v.x, v.y, v.z, v.w};
}

template <typename T, glm::qualifier Q>
inline glm::vec<3, T, Q> operator * (const glm::mat<4, 4, T, Q>& m,  const glm::vec<3, T, Q>& v) {
    return glm::vec<3, T, Q>(m * glm::vec<4,T,Q>(v, 1));
}

//stream operators

namespace std {

    template<int N, typename T, glm::qualifier Q>
    ostream &operator<<(ostream &out, const glm::vec<N, T, Q> &v) {
        for (int i = 0; i < N; i++)
            out << (i > 0 ? " " : "") << v[i];
        return out;
    }

    template<int N, typename T, glm::qualifier Q>
    istream &operator>>(istream &in, glm::vec<N, T, Q> &v) {
        for (int i = 0; i < N; i++)
            in >> v[i];
        return in;
    }

    inline ostream &operator<<(ostream &out, const glm::quat&v) {
        for (int i = 0; i < 4; i++)
            out << (i > 0 ? " " : "") << v[i];
        return out;
    }

    inline istream &operator>>(istream &in, glm::quat& v) {
        for (int i = 0; i < 4; i++)
            in >> v[i];
        return in;
    }
}

template<int N, typename T, glm::qualifier Q>
std::string to_string(const glm::vec<N, T, Q> &v) {
    std::stringstream ss;
    ss << v;
    return ss.str();
}

inline std::string to_string(const glm::quat& v) {
    std::stringstream ss;
    ss << v;
    return ss.str();
}

#ifdef AI_MATRIX4X4_H_INC

inline glm::mat4 glmMat4(const aiMatrix4x4& m){
    glm::mat4 ret;
    for(int i=0; i<4; i++)
        for(int j=0; j<4; j++)
            ret[j][i] = m[i][j];
    return ret;
}

#endif
