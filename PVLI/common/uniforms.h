#pragma once

#include <GL/glew.h>
#include <glm/glm.hpp>

// helper for setting OpenGL uniforms based on their type
// support: base types, GLM vectors and matrices (currently not all)

namespace gl {

    template<typename T>
    inline bool setUniform(GLint, const T&) { return false; }

    // common
    #define FUN template<> inline bool setUniform
    #define RET return true

    // scalars/vectors
    #define BIND(T, V, S)\
    FUN <T>(GLint l, const T& v){ glUniform1##S(l,v); RET; }\
    FUN <glm::V##2>(GLint l, const glm::V##2& v){ glUniform2##S(l,v.x, v.y); RET; }\
    FUN <glm::V##3>(GLint l, const glm::V##3& v){ glUniform3##S(l,v.x, v.y, v.z); RET; }\
    FUN <glm::V##4>(GLint l, const glm::V##4& v){ glUniform4##S(l,v.x, v.y, v.z, v.w); RET; }\

    BIND(float, vec, f)
    BIND(double, dvec, d)
    BIND(int, ivec, i)
    BIND(unsigned int, uvec, ui)
    #undef BIND

    // matrices
    #define BIND(T, M, S, N)\
    FUN <glm::M##N> (GLint l, const glm::M##N& m){ \
        glUniformMatrix##N##S##v(l,1, GL_FALSE, &(m[0][0])); RET; }

    #define BINDS(T, M, S)\
    BIND(T, M, S, 2)\
    BIND(T, M, S, 3)\
    BIND(T, M, S, 4)

    BINDS(float, mat, f)
    BINDS(double, dmat, d)
    #undef BIND
    #undef BINDS

    #undef RET
    #undef FUN
}