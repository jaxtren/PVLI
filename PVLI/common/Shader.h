#pragma once

#include <GL/glew.h>
#include <string>
#include <map>
#include <graphic.h>
#include <uniforms.h>

namespace gl {

    class Shader {
        GLuint program = 0;
        std::map<std::string, GLint> uniforms;
        glm::ivec3 computeWorkGroupSize = {-1, -1, -1}; // for compute shaders only

    public:
        Shader() = default;
        ~Shader() { free(); }

        inline GLuint id() const { return program; }

        inline bool load(const std::string& fileName, const std::string& basePath = "./",
                       const std::vector<std::string>& defs = std::vector<std::string>()) {
            free();
            program = gl::loadProgram(fileName, basePath, defs);
            return program;
        }

        inline void free() {
            gl::deleteProgram(program);
            uniforms.clear();
        }

        inline void use() const { glUseProgram(program); }

        inline GLint uniformLocation(const std::string& name) {
            auto it = uniforms.find(name);
            if (it != uniforms.end()) return it->second;
            auto loc = glGetUniformLocation(program, name.c_str());
            uniforms.emplace(name, loc);
            return loc;
        }

        template<typename T> bool uniform(const std::string& name, const T& value) {
            auto loc = uniformLocation(name);
            return loc >= 0 && gl::setUniform(loc, value);
        }

        inline operator bool() const { return program; }

        // for compute shaders only

        inline glm::ivec3 workGroupSize() {
            if (computeWorkGroupSize.x < 0)
                glGetProgramiv(program, GL_COMPUTE_WORK_GROUP_SIZE, (int*)&computeWorkGroupSize);
            return computeWorkGroupSize;
        }

        inline void dispatchCompute(const glm::ivec3& size) {
            auto s = (size - 1) / workGroupSize() + 1;
            glDispatchCompute(s.x, s.y, s.z);
        }

        inline void dispatchCompute(const glm::ivec2& size) {
            dispatchCompute({size.x, size.y, 1});
        }
    };
}