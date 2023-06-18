#pragma once

#include <GL/glew.h>
#include <utility>
#include <iostream>
#include <string>
#include <vector>

namespace gl {

    /// error & log
    bool checkErrors(const char *file, int line);
    #define GLC gl::checkErrors( __FILE__, __LINE__ )
    std::string getInfoLog(GLuint object, PFNGLGETSHADERIVPROC glGet__iv, PFNGLGETSHADERINFOLOGPROC glGet__InfoLog);
    void debugMessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity,
                              GLsizei length, const GLchar *message, const void *userParam);

    /**
     * load shader with ability to #include another files and inserting #defines
     * @param basePath path for searching #include files
     * @param fileName fileName of root shader to load (without path)
     * @param defs #defines to insert into shader string
     * @return
     */
    std::string loadAndPreprocessShaderString(const std::string& basePath, const std::string& fileName,
                                              const std::vector<std::string>& defs = std::vector<std::string>());

    /// shader & program
    GLuint createShaderFromString(GLenum type, const std::string &source, std::ostream& out = std::cout, const std::string& label = "");
    void linkProgram(GLuint& program);
    void linkProgramAndClean(GLuint& program);

    /**
     * create opengl program from shaders
     * @tparam Container iteratable object with GLuint type, offen autodetected
     * @param shaders GLuint list of shaders to attach
     * @return opengl program
     */
    template<typename Container>
    GLuint createProgram(const Container &shaders) {
        GLuint program = glCreateProgram();
        for (const auto &shader : shaders)
            glAttachShader(program, shader);
        return program;
    }

    /**
     * create opengl program from shaders
     * @param shaders GLuint initializer list of shaders to attach
     * @return opengl program
     */
    GLuint createProgram(std::initializer_list<GLuint>);

    inline void deleteProgram(GLuint& program) {
        if (program) {
            glDeleteProgram(program);
            program = 0;
        }
    }

    /**
     * load program from multiple files with ability to #include another files and inserting #defines
     * @param fileName fileName without suffix and path
     * @param basePath path for searching #include files
     * @param defs #defines to insert into shader string
     * @return program or 0 (on fail)
     */
    GLuint loadProgram(const std::string& fileName, const std::string& basePath = "./",
                       const std::vector<std::string>& defs = std::vector<std::string>());


    template<typename T>
    inline void bufferData(GLuint buffer, GLsizei size, const T* data, GLenum usage){
        glBufferData(buffer, size * sizeof(T), reinterpret_cast<const void*>(data), usage);
    }

    template<typename T>
    inline void bufferSubData(GLuint buffer, GLsizei offset, GLsizei size, const T* data){
        glBufferSubData(buffer, offset * sizeof(T), size * sizeof(T), reinterpret_cast<const void*>(data));
    }

    inline void vertexAttribPointer(GLint index, size_t size, GLenum type, size_t stride, size_t offset, bool normalized = false){
        glVertexAttribPointer((GLuint)index, (GLint)size, type, normalized, (GLsizei)stride, reinterpret_cast<GLvoid*>(static_cast<intptr_t>(offset)));
    }

    inline void vertexAttribIPointer(GLint index, size_t size, GLenum type, size_t stride, size_t offset){
        glVertexAttribIPointer((GLuint)index, (GLint)size, type, (GLsizei)stride, reinterpret_cast<GLvoid*>(static_cast<intptr_t>(offset)));
    }

    inline void drawElementsInt (GLenum mode, GLsizei count, size_t offset){
        glDrawElements(mode, count, GL_UNSIGNED_INT, reinterpret_cast<GLvoid*>(static_cast<intptr_t>(offset * sizeof(unsigned int))));
    }
}
