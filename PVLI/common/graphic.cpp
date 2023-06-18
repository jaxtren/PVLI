#include "graphic.h"
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cassert>

using namespace std;

namespace gl {

    static const char* debugType(GLenum type) {
        switch (type) {
            case GL_DEBUG_SOURCE_API:
                return "GL_DEBUG_SOURCE_API";
            case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
                return "GL_DEBUG_SOURCE_WINDOW_SYSTEM";
            case GL_DEBUG_SOURCE_SHADER_COMPILER:
                return "GL_DEBUG_SOURCE_SHADER_COMPILER";
            case GL_DEBUG_SOURCE_THIRD_PARTY:
                return "GL_DEBUG_SOURCE_THIRD_PARTY";
            case GL_DEBUG_SOURCE_APPLICATION:
                return "GL_DEBUG_SOURCE_APPLICATION";
            case GL_DEBUG_SOURCE_OTHER:
                return "GL_DEBUG_SOURCE_OTHER";

            case GL_DEBUG_TYPE_ERROR:
                return "GL_DEBUG_TYPE_ERROR";
            case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
                return "GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR";
            case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
                return "GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR";
            case GL_DEBUG_TYPE_PORTABILITY:
                return "GL_DEBUG_TYPE_PORTABILITY";
            case GL_DEBUG_TYPE_PERFORMANCE:
                return "GL_DEBUG_TYPE_PERFORMANCE";
            case GL_DEBUG_TYPE_OTHER:
                return "GL_DEBUG_TYPE_OTHER";
            case GL_DEBUG_TYPE_MARKER:
                return "GL_DEBUG_TYPE_MARKER";
            case GL_DEBUG_TYPE_PUSH_GROUP:
                return "GL_DEBUG_TYPE_PUSH_GROUP";
            case GL_DEBUG_TYPE_POP_GROUP:
                return "GL_DEBUG_TYPE_POP_GROUP";

            case GL_DEBUG_SEVERITY_HIGH:
                return "GL_DEBUG_SEVERITY_HIGH ";
            case GL_DEBUG_SEVERITY_MEDIUM:
                return "GL_DEBUG_SEVERITY_MEDIUM ";
            case GL_DEBUG_SEVERITY_LOW:
                return "GL_DEBUG_SEVERITY_LOW";

            default:
                return "UNKNOWN";
        }
    }

    void debugMessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity,
                              GLsizei length, const GLchar *message, const void *userParam) {
        string so = debugType(source);
        string t = debugType(type);
        string se = debugType(severity);
        if (so != "UNKNOWN" && t != "UNKNOWN" && se != "UNKNOWN")
            cerr << so << ' ' << t << ' ' << se  << " (" << id << ") " << message << endl;
    }

    static bool startsWith(const string& str, const string& start) {
        if (&start == &str) return true;
        if (start.length() > str.length()) return false;
        for (size_t i = 0; i < start.length(); ++i)
            if (start[i] != str[i]) return false;
        return true;
    }

    string loadAndPreprocess(const string& base, const string& fileName){
        string line;
        stringstream ret;

        ifstream in(base + (base.empty() ? "" : "/") + fileName);
        if(!in.is_open() || !in) return "";

        //process remainding data with #include support
        while(getline(in,line)){
            if(startsWith(line,"#include")){
                stringstream ss(line);
                string includeFile;
                ss >> includeFile >> includeFile; //first skips "#include";
                ret << loadAndPreprocess(base, includeFile) << endl;
            }else ret << line << endl;
        };

        return ret.str();
    }

    string loadAndPreprocessShaderString(const string& base, const string& fileName, const vector<string>& defs) {
        stringstream ret;
        string version = "#version 140"; //TODO provide way to change it
        string code = loadAndPreprocess(base, fileName);

        if(code.empty()) return "";

        if(startsWith(code,"#version")) {
            version = code.substr(0,code.find_first_of('\n') + 1);
            code = code.substr(code.find_first_of('\n') + 1);
        }

        ret << version << endl;
        for(const string& d : defs)
            ret << "#define " << d << endl;
        //TODO add #line directive

        ret << code << endl;

        return ret.str();
    }

    string getInfoLog(
            GLuint object,
            PFNGLGETSHADERIVPROC glGet__iv,
            PFNGLGETSHADERINFOLOGPROC glGet__InfoLog
    ) {
        string ret;
        GLint length;
        glGet__iv(object, GL_INFO_LOG_LENGTH, &length);
        if (length > 0) {
            vector<char> log(length);
            glGet__InfoLog(object, length, nullptr, log.data());
            ret = log.data();
        }
        return ret;
    }


    bool checkErrors(const char *file, int line) {
        bool ret = false;
        for (GLenum err; (err = glGetError()) != GL_NO_ERROR;) {
            string error;
            switch(err) {
                case GL_INVALID_OPERATION:              error="INVALID_OPERATION";              break;
                case GL_INVALID_ENUM:                   error="INVALID_ENUM";                   break;
                case GL_INVALID_VALUE:                  error="INVALID_VALUE";                  break;
                case GL_OUT_OF_MEMORY:                  error="OUT_OF_MEMORY";                  break;
                case GL_INVALID_FRAMEBUFFER_OPERATION:  error="INVALID_FRAMEBUFFER_OPERATION";  break;
                default:                                error="UNKNOWN";                        break;
            }
            cerr << "GL ERROR: " << error << " (" << hex << err << ")" << " in " << file << " at line " << dec << line << endl;
            ret = true;
        }
        return ret;
    }

    GLuint createShaderFromString(GLenum type, const string &source, ostream& out, const string& label) {
        GLuint shader = glCreateShader(type);
        const char *src = source.c_str();
        GLint length = (GLint)source.length();
        glShaderSource(shader, 1, &src, &length);
        glCompileShader(shader);

        GLint shader_ok;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &shader_ok);
        if (!shader_ok) {
            if(label.empty()) {
                stringstream ss(source);
                string line;
                for (int i = 1; getline(ss, line); i++)
                    out << setw(4) << i << " | " << line << endl;
            }else out << "Error in shader: \"" << label << '"' << endl;
            out << getInfoLog(shader, glGetShaderiv, glGetShaderInfoLog) << endl;
            glDeleteShader(shader);
            shader = 0;
        }

        return shader;
    }

    GLuint createProgram(initializer_list<GLuint> shaders) {
        GLuint program = glCreateProgram();
        for (const auto &shader : shaders)
            glAttachShader(program, shader);
        return program;
    }

    void linkProgram(GLuint& program) {
        glLinkProgram(program);
        GLint program_ok;
        glGetProgramiv(program, GL_LINK_STATUS, &program_ok);
        if (!program_ok) {
            cout << getInfoLog(program, glGetProgramiv, glGetProgramInfoLog);
            glDeleteProgram(program);
            program = 0;
        }
    }

    void linkProgramAndClean(GLuint& program) {
        glLinkProgram(program);

        GLint program_ok;
        glGetProgramiv(program, GL_LINK_STATUS, &program_ok);

        GLint shader_count;
        glGetProgramiv(program, GL_ATTACHED_SHADERS, &shader_count);
        vector<GLuint> shaders(shader_count);

        glGetAttachedShaders(program,shader_count,&shader_count,&shaders[0]);
        for(int i=0; i<shader_count; i++){
            glDetachShader(program,shaders[i]);
            glDeleteShader(shaders[i]);
        }

        if (!program_ok) {
            cout << getInfoLog(program, glGetProgramiv, glGetProgramInfoLog);
            glDeleteProgram(program);
            program = 0;
        }
    }

    GLuint loadProgram(const std::string& fileName, const std::string& basePath, const std::vector<std::string>& defs){
        const GLenum types[] = { GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER, GL_COMPUTE_SHADER };
        const std::vector<std::string> names = { "vert", "tessc", "tesse", "geom", "frag", "comp" };
        std::vector<GLuint> shaders;
        for(int i=0; i<6; i++){
            string file = fileName + "." + names[i];
            auto str = gl::loadAndPreprocessShaderString(basePath, file, defs);
            if(str.empty()) continue;
            auto sh = gl::createShaderFromString(types[i], str, cout, file);
            if(sh == 0) {
                for(GLuint s : shaders) glDeleteShader(s);
                return 0;
            }
            shaders.push_back(sh);
        }
        GLuint program = gl::createProgram(shaders);
        gl::linkProgramAndClean(program);
        return program;
    }

}
