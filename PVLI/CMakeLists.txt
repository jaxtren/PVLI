cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(PVLI LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_SOURCE_DIR}/cmake_modules/")

#GLM
find_package(glm REQUIRED)

#threads
find_package(Threads REQUIRED)

#GL
find_package(OpenGL REQUIRED COMPONENTS OpenGL OPTIONAL_COMPONENTS EGL)
find_package(GLEW REQUIRED)
find_package(glfw3)
set(GL_INCLUDE_DIRS ${OPENGL_INCLUDE_DIR} ${OPENGL_EGL_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS})

if (WIN32)
    # windows 10
    add_definitions(-D_WIN32_WINNT=0x0A00)
    set(GL_USED_LIBS ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY} ${GLEW_LIBRARIES} glfw)
else()
    set(GL_USED_LIBS OpenGL::OpenGL OpenGL::EGL ${GLEW_LIBRARIES} glfw)
endif()

#GLFW
set(ENABLE_GLFW ${glfw3_FOUND} CACHE BOOL "Enable support for GLFW")
if(${ENABLE_GLFW})
    if (${glfw3_FOUND})
        add_definitions(-DENABLE_GLFW)
        message(STATUS "Enable GLFW support")
    else()
        message(FATAL_ERROR "Cannot enable GLFW support - library not found")
    endif()
endif ()

#EGL
set(ENABLE_EGL ${OpenGL_EGL_FOUND} CACHE BOOL "Enable support for EGL")
if(${ENABLE_EGL})
    if (${OpenGL_EGL_FOUND})
        add_definitions(-DENABLE_EGL)
        message(STATUS "Enable EGL support")
    else()
        message(FATAL_ERROR "Cannot enable EGL support - library not found")
    endif()
endif ()

#Boost
set(Boost_USE_MULTITHREAD ON)
find_package(Boost REQUIRED COMPONENTS system)

#common
set(COMMON_INCLUDE "${CMAKE_SOURCE_DIR}/common" "${CMAKE_SOURCE_DIR}/lib")
file(GLOB_RECURSE SRC_COMMON "${CMAKE_SOURCE_DIR}/common/*.cpp")

set(COMMON_LIBRARIES  "Threads::Threads")

#C++ parallel algorithms
set(DISABLE_PARALLEL_ALGORITHMS FALSE CACHE BOOL "Disable C++ parallel algorithms")
if(${DISABLE_PARALLEL_ALGORITHMS})
    add_definitions(-DDISABLE_PARALLEL_ALGORITHMS)
    message(STATUS "Disable C++ parallel algorithms")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(COMMON_LIBRARIES "${COMMON_LIBRARIES}" "tbb")
endif()

#FSE
set(COMMON_INCLUDE "${COMMON_INCLUDE}" "${CMAKE_SOURCE_DIR}/lib/FSE/lib/")
file(GLOB SRC_FSE "${CMAKE_SOURCE_DIR}/lib/FSE/lib/*.c")
set(SRC_COMMON "${SRC_COMMON}" "${SRC_FSE}")

#PX
set(COMMON_INCLUDE "${COMMON_INCLUDE}" "${CMAKE_SOURCE_DIR}/lib/px/")

#ImGUI
set(IMGUI_INCLUDE "${CMAKE_SOURCE_DIR}/lib/imgui")
file(GLOB IMGUI_SRC "${CMAKE_SOURCE_DIR}/lib/imgui/*.cpp")
set(IMGUI_SRC "${IMGUI_SRC}"
    "${CMAKE_SOURCE_DIR}/lib/imgui/misc/cpp/imgui_stdlib.cpp"
    "${CMAKE_SOURCE_DIR}/lib/imgui/examples/imgui_impl_glfw.cpp"
    "${CMAKE_SOURCE_DIR}/lib/imgui/examples/imgui_impl_opengl3.cpp")

#FFmpeg
find_package(FFmpeg REQUIRED)

set(BUILD_SERVER TRUE CACHE BOOL "Enable server build")
if(BUILD_SERVER)
    add_subdirectory(server)
endif()

set(BUILD_CLIENT TRUE CACHE BOOL "Enable client build")
if(BUILD_CLIENT)
    add_subdirectory(client)
endif()

set(BUILD_TESTS TRUE CACHE BOOL "Enable tests build")
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
