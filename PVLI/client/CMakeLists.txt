cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(client LANGUAGES CXX C)

file(GLOB_RECURSE SRC_CLIENT "*.cpp" "*.h")

if (WIN32)
    set(SRC_CLIENT ${SRC_CLIENT} ../platform/dxva2_opengl.h)
    add_definitions(-DDECODE_DXVA2)
else()
    add_definitions(-DDECODE_SW)
endif()

add_executable(${PROJECT_NAME} ${SRC_CLIENT} ${SRC_COMMON} ${IMGUI_SRC})
set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/data/client")

target_include_directories(${PROJECT_NAME} PRIVATE
    ${GLM_INCLUDE_DIRS}
    ${COMMON_INCLUDE}
    ${IMGUI_INCLUDE}
    ${Boost_INCLUDE_DIRS}
    ${GL_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_DIR})

if (WIN32)
    target_link_directories(${PROJECT_NAME} PRIVATE "${Boost_LIBRARIES}")
    target_link_libraries(${PROJECT_NAME} PRIVATE Boost::boost D3D9.lib)
else()
    target_link_directories(${PROJECT_NAME} PRIVATE "${Boost_LIBRARY_DIRS}")
    target_link_libraries(${PROJECT_NAME} PRIVATE stdc++fs)
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${COMMON_LIBRARIES}
    ${GL_USED_LIBS}
    ${Boost_SYSTEM_LIBRARY}
    ${FFMPEG_LIBRARIES})
    
find_package(TurboJPEG QUIET)
if (TurboJPEG_FOUND)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_TURBOJPEG)
    target_include_directories(${PROJECT_NAME} PRIVATE ${TURBOJPEG_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TURBOJPEG_LIBRARIES})
endif()