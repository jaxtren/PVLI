cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(tests LANGUAGES CXX C)

file(GLOB_RECURSE SRC_TESTS "*.cpp" "*.h")

add_executable(${PROJECT_NAME} ${SRC_TESTS} ${SRC_COMMON} ${IMGUI_SRC})
target_include_directories(${PROJECT_NAME} PUBLIC
    ${GLM_INCLUDE_DIRS}
    ${COMMON_INCLUDE}
    ${IMGUI_INCLUDE}
    ${Boost_INCLUDE_DIRS}
    ${GL_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_DIR})

link_directories(
    "${Boost_LIBRARY_DIRS}")
    
target_link_libraries(${PROJECT_NAME} PUBLIC
    ${COMMON_LIBRARIES}
    ${GL_USED_LIBS}
    ${Boost_SYSTEM_LIBRARY}
    ${FFMPEG_LIBRARIES})

if (WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE Boost::boost)
else ()
    target_link_libraries(${PROJECT_NAME} PUBLIC stdc++fs)
endif()