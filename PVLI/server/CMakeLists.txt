cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(server LANGUAGES CXX C CUDA)

#CUDA
set(CMAKE_CUDA_TOOLKIT_LIB_DIRECTORIES ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib) #WORKAROUND
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
add_definitions(-DENABLE_CUDA)

if (NOT WIN32)
    #FIXME workaround for linking latest libstd++ with cuda g++ compiler
    # generate project with: -DCMAKE_CXX_COMPILER=/opt/cuda/bin/g++
    # in project set compiler to g++
    set(CMAKE_CXX_COMPILER g++)
endif()

#FIXME workaround to supress GLM CUDA warnings, e.g.:
#warning: __device__ annotation is ignored on a function("mat") that is explicitly defaulted on its first declaration
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe=\"--diag_suppress=20012\"") #--display_error_number

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G") #debug for CUDA objects
else()
    # https://docs.nvidia.com/cuda/cuda-memcheck/index.html#compilation-options
    # Add lineinfo and symbols to the assembly, for improved debugging.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -restrict -use_fast_math")
endif()

#NvEncode
find_package(NVCodec REQUIRED)

#Lighthouse2 and dependencies
set(LIGHTHOUSE2_SRC_DIR "${CMAKE_SOURCE_DIR}/../lighthouse2")

if (WIN32)
    add_definitions(-DNOMINMAX)
    add_definitions(-DWIN32_LEAN_AND_MEAN)
endif()

set(FREEIMAGE_ROOT "${LIGHTHOUSE2_SRC_DIR}/lib/FreeImage" CACHE PATH "Additional directory to search for FreeImage")
find_package(FreeImage REQUIRED)

set(FreeType2_DIR "${LIGHTHOUSE2_SRC_DIR}/lib/FreeType2" CACHE PATH "Directory containing FreeType2")
find_package(FreeType2 REQUIRED)

if (WIN32)
    set(Lighthouse2_BUILD_ALWAYS false)
else()
    set(Lighthouse2_BUILD_ALWAYS true)
endif()

include(ExternalProject)
ExternalProject_Add(Lighthouse2
        PREFIX "${CMAKE_CURRENT_BINARY_DIR}/lighthouse2"
        BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/lighthouse2"
        SOURCE_DIR "${LIGHTHOUSE2_SRC_DIR}"
        CMAKE_ARGS "-DCMAKE_BUILD_TYPE=$<CONFIG>"
        INSTALL_COMMAND ""
        BUILD_COMMAND ${CMAKE_COMMAND} --build . --config $<CONFIG> --target RenderCore_Optix7Filter
        BUILD_ALWAYS ${Lighthouse2_BUILD_ALWAYS}
)

function(lighthouseLib libName dirName)
    add_library(${libName} STATIC IMPORTED)
    if (WIN32)
        set_property(TARGET ${libName} PROPERTY IMPORTED_LOCATION_DEBUG ${CMAKE_CURRENT_BINARY_DIR}/lighthouse2/lib/${dirName}/Debug/${libName}.lib)
        set_property(TARGET ${libName} PROPERTY IMPORTED_LOCATION_RELEASE ${CMAKE_CURRENT_BINARY_DIR}/lighthouse2/lib/${dirName}/Release/${libName}.lib)
        set_property(TARGET ${libName} PROPERTY IMPORTED_LOCATION_RELWITHDEBINFO ${CMAKE_CURRENT_BINARY_DIR}/lighthouse2/lib/${dirName}/RelWithDebInfo/${libName}.lib)
    else()
        set_property(TARGET ${libName} PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/lighthouse2/lib/${dirName}/lib${libName}.a)
    endif()
endfunction()

lighthouseLib(RenderSystem RenderSystem)
lighthouseLib(Platform platform)
lighthouseLib(glad glad)
lighthouseLib(tinyxml2 tinyxml2)

#determine directories from LIGHTHOUSE2_SRC_DIR because structure in CMAKE_CURRENT_BINARY_DIR is not fully created yet
file(GLOB LIGHTHOUSE_LINK_DIRS "${LIGHTHOUSE2_SRC_DIR}/lib/*/")
string(REPLACE ${LIGHTHOUSE2_SRC_DIR}/lib/ ${CMAKE_CURRENT_BINARY_DIR}/lighthouse2/lib/
    LIGHTHOUSE_LINK_DIRS "${LIGHTHOUSE_LINK_DIRS}")

set(LIGHTHOUSE2_INCLUDE_DIRS
    ${LIGHTHOUSE2_SRC_DIR}/lib/RenderSystem/
    ${LIGHTHOUSE2_SRC_DIR}/lib/platform/
    ${LIGHTHOUSE2_SRC_DIR}/lib/half2.1.0/
    ${LIGHTHOUSE2_SRC_DIR}/lib/CUDA/
    ${LIGHTHOUSE2_SRC_DIR}/lib/tinyxml2/
    ${LIGHTHOUSE2_SRC_DIR}/lib/glad/include/)

set(LIGHTHOUSE2_LIBS
    RenderSystem
    Platform
    tinyxml2
    glad
    FreeType2::freetype2
    FreeImage::freeimage)

#project
file(GLOB_RECURSE SRC_SERVER "*.cpp" "*.h" "*.cu" "*.cuh")
add_executable(${PROJECT_NAME} ${SRC_SERVER} ${SRC_COMMON})
set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/data/server")

add_dependencies(${PROJECT_NAME} Lighthouse2)
target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_LIGHTHOUSE)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${GLM_INCLUDE_DIRS}
    ${COMMON_INCLUDE}
    ${IMGUI_INCLUDE}
    ${Boost_INCLUDE_DIRS}
    ${GL_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${FFMPEG_INCLUDE_DIR}
    ${LIGHTHOUSE2_INCLUDE_DIRS}
    ${NVCODEC_INCLUDE_DIR})

target_link_directories(${PROJECT_NAME} PRIVATE
    "${LIGHTHOUSE_LINK_DIRS}"
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    cuda
    "${COMMON_LIBRARIES}"
    "${GL_USED_LIBS}"
    "${Boost_SYSTEM_LIBRARY}"
    "${FFMPEG_LIBRARIES}"
    "${LIGHTHOUSE2_LIBS}"
    "${CUDA_nppicc_LIBRARY}"
    "${NVCODEC_LIBRARIES}")

    
#Assimp
set(ON 1) #fix for assimp
set(ASSIMP_SDK_PATH "" CACHE PATH "Directory containing installed Assimp SDK")
if (ASSIMP_SDK_PATH)
    set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${ASSIMP_SDK_PATH}/cmake-modules")
endif()
find_package(assimp QUIET)
if (asssimp_FOUND)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_ASSIMP)
    target_include_directories(${PROJECT_NAME} PRIVATE ${ASSIMP_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE "${ASSIMP_LIBRARIES}")
    target_link_directories(${PROJECT_NAME} PRIVATE "${ASSIMP_LIBRARY_DIRS}")
endif()

find_package(NVJPEG QUIET)
if (NVJPEG_FOUND)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_NVJPEG)
    target_include_directories(${PROJECT_NAME} PRIVATE ${NVJPEG_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE "${NVJPEG_LIBRARIES}")
endif()

if (WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE Boost::boost)
else ()
    target_link_libraries(${PROJECT_NAME} PUBLIC stdc++fs)
endif()

target_compile_definitions(${PROJECT_NAME} PRIVATE GLM_FORCE_CUDA)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:${PROJECT_NAME}> $<TARGET_FILE_DIR:${PROJECT_NAME}>
  COMMAND_EXPAND_LISTS)