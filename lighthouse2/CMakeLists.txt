# https://cmake.org/cmake/help/latest/module/FindCUDA.html: 3.10 supports CUDA
# asa first-class language:
cmake_minimum_required(VERSION 3.10)

cmake_policy(SET CMP0074 NEW)

project(Lighthouse2)

# Module and Config searches look in the cmake subdirectory first:
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")
set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/cmake" "${CMAKE_PREFIX_PATH}")

# Install binaries, libraries and data files (scenes, models and shaders) to the
# same root directory.
set(CMAKE_INSTALL_BINDIR ".")
set(CMAKE_INSTALL_LIBDIR ".")
set(CMAKE_INSTALL_DATADIR ".")
if(APPLE)
  # Allow the linker to look for libraries in the current working directory
  set(CMAKE_INSTALL_RPATH ".")
else()
  # Allow the linker to look for libraries in the same directory
  set(CMAKE_INSTALL_RPATH "$ORIGIN")
  # Link build-time libraries with $ORIGIN RPATH token. This allows the build
  # directory to be relocated.
  set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
endif()

if(WIN32)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
endif()

# External dependencies bundled with the source:
add_subdirectory("lib/glad")
add_subdirectory("lib/half2.1.0")
add_subdirectory("lib/tinyxml2")
add_subdirectory("lib/tinygltf")
add_subdirectory("lib/tinyobjloader")
add_subdirectory("lib/imgui")
add_subdirectory("lib/taskflow")

# Platform configuration goes first:
add_subdirectory("lib/platform")
add_subdirectory("lib/RenderSystem")
# HACK! Circular dependency on project *headers*
target_include_directories(Platform PRIVATE "lib/RenderSystem")
# Note that there is a rather peculiar dependency mechanism. Things defined in
# system.h are implemented in platform.cpp, and headers are shared between
# lib/platform and lib/RenderSystem making dependency management ever so
# slightly less elegant.

# Include cross-compatible BSDF implementation:
add_subdirectory("lib/sharedBSDFs")

macro(add_render_core core_path)
  if(NOT EXISTS "${core_path}/CMakeLists.txt")
    message(
      WARNING
        "${core_path} does not contain CMakeLists.txt, it will not be built.")
  else()
    add_subdirectory(${core_path})
    get_filename_component(core_name ${core_path} NAME)
    # Ensure rendercore is named just like the folder:
    if(NOT TARGET ${core_name})
      message(FATAL_ERROR "${core_name} Does not name a target!")
    endif()
    # Collect compiled rendercores for proper loading from apps:
    list(APPEND render_cores ${core_name})
  endif()
endmacro()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  # Allow constexpr usage on the device, without __device__ (eg. std::max):
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G") #debug for CUDA objects
  else()
    # https://docs.nvidia.com/cuda/cuda-memcheck/index.html#compilation-options
    # Add lineinfo and symbols to the assembly, for improved debugging.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -restrict -use_fast_math")
  endif()

  if(NOT MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -rdynamic")
  endif()

  # Include CUDA support headers:
  add_subdirectory("lib/CUDA")

  # Include CUDA+Optix RenderCores:
  file(GLOB cores "lib/RenderCore_Opti*" "lib/RenderCore_Prime*")
  foreach(core IN LISTS cores)
    add_render_core(${core})
  endforeach()
endif()

find_package(Vulkan QUIET)
find_package(Shaderc QUIET)
if(Vulkan_FOUND AND Shaderc_FOUND)
  #add_render_core("${CMAKE_SOURCE_DIR}/lib/RenderCore_Vulkan_RT")
endif()

# Include default RenderCores:
add_render_core("${CMAKE_SOURCE_DIR}/lib/RenderCore_SoftRasterizer")
add_render_core("${CMAKE_SOURCE_DIR}/lib/RenderCore_Minimal")

message(STATUS "Found RenderCores: ${render_cores}")

foreach(core_name IN LISTS render_cores)
  list(APPEND core_folders $<TARGET_FILE_DIR:${core_name}>)
endforeach()

# Create a macro for app cmake definitions to call. Their folder name need not
# match their project/executable name.
function(register_app app_target)
  # Ensure RenderCores are built as app dependency, in particular when running
  # (debugging) the startup project (an app) in Visual Studio.
  add_dependencies(${app_target} ${render_cores})

  if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13" AND UNIX)
    # Set rpath for dlopen to find the cores:
    target_link_directories(${PROJECT_NAME} PRIVATE ${core_folders})
  elseif(MSVC)
    # Note that target_link_directories has no effect, since libraries need
    # either be in the same directory, or on the PATH (CMAKE_MSVCIDE_RUN_PATH).
    # For now, all executables and libraries are compiled to the same folder in
    # the project root (only on WIN32).
  else()
    message(FATAL_ERROR "CMake < 3.13 doesn't support target_link_directories!")
    # TODO: Find a better solution on these platforms: Also move all output to
    # the same directory? Use PATH variables during debugging? Rely on
    # install/export? (not efficient for debugging sessions)
  endif()
endfunction()

# Include applications:
file(GLOB app_directories "apps/*")
foreach(app IN LISTS app_directories)
  if(NOT EXISTS "${app}/CMakeLists.txt")
    message(
      WARNING "${app} does not contain CMakeLists.txt, it will not be built.")
    continue()
  endif()

  add_subdirectory(${app})
endforeach()

if(WIN32)
  set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ImguiApp)
endif()
