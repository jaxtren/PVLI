project(RenderSystem)

file(GLOB_RECURSE sources "*.cpp")
file(GLOB_RECURSE headers "*.h")
add_library(${PROJECT_NAME} ${sources} ${headers})
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17
                                                 CXX_STANDARD_REQUIRED ON)
target_compile_definitions(${PROJECT_NAME} PRIVATE RENDERSYSTEMBUILD)
#target_compile_options(
#  ${PROJECT_NAME} PRIVATE $<$<BOOL:${MSVC}>:/arch:AVX2>
#                          $<$<NOT:$<BOOL:${MSVC}>>:-mavx2 -mfma>)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR})
if(NOT WIN32)
  target_link_libraries(${PROJECT_NAME} PRIVATE pthread dl)
endif()
if(MSVC)
  # Prevent CUDA projects from inheriting this flag.
  target_compile_options(${PROJECT_NAME} PRIVATE /MP)
endif()

# External dependencies, PUBLIC because exposed through header:
target_link_libraries(${PROJECT_NAME} PUBLIC tinyxml2 TinyGLTF::tinygltf
                                             TinyOBJLoader::tinyobjloader)

target_link_libraries(${PROJECT_NAME} PUBLIC Platform)

include(CheckCXXSourceCompiles)

check_cxx_source_compiles(
  "
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
int main() {
   int fd = open(\"foo\", O_RDONLY);
   struct stat s;
   fstat(fd, &s);
   size_t len = s.st_size;
   void *ptr = mmap(0, len, PROT_READ, MAP_FILE | MAP_SHARED, fd, 0);
   munmap(ptr, len);
}
"
  HAVE_MMAP)

target_compile_definitions(
  ${PROJECT_NAME}
  PUBLIC $<$<BOOL:${HAVE_MMAP}>:PBRT_HAVE_MMAP>
         $<$<OR:$<BOOL:${MSVC}>,$<BOOL:${WIN32}>>:PBRT_IS_WINDOWS>)

if(BUILD_SHARED_LIBS)
  install(TARGETS ${PROJECT_NAME})
endif()
