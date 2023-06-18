# Derived from: caffe2/cmake/Modules/FindJpegTurbo.cmake
#
# - Try to find nvjpeg
#
# The following variables are optionally searched for defaults
#  NVJPEG_ROOT_DIR:            Base directory where all NVJPEG components are found
#
# The following are set after configuration is done:
#  NVJPEG_FOUND
#  NVJPEG_INCLUDE_DIRS
#  NVJPEG_LIBRARIES

set(NVJPEG_ROOT_DIR "" CACHE PATH "Folder contains NVJPEG")

find_path(NVJPEG_INCLUDE_DIRS nvjpeg.h
    PATHS ${NVJPEG_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    PATH_SUFFIXES / include)

find_library(NVJPEG_LIBRARIES nvjpeg
    PATHS ${NVJPEG_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR} ${CMAKE_CUDA_TOOLKIT_LIB_DIRECTORIES}
    PATH_SUFFIXES / lib lib64)

# nvJPEG 9.0 calls itself 0.1.x via API calls, and the header file doesn't tell you which
# version it is. There's not a super clean way to determine which CUDA's nvJPEG we have.
execute_process(COMMAND strings ${NVJPEG_LIBRARIES} COMMAND grep /toolkit/
                COMMAND sed "s;^.*toolkit/r\\(\[^/\]\\+\\\).*$;\\1;" COMMAND sort -u
                OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NVJPEG_VERSION)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVJPEG
    REQUIRED_VARS NVJPEG_INCLUDE_DIRS NVJPEG_LIBRARIES
    VERSION_VAR NVJPEG_VERSION)

if(NVJPEG_FOUND)
  # set includes and link libs for nvJpeg
  set(CMAKE_REQUIRED_INCLUDES ${CUDA_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${NVJPEG_LIBRARY} "-lnvjpeg" "-lcudart_static" "-lculibos" "dl" "-pthread" "rt")
  #check_symbol_exists("nvjpegCreateEx" "nvjpeg.h" NVJPEG_LIBRARY_0_2_0)
  #check_symbol_exists("nvjpegBufferPinnedCreate" "nvjpeg.h" NVJPEG_DECOUPLED_API)

  mark_as_advanced(NVJPEG_ROOT_DIR NVJPEG_LIBRARY_RELEASE NVJPEG_LIBRARY_DEBUG)
  if (NVJPEG_DECOUPLED_API)
    message(STATUS "nvJPEG is using new API")
  endif()
endif()