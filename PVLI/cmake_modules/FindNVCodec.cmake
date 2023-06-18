# - Try to find nvcodec
#
# The following are set after configuration is done:
#  NVCODEC_FOUND
#  NVCODEC_INCLUDE_DIR
#  NVCODEC_LIBRARIES

set(NVCODEC_ROOT_DIR "" CACHE PATH "Directory containing NVIDIA Video Codec SDK")

find_path(NVCODEC_INCLUDE_DIR
    NAMES nvEncodeAPI.h nvcuvid.h cuviddec.h
    PATHS 
      ${NVCODEC_ROOT_DIR}/Interface
      /usr/include
      /usr/local/include
      /opt/local/include
      /sw/include)

find_library(NVCODEC_LIBNVENCODE
    NAMES nvencodeapi
    PATHS
      ${NVCODEC_ROOT_DIR}/Lib/x64
      /usr/lib
      /usr/local/lib
      /opt/local/lib
      /sw/lib)

find_library(NVCODEC_LIBNVCUVID
    NAMES nvcuvid
    PATHS
      ${NVCODEC_ROOT_DIR}/Lib/x64
      /usr/lib
      /usr/local/lib
      /opt/local/lib
      /sw/lib)

if(NVCODEC_INCLUDE_DIR AND NVCODEC_LIBNVENCODE AND NVCODEC_LIBNVCUVID)
    set(NVCODEC_FOUND TRUE)
endif()

if(NVCODEC_FOUND)
    set(NVCODEC_LIBRARIES
      ${NVCODEC_LIBNVENCODE}
      ${NVCODEC_LIBNVCUVID})
endif()

include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(NVCODEC
        FOUND_VAR NVCODEC_FOUND
        REQUIRED_VARS NVCODEC_INCLUDE_DIR NVCODEC_LIBRARIES
        VERSION_VAR NVCODEC_VERSION)