# FindTurboJPEG.cmake
# Uses environment variable TurboJPEG_ROOT as backup
# - TURBOJPEG_FOUND
# - TURBOJPEG_INCLUDE_DIRS
# - TURBOJPEG_LIBRARIES

FIND_PATH(TURBOJPEG_INCLUDE_DIRS
  turbojpeg.h
  DOC "Found TurboJPEG include directory"
  PATHS
    "${DEPENDS_DIR}/libjpeg_turbo"
    "${DEPENDS_DIR}/libjpeg-turbo64"
    "/usr/local/opt/jpeg-turbo" # homebrew
    "/opt/local" # macports
    "C:/libjpeg-turbo64"
    "/opt/libjpeg-turbo"
    ENV TurboJPEG_ROOT
  PATH_SUFFIXES
    include
)

#Library names:
# debian sid,strech: libturbojpeg0
# debian/ubuntu else: libturbojpeg1-dev #provided by libjpeg-turbo8-dev (ubuntu)
FIND_LIBRARY(TURBOJPEG_LIBRARIES
  NAMES libturbojpeg.so.1 libturbojpeg.so.0 turbojpeg
  DOC "Found TurboJPEG library path"
  PATHS
    "${DEPENDS_DIR}/libjpeg_turbo"
    "${DEPENDS_DIR}/libjpeg-turbo64"
    "/usr/local/opt/jpeg-turbo" # homebrew
    "/opt/local" # macports
    "C:/libjpeg-turbo64"
    "/opt/libjpeg-turbo"
    ENV TurboJPEG_ROOT
  PATH_SUFFIXES
    lib
    lib64
)

IF(WIN32)
FIND_FILE(TurboJPEG_DLL
  turbojpeg.dll
  DOC "Found TurboJPEG DLL path"
  PATHS
    "${DEPENDS_DIR}/libjpeg_turbo"
    "${DEPENDS_DIR}/libjpeg-turbo64"
    "C:/libjpeg-turbo64"
    ENV TurboJPEG_ROOT
  PATH_SUFFIXES
    bin
)
ENDIF()

IF(TURBOJPEG_INCLUDE_DIRS AND TURBOJPEG_LIBRARIES)
INCLUDE(CheckCSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${TURBOJPEG_INCLUDE_DIRS})
set(CMAKE_REQUIRED_LIBRARIES ${TURBOJPEG_LIBRARIES})
check_c_source_compiles("#include <turbojpeg.h>\nint main(void) { tjhandle h=tjInitCompress(); return 0; }" TURBOJPEG_WORKS)
set(CMAKE_REQUIRED_DEFINITIONS)
set(CMAKE_REQUIRED_INCLUDES)
set(CMAKE_REQUIRED_LIBRARIES)
ENDIF()

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(TurboJPEG FOUND_VAR TURBOJPEG_FOUND
  REQUIRED_VARS TURBOJPEG_LIBRARIES TURBOJPEG_INCLUDE_DIRS TURBOJPEG_WORKS)