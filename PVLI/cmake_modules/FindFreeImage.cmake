# Find the FreeImage library.
#
# This module defines FreeImage::freeimage        - Imported target; link
# against this in your project. FreeImage_FOUND             - True if FreeImage
# was found.

# Additionaly, these variables are defined and can be used:
# FREEIMAGE_INCLUDE_DIRS      - Include directories for FreeImage headers.
# FREEIMAGE_LIBRARY           - Libraries for FreeImage.
#
# To specify an additional directory to search, set FREEIMAGE_ROOT.
#
# Copyright (c) 2010, Ewen Cheslack-Postava Based on FindSQLite3.cmake by:
# Copyright (c) 2006, Jaroslaw Staniek, <js@iidea.pl> Extended by Siddhartha
# Chaudhuri, 2008.
#
# Redistribution and use is allowed according to the terms of the BSD license.
#

set(SEARCH_PATHS
    $ENV{ProgramFiles}/freeimage/include $ENV{SystemDrive}/freeimage/include
    $ENV{ProgramFiles}/freeimage $ENV{SystemDrive}/freeimage)
if(FREEIMAGE_ROOT)
  set(SEARCH_PATHS ${FREEIMAGE_ROOT} ${FREEIMAGE_ROOT}/include
                   ${FREEIMAGE_ROOT}/inc ${SEARCH_PATHS})
endif()

find_path(
  FREEIMAGE_INCLUDE_DIRS
  NAMES FreeImage.h
  PATHS ${SEARCH_PATHS}
  NO_DEFAULT_PATH)
if(NOT FREEIMAGE_INCLUDE_DIRS) # now look in system locations
  find_path(FREEIMAGE_INCLUDE_DIRS NAMES FreeImage.h)
endif(NOT FREEIMAGE_INCLUDE_DIRS)

if(FREEIMAGE_ROOT)
  set(FREEIMAGE_LIBRARY_DIRS
      ${FREEIMAGE_ROOT} ${FREEIMAGE_ROOT}/lib ${FREEIMAGE_ROOT}/lib/static
      ${FREEIMAGE_ROOT}/lib64 ${FREEIMAGE_ROOT}/lib64/static)
endif()

# FREEIMAGE Without system dirs
find_library(
  FREEIMAGE_LIBRARY
  NAMES freeimage
  PATHS ${FREEIMAGE_LIBRARY_DIRS}
  NO_DEFAULT_PATH)
if(NOT FREEIMAGE_LIBRARY) # now look in system locations
  find_library(FREEIMAGE_LIBRARY NAMES freeimage)
endif(NOT FREEIMAGE_LIBRARY)

if(WIN32 AND FREEIMAGE_LIBRARY)
  find_file(
    FREEIMAGE_RUNTIME_LIBRARY
    NAMES FreeImage.dll
    PATHS ${FREEIMAGE_LIBRARY_DIRS})
endif(WIN32 AND FREEIMAGE_LIBRARY)

# Handle QUIET & REQUIRED flags. Set FreeImage_FOUND properly:
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FreeImage DEFAULT_MSG FREEIMAGE_LIBRARY
                                  FREEIMAGE_INCLUDE_DIRS)

if(FreeImage_FOUND)
  if(NOT TARGET FreeImage::freeimage)
    message(STATUS "${FREEIMAGE_INCLUDE_DIRS}")
    add_library(FreeImage::freeimage SHARED IMPORTED)
    if(WIN32)
      set_target_properties(
        FreeImage::freeimage
        PROPERTIES IMPORTED_IMPLIB "${FREEIMAGE_LIBRARY}"
                   IMPORTED_LOCATION "${FREEIMAGE_RUNTIME_LIBRARY}")
    else(WIN32)
      set_target_properties(FreeImage::freeimage
                            PROPERTIES IMPORTED_LOCATION "${FREEIMAGE_LIBRARY}")
    endif(WIN32)
    set_target_properties(
      FreeImage::freeimage PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                      "${FREEIMAGE_INCLUDE_DIRS}")
  endif(NOT TARGET FreeImage::freeimage)
endif(FreeImage_FOUND)

mark_as_advanced(FREEIMAGE_INCLUDE_DIRS FREEIMAGE_LIBRARY)