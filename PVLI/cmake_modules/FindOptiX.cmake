#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met: *
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer. * Redistributions in binary
# form must reproduce the above copyright notice, this list of conditions and
# the following disclaimer in the documentation and/or other materials provided
# with the distribution. * Neither the name of NVIDIA CORPORATION nor the names
# of its contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO
# EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# Locate the OptiX distribution.  Search relative to the SDK first, then look in
# the system.

# Our initial guess will be within the SDK.
set(OptiX_INSTALL_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../"
    CACHE PATH "Path to OptiX installed location.")
set(OptiX_VERSION
    "6.5.0"
    CACHE STRING "The full OptiX6 version string")

# The distribution contains only 64 bit libraries.  Error when we have been mis-
# configured.
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  if(WIN32)
    message(
      SEND_ERROR
        "Make sure when selecting the generator, you select one with Win64 or x64."
    )
  endif()
  message(FATAL_ERROR "OptiX only supports builds configured for 64 bits.")
endif()

# search path based on the bit-ness of the build.  (i.e. 64: bin64, lib64; 32:
# bin, lib).  Note that on Mac, the OptiX library is a universal binary, so we
# only need to look in lib and not lib64 for 64 bit builds.
if(NOT APPLE)
  set(bit_dest "64")
else()
  set(bit_dest "")
endif()

macro(OPTIX_find_api_library name version)
  find_library(
    ${name}_LIBRARY
    NAMES ${name}.${version} ${name}
    PATHS "${OptiX_INSTALL_DIR}/lib${bit_dest}"
    NO_DEFAULT_PATH)
  find_library(${name}_LIBRARY NAMES ${name}.${version} ${name})
  if(WIN32)
    find_file(
      ${name}_DLL
      NAMES ${name}.${version}.dll
      PATHS "${OptiX_INSTALL_DIR}/bin${bit_dest}"
      NO_DEFAULT_PATH)
    find_file(${name}_DLL NAMES ${name}.${version}.dll)
    # Create a list of _variable names_ containing the DLLs. This allows FPHSA
    # to check whether the DLLs are available, only if the target is WIN32.
    list(APPEND OptiX_RUNTIME_LIBRARY_VARS ${name}_DLL)
  endif()
endmacro()

optix_find_api_library(optix ${OptiX_VERSION})
optix_find_api_library(optixu ${OptiX_VERSION})
optix_find_api_library(optix_prime ${OptiX_VERSION})

# Include
find_path(
  OptiX_INCLUDE
  NAMES optix.h
  PATHS "${OptiX_INSTALL_DIR}/include"
  NO_DEFAULT_PATH)
find_path(OptiX_INCLUDE NAMES optix.h)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  OptiX
  DEFAULT_MSG
  OptiX_INCLUDE
  optix_LIBRARY
  optixu_LIBRARY
  optix_prime_LIBRARY
  ${OptiX_RUNTIME_LIBRARY_VARS})

# Macro for setting up dummy targets
function(OptiX_add_imported_library name lib_location dll_lib dependent_libs)
  set(CMAKE_IMPORT_FILE_VERSION 1)

  # Create imported target
  add_library(${name} SHARED IMPORTED)

  # Import target "optix" for configuration "Debug"
  if(WIN32)
    set_target_properties(
      ${name}
      PROPERTIES IMPORTED_IMPLIB "${lib_location}"
                 # IMPORTED_LINK_INTERFACE_LIBRARIES "glu32;opengl32"
                 IMPORTED_LOCATION "${dll_lib}"
                 IMPORTED_LINK_INTERFACE_LIBRARIES "${dependent_libs}")
  elseif(UNIX)
    set_target_properties(
      ${name}
      PROPERTIES # IMPORTED_LINK_INTERFACE_LIBRARIES "glu32;opengl32"
                 IMPORTED_LOCATION "${lib_location}"
                 # We don't have versioned filenames for now, and it may not
                 # even matter.
                 # IMPORTED_SONAME "${optix_soname}"
                 IMPORTED_LINK_INTERFACE_LIBRARIES "${dependent_libs}")
  else()
    # Unknown system, but at least try and provide the minimum required
    # information.
    set_target_properties(
      ${name} PROPERTIES IMPORTED_LOCATION "${lib_location}"
                         IMPORTED_LINK_INTERFACE_LIBRARIES "${dependent_libs}")
  endif()

  # Commands beyond this point should not need to know the version.
  set(CMAKE_IMPORT_FILE_VERSION)
endfunction()

# Sets up a dummy target
optix_add_imported_library(optix "${optix_LIBRARY}" "${optix_DLL}"
                           "${OPENGL_LIBRARIES}")
optix_add_imported_library(optixu "${optixu_LIBRARY}" "${optixu_DLL}" "")
optix_add_imported_library(optix_prime "${optix_prime_LIBRARY}"
                           "${optix_prime_DLL}" "")

macro(OptiX_check_same_path libA libB)
  if(_optix_path_to_${libA})
    if(NOT _optix_path_to_${libA} STREQUAL _optix_path_to_${libB})
      # ${libA} and ${libB} are in different paths.  Make sure there isn't a
      # ${libA} next to the ${libB}.
      get_filename_component(_optix_name_of_${libA} "${${libA}_LIBRARY}" NAME)
      if(EXISTS "${_optix_path_to_${libB}}/${_optix_name_of_${libA}}")
        message(
          WARNING
            " ${libA} library found next to ${libB} library that is not being used.  Due to the way we are using rpath, the copy of ${libA} next to ${libB} will be used during loading instead of the one you intended.  Consider putting the libraries in the same directory or moving ${_optix_path_to_${libB}}/${_optix_name_of_${libA}} out of the way."
        )
      endif()
    endif()
    set(_${libA}_rpath "-Wl,-rpath,${_optix_path_to_${libA}}")
  endif()
endmacro()

# Since liboptix.1.dylib is built with an install name of @rpath, we need to
# compile our samples with the rpath set to where optix exists.
if(APPLE)
  get_filename_component(_optix_path_to_optix "${optix_LIBRARY}" PATH)
  if(_optix_path_to_optix)
    set(_optix_rpath "-Wl,-rpath,${_optix_path_to_optix}")
  endif()
  get_filename_component(_optix_path_to_optixu "${optixu_LIBRARY}" PATH)
  optix_check_same_path(optixu optix)
  get_filename_component(_optix_path_to_optix_prime "${optix_prime_LIBRARY}"
                         PATH)
  optix_check_same_path(optix_prime optix)
  optix_check_same_path(optix_prime optixu)

  set(optix_rpath ${_optix_rpath} ${_optixu_rpath} ${_optix_prime_rpath})
  list(REMOVE_DUPLICATES optix_rpath)
endif()
