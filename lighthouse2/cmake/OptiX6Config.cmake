# A copy of FindOptiX.cmake resides in the local tree:
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/lib/OptiX/CMake")
find_package(OptiX)
list(POP_FRONT CMAKE_MODULE_PATH)

# Extend libraries with include directories:
target_include_directories(optix INTERFACE ${OptiX_INCLUDE})
target_include_directories(optixu INTERFACE ${OptiX_INCLUDE})
target_include_directories(optix_prime INTERFACE ${OptiX_INCLUDE})

foreach(lib IN ITEMS optix optixu optix_prime)
  list(APPEND install_libs "$<TARGET_PROPERTY:${lib},IMPORTED_LOCATION>")
  if(UNIX)
    # Add the linked library. Unfortunately: - file(INSTALL) with
    # FOLLOW_SYMLINK_CHAIN doesn't allow generator expressions -
    # get_filename_component(REALPATH) doesn't allow generator expressions -
    # These files might be installation-dependent.
    list(APPEND install_libs
         "$<TARGET_PROPERTY:${lib},IMPORTED_LOCATION>.${OptiX_VERSION}")
  endif()
endforeach()

install(FILES ${install_libs} TYPE LIB)
# file(INSTALL ${install_libs} DESTINATION ${CMAKE_INSTALL_LIBDIR}
# FOLLOW_SYMLINK_CHAIN)
