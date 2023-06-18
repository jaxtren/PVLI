if(WIN32)
  # Use prebuilts on Windows:
  set(FREEIMAGE_ROOT "${CMAKE_SOURCE_DIR}/lib/FreeImage")
endif()
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/lib/FreeImage")
find_package(FreeImage)
list(POP_FRONT CMAKE_MODULE_PATH)

if(WIN32)
  install(FILES "$<TARGET_PROPERTY:FreeImage::freeimage,IMPORTED_LOCATION>"
          TYPE LIB)
endif()
