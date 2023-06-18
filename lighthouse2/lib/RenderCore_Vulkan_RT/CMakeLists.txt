project("RenderCore_Vulkan_RT")

file(GLOB_RECURSE sources "*.cpp")
file(GLOB_RECURSE headers "*.h")
add_library(${PROJECT_NAME} MODULE ${sources} ${headers})
if(MSVC)
  set_target_properties(
    ${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                               "${CMAKE_SOURCE_DIR}/coredlls")
endif()
target_compile_definitions(${PROJECT_NAME} PRIVATE COREDLL_EXPORTS=1)
target_link_libraries(${PROJECT_NAME} PRIVATE RenderSystem)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS
                                                                 OFF)

find_package(Vulkan REQUIRED)
find_package(Shaderc REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan Shaderc::shaderc)

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
