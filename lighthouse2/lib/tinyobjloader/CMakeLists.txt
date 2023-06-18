project(tinyobjloader)

add_library(${PROJECT_NAME} INTERFACE IMPORTED GLOBAL)
add_library(TinyOBJLoader::tinyobjloader ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} INTERFACE ${PROJECT_SOURCE_DIR})
