project(tinygltf)

add_library(${PROJECT_NAME} INTERFACE IMPORTED GLOBAL)
add_library(TinyGLTF::tinygltf ALIAS ${PROJECT_NAME})
target_include_directories(
  ${PROJECT_NAME} INTERFACE "${PROJECT_SOURCE_DIR}"
                            "${PROJECT_SOURCE_DIR}/rapidjson")
