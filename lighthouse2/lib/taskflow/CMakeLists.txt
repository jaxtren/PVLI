project(Taskflow LANGUAGES CXX)

add_library(${PROJECT_NAME} INTERFACE IMPORTED GLOBAL)
add_library(Taskflow::Taskflow ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} INTERFACE ${PROJECT_SOURCE_DIR})

# TODO Transitive cxx 17 compile feature breaks CUDA version:
# https://gitlab.kitware.com/cmake/cmake/issues/19123
# target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
