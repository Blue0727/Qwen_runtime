cmake_minimum_required(VERSION 3.20)

project(MiniEdgeLLM LANGUAGES CXX)

find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(qwen_runtime
    src/main.cpp
    src/model_config.cpp
    src/config_loader.cpp
    src/safetensors_loader.cpp
    src/operators.cpp
    src/model.cpp
)

target_compile_features(qwen_runtime PRIVATE cxx_std_17)
set_target_properties(qwen_runtime PROPERTIES CXX_EXTENSIONS OFF)

target_include_directories(qwen_runtime PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)

target_link_libraries(qwen_runtime PRIVATE OpenMP::OpenMP_CXX)

if(MSVC)
    target_compile_options(qwen_runtime PRIVATE /W4 /utf-8 /EHsc)
endif()