cmake_minimum_required(VERSION 3.20)

project(MiniEdgeLLM LANGUAGES CXX)

option(QWEN_ENABLE_AVX2 "Enable AVX2-targeted SIMD operators" OFF)
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

    if(QWEN_ENABLE_AVX2)
        set_source_files_properties(
            src/operators.cpp
            PROPERTIES COMPILE_OPTIONS "/arch:AVX2"
        )
    endif()
endif()