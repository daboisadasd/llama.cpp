cmake_minimum_required(VERSION 3.20)
project(llama_dspine_graph_smoke LANGUAGES C CXX ASM)

# Build GGML's CPU backend in isolation; this test does not depend on
# CUDA, the 27B GGUF, SGLang, or any RPC worker.
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_RPC OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../ggml" "${CMAKE_CURRENT_BINARY_DIR}/ggml")

add_executable(test-dspine-bridge test_native_bridge.cpp)
target_compile_features(test-dspine-bridge PRIVATE cxx_std_17)
target_include_directories(test-dspine-bridge PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/models"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include")
target_link_libraries(test-dspine-bridge PRIVATE ggml)

enable_testing()
add_test(NAME dspine-native-bridge COMMAND test-dspine-bridge)
