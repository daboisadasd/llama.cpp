cmake_minimum_required(VERSION 3.20)
project(llama_dspine_graph_smoke LANGUAGES C CXX ASM)

# CPU test remains available without CUDA/RPC. An optional second executable
# can directly execute the learned code bridge on a remote native GGML RPC GPU.
option(DSPINE_BRIDGE_RPC "Build experimental real-checkpoint RPC bridge test" OFF)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
if (DSPINE_BRIDGE_RPC)
    set(GGML_RPC ON CACHE BOOL "" FORCE)
else()
    set(GGML_RPC OFF CACHE BOOL "" FORCE)
endif()
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../ggml" "${CMAKE_CURRENT_BINARY_DIR}/ggml")

add_executable(test-dspine-bridge test_native_bridge.cpp)
target_compile_features(test-dspine-bridge PRIVATE cxx_std_17)
target_include_directories(test-dspine-bridge PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/models"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include")
target_link_libraries(test-dspine-bridge PRIVATE ggml)

add_executable(test-dspine-refine test_native_refine.cpp)
target_compile_features(test-dspine-refine PRIVATE cxx_std_17)
target_include_directories(test-dspine-refine PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/models"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include")
target_link_libraries(test-dspine-refine PRIVATE ggml)

add_executable(test-dspine-tail test_native_tail.cpp)
target_compile_features(test-dspine-tail PRIVATE cxx_std_17)
target_include_directories(test-dspine-tail PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/models"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include")
target_link_libraries(test-dspine-tail PRIVATE ggml)

add_executable(test-dspine-tables test_native_tables.cpp)
target_compile_features(test-dspine-tables PRIVATE cxx_std_17)
target_include_directories(test-dspine-tables PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/models"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include")
target_link_libraries(test-dspine-tables PRIVATE ggml)

add_executable(test-dspine-gguf-names test_gguf_names.cpp)
target_compile_features(test-dspine-gguf-names PRIVATE cxx_std_17)
target_include_directories(test-dspine-gguf-names PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include"
    "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test-dspine-gguf-names PRIVATE ggml)

// The real RPC draft path carries BF16 inputs; this CPU reference checks
// CUDA-incompatible RMSNorm input dtypes before a remote graph is submitted.
add_executable(test-dspine-bridge-bf16 test_native_bridge_bf16.cpp)
target_compile_features(test-dspine-bridge-bf16 PRIVATE cxx_std_17)
target_include_directories(test-dspine-bridge-bf16 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/models"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include")
target_link_libraries(test-dspine-bridge-bf16 PRIVATE ggml)

enable_testing()
add_test(NAME dspine-native-bridge-bf16 COMMAND test-dspine-bridge-bf16)
add_test(NAME dspine-native-gguf-names COMMAND test-dspine-gguf-names)
add_test(NAME dspine-native-tables COMMAND test-dspine-tables)
add_test(NAME dspine-native-bridge COMMAND test-dspine-bridge)
add_test(NAME dspine-native-refine COMMAND test-dspine-refine)
add_test(NAME dspine-native-tail COMMAND test-dspine-tail)

if (DSPINE_BRIDGE_RPC)
    add_executable(test-dspine-rpc-bridge test_remote_bridge.cpp)
    target_compile_features(test-dspine-rpc-bridge PRIVATE cxx_std_17)
    target_include_directories(test-dspine-rpc-bridge PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../src/models"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include")
    target_link_libraries(test-dspine-rpc-bridge PRIVATE ggml ggml-rpc)
endif()
