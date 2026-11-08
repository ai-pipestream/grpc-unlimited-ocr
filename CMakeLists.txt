cmake_minimum_required(VERSION 3.25)
project(grpc-unlimited-ocr VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF)

option(UOCR_WERROR "Treat warnings as errors for first-party targets" OFF)
set(UOCR_SANITIZE "" CACHE STRING "Comma list for -fsanitize on first-party targets")

add_library(uocr_flags INTERFACE)
target_compile_options(uocr_flags INTERFACE -Wall -Wextra -Wnon-virtual-dtor -Wconversion -Wno-sign-conversion)
if(UOCR_WERROR)
  target_compile_options(uocr_flags INTERFACE -Werror)
endif()
if(UOCR_SANITIZE)
  target_compile_options(uocr_flags INTERFACE -fsanitize=${UOCR_SANITIZE} -fno-omit-frame-pointer -g)
  target_link_options(uocr_flags INTERFACE -fsanitize=${UOCR_SANITIZE})
endif()

find_package(Threads REQUIRED)

include(FetchContent)

# JSON appears in exactly one place: the safetensors file header and the
# shard index, read once at startup. It never crosses the gRPC wire.
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_Declare(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.12.0
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(nlohmann_json)

# stb_image (public domain or MIT): PNG and JPEG decode of page rasters.
FetchContent_Declare(stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG 003501f6d7a8668ce8b6d875e34dd97ad0d5d5b5)
FetchContent_MakeAvailable(stb)

add_library(uocr_weights STATIC
  src/tensor/dtype.cpp
  src/weights/safetensors.cpp
  src/weights/sharded_weights.cpp
  src/weights/architecture.cpp)
target_include_directories(uocr_weights PUBLIC src)
target_link_libraries(uocr_weights PUBLIC uocr_flags PRIVATE nlohmann_json::nlohmann_json)

# The reference backend and the model code written against it.
add_library(uocr_reference STATIC
  src/backend/reference/ops.cpp
  src/model/weight_source.cpp
  src/model/sam_encoder.cpp
  src/model/clip_encoder.cpp
  src/model/vision_encoder.cpp
  src/model/page_rope.cpp
  src/model/page_layout.cpp
  src/model/block_router.cpp
  src/model/feed_forward.cpp
  src/model/decoder.cpp
  src/model/attention_plan.cpp
  src/tokenizer/tokenizer.cpp
  src/runtime/prompt_builder.cpp
  src/runtime/logit_policy.cpp
  src/runtime/generator.cpp
  src/image/resample.cpp
  src/image/page_preprocess.cpp)
target_link_libraries(uocr_reference PUBLIC uocr_weights Threads::Threads PRIVATE nlohmann_json::nlohmann_json)
# stb is third-party C; keep its headers out of the first-party warning set.
target_include_directories(uocr_reference SYSTEM PRIVATE ${stb_SOURCE_DIR})

add_executable(uocr-run tools/uocr_run.cpp)
target_link_libraries(uocr-run PRIVATE uocr_reference)

# Native Metal kernels and model backend.
option(UOCR_WITH_METAL "Build the native Apple Metal backend" OFF)
if(UOCR_WITH_METAL)
  include(cmake/metal.cmake)
endif()

if(BUILD_TESTING)
  foreach(test_name dtype_test safetensors_test architecture_test checkpoint_test ops_test vision_test
                    vision_checkpoint_test page_layout_test router_test logit_policy_test tokenizer_test
                    decoder_test generator_recovery_test image_test golden_test)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE uocr_reference)
    target_include_directories(${test_name} PRIVATE tests)
    target_compile_definitions(${test_name} PRIVATE UOCR_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/tests/goldens")
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES SKIP_RETURN_CODE 77
      ENVIRONMENT "UOCR_TEST_TMP=${CMAKE_CURRENT_BINARY_DIR}/test-tmp")
  endforeach()
endif()

# ---------------------------------------------------------------------------
# CUDA backend (optional). Kernels build with nvcc; nothing CUDA leaks into
# the C++23 host headers, so nvcc's host compiler may be older than ours
# (pass -DCMAKE_CUDA_HOST_COMPILER=g++-13 with a CUDA 12 toolkit).
# ---------------------------------------------------------------------------
option(UOCR_WITH_CUDA "Build the CUDA backend" OFF)
if(UOCR_WITH_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87 89)  # Jetson Orin, Ada
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)
  add_library(uocr_cuda STATIC
    src/backend/cuda/context.cu
    src/backend/cuda/ops.cu
    src/backend/cuda/decode_ops.cu
    src/backend/cuda/gemv_ops.cu)
  target_include_directories(uocr_cuda PUBLIC src)
  target_link_libraries(uocr_cuda PUBLIC CUDA::cudart CUDA::cublas CUDA::cublasLt)
  # Model code on the CUDA backend: C++23 host code over the op set.
  add_library(uocr_cuda_model STATIC
    src/model/cuda/cuda_vision.cpp
    src/model/cuda/cuda_decoder.cpp
    src/model/cuda/cuda_decode_step.cpp
    src/model/cuda/device_weight.cpp)
  target_link_libraries(uocr_cuda_model PUBLIC uocr_reference uocr_cuda)
  target_link_libraries(uocr-run PRIVATE uocr_cuda_model)
  target_compile_definitions(uocr-run PRIVATE UOCR_WITH_CUDA=1)
  if(BUILD_TESTING)
    foreach(test_name cuda_ops_test cuda_vision_test cuda_decoder_test cuda_golden_test)
      add_executable(${test_name} tests/${test_name}.cpp)
      target_link_libraries(${test_name} PRIVATE uocr_cuda_model)
      target_include_directories(${test_name} PRIVATE tests)
      target_compile_definitions(${test_name} PRIVATE UOCR_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/tests/goldens")
      add_test(NAME ${test_name} COMMAND ${test_name})
      set_tests_properties(${test_name} PROPERTIES SKIP_RETURN_CODE 77)
    endforeach()
  endif()
endif()

# ---------------------------------------------------------------------------
# The gRPC service: contract, block parser, Document fold and server. gRPC
# is pinned and wired the way grpc-vlm-convert does it (fetched, static,
# protoc and the plugin built from source, no host protoc). Off builds the
# engine and its tests only.
# ---------------------------------------------------------------------------
option(UOCR_WITH_SERVER "Build the gRPC service" ON)
if(UOCR_WITH_SERVER)
  include(cmake/server.cmake)
endif()

# Initial Intel primitive qualification only. No Intel model/service advertised
# until vision, decoder and real-page gates are implemented and measured.
option(UOCR_WITH_INTEL "Build native Intel SYCL primitive kernels" OFF)
if(UOCR_WITH_INTEL)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-fsycl" UOCR_COMPILER_HAS_SYCL)
  if(NOT UOCR_COMPILER_HAS_SYCL)
    message(FATAL_ERROR "UOCR_WITH_INTEL requires a SYCL compiler (icpx)")
  endif()
  add_library(uocr_intel STATIC src/backend/intel/ops.cpp)
  target_include_directories(uocr_intel PUBLIC src)
  target_compile_options(uocr_intel PRIVATE -fsycl)
  target_link_options(uocr_intel PUBLIC -fsycl)
  target_link_libraries(uocr_intel PUBLIC uocr_flags)
  if(BUILD_TESTING)
    add_executable(intel_ops_test tests/intel_ops_test.cpp)
    target_link_libraries(intel_ops_test PRIVATE uocr_intel uocr_reference)
    add_test(NAME intel_ops_test COMMAND intel_ops_test)
    set_tests_properties(intel_ops_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
  endif()
endif()

if(UOCR_WITH_INTEL)
  find_package(DNNL CONFIG REQUIRED)
  add_library(uocr_intel_native STATIC
    src/backend/intel/native/context.cpp src/backend/intel/native/ops.cpp src/backend/intel/native/layer_norm.cpp
    src/backend/intel/native/gemm.cpp src/backend/intel/native/routing.cpp
    src/backend/intel/native/position_tables.cpp)
  target_include_directories(uocr_intel_native PUBLIC src)
  target_compile_options(uocr_intel_native PRIVATE -fsycl)
  # Do not reassociate the centered LayerNorm expression back into cancellation.
  set_source_files_properties(src/backend/intel/native/layer_norm.cpp PROPERTIES COMPILE_OPTIONS "-fp-model=precise")
  target_link_options(uocr_intel_native PUBLIC -fsycl)
  target_link_libraries(uocr_intel_native PUBLIC DNNL::dnnl)
  add_library(uocr_intel_model STATIC src/model/intel/intel_vision.cpp
    src/model/intel/intel_decoder.cpp src/model/intel/device_weight.cpp)
  target_link_libraries(uocr_intel_model PUBLIC uocr_reference uocr_intel_native)
  target_link_libraries(uocr-run PRIVATE uocr_intel_model)
  target_compile_definitions(uocr-run PRIVATE UOCR_WITH_INTEL=1)
  if(UOCR_WITH_SERVER)
    target_link_libraries(uocr_model_engine PRIVATE uocr_intel_model)
    target_compile_definitions(uocr_model_engine PRIVATE UOCR_WITH_INTEL=1)
  endif()
endif()

if(UOCR_WITH_INTEL AND BUILD_TESTING)
  add_executable(intel_vision_diagnostic tests/intel_vision_diagnostic.cpp)
  target_include_directories(intel_vision_diagnostic PRIVATE tests)
  target_compile_definitions(intel_vision_diagnostic PRIVATE UOCR_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/tests/goldens")
  target_link_libraries(intel_vision_diagnostic PRIVATE uocr_intel_model)
  find_path(UOCR_LEVEL_ZERO_INCLUDE_DIR level_zero/zes_api.h)
  find_library(UOCR_LEVEL_ZERO_LOADER ze_loader)
  if(UOCR_LEVEL_ZERO_INCLUDE_DIR AND UOCR_LEVEL_ZERO_LOADER)
    add_executable(intel_memory_check tools/intel_memory_check.cpp)
    target_include_directories(intel_memory_check PRIVATE ${UOCR_LEVEL_ZERO_INCLUDE_DIR})
    target_link_libraries(intel_memory_check PRIVATE uocr_flags ${UOCR_LEVEL_ZERO_LOADER})
  endif()
  foreach(test_name intel_numeric_guard_test intel_native_ops_test intel_native_vision_test intel_native_decoder_test intel_native_golden_test)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_include_directories(${test_name} PRIVATE tests)
    target_link_libraries(${test_name} PRIVATE uocr_intel_model)
    target_compile_definitions(${test_name} PRIVATE UOCR_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/tests/goldens")
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES SKIP_RETURN_CODE 77)
  endforeach()
endif()

if(UOCR_WITH_INTEL AND UOCR_WITH_SERVER AND BUILD_TESTING)
  add_executable(intel_native_rpc_test tests/intel_native_rpc_test.cpp)
  target_include_directories(intel_native_rpc_test PRIVATE tests)
  target_link_libraries(intel_native_rpc_test PRIVATE uocr_model_engine)
  target_compile_definitions(intel_native_rpc_test PRIVATE UOCR_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/tests/goldens")
  add_test(NAME intel_native_rpc_test COMMAND intel_native_rpc_test)
  add_test(NAME intel_native_rpc_assertions_test COMMAND intel_native_rpc_test --check-golden-assertions)
  set_tests_properties(intel_native_rpc_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)
endif()
