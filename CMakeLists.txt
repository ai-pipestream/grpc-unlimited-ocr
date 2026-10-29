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
  src/model/vision_encoder.cpp)
target_link_libraries(uocr_reference PUBLIC uocr_weights Threads::Threads)

if(BUILD_TESTING)
  foreach(test_name dtype_test safetensors_test architecture_test checkpoint_test ops_test vision_test
                    vision_checkpoint_test)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE uocr_reference)
    target_include_directories(${test_name} PRIVATE tests)
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES SKIP_RETURN_CODE 77
      ENVIRONMENT "UOCR_TEST_TMP=${CMAKE_CURRENT_BINARY_DIR}/test-tmp")
  endforeach()
endif()
