# The gRPC service, included from the top-level CMakeLists when
# UOCR_WITH_SERVER is on.

set(gRPC_INSTALL OFF CACHE BOOL "" FORCE)
set(gRPC_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(gRPC_BUILD_CSHARP_EXT OFF CACHE BOOL "" FORCE)
foreach(plugin CSHARP NODE OBJECTIVE_C PHP PYTHON RUBY)
  set(gRPC_BUILD_GRPC_${plugin}_PLUGIN OFF CACHE BOOL "" FORCE)
endforeach()
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
# The fetched re2 and zlib register their own tests with CTest; keep them
# out of ours.
set(RE2_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(grpc
  GIT_REPOSITORY https://github.com/grpc/grpc.git
  GIT_TAG v1.83.0
  GIT_SHALLOW TRUE
  GIT_SUBMODULES_RECURSE TRUE
  SYSTEM)
# An ASan build instruments gRPC, protobuf and abseil too: their inline
# code is compiled into our objects with ASan's string and container
# annotations, and an uninstrumented library writing the same objects
# reports false use-after-poison and container-overflow errors. Only
# address: abseil does not compile under GCC's -fsanitize=undefined.
if(UOCR_SANITIZE MATCHES "address")
  set(UOCR_SAVED_C_FLAGS "${CMAKE_C_FLAGS}")
  set(UOCR_SAVED_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  set(UOCR_SAVED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
  string(APPEND CMAKE_C_FLAGS " -fsanitize=address -fno-omit-frame-pointer")
  string(APPEND CMAKE_CXX_FLAGS " -fsanitize=address -fno-omit-frame-pointer")
  string(APPEND CMAKE_EXE_LINKER_FLAGS " -fsanitize=address")
endif()
FetchContent_MakeAvailable(grpc)
if(UOCR_SANITIZE MATCHES "address")
  set(CMAKE_C_FLAGS "${UOCR_SAVED_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${UOCR_SAVED_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${UOCR_SAVED_LINKER_FLAGS}")
endif()

# Generated code. protoc and grpc_cpp_plugin are the fetched targets.
set(UOCR_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(UOCR_PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/proto")
set(UOCR_PROTOS
  "${UOCR_PROTO_DIR}/ai/pipestream/document/v1/document.proto"
  # Vendored byte-identical from gRParse beside document.proto, which
  # imports them for Document.analyses (this service never fills them).
  "${UOCR_PROTO_DIR}/org/apache/opennlp/grpc/v1/opennlp_document.proto"
  "${UOCR_PROTO_DIR}/org/apache/opennlp/grpc/v1/opennlp_annotations.proto"
  "${UOCR_PROTO_DIR}/ai/pipestream/unlimitedocr/v1/unlimited_ocr.proto")
set(UOCR_PROTO_SOURCES
  "${UOCR_GENERATED_DIR}/ai/pipestream/document/v1/document.pb.cc"
  "${UOCR_GENERATED_DIR}/org/apache/opennlp/grpc/v1/opennlp_document.pb.cc"
  "${UOCR_GENERATED_DIR}/org/apache/opennlp/grpc/v1/opennlp_annotations.pb.cc"
  "${UOCR_GENERATED_DIR}/ai/pipestream/unlimitedocr/v1/unlimited_ocr.pb.cc"
  "${UOCR_GENERATED_DIR}/ai/pipestream/unlimitedocr/v1/unlimited_ocr.grpc.pb.cc")
set(UOCR_PROTO_HEADERS ${UOCR_PROTO_SOURCES})
list(TRANSFORM UOCR_PROTO_HEADERS REPLACE "\\.cc$" ".h")

add_custom_command(
  OUTPUT ${UOCR_PROTO_SOURCES} ${UOCR_PROTO_HEADERS}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${UOCR_GENERATED_DIR}"
  COMMAND $<TARGET_FILE:protoc>
  ARGS --proto_path=${UOCR_PROTO_DIR}
       --proto_path=${protobuf_SOURCE_DIR}/src
       --cpp_out=${UOCR_GENERATED_DIR}
       --grpc_out=${UOCR_GENERATED_DIR}
       --plugin=protoc-gen-grpc=$<TARGET_FILE:grpc_cpp_plugin>
       ${UOCR_PROTOS}
  DEPENDS ${UOCR_PROTOS} protoc grpc_cpp_plugin
  VERBATIM)

add_library(uocr_proto STATIC ${UOCR_PROTO_SOURCES})
# Generated code is third-party style: its headers stay out of the
# first-party warning set.
target_include_directories(uocr_proto SYSTEM PUBLIC "${UOCR_GENERATED_DIR}")
target_link_libraries(uocr_proto PUBLIC grpc++)

# The output side: block grammar, incremental parser, HTML tables,
# continuation links and the Document fold. No engine, no gRPC runtime.
add_library(uocr_output STATIC
  src/output/page_text.cpp
  src/output/block_grammar.cpp
  src/output/block_parser.cpp
  src/output/html_table.cpp
  src/output/continuations.cpp
  src/output/document_builder.cpp
  src/output/document_fold.cpp)
target_include_directories(uocr_output PUBLIC src)
target_link_libraries(uocr_output PUBLIC uocr_proto PRIVATE uocr_flags)

# The service: configuration, the engine interface, one document's session
# and the gRPC handlers.
add_library(uocr_service STATIC
  src/service/config.cpp
  src/service/document_session.cpp
  src/service/page_intake.cpp
  src/service/unlimited_ocr_service.cpp)
target_include_directories(uocr_service PUBLIC src)
target_compile_definitions(uocr_service PRIVATE UOCR_VERSION="${PROJECT_VERSION}")
target_link_libraries(uocr_service PUBLIC uocr_output Threads::Threads PRIVATE uocr_flags)

# The real engine behind the interface.
add_library(uocr_model_engine STATIC src/service/model_engine.cpp)
target_link_libraries(uocr_model_engine PUBLIC uocr_service uocr_reference PRIVATE uocr_flags)
target_include_directories(uocr_model_engine SYSTEM PRIVATE ${stb_SOURCE_DIR})
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/models/MANIFEST" UOCR_MANIFEST_SHA256)
target_compile_definitions(uocr_model_engine PRIVATE UOCR_MANIFEST_SHA256="${UOCR_MANIFEST_SHA256}")
if(UOCR_WITH_CUDA)
  target_link_libraries(uocr_model_engine PRIVATE uocr_cuda_model)
  target_compile_definitions(uocr_model_engine PRIVATE UOCR_WITH_CUDA=1)
endif()

add_executable(grpc-unlimited-ocr-server src/server_main.cpp)
target_link_libraries(grpc-unlimited-ocr-server PRIVATE uocr_model_engine grpc++_reflection uocr_flags)

if(BUILD_TESTING)
  foreach(test_name block_parser_test block_fuzz_test html_table_test document_fold_test lossless_test
                    service_test config_test)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE uocr_service uocr_flags)
    target_include_directories(${test_name} PRIVATE tests)
    target_compile_definitions(${test_name} PRIVATE UOCR_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/tests/goldens")
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES SKIP_RETURN_CODE 77)
  endforeach()
  # The golden page through the real server; skips (77) without the weights.
  add_executable(server_golden_test tests/server_golden_test.cpp)
  target_link_libraries(server_golden_test PRIVATE uocr_model_engine uocr_flags)
  target_include_directories(server_golden_test PRIVATE tests)
  target_compile_definitions(server_golden_test PRIVATE UOCR_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/tests/goldens")
  if(UOCR_WITH_CUDA)
    target_compile_definitions(server_golden_test PRIVATE UOCR_WITH_CUDA=1)
  endif()
  add_test(NAME server_golden_test COMMAND server_golden_test)
  set_tests_properties(server_golden_test PROPERTIES SKIP_RETURN_CODE 77)
endif()
