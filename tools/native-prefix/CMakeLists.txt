cmake_minimum_required(VERSION 3.20)
project(jev_native_prefix LANGUAGES C CXX)

set(LLAMA_CPP_DIR "" CACHE PATH "Pristine checkout of the pinned llama.cpp source")
set(LLAMA_CPP_REVISION "5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd")
find_package(Git REQUIRED)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${LLAMA_CPP_DIR}" rev-parse HEAD
  RESULT_VARIABLE revision_result OUTPUT_VARIABLE actual_revision OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT revision_result EQUAL 0 OR NOT actual_revision STREQUAL LLAMA_CPP_REVISION)
  message(FATAL_ERROR "LLAMA_CPP_DIR must be pinned to ${LLAMA_CPP_REVISION}")
endif()
execute_process(COMMAND "${GIT_EXECUTABLE}" -c core.fsmonitor=false -C "${LLAMA_CPP_DIR}" diff --quiet HEAD --
  RESULT_VARIABLE dirty_result)
if(NOT dirty_result EQUAL 0)
  message(FATAL_ERROR "Pinned llama.cpp tracked source must be pristine")
endif()

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
add_subdirectory("${LLAMA_CPP_DIR}" "${CMAKE_BINARY_DIR}/llama.cpp")

add_executable(s1-gguf-prefix s1_prefix.cpp)
target_compile_features(s1-gguf-prefix PRIVATE cxx_std_17)
target_compile_definitions(s1-gguf-prefix PRIVATE S1_LLAMA_CPP_REVISION="${LLAMA_CPP_REVISION}")
target_include_directories(s1-gguf-prefix PRIVATE "${LLAMA_CPP_DIR}/vendor" "${LLAMA_CPP_DIR}/tools/mtmd")
target_link_libraries(s1-gguf-prefix PRIVATE llama mtmd ggml ggml-base)

enable_testing()
add_executable(test-prefix-plan test_prefix_plan.cpp)
target_compile_features(test-prefix-plan PRIVATE cxx_std_17)
add_test(NAME prefix-plan COMMAND test-prefix-plan)

# This executable includes the real scorer but supplies fake C API definitions.
# It deliberately does not link llama, mtmd or any other inference library.
add_executable(test-native-runtime test_native_runtime.cpp)
target_compile_features(test-native-runtime PRIVATE cxx_std_17)
target_compile_definitions(test-native-runtime PRIVATE S1_LLAMA_CPP_REVISION="${LLAMA_CPP_REVISION}")
target_include_directories(test-native-runtime PRIVATE
  "${LLAMA_CPP_DIR}/include" "${LLAMA_CPP_DIR}/ggml/include"
  "${LLAMA_CPP_DIR}/tools/mtmd" "${LLAMA_CPP_DIR}/vendor")
add_test(NAME native-runtime-offline COMMAND test-native-runtime)
