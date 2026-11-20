# Standalone benchmark for the ggml CPU backend and the QNN sidecars (docs/qnn.md) on Android.
# Builds sd-bench (examples/ios-bench/bench.cpp), sd-cli and test-qnn as command-line tools for arm64-v8a,
# which run_on_device.sh pushes to /data/local/tmp and runs over adb.
cmake_minimum_required(VERSION 3.21)
project(sdbench_android C CXX)

if(NOT ANDROID)
    message(FATAL_ERROR "configure with the Android NDK toolchain: "
                        "-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a")
endif()

set(SD_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

set(SD_QNN ON CACHE BOOL "" FORCE)
set(SD_BUILD_EXAMPLES ON CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(${SD_ROOT} sd)

add_executable(sd-bench ${SD_ROOT}/examples/ios-bench/bench.cpp)
target_include_directories(sd-bench PRIVATE ${SD_ROOT}/thirdparty)
target_link_libraries(sd-bench PRIVATE stable-diffusion)
target_compile_features(sd-bench PRIVATE cxx_std_17)
set_target_properties(sd-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
