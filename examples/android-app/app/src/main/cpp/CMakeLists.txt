# Native part of the app: sd.cpp with the QNN sidecars (SD_QNN) and the JNI bridge, as libsdjni.so.
cmake_minimum_required(VERSION 3.22)
project(sdjni C CXX)

# The Android Gradle plugin compiles with -g; the debug info is stripped from the APK anyway, and it makes
# src/vocab/vocab.cpp (huge string tables) need several GB of RAM to compile.
string(REGEX REPLACE "(^| )-g( |$)" " " CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
string(REGEX REPLACE "(^| )-g( |$)" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

get_filename_component(SD_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../.. ABSOLUTE)

set(SD_QNN ON CACHE BOOL "" FORCE)
set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
# Every phone the QNN sidecars support (Snapdragon 8 Gen 2 and newer) has dot-product and fp16 instructions,
# which make the CPU fallback several times faster.
set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)
add_subdirectory(${SD_ROOT} sd)

add_library(sdjni SHARED sd_jni.cpp)
target_link_libraries(sdjni PRIVATE stable-diffusion log)
target_compile_features(sdjni PRIVATE cxx_std_17)
