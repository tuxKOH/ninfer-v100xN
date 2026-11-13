cmake_minimum_required(VERSION 3.28)

# The implementation supports Volta sm_70, Ampere sm_86 and early Ada sm_89. Set and validate the
# architecture before CUDA compiler detection so CMake never injects a generic
# host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^(70|86|89)$")
  message(FATAL_ERROR
    "NInfer supports CMAKE_CUDA_ARCHITECTURES=70, 86 or 89; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(ninfer LANGUAGES C CXX CUDA)

if(CMAKE_CUDA_ARCHITECTURES STREQUAL "70")
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
    message(FATAL_ERROR "Volta sm_70 requires CUDA 12.8; CUDA 13 removed Volta compilation")
  endif()
  add_compile_definitions(NINFER_VOLTA_BUILD=1)
  include(FetchContent)
  set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")
  set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "Disable CUTLASS tests")
  set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "Disable CUTLASS examples")
  FetchContent_Declare(cutlass
    GIT_REPOSITORY https://github.com/nvidia/cutlass.git
    GIT_TAG v4.4.2
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(cutlass)
endif()

# NINFER_SM8X_COMPAT gates every Ampere/early-Ada compatibility path. ninfer_ops also sets it
# PUBLIC for its own consumers, but that link is PRIVATE through ninfer_engine, so apps and tests
# would not otherwise see it. Defining it globally keeps a single consistent answer for every
# target, including apps/ninfer-perplexity, whose upstream default KV profile is FP8.
if(CMAKE_CUDA_ARCHITECTURES MATCHES "^(86|89)$")
  add_compile_definitions(NINFER_SM8X_COMPAT=1)
endif()

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI and server" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)
# CUDA device selection is a runtime concern. When set, CTest injects this value as
# CUDA_VISIBLE_DEVICES so tests using logical devices 0/1 can run on a selected pair on
# multi-GPU hosts without changing the product's device numbering or compile target.
set(NINFER_TEST_CUDA_VISIBLE_DEVICES "" CACHE STRING
  "CUDA_VISIBLE_DEVICES value injected into CTest (for example 4,6)")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(MSVC)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
  # /Zc:__cplusplus makes MSVC report the standard it is actually compiling. Without it the macro
  # stays at 199711L even under /std:c++20, and CUDA's <vector_types.h> then hides the constexpr
  # dim3 constructors behind its own __cplusplus >= 201103L test, which fails to compile every
  # translation unit that builds a dim3 in a constant expression.
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:__cplusplus>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>)
  # CUDA's static runtime requests LIBCMT even when the application and vcpkg dependencies use
  # the project-wide /MD runtime. Keep one CRT in the final process.
  add_link_options(/NODEFAULTLIB:LIBCMT)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS cuda_link=1)
  set(CMAKE_JOB_POOL_LINK cuda_link)
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
  message(FATAL_ERROR
    "NInfer requires CUDA 12.8 or newer; found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()

set(NINFER_BUILD_MEDIA_ACQUIRE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_MEDIA_ACQUIRE ON)
endif()

set(NINFER_BUILD_PROMPT_INPUT OFF)
if(NINFER_BUILD_APPS)
  set(NINFER_BUILD_PROMPT_INPUT ON)
endif()

set(NINFER_BUILD_SERVE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_SERVE ON)
endif()

find_package(CUDAToolkit REQUIRED)
if(WIN32)
  set(NINFER_CUDART_TARGET CUDA::cudart_static)
else()
  set(NINFER_CUDART_TARGET CUDA::cudart)
endif()
if(WIN32 OR DEFINED VCPKG_TARGET_TRIPLET)
  find_package(FFMPEG REQUIRED)
  add_library(ninfer_ffmpeg_dependencies INTERFACE)
  target_include_directories(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_INCLUDE_DIRS})
  target_link_directories(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_LIBRARY_DIRS})
  target_link_libraries(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_LIBRARIES})
  set(NINFER_FFMPEG_TARGET ninfer_ffmpeg_dependencies)
  if(NINFER_BUILD_MEDIA_ACQUIRE)
    find_package(CURL 7.85 REQUIRED)
    set(NINFER_CURL_TARGET CURL::libcurl)
  endif()
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
  set(NINFER_FFMPEG_TARGET PkgConfig::FFMPEG)
  if(NINFER_BUILD_MEDIA_ACQUIRE)
    pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
    set(NINFER_CURL_TARGET PkgConfig::LIBCURL)
  endif()
endif()
find_package(Threads REQUIRED)
set(BLA_SIZEOF_INTEGER 4)
set(BLA_PREFER_PKGCONFIG ON)
find_package(BLAS REQUIRED)

# Grammar-constrained decoding for structured output. Only xgrammar's C++ core is compiled, as a
# private static library; its own build script (Python bindings, -flto, -Werror) is not used.
include(FetchContent)
FetchContent_Declare(xgrammar
  GIT_REPOSITORY https://github.com/mlc-ai/xgrammar.git
  GIT_TAG v0.2.8
  GIT_SHALLOW TRUE
  GIT_SUBMODULES 3rdparty/dlpack 3rdparty/picojson
  SOURCE_SUBDIR include)
FetchContent_MakeAvailable(xgrammar)
file(GLOB_RECURSE NINFER_XGRAMMAR_SOURCES CONFIGURE_DEPENDS "${xgrammar_SOURCE_DIR}/cpp/*.cc")
list(FILTER NINFER_XGRAMMAR_SOURCES EXCLUDE REGEX "/cpp/tvm_ffi/.*\\.cc$")
add_library(xgrammar STATIC ${NINFER_XGRAMMAR_SOURCES})
target_include_directories(xgrammar PUBLIC "${xgrammar_SOURCE_DIR}/include")
target_include_directories(xgrammar SYSTEM PUBLIC
  "${xgrammar_SOURCE_DIR}/3rdparty/picojson"
  "${xgrammar_SOURCE_DIR}/3rdparty/dlpack/include")
target_compile_definitions(xgrammar PUBLIC XGRAMMAR_ENABLE_CPPTRACE=0)
set_target_properties(xgrammar PROPERTIES CXX_STANDARD 17 POSITION_INDEPENDENT_CODE ON)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
