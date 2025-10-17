cmake_minimum_required(VERSION 3.14)
project(bigwig_cpp_reader LANGUAGES CXX)

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# Find include directory and library of libBigWig
# libBigWig is always installed in /usr/local/include and /usr/local/lib.
# If libBigWig would installed in other place, adjust below.
find_library(BIGWIG_LIBRARY bigWig HINTS /usr/local/lib)
find_path(BIGWIG_INCLUDE_DIR bigWig.h HINTS /usr/local/include)

if(NOT BIGWIG_LIBRARY)
    message(FATAL_ERROR "libBigWig library not found. Please install it or specify its path.")
endif()
if(NOT BIGWIG_INCLUDE_DIR)
    message(FATAL_ERROR "libBigWig include directory not found. Please install it or specify its path.")
endif()

include_directories(${BIGWIG_INCLUDE_DIR})

# C++17 を使用（pybind11 の要件）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Python モジュールを作成
pybind11_add_module(bigwig_cpp_reader bigwig_reader.cpp)

# libBigWig ライブラリをリンク
target_link_libraries(bigwig_cpp_reader PRIVATE ${BIGWIG_LIBRARY})

# ビルドタイプをリリースに設定（最適化のため）
set(CMAKE_BUILD_TYPE Release)

# Linux での共有ライブラリの命名規則
if (UNIX AND NOT APPLE)
    set_target_properties(bigwig_cpp_reader PROPERTIES SUFFIX ".so")
endif()