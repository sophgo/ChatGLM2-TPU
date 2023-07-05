cmake_minimum_required(VERSION 2.8)
project(chatglm2)

set(LIBSOPHON /opt/sophon/libsophon-current)
include_directories(${LIBSOPHON}/include)
include_directories(${PROJECT_SOURCE_DIR}/include)
link_directories(${LIBSOPHON}/lib)
link_directories(${PROJECT_SOURCE_DIR}/lib)

add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
set(CMAKE_BUILD_TYPE "Debug")

add_executable(chatglm2 demo.cpp)
target_link_libraries(chatglm2 bmrt bmlib sentencepiece)
