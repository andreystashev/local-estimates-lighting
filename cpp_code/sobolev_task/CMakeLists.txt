cmake_minimum_required(VERSION 3.10)

project(LocalEstimatesLighting)

set(CMAKE_CXX_STANDARD 14)

# Найти библиотеку Boost
find_package(Boost REQUIRED COMPONENTS system)

include_directories(${Boost_INCLUDE_DIRS})

add_executable(LocalEstimatesLighting main.cpp)

target_link_libraries(LocalEstimatesLighting ${Boost_LIBRARIES})