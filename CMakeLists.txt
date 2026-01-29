cmake_minimum_required(VERSION 3.30)
project(RNNG LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include)

find_package(MPI REQUIRED)

add_executable(systolic drivers/systolic.cpp)
target_link_libraries(systolic PRIVATE MPI::MPI_CXX)
target_compile_definitions(systolic PRIVATE ${DEFS})

add_executable(landmark drivers/landmark.cpp)
target_link_libraries(landmark PRIVATE MPI::MPI_CXX)
target_compile_definitions(landmark PRIVATE ${DEFS})

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -fsanitize=undefined -fsanitize=address")
  target_link_options(systolic BEFORE PUBLIC -fsanitize=undefined PUBLIC -fsanitize=address)
endif()

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -fsanitize=undefined -fsanitize=address")
  target_link_options(landmark BEFORE PUBLIC -fsanitize=undefined PUBLIC -fsanitize=address)
endif()
