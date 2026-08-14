cmake_minimum_required(VERSION 4.2.3)
project(Boilerplate)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O0 -g -Werror -Wall -Wno-unused)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -Werror -Wall -Wno-unused)
endif()


find_package(SFML 3.1 COMPONENTS Graphics Audio REQUIRED)

add_executable(Boilerplate src/main.cpp)

target_link_libraries(Boilerplate PRIVATE SFML::Graphics SFML::Audio)