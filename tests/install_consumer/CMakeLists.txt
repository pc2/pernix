cmake_minimum_required(VERSION 3.24)
project(pernix_install_consumer LANGUAGES C CXX)

find_package(pernix CONFIG REQUIRED)

add_executable(pernix_consumer_cpp main.cpp)
target_link_libraries(pernix_consumer_cpp PRIVATE pernix::pernix)

add_executable(pernix_consumer_c main.c)
target_link_libraries(pernix_consumer_c PRIVATE pernix::pernix)

enable_testing()
add_test(NAME pernix_consumer_cpp COMMAND pernix_consumer_cpp)
add_test(NAME pernix_consumer_c COMMAND pernix_consumer_c)
