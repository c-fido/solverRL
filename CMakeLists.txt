cmake_minimum_required(VERSION 3.18)
project(solverrl_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_library(solverrl_vocab src/vocabulary.cpp)
target_include_directories(solverrl_vocab PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(solverrl_foil src/foil.cpp)
target_include_directories(solverrl_foil PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_vocabulary tests/cpp/test_vocabulary.cpp)
target_link_libraries(test_vocabulary PRIVATE solverrl_vocab GTest::gtest_main)

add_executable(test_foil tests/cpp/test_foil.cpp)
target_link_libraries(test_foil PRIVATE solverrl_foil GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_vocabulary)
gtest_discover_tests(test_foil)

# ponytail: pybind11 comes from the pinned venv, not FetchContent — one source of truth for the ABI
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 REQUIRED)

pybind11_add_module(solverrl_core src/core.cpp src/vocabulary.cpp src/foil.cpp)
target_include_directories(solverrl_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
