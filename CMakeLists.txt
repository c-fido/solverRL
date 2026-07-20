cmake_minimum_required(VERSION 3.18)
project(solverrl_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ponytail: pybind11 comes from the pinned venv, not FetchContent — one source of truth for the ABI
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 REQUIRED)

pybind11_add_module(solverrl_core src/core.cpp)
