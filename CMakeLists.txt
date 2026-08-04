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

add_library(solverrl_prolog src/prolog_emit.cpp)
target_include_directories(solverrl_prolog PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(solverrl_prolog PUBLIC solverrl_foil)

add_library(solverrl_keydoor_ground src/keydoor_ground.cpp)
target_include_directories(solverrl_keydoor_ground PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(solverrl_keydoor_ground PUBLIC solverrl_vocab)

add_library(solverrl_exact_eval src/exact_eval.cpp)
target_include_directories(solverrl_exact_eval PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(solverrl_rule_learner src/rule_learner.cpp)
target_include_directories(solverrl_rule_learner PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(solverrl_rule_learner PUBLIC solverrl_foil solverrl_prolog solverrl_keydoor_ground)

add_executable(test_exact_eval tests/cpp/test_exact_eval.cpp)
target_link_libraries(test_exact_eval PRIVATE solverrl_exact_eval GTest::gtest_main)

add_executable(test_keydoor_ground tests/cpp/test_keydoor_ground.cpp)
target_link_libraries(test_keydoor_ground PRIVATE solverrl_keydoor_ground GTest::gtest_main)

add_executable(test_rule_learner tests/cpp/test_rule_learner.cpp)
target_link_libraries(test_rule_learner PRIVATE solverrl_rule_learner GTest::gtest_main)

add_executable(test_vocabulary tests/cpp/test_vocabulary.cpp)
target_link_libraries(test_vocabulary PRIVATE solverrl_vocab GTest::gtest_main)

add_executable(test_foil tests/cpp/test_foil.cpp)
target_link_libraries(test_foil PRIVATE solverrl_foil GTest::gtest_main)

add_executable(test_prolog_emit tests/cpp/test_prolog_emit.cpp)
target_link_libraries(test_prolog_emit PRIVATE solverrl_prolog GTest::gtest_main)

add_executable(test_synthetic_vocab tests/cpp/test_synthetic_vocab.cpp)
target_link_libraries(test_synthetic_vocab PRIVATE solverrl_vocab solverrl_prolog GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_exact_eval)
gtest_discover_tests(test_keydoor_ground)
gtest_discover_tests(test_rule_learner)
gtest_discover_tests(test_vocabulary)
gtest_discover_tests(test_foil)
gtest_discover_tests(test_prolog_emit)
gtest_discover_tests(test_synthetic_vocab)

# Resolve pybind11 from the active Python environment (matches pip/venv ABI).
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 REQUIRED)

pybind11_add_module(solverrl_core src/core.cpp)
target_include_directories(solverrl_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(solverrl_core PRIVATE solverrl_rule_learner solverrl_vocab solverrl_exact_eval)
