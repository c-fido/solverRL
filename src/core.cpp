#include <pybind11/pybind11.h>

namespace py = pybind11;

// ponytail: skeleton only — RuleLearner lands in Week 2
PYBIND11_MODULE(solverrl_core, m) {
  m.doc() = "SolverRL C++ core";
  m.def("ping", []() { return "pong"; });
}
