#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "solverrl/vocabulary.hpp"

namespace py = pybind11;

PYBIND11_MODULE(solverrl_core, m) {
  m.doc() = "SolverRL C++ core";
  m.def("ping", []() { return "pong"; });

  py::enum_<solverrl::ArgSort>(m, "ArgSort")
      .value("State", solverrl::ArgSort::State)
      .value("Object", solverrl::ArgSort::Object)
      .value("Direction", solverrl::ArgSort::Direction);

  py::class_<solverrl::PredicateSpec>(m, "PredicateSpec")
      .def_readonly("name", &solverrl::PredicateSpec::name)
      .def_readonly("arg_sorts", &solverrl::PredicateSpec::arg_sorts)
      .def_readonly("allows_negation", &solverrl::PredicateSpec::allows_negation)
      .def_readonly("binds_direction_var", &solverrl::PredicateSpec::binds_direction_var)
      .def_property_readonly("arity", &solverrl::PredicateSpec::arity);

  py::class_<solverrl::Vocabulary>(m, "Vocabulary")
      .def_static("keydoor", &solverrl::Vocabulary::KeyDoor)
      .def_readonly("name", &solverrl::Vocabulary::name)
      .def_readonly("objects", &solverrl::Vocabulary::objects)
      .def_readonly("directions", &solverrl::Vocabulary::directions)
      .def_readonly("predicates", &solverrl::Vocabulary::predicates)
      .def("find",
           [](const solverrl::Vocabulary& v, const std::string& name) -> py::object {
             const auto* p = v.find(name);
             if (!p) {
               return py::none();
             }
             return py::cast(*p);
           })
      .def("__len__", &solverrl::Vocabulary::size);
}
