#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <vector>

#include "solverrl/keydoor_ground.hpp"
#include "solverrl/rule_learner.hpp"
#include "solverrl/vocabulary.hpp"

namespace py = pybind11;

namespace {

std::vector<solverrl::Example> examples_from_numpy(py::array atoms, py::array actions,
                                                   py::object weights) {
  if (atoms.ndim() != 2) {
    throw std::invalid_argument("atoms must be 2-D (n_states, n_atoms)");
  }
  if (actions.ndim() != 1) {
    throw std::invalid_argument("actions must be 1-D (n_states,)");
  }
  const int n_states = static_cast<int>(atoms.shape(0));
  const int n_atoms = static_cast<int>(atoms.shape(1));
  if (static_cast<int>(actions.shape(0)) != n_states) {
    throw std::invalid_argument("actions length must match atoms rows");
  }

  auto atoms_req = atoms.request();
  auto actions_req = actions.request();
  std::vector<bool> flat(static_cast<std::size_t>(n_states * n_atoms));
  const auto* atoms_ptr = static_cast<const uint8_t*>(atoms_req.ptr);
  for (std::size_t i = 0; i < flat.size(); ++i) {
    flat[i] = atoms_ptr[i] != 0;
  }

  std::vector<int> acts(static_cast<std::size_t>(n_states));
  if (actions_req.format == py::format_descriptor<int64_t>::format()) {
    const auto* p = static_cast<const int64_t*>(actions_req.ptr);
    for (int i = 0; i < n_states; ++i) {
      acts[static_cast<std::size_t>(i)] = static_cast<int>(p[i]);
    }
  } else if (actions_req.format == py::format_descriptor<int32_t>::format()) {
    const auto* p = static_cast<const int32_t*>(actions_req.ptr);
    for (int i = 0; i < n_states; ++i) {
      acts[static_cast<std::size_t>(i)] = p[i];
    }
  } else {
    throw std::invalid_argument("actions must be int64 or int32");
  }

  std::vector<double> w;
  const double* w_ptr = nullptr;
  if (!weights.is_none()) {
    py::array w_arr = py::cast<py::array>(weights);
    if (w_arr.ndim() != 1 || static_cast<int>(w_arr.shape(0)) != n_states) {
      throw std::invalid_argument("weights must be 1-D with length n_states");
    }
    w.resize(static_cast<std::size_t>(n_states));
    const auto w_req = w_arr.request();
    const auto* wp = static_cast<const double*>(w_req.ptr);
    for (int i = 0; i < n_states; ++i) {
      w[static_cast<std::size_t>(i)] = wp[i];
    }
    w_ptr = w.data();
  }

  return solverrl::examples_from_arrays(flat, n_states, n_atoms, acts.data(), w_ptr);
}

py::array ground_keydoor_states(py::array states) {
  if (states.ndim() != 2 || states.shape(1) != solverrl::kKeyDoorStateFields) {
    throw std::invalid_argument(
        "states must be (n, 9) int32: door_row, agent_r/c, key_r/c, goal_r/c, "
        "door_open, carrying");
  }
  const int n = static_cast<int>(states.shape(0));
  py::array_t<uint8_t> atoms({n, solverrl::kKeyDoorNumAtoms});
  auto atoms_req = atoms.request();
  auto* out = static_cast<uint8_t*>(atoms_req.ptr);

  const auto states_req = states.request();
  const auto* in = static_cast<const int32_t*>(states_req.ptr);
  for (int i = 0; i < n; ++i) {
    const solverrl::KeyDoorState s =
        solverrl::decode_keydoor_state(in + static_cast<std::ptrdiff_t>(i) * 9);
    const auto row = solverrl::ground_atoms(s);
    for (int j = 0; j < solverrl::kKeyDoorNumAtoms; ++j) {
      out[static_cast<std::size_t>(i) * solverrl::kKeyDoorNumAtoms + j] =
          row[static_cast<std::size_t>(j)] ? 1 : 0;
    }
  }
  return atoms;
}

}  // namespace

PYBIND11_MODULE(solverrl_core, m) {
  m.doc() = "SolverRL C++ core";
  m.def("ping", []() { return "pong"; });

  m.attr("KEYDOOR_NUM_ATOMS") = solverrl::kKeyDoorNumAtoms;
  m.attr("KEYDOOR_NUM_ACTIONS") = solverrl::kKeyDoorNumActions;
  m.attr("KEYDOOR_STATE_FIELDS") = solverrl::kKeyDoorStateFields;
  m.def("ground_keydoor_states", &ground_keydoor_states,
        py::arg("states"),
        "Ground KeyDoor states to atom matrix (n, 17) uint8.");

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

  py::class_<solverrl::RuleLearner>(m, "RuleLearner")
      .def_static("keydoor", &solverrl::RuleLearner::keydoor, py::arg("min_score") = 1e-12,
                  py::arg("max_body_literals") = 3)
      .def(
          "fit",
          [](solverrl::RuleLearner& self, py::array atoms, py::array actions,
             py::object weights) {
            self.fit(examples_from_numpy(atoms, actions, weights));
          },
          py::arg("atoms"), py::arg("actions"), py::arg("weights") = py::none())
      .def(
          "fidelity",
          [](const solverrl::RuleLearner& self, py::array atoms, py::array actions,
             py::object weights) {
            return self.fidelity(examples_from_numpy(atoms, actions, weights));
          },
          py::arg("atoms"), py::arg("actions"), py::arg("weights") = py::none())
      .def(
          "predict",
          [](const solverrl::RuleLearner& self, py::array atoms) {
            if (atoms.ndim() != 2) {
              throw std::invalid_argument("atoms must be 2-D (n_states, n_atoms)");
            }
            const int n = static_cast<int>(atoms.shape(0));
            const int n_atoms = static_cast<int>(atoms.shape(1));
            py::array_t<int64_t> actions(n);
            auto acts_req = actions.request();
            auto* out = static_cast<int64_t*>(acts_req.ptr);

            const auto atoms_req = atoms.request();
            const auto* atoms_ptr = static_cast<const uint8_t*>(atoms_req.ptr);
            for (int i = 0; i < n; ++i) {
              std::vector<bool> row(static_cast<std::size_t>(n_atoms));
              for (int j = 0; j < n_atoms; ++j) {
                row[static_cast<std::size_t>(j)] =
                    atoms_ptr[static_cast<std::size_t>(i * n_atoms + j)] != 0;
              }
              out[i] = self.predict_atoms(row);
            }
            return actions;
          },
          py::arg("atoms"))
      .def("to_prolog", &solverrl::RuleLearner::to_prolog)
      .def_property_readonly("is_fitted", &solverrl::RuleLearner::is_fitted)
      .def_property_readonly("n_clauses", &solverrl::RuleLearner::n_clauses);
}
