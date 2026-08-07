#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <vector>

#include "solverrl/exact_eval.hpp"
#include "solverrl/expand.hpp"
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

std::vector<int> policy_from_numpy(py::array policy) {
  if (policy.ndim() != 1) {
    throw std::invalid_argument("policy must be 1-D (n_states,)");
  }
  const int n = static_cast<int>(policy.shape(0));
  std::vector<int> out(static_cast<std::size_t>(n));
  const auto req = policy.request();
  if (req.format == py::format_descriptor<int64_t>::format()) {
    const auto* p = static_cast<const int64_t*>(req.ptr);
    for (int i = 0; i < n; ++i) {
      out[static_cast<std::size_t>(i)] = static_cast<int>(p[i]);
    }
  } else if (req.format == py::format_descriptor<int32_t>::format()) {
    const auto* p = static_cast<const int32_t*>(req.ptr);
    for (int i = 0; i < n; ++i) {
      out[static_cast<std::size_t>(i)] = p[i];
    }
  } else {
    throw std::invalid_argument("policy must be int64 or int32");
  }
  return out;
}

solverrl::ExactEvaluator exact_evaluator_from_numpy(py::array transition, py::array reward,
                                                    py::array initial, double gamma,
                                                    int done_index, int horizon) {
  if (transition.ndim() != 3) {
    throw std::invalid_argument("transition must be 3-D (n_actions, n_states, n_states)");
  }
  if (reward.ndim() != 2) {
    throw std::invalid_argument("reward must be 2-D (n_actions, n_states)");
  }
  if (initial.ndim() != 1) {
    throw std::invalid_argument("initial must be 1-D (n_states,)");
  }
  const int a = static_cast<int>(transition.shape(0));
  const int n = static_cast<int>(transition.shape(1));
  if (static_cast<int>(transition.shape(2)) != n) {
    throw std::invalid_argument("transition last dim must match n_states");
  }
  if (static_cast<int>(reward.shape(0)) != a || static_cast<int>(reward.shape(1)) != n) {
    throw std::invalid_argument("reward shape mismatch");
  }
  if (static_cast<int>(initial.shape(0)) != n) {
    throw std::invalid_argument("initial length must match n_states");
  }

  std::vector<double> P(static_cast<std::size_t>(a * n * n));
  std::vector<double> r(static_cast<std::size_t>(a * n));
  std::vector<double> mu0(static_cast<std::size_t>(n));

  const auto p_req = transition.request();
  const auto r_req = reward.request();
  const auto mu_req = initial.request();
  const auto* p_ptr = static_cast<const double*>(p_req.ptr);
  const auto* r_ptr = static_cast<const double*>(r_req.ptr);
  const auto* mu_ptr = static_cast<const double*>(mu_req.ptr);
  if (p_req.strides[2] != static_cast<py::ssize_t>(sizeof(double))) {
    throw std::invalid_argument("transition must be C-contiguous float64");
  }
  std::copy(p_ptr, p_ptr + P.size(), P.begin());
  std::copy(r_ptr, r_ptr + r.size(), r.begin());
  std::copy(mu_ptr, mu_ptr + mu0.size(), mu0.begin());

  return solverrl::ExactEvaluator(n, a, std::move(P), std::move(r), std::move(mu0), gamma,
                                  done_index, horizon);
}

py::array rollout_policy_array(const solverrl::DecisionList& list, py::array atoms) {
  if (atoms.ndim() != 2) {
    throw std::invalid_argument("atoms must be 2-D (n_states, n_atoms)");
  }
  const int n = static_cast<int>(atoms.shape(0));
  py::array_t<int64_t> dummy(n);
  auto dummy_req = dummy.request();
  auto* dummy_ptr = static_cast<int64_t*>(dummy_req.ptr);
  for (int i = 0; i < n; ++i) {
    dummy_ptr[i] = 0;
  }
  const auto examples = examples_from_numpy(atoms, dummy, py::none());
  const auto policy = solverrl::rollout_policy(list, examples);

  py::array_t<int64_t> out(n);
  auto out_req = out.request();
  auto* out_ptr = static_cast<int64_t*>(out_req.ptr);
  for (int i = 0; i < n; ++i) {
    out_ptr[i] = policy[static_cast<std::size_t>(i)];
  }
  return out;
}

const char* edit_kind_name(solverrl::EditKind kind) {
  switch (kind) {
    case solverrl::EditKind::Specialize:
      return "specialize";
    case solverrl::EditKind::Reorder:
      return "reorder";
    case solverrl::EditKind::Prune:
      return "prune";
  }
  return "unknown";
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
      .def("load_expansion_result",
           [](solverrl::RuleLearner& self, const solverrl::ExpansionResult& result) {
             self.load_decision_list(result.final_list);
           })
      .def_property_readonly("is_fitted", &solverrl::RuleLearner::is_fitted)
      .def_property_readonly("n_clauses", &solverrl::RuleLearner::n_clauses);

  py::class_<solverrl::AdvantageGapCert>(m, "AdvantageGapCert")
      .def_readonly("weighted_gap", &solverrl::AdvantageGapCert::weighted_gap)
      .def_readonly("return_gap", &solverrl::AdvantageGapCert::return_gap)
      .def_readonly("n_disagree", &solverrl::AdvantageGapCert::n_disagree)
      .def_readonly("max_gap", &solverrl::AdvantageGapCert::max_gap);

  py::class_<solverrl::ExactEvaluator>(m, "ExactEvaluator")
      .def(py::init(&exact_evaluator_from_numpy), py::arg("transition"), py::arg("reward"),
           py::arg("initial"), py::arg("gamma"), py::arg("done_index"), py::arg("horizon"))
      .def_property_readonly("n_states", &solverrl::ExactEvaluator::n_states)
      .def_property_readonly("n_actions", &solverrl::ExactEvaluator::n_actions)
      .def(
          "exact_return",
          [](const solverrl::ExactEvaluator& self, py::array policy) {
            return self.exact_return(policy_from_numpy(policy));
          },
          py::arg("policy"))
      .def(
          "success",
          [](const solverrl::ExactEvaluator& self, py::array policy) {
            return self.success_probability(policy_from_numpy(policy));
          },
          py::arg("policy"))
      .def(
          "advantage_gap_cert",
          [](const solverrl::ExactEvaluator& self, py::array student, py::array teacher) {
            return self.advantage_gap_cert(policy_from_numpy(student),
                                           policy_from_numpy(teacher));
          },
          py::arg("student"), py::arg("teacher"));

  py::enum_<solverrl::EditKind>(m, "EditKind")
      .value("Specialize", solverrl::EditKind::Specialize)
      .value("Reorder", solverrl::EditKind::Reorder)
      .value("Prune", solverrl::EditKind::Prune);

  py::class_<solverrl::EditProposal>(m, "EditProposal")
      .def_property_readonly("kind", [](const solverrl::EditProposal& p) { return p.kind; })
      .def_property_readonly("kind_name",
                             [](const solverrl::EditProposal& p) {
                               return edit_kind_name(p.kind);
                             })
      .def_readonly("clause_index", &solverrl::EditProposal::clause_index)
      .def_readonly("other_index", &solverrl::EditProposal::other_index)
      .def(
          "rollout",
          [](const solverrl::EditProposal& p, py::array atoms) {
            return rollout_policy_array(p.list, atoms);
          },
          py::arg("atoms"));

  py::class_<solverrl::ExpandEditor>(m, "ExpandEditor")
      .def(py::init<int, int>(), py::arg("n_atoms"),
           py::arg("max_body_literals") = 3)
      .def(
          "propose",
          [](const solverrl::ExpandEditor& self, const solverrl::RuleLearner& learner) {
            if (!learner.is_fitted()) {
              throw std::runtime_error("ExpandEditor.propose: RuleLearner not fitted");
            }
            return self.propose(learner.decision_list());
          },
          py::arg("learner"));

  py::class_<solverrl::ExpansionResult>(m, "ExpansionResult")
      .def_readonly("return_curve", &solverrl::ExpansionResult::return_curve)
      .def_readonly("success_curve", &solverrl::ExpansionResult::success_curve)
      .def_readonly("iterations", &solverrl::ExpansionResult::iterations)
      .def_readonly("final_return", &solverrl::ExpansionResult::final_return)
      .def_readonly("final_success", &solverrl::ExpansionResult::final_success)
      .def_readonly("accepted_any_edit", &solverrl::ExpansionResult::accepted_any_edit);

  py::class_<solverrl::ExpansionLoop>(m, "ExpansionLoop")
      .def(
          py::init([](const solverrl::ExactEvaluator& evaluator, py::array atoms, int max_lit,
                      double tau, int max_iterations) {
            if (atoms.ndim() != 2) {
              throw std::invalid_argument("atoms must be 2-D (n_states, n_atoms)");
            }
            const int n = static_cast<int>(atoms.shape(0));
            py::array_t<int64_t> dummy(n);
            auto dummy_req = dummy.request();
            auto* dummy_ptr = static_cast<int64_t*>(dummy_req.ptr);
            for (int i = 0; i < n; ++i) {
              dummy_ptr[i] = 0;
            }
            auto examples = examples_from_numpy(atoms, dummy, py::none());
            solverrl::ExpandEditor editor(solverrl::kKeyDoorNumAtoms, max_lit);
            return solverrl::ExpansionLoop(evaluator, std::move(examples), std::move(editor),
                                           tau, max_iterations);
          }),
          py::arg("evaluator"), py::arg("atoms"), py::arg("max_body_literals") = 3,
          py::arg("tau") = 1e-9, py::arg("max_iterations") = 200)
      .def(
          "run",
          [](const solverrl::ExpansionLoop& self, const solverrl::RuleLearner& learner) {
            if (!learner.is_fitted()) {
              throw std::runtime_error("ExpansionLoop.run: RuleLearner not fitted");
            }
            return self.run(learner.decision_list());
          },
          py::arg("learner"));
}
