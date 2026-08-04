#include "solverrl/rule_learner.hpp"

#include <stdexcept>

namespace solverrl {

RuleLearner RuleLearner::keydoor(double min_score, int max_body_literals) {
  return RuleLearner(kKeyDoorNumAtoms, kKeyDoorNumActions, keydoor_emit_config(), min_score,
                     max_body_literals);
}

RuleLearner::RuleLearner(int n_atoms, int n_actions, EmitConfig emit_cfg, double min_score,
                         int max_body_literals)
    : learner_(n_atoms, n_actions, min_score, max_body_literals),
      emit_cfg_(std::move(emit_cfg)) {}

void RuleLearner::fit(const std::vector<Example>& examples) {
  if (examples.empty()) {
    throw std::invalid_argument("RuleLearner.fit: empty examples");
  }
  list_ = learner_.fit(examples);
  fitted_ = true;
}

double RuleLearner::fidelity(const std::vector<Example>& examples) const {
  if (!fitted_) {
    throw std::runtime_error("RuleLearner.fidelity: call fit() first");
  }
  if (examples.empty()) {
    return 0.0;
  }
  double correct = 0.0;
  double total = 0.0;
  for (const auto& ex : examples) {
    total += ex.weight;
    if (predict(ex) == ex.action) {
      correct += ex.weight;
    }
  }
  return total > 0.0 ? correct / total : 0.0;
}

int RuleLearner::predict(const Example& example) const {
  if (!fitted_) {
    throw std::runtime_error("RuleLearner.predict: call fit() first");
  }
  return list_.predict(example);
}

int RuleLearner::predict_atoms(const std::vector<bool>& atoms) const {
  Example ex;
  ex.atoms = atoms;
  ex.action = 0;
  ex.weight = 1.0;
  return predict(ex);
}

std::vector<int> RuleLearner::predict_batch(const std::vector<Example>& examples) const {
  std::vector<int> out;
  out.reserve(examples.size());
  for (const auto& ex : examples) {
    out.push_back(predict(ex));
  }
  return out;
}

std::string RuleLearner::to_prolog() const {
  if (!fitted_) {
    throw std::runtime_error("RuleLearner.to_prolog: call fit() first");
  }
  return emit_prolog(list_, emit_cfg_);
}

std::vector<Example> examples_from_arrays(const std::vector<bool>& atoms_flat, int n_states,
                                          int n_atoms, const int* actions,
                                          const double* weights) {
  if (n_states <= 0 || n_atoms <= 0) {
    throw std::invalid_argument("examples_from_arrays: invalid shape");
  }
  const std::size_t row = static_cast<std::size_t>(n_atoms);
  const std::size_t need = static_cast<std::size_t>(n_states) * row;
  if (atoms_flat.size() < need) {
    throw std::invalid_argument("examples_from_arrays: atoms buffer too small");
  }

  std::vector<Example> examples;
  examples.reserve(static_cast<std::size_t>(n_states));
  for (int i = 0; i < n_states; ++i) {
    Example ex;
    ex.atoms.assign(atoms_flat.begin() + static_cast<std::size_t>(i) * row,
                    atoms_flat.begin() + static_cast<std::size_t>(i + 1) * row);
    ex.action = actions[i];
    ex.weight = weights ? weights[i] : 1.0;
    examples.push_back(std::move(ex));
  }
  return examples;
}

}  // namespace solverrl
