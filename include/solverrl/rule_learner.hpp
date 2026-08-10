#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "solverrl/foil.hpp"
#include "solverrl/keydoor_ground.hpp"
#include "solverrl/prolog_emit.hpp"

namespace solverrl {

class RuleLearner {
 public:
  static RuleLearner keydoor(double min_score = 1e-12, int max_body_literals = 3);

  RuleLearner(int n_atoms, int n_actions, EmitConfig emit_cfg, double min_score = 1e-12,
              int max_body_literals = 3, SharedDirConfig shared_dir = {});

  void fit(const std::vector<Example>& examples);
  void load_decision_list(DecisionList list);
  double fidelity(const std::vector<Example>& examples) const;
  int predict(const Example& example) const;
  int predict_atoms(const std::vector<bool>& atoms) const;
  std::vector<int> predict_batch(const std::vector<Example>& examples) const;
  std::string to_prolog() const;

  bool is_fitted() const { return fitted_; }
  std::size_t n_clauses() const { return list_.clauses.size(); }
  const DecisionList& decision_list() const { return list_; }
  const SharedDirConfig& shared_dir() const { return shared_dir_; }

 private:
  OrderedCovering learner_;
  EmitConfig emit_cfg_;
  SharedDirConfig shared_dir_;
  DecisionList list_;
  bool fitted_ = false;
};

std::vector<Example> examples_from_arrays(const std::vector<bool>& atoms_flat, int n_states,
                                          int n_atoms, const int* actions,
                                          const double* weights);

}  // namespace solverrl
