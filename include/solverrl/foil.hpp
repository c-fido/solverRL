#pragma once

#include <cstddef>
#include <vector>

namespace solverrl {

struct Example {
  std::vector<bool> atoms;  // ground atom truth vector for this state
  int action = 0;
  double weight = 1.0;
};

struct Literal {
  int atom_id = 0;
  bool negated = false;

  bool holds(const Example& ex) const {
    if (atom_id < 0 || static_cast<std::size_t>(atom_id) >= ex.atoms.size()) {
      return false;
    }
    const bool v = ex.atoms[static_cast<std::size_t>(atom_id)];
    return negated ? !v : v;
  }
};

struct Clause {
  std::vector<Literal> body;
  int head_action = 0;

  bool covers(const Example& ex) const {
    for (const auto& lit : body) {
      if (!lit.holds(ex)) {
        return false;
      }
    }
    return true;
  }
};

struct DecisionList {
  std::vector<Clause> clauses;
};

// Visitation-weighted FOIL-style score used by the paper's induction stage:
// coverage = w_pos_covered / w_pos_remaining
// precision = w_pos_covered / w_all_covered
// score = coverage * precision
double coverage_precision_score(double w_pos_covered, double w_all_covered,
                                double w_pos_remaining);

class OrderedCovering {
 public:
  OrderedCovering(int n_atoms, int n_actions, double min_score = 1e-12);

  DecisionList fit(const std::vector<Example>& examples) const;

  static double score_literal(const std::vector<Example>& remaining, int target_action,
                              const Literal& lit);

 private:
  int n_atoms_;
  int n_actions_;
  double min_score_;

  Clause learn_one_clause(const std::vector<Example>& remaining) const;
};

}  // namespace solverrl
