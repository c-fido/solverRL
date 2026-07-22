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
  bool is_default = false;

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

  // First-match semantics (empty-body default covers everything).
  int predict(const Example& ex) const {
    for (const auto& c : clauses) {
      if (c.covers(ex)) {
        return c.head_action;
      }
    }
    return -1;
  }
};

// Visitation-weighted FOIL-style score used by the paper's induction stage:
// coverage = w_pos_covered / w_pos_remaining
// precision = w_pos_covered / w_all_covered
// score = coverage * precision
double coverage_precision_score(double w_pos_covered, double w_all_covered,
                                double w_pos_remaining);

class OrderedCovering {
 public:
  // max_body_literals: paper specializes with up to ~3 literals.
  OrderedCovering(int n_atoms, int n_actions, double min_score = 1e-12,
                  int max_body_literals = 3);

  DecisionList fit(const std::vector<Example>& examples) const;

  static double score_literal(const std::vector<Example>& remaining, int target_action,
                              const Literal& lit);
  static double score_clause(const std::vector<Example>& remaining, int target_action,
                             const Clause& clause);
  static int weighted_majority_action(const std::vector<Example>& examples, int n_actions);

 private:
  int n_atoms_;
  int n_actions_;
  double min_score_;
  int max_body_literals_;

  Clause grow_clause(const std::vector<Example>& remaining, int target_action) const;
  Clause learn_one_clause(const std::vector<Example>& remaining) const;
};

}  // namespace solverrl
