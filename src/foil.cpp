#include "solverrl/foil.hpp"

namespace solverrl {

double coverage_precision_score(double w_pos_covered, double w_all_covered,
                                double w_pos_remaining) {
  if (w_pos_covered <= 0.0 || w_all_covered <= 0.0 || w_pos_remaining <= 0.0) {
    return 0.0;
  }
  const double coverage = w_pos_covered / w_pos_remaining;
  const double precision = w_pos_covered / w_all_covered;
  return coverage * precision;
}

OrderedCovering::OrderedCovering(int n_atoms, int n_actions, double min_score)
    : n_atoms_(n_atoms), n_actions_(n_actions), min_score_(min_score) {}

double OrderedCovering::score_literal(const std::vector<Example>& remaining,
                                      int target_action, const Literal& lit) {
  double w_pos_rem = 0.0;
  double w_pos_cov = 0.0;
  double w_all_cov = 0.0;
  for (const auto& ex : remaining) {
    if (ex.action == target_action) {
      w_pos_rem += ex.weight;
    }
    if (!lit.holds(ex)) {
      continue;
    }
    w_all_cov += ex.weight;
    if (ex.action == target_action) {
      w_pos_cov += ex.weight;
    }
  }
  return coverage_precision_score(w_pos_cov, w_all_cov, w_pos_rem);
}

Clause OrderedCovering::learn_one_clause(const std::vector<Example>& remaining) const {
  // This bullet: singleton bodies scored by coverage×precision.
  // Multi-literal growing lands in the next task.
  Clause best;
  double best_score = -1.0;
  bool found = false;

  for (int action = 0; action < n_actions_; ++action) {
    for (int atom = 0; atom < n_atoms_; ++atom) {
      for (bool neg : {false, true}) {
        Literal lit{atom, neg};
        const double s = score_literal(remaining, action, lit);
        if (s > best_score) {
          best_score = s;
          best.body = {lit};
          best.head_action = action;
          found = true;
        }
      }
    }
  }

  if (!found || best_score < min_score_) {
    return Clause{};  // empty body sentinel ⇒ stop covering
  }
  return best;
}

DecisionList OrderedCovering::fit(const std::vector<Example>& examples) const {
  DecisionList list;
  std::vector<Example> remaining = examples;

  while (!remaining.empty()) {
    Clause clause = learn_one_clause(remaining);
    if (clause.body.empty()) {
      break;
    }

    std::vector<Example> next;
    next.reserve(remaining.size());
    for (const auto& ex : remaining) {
      if (!clause.covers(ex)) {
        next.push_back(ex);
      }
      // Covered examples are removed from the pool (classic sequential covering),
      // whether or not their label matches — first-match list semantics.
    }
    list.clauses.push_back(std::move(clause));
    if (next.size() == remaining.size()) {
      break;  // no progress
    }
    remaining.swap(next);
  }

  return list;
}

}  // namespace solverrl
