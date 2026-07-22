#include "solverrl/foil.hpp"

#include <vector>

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

OrderedCovering::OrderedCovering(int n_atoms, int n_actions, double min_score,
                                 int max_body_literals)
    : n_atoms_(n_atoms),
      n_actions_(n_actions),
      min_score_(min_score),
      max_body_literals_(max_body_literals) {}

double OrderedCovering::score_literal(const std::vector<Example>& remaining,
                                      int target_action, const Literal& lit) {
  Clause c;
  c.body = {lit};
  c.head_action = target_action;
  return score_clause(remaining, target_action, c);
}

double OrderedCovering::score_clause(const std::vector<Example>& remaining,
                                     int target_action, const Clause& clause) {
  double w_pos_rem = 0.0;
  double w_pos_cov = 0.0;
  double w_all_cov = 0.0;
  for (const auto& ex : remaining) {
    if (ex.action == target_action) {
      w_pos_rem += ex.weight;
    }
    if (!clause.covers(ex)) {
      continue;
    }
    w_all_cov += ex.weight;
    if (ex.action == target_action) {
      w_pos_cov += ex.weight;
    }
  }
  return coverage_precision_score(w_pos_cov, w_all_cov, w_pos_rem);
}

int OrderedCovering::weighted_majority_action(const std::vector<Example>& examples,
                                              int n_actions) {
  std::vector<double> mass(static_cast<std::size_t>(n_actions), 0.0);
  for (const auto& ex : examples) {
    if (ex.action >= 0 && ex.action < n_actions) {
      mass[static_cast<std::size_t>(ex.action)] += ex.weight;
    }
  }
  int best = 0;
  double best_w = -1.0;
  for (int a = 0; a < n_actions; ++a) {
    if (mass[static_cast<std::size_t>(a)] > best_w) {
      best_w = mass[static_cast<std::size_t>(a)];
      best = a;
    }
  }
  return best;
}

Clause OrderedCovering::grow_clause(const std::vector<Example>& remaining,
                                    int target_action) const {
  Clause clause;
  clause.head_action = target_action;
  double best_score = 0.0;

  auto literal_used = [&](const Literal& cand) {
    for (const auto& lit : clause.body) {
      if (lit.atom_id == cand.atom_id && lit.negated == cand.negated) {
        return true;
      }
      // Also skip the opposite polarity on the same atom (contradiction).
      if (lit.atom_id == cand.atom_id && lit.negated != cand.negated) {
        return true;
      }
    }
    return false;
  };

  while (static_cast<int>(clause.body.size()) < max_body_literals_) {
    Literal best_lit;
    double improved = best_score;
    bool found = false;

    for (int atom = 0; atom < n_atoms_; ++atom) {
      for (bool neg : {false, true}) {
        Literal lit{atom, neg};
        if (literal_used(lit)) {
          continue;
        }
        Clause trial = clause;
        trial.body.push_back(lit);
        const double s = score_clause(remaining, target_action, trial);
        if (s > improved + 1e-15) {
          improved = s;
          best_lit = lit;
          found = true;
        }
      }
    }

    if (!found) {
      break;
    }
    clause.body.push_back(best_lit);
    best_score = improved;
  }

  if (clause.body.empty() || best_score < min_score_) {
    return Clause{};
  }
  return clause;
}

Clause OrderedCovering::learn_one_clause(const std::vector<Example>& remaining) const {
  Clause best;
  double best_score = -1.0;

  for (int action = 0; action < n_actions_; ++action) {
    Clause grown = grow_clause(remaining, action);
    if (grown.body.empty()) {
      continue;
    }
    const double s = score_clause(remaining, action, grown);
    if (s > best_score) {
      best_score = s;
      best = std::move(grown);
    }
  }

  if (best.body.empty() || best_score < min_score_) {
    return Clause{};
  }
  return best;
}

DecisionList OrderedCovering::fit(const std::vector<Example>& examples) const {
  DecisionList list;
  std::vector<Example> remaining = examples;
  const std::vector<Example> all = examples;

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
    }
    list.clauses.push_back(std::move(clause));
    if (next.size() == remaining.size()) {
      break;
    }
    remaining.swap(next);
  }

  // Default clause: empty body, residual majority (paper); if none left, use full-data majority.
  const auto& pool = remaining.empty() ? all : remaining;
  Clause def;
  def.body.clear();
  def.head_action = weighted_majority_action(pool, n_actions_);
  def.is_default = true;
  list.clauses.push_back(std::move(def));

  return list;
}

}  // namespace solverrl
