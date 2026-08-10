#include "solverrl/foil.hpp"

#include <algorithm>
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
                                 int max_body_literals, SharedDirConfig shared_dir)
    : n_atoms_(n_atoms),
      n_actions_(n_actions),
      min_score_(min_score),
      max_body_literals_(max_body_literals),
      shared_dir_(std::move(shared_dir)) {}

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

double OrderedCovering::score_shared_dir_clause(const std::vector<Example>& remaining,
                                                const Clause& clause,
                                                const SharedDirConfig& cfg) {
  if (!clause.binds_direction || !cfg.enabled) {
    return 0.0;
  }
  double w_pos_rem = 0.0;
  double w_pos_cov = 0.0;
  double w_all_cov = 0.0;

  for (const auto& ex : remaining) {
    // Positives: teacher chose a move that lies on a shortest path to Obj.
    // (dir_to may be multi-true; any matching teacher dir counts.)
    const bool explainable =
        ex.action >= 0 && ex.action < cfg.n_directions &&
        clause.dir_holds(ex, cfg, ex.action);
    if (explainable && clause.body_holds(ex)) {
      w_pos_rem += ex.weight;
    }

    if (!clause.covers(ex, cfg)) {
      continue;
    }
    w_all_cov += ex.weight;
    // Credit a cover when the teacher's action is among the true dirs, even if
    // resolve_direction() would pick a different tied shortest-path step.
    if (explainable) {
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
      if (lit.atom_id == cand.atom_id) {
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

Clause OrderedCovering::grow_shared_dir_clause(const std::vector<Example>& remaining,
                                               int dir_object) const {
  Clause clause;
  clause.binds_direction = true;
  clause.dir_object = dir_object;
  clause.head_action = 0;  // unused; prediction comes from D
  double best_score = score_shared_dir_clause(remaining, clause, shared_dir_);

  // dir_to(..., D) counts as one body literal; grow filters up to the remaining budget.
  const int max_filters = std::max(0, max_body_literals_ - 1);

  auto literal_used = [&](const Literal& cand) {
    for (const auto& lit : clause.body) {
      if (lit.atom_id == cand.atom_id) {
        return true;
      }
    }
    return false;
  };

  // Only unary perception atoms as filters (not other dir_to groundings).
  const int unary_end = shared_dir_.atom_base;

  while (static_cast<int>(clause.body.size()) < max_filters) {
    Literal best_lit;
    double improved = best_score;
    bool found = false;

    for (int atom = 0; atom < unary_end && atom < n_atoms_; ++atom) {
      for (bool neg : {false, true}) {
        Literal lit{atom, neg};
        if (literal_used(lit)) {
          continue;
        }
        Clause trial = clause;
        trial.body.push_back(lit);
        const double s = score_shared_dir_clause(remaining, trial, shared_dir_);
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

  if (best_score < min_score_) {
    return Clause{};
  }
  return clause;
}

Clause OrderedCovering::learn_one_clause(const std::vector<Example>& remaining,
                                         bool relational_only) const {
  Clause best;
  double best_score = -1.0;

  // Prefer shared-D (key → door → goal) before fixed-head clauses.
  if (shared_dir_.enabled) {
    for (int obj = 0; obj < shared_dir_.n_objects; ++obj) {
      Clause grown = grow_shared_dir_clause(remaining, obj);
      if (!grown.binds_direction) {
        continue;
      }
      const double s = score_shared_dir_clause(remaining, grown, shared_dir_);
      if (s > best_score) {
        best_score = s;
        best = std::move(grown);
      }
    }
  }

  for (int action = 0; action < n_actions_; ++action) {
    if (relational_only && shared_dir_.enabled && action < shared_dir_.n_directions) {
      // Skip fixed move(up/down/left/right) while learning the relational core.
      continue;
    }
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

  if ((!best.binds_direction && best.body.empty()) || best_score < min_score_) {
    return Clause{};
  }
  return best;
}

DecisionList OrderedCovering::fit(const std::vector<Example>& examples) const {
  DecisionList list;
  list.shared_dir = shared_dir_;
  std::vector<Example> remaining = examples;
  const std::vector<Example> all = examples;

  // Phase 0: shared-D + pickup/toggle. Phase 1: residual fixed-head moves.
  bool relational_only = shared_dir_.enabled;
  while (!remaining.empty()) {
    Clause clause = learn_one_clause(remaining, relational_only);
    if (!clause.binds_direction && clause.body.empty()) {
      if (relational_only) {
        relational_only = false;
        continue;
      }
      break;
    }

    std::vector<Example> next;
    next.reserve(remaining.size());
    for (const auto& ex : remaining) {
      if (!clause.covers(ex, shared_dir_)) {
        next.push_back(ex);
      }
    }
    list.clauses.push_back(std::move(clause));
    if (next.size() == remaining.size()) {
      if (relational_only) {
        relational_only = false;
        continue;
      }
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
