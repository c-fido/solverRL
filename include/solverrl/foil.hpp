#pragma once

#include <cstddef>
#include <string>
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

// Optional shared-direction binding: act(S, move(D)) :- dir_to(S, Obj, D), ...
// Ground atom layout: atoms[atom_base + object*n_directions + d] = dir_to(obj, dir_d).
struct SharedDirConfig {
  bool enabled = false;
  int n_objects = 0;
  int atom_base = 0;
  int n_directions = 4;  // move actions are 0 .. n_directions-1
  std::vector<std::string> object_names;
};

struct Clause {
  std::vector<Literal> body;
  int head_action = 0;
  bool is_default = false;
  // If true: head is move(D) bound by dir_to(S, object_names[dir_object], D).
  bool binds_direction = false;
  int dir_object = -1;

  bool body_holds(const Example& ex) const {
    for (const auto& lit : body) {
      if (!lit.holds(ex)) {
        return false;
      }
    }
    return true;
  }

  // True iff dir_to(obj, direction d) holds.
  bool dir_holds(const Example& ex, const SharedDirConfig& cfg, int d) const {
    if (!binds_direction || dir_object < 0 || !cfg.enabled || d < 0 ||
        d >= cfg.n_directions) {
      return false;
    }
    const int idx = cfg.atom_base + dir_object * cfg.n_directions + d;
    return idx >= 0 && static_cast<std::size_t>(idx) < ex.atoms.size() &&
           ex.atoms[static_cast<std::size_t>(idx)];
  }

  // Resolve move(D): if multiple shortest-path dirs are true, prefer a stable
  // order (up, down, left, right). Any true dir is a valid shortest-path step.
  int resolve_direction(const Example& ex, const SharedDirConfig& cfg) const {
    if (!binds_direction || dir_object < 0 || !cfg.enabled) {
      return -1;
    }
    for (int d = 0; d < cfg.n_directions; ++d) {
      if (dir_holds(ex, cfg, d)) {
        return d;
      }
    }
    return -1;
  }

  bool covers(const Example& ex, const SharedDirConfig& cfg = {}) const {
    if (!body_holds(ex)) {
      return false;
    }
    if (binds_direction) {
      return resolve_direction(ex, cfg) >= 0;
    }
    return true;
  }

  int action_for(const Example& ex, const SharedDirConfig& cfg = {}) const {
    if (!covers(ex, cfg)) {
      return -1;
    }
    if (binds_direction) {
      return resolve_direction(ex, cfg);
    }
    return head_action;
  }
};

struct DecisionList {
  std::vector<Clause> clauses;
  SharedDirConfig shared_dir;

  // First-match semantics (empty-body default covers everything).
  int predict(const Example& ex) const {
    for (const auto& c : clauses) {
      const int a = c.action_for(ex, shared_dir);
      if (a >= 0) {
        return a;
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
                  int max_body_literals = 3, SharedDirConfig shared_dir = {});

  DecisionList fit(const std::vector<Example>& examples) const;

  static double score_literal(const std::vector<Example>& remaining, int target_action,
                              const Literal& lit);
  static double score_clause(const std::vector<Example>& remaining, int target_action,
                             const Clause& clause);
  static double score_shared_dir_clause(const std::vector<Example>& remaining,
                                        const Clause& clause, const SharedDirConfig& cfg);
  static int weighted_majority_action(const std::vector<Example>& examples, int n_actions);

 private:
  int n_atoms_;
  int n_actions_;
  double min_score_;
  int max_body_literals_;
  SharedDirConfig shared_dir_;

  Clause grow_clause(const std::vector<Example>& remaining, int target_action) const;
  Clause grow_shared_dir_clause(const std::vector<Example>& remaining, int dir_object) const;
  // relational_only: shared-D + non-move heads (pickup/toggle); skips fixed move(up/…)
  Clause learn_one_clause(const std::vector<Example>& remaining,
                          bool relational_only) const;
};

}  // namespace solverrl
