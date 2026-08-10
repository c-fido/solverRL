#include "solverrl/expand.hpp"

#include <algorithm>
#include <stdexcept>

namespace solverrl {
namespace {

void dedupe_proposals(std::vector<EditProposal>& proposals) {
  std::vector<EditProposal> unique;
  unique.reserve(proposals.size());
  for (const auto& p : proposals) {
    bool seen = false;
    for (const auto& u : unique) {
      if (decision_lists_equal(u.list, p.list)) {
        seen = true;
        break;
      }
    }
    if (!seen) {
      unique.push_back(p);
    }
  }
  proposals.swap(unique);
}

}  // namespace

std::vector<int> rollout_policy(const DecisionList& list,
                                const std::vector<Example>& examples) {
  std::vector<int> policy;
  policy.reserve(examples.size());
  for (const auto& ex : examples) {
    policy.push_back(list.predict(ex));
  }
  return policy;
}

bool policies_equal(const std::vector<int>& a, const std::vector<int>& b) {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin());
}

bool decision_lists_equal(const DecisionList& a, const DecisionList& b) {
  if (a.clauses.size() != b.clauses.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.clauses.size(); ++i) {
    const Clause& ca = a.clauses[i];
    const Clause& cb = b.clauses[i];
    if (ca.head_action != cb.head_action || ca.is_default != cb.is_default ||
        ca.binds_direction != cb.binds_direction || ca.dir_object != cb.dir_object ||
        ca.body.size() != cb.body.size()) {
      return false;
    }
    for (std::size_t j = 0; j < ca.body.size(); ++j) {
      if (ca.body[j].atom_id != cb.body[j].atom_id ||
          ca.body[j].negated != cb.body[j].negated) {
        return false;
      }
    }
  }
  return true;
}

ExpandEditor::ExpandEditor(int n_atoms, int n_actions, int max_body_literals)
    : n_atoms_(n_atoms), n_actions_(n_actions), max_body_literals_(max_body_literals) {
  if (n_atoms_ <= 0) {
    throw std::invalid_argument("ExpandEditor: n_atoms must be positive");
  }
  if (n_actions_ <= 0) {
    throw std::invalid_argument("ExpandEditor: n_actions must be positive");
  }
  if (max_body_literals_ <= 0) {
    throw std::invalid_argument("ExpandEditor: max_body_literals must be positive");
  }
}

bool ExpandEditor::is_default_clause(const Clause& clause) {
  return clause.is_default || (!clause.binds_direction && clause.body.empty());
}

bool ExpandEditor::literal_present(const Clause& clause, const Literal& lit) {
  for (const auto& existing : clause.body) {
    if (existing.atom_id == lit.atom_id) {
      return true;
    }
  }
  return false;
}

int ExpandEditor::teacher_majority_on_clause(const Clause& clause,
                                             const std::vector<Example>& examples,
                                             int n_actions, const SharedDirConfig& cfg) {
  std::vector<double> mass(static_cast<std::size_t>(n_actions), 0.0);
  for (const auto& ex : examples) {
    if (!clause.covers(ex, cfg)) {
      continue;
    }
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

std::vector<EditProposal> ExpandEditor::propose(const DecisionList& list,
                                                const std::vector<Example>& examples) const {
  if (list.clauses.empty()) {
    return {};
  }

  std::vector<EditProposal> proposals;
  const int n = static_cast<int>(list.clauses.size());
  const int default_index = is_default_clause(list.clauses.back()) ? n - 1 : -1;

  // Specialize-at-position: add one literal to a non-default clause.
  for (int i = 0; i < n; ++i) {
    if (i == default_index || is_default_clause(list.clauses[static_cast<std::size_t>(i)])) {
      continue;
    }
    const Clause& clause = list.clauses[static_cast<std::size_t>(i)];
    if (static_cast<int>(clause.body.size()) >= max_body_literals_) {
      continue;
    }
    for (int atom = 0; atom < n_atoms_; ++atom) {
      for (const bool neg : {false, true}) {
        const Literal lit{atom, neg};
        if (literal_present(clause, lit)) {
          continue;
        }
        EditProposal prop;
        prop.kind = EditKind::Specialize;
        prop.clause_index = i;
        prop.added_literal = lit;
        prop.has_added_literal = true;
        prop.list = list;
        prop.list.clauses[static_cast<std::size_t>(i)].body.push_back(lit);
        proposals.push_back(std::move(prop));
      }
    }
  }

  // Reorder: swap two non-default clauses (default stays last).
  for (int i = 0; i < n; ++i) {
    if (i == default_index) {
      continue;
    }
    for (int j = i + 1; j < n; ++j) {
      if (j == default_index) {
        continue;
      }
      EditProposal prop;
      prop.kind = EditKind::Reorder;
      prop.clause_index = i;
      prop.other_index = j;
      prop.list = list;
      std::swap(prop.list.clauses[static_cast<std::size_t>(i)],
                prop.list.clauses[static_cast<std::size_t>(j)]);
      proposals.push_back(std::move(prop));
    }
  }

  // Prune-non-default: drop one conditional clause.
  for (int i = 0; i < n; ++i) {
    if (i == default_index || is_default_clause(list.clauses[static_cast<std::size_t>(i)])) {
      continue;
    }
    EditProposal prop;
    prop.kind = EditKind::Prune;
    prop.clause_index = i;
    prop.list = list;
    prop.list.clauses.erase(prop.list.clauses.begin() + i);
    proposals.push_back(std::move(prop));
  }

  // Retarget-head: try a different action on fixed-head clauses (including default).
  for (int i = 0; i < n; ++i) {
    if (list.clauses[static_cast<std::size_t>(i)].binds_direction) {
      continue;
    }
    const int current = list.clauses[static_cast<std::size_t>(i)].head_action;
    for (int action = 0; action < n_actions_; ++action) {
      if (action == current) {
        continue;
      }
      EditProposal prop;
      prop.kind = EditKind::RetargetHead;
      prop.clause_index = i;
      prop.head_action = action;
      prop.list = list;
      prop.list.clauses[static_cast<std::size_t>(i)].head_action = action;
      proposals.push_back(std::move(prop));
    }
  }

  // Add teacher-aligned single-literal clause at any position before the default.
  const int insert_hi = default_index >= 0 ? default_index : n;
  for (int insert_at = 0; insert_at <= insert_hi; ++insert_at) {
    for (int atom = 0; atom < n_atoms_; ++atom) {
      for (const bool neg : {false, true}) {
        const Literal lit{atom, neg};
        Clause probe;
        probe.body = {lit};
        probe.head_action =
            teacher_majority_on_clause(probe, examples, n_actions_, list.shared_dir);
        EditProposal prop;
        prop.kind = EditKind::AddClause;
        prop.clause_index = insert_at;
        prop.head_action = probe.head_action;
        prop.added_literal = lit;
        prop.has_added_literal = true;
        prop.list = list;
        Clause added;
        added.body = {lit};
        added.head_action = probe.head_action;
        prop.list.clauses.insert(prop.list.clauses.begin() + insert_at, std::move(added));
        proposals.push_back(std::move(prop));
      }
    }
  }

  dedupe_proposals(proposals);
  return proposals;
}

ExpansionLoop::ExpansionLoop(const ExactEvaluator& evaluator, std::vector<Example> examples,
                             ExpandEditor editor, double tau, int max_iterations)
    : evaluator_(evaluator),
      examples_(std::move(examples)),
      editor_(std::move(editor)),
      tau_(tau),
      max_iterations_(max_iterations) {
  if (examples_.empty()) {
    throw std::invalid_argument("ExpansionLoop: examples must be non-empty");
  }
  if (tau_ < 0.0) {
    throw std::invalid_argument("ExpansionLoop: tau must be non-negative");
  }
  if (max_iterations_ <= 0) {
    throw std::invalid_argument("ExpansionLoop: max_iterations must be positive");
  }
}

ExpansionResult ExpansionLoop::run(const DecisionList& initial) const {
  if (initial.clauses.empty()) {
    throw std::invalid_argument("ExpansionLoop.run: empty decision list");
  }

  ExpansionResult result;
  result.final_list = initial;

  auto eval_list = [&](const DecisionList& list) {
    const auto policy = rollout_policy(list, examples_);
    return std::pair<double, double>{evaluator_.exact_return(policy),
                                     evaluator_.success_probability(policy)};
  };

  auto [j, succ] = eval_list(result.final_list);
  result.return_curve.push_back(j);
  result.success_curve.push_back(succ);
  result.final_return = j;
  result.final_success = succ;

  for (int iter = 0; iter < max_iterations_; ++iter) {
    const auto proposals = editor_.propose(result.final_list, examples_);
    const auto current_policy = rollout_policy(result.final_list, examples_);
    double best_delta = 0.0;
    int best_index = -1;

    for (int i = 0; i < static_cast<int>(proposals.size()); ++i) {
      const auto& candidate = proposals[static_cast<std::size_t>(i)].list;
      const auto candidate_policy = rollout_policy(candidate, examples_);
      if (policies_equal(current_policy, candidate_policy)) {
        continue;
      }
      const auto [j_candidate, _succ] = eval_list(candidate);
      (void)_succ;
      const double delta = j_candidate - result.final_return;
      if (delta + 1e-15 >= tau_ && delta > best_delta + 1e-15) {
        best_delta = delta;
        best_index = i;
      }
    }

    if (best_index < 0) {
      break;
    }

    result.final_list = proposals[static_cast<std::size_t>(best_index)].list;
    result.final_return += best_delta;
    const auto [j_new, succ_new] = eval_list(result.final_list);
    result.final_return = j_new;
    result.final_success = succ_new;
    result.return_curve.push_back(j_new);
    result.success_curve.push_back(succ_new);
    result.accepted_any_edit = true;
    result.iterations = iter + 1;
  }

  return result;
}

}  // namespace solverrl
