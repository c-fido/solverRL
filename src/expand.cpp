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

bool decision_lists_equal(const DecisionList& a, const DecisionList& b) {
  if (a.clauses.size() != b.clauses.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.clauses.size(); ++i) {
    const Clause& ca = a.clauses[i];
    const Clause& cb = b.clauses[i];
    if (ca.head_action != cb.head_action || ca.is_default != cb.is_default ||
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

ExpandEditor::ExpandEditor(int n_atoms, int max_body_literals)
    : n_atoms_(n_atoms), max_body_literals_(max_body_literals) {
  if (n_atoms_ <= 0) {
    throw std::invalid_argument("ExpandEditor: n_atoms must be positive");
  }
  if (max_body_literals_ <= 0) {
    throw std::invalid_argument("ExpandEditor: max_body_literals must be positive");
  }
}

bool ExpandEditor::is_default_clause(const Clause& clause) {
  return clause.is_default || clause.body.empty();
}

bool ExpandEditor::literal_present(const Clause& clause, const Literal& lit) {
  for (const auto& existing : clause.body) {
    if (existing.atom_id == lit.atom_id) {
      return true;
    }
  }
  return false;
}

std::vector<EditProposal> ExpandEditor::propose(const DecisionList& list) const {
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

  dedupe_proposals(proposals);
  return proposals;
}

}  // namespace solverrl
