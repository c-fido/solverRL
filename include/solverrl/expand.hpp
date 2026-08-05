#pragma once

#include <cstddef>
#include <vector>

#include "solverrl/foil.hpp"

namespace solverrl {

enum class EditKind { Specialize, Reorder, Prune };

struct EditProposal {
  EditKind kind = EditKind::Specialize;
  DecisionList list;
  int clause_index = -1;
  int other_index = -1;
  Literal added_literal{};
  bool has_added_literal = false;
};

// Roll out first-match decision-list actions over grounded examples.
std::vector<int> rollout_policy(const DecisionList& list,
                                const std::vector<Example>& examples);

class ExpandEditor {
 public:
  ExpandEditor(int n_atoms, int max_body_literals = 3);

  std::vector<EditProposal> propose(const DecisionList& list) const;

 private:
  int n_atoms_;
  int max_body_literals_;

  static bool is_default_clause(const Clause& clause);
  static bool literal_present(const Clause& clause, const Literal& lit);
};

bool decision_lists_equal(const DecisionList& a, const DecisionList& b);

}  // namespace solverrl
