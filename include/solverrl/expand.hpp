#pragma once

#include <cstddef>
#include <vector>

#include "solverrl/foil.hpp"
#include "solverrl/exact_eval.hpp"

namespace solverrl {

enum class EditKind { Specialize, Reorder, Prune, RetargetHead, AddClause };

struct EditProposal {
  EditKind kind = EditKind::Specialize;
  DecisionList list;
  int clause_index = -1;
  int other_index = -1;
  int head_action = -1;
  Literal added_literal{};
  bool has_added_literal = false;
};

// Roll out first-match decision-list actions over grounded examples.
std::vector<int> rollout_policy(const DecisionList& list,
                                const std::vector<Example>& examples);

bool policies_equal(const std::vector<int>& a, const std::vector<int>& b);

class ExpandEditor {
 public:
  ExpandEditor(int n_atoms, int n_actions, int max_body_literals = 3);

  std::vector<EditProposal> propose(const DecisionList& list,
                                    const std::vector<Example>& examples) const;

 private:
  int n_atoms_;
  int n_actions_;
  int max_body_literals_;

  static bool is_default_clause(const Clause& clause);
  static bool literal_present(const Clause& clause, const Literal& lit);
  static int teacher_majority_on_clause(const Clause& clause,
                                        const std::vector<Example>& examples,
                                        int n_actions, const SharedDirConfig& cfg);
};

bool decision_lists_equal(const DecisionList& a, const DecisionList& b);

struct ExpansionResult {
  DecisionList final_list;
  std::vector<double> return_curve;
  std::vector<double> success_curve;
  int iterations = 0;
  double final_return = 0.0;
  double final_success = 0.0;
  bool accepted_any_edit = false;
};

class ExpansionLoop {
 public:
  ExpansionLoop(const ExactEvaluator& evaluator, std::vector<Example> examples,
                ExpandEditor editor, double tau, int max_iterations);

  ExpansionResult run(const DecisionList& initial) const;

 private:
  const ExactEvaluator& evaluator_;
  std::vector<Example> examples_;
  ExpandEditor editor_;
  double tau_;
  int max_iterations_;
};

}  // namespace solverrl
