#include <gtest/gtest.h>

#include "solverrl/exact_eval.hpp"
#include "solverrl/expand.hpp"

using solverrl::Clause;
using solverrl::DecisionList;
using solverrl::EditKind;
using solverrl::EditProposal;
using solverrl::Example;
using solverrl::ExpandEditor;
using solverrl::Literal;
using solverrl::rollout_policy;

namespace {

DecisionList sample_list() {
  DecisionList list;
  Clause c0;
  c0.body = {Literal{0, false}};
  c0.head_action = 1;
  list.clauses.push_back(c0);

  Clause c1;
  c1.body = {Literal{1, false}};
  c1.head_action = 2;
  list.clauses.push_back(c1);

  Clause def;
  def.is_default = true;
  def.head_action = 0;
  list.clauses.push_back(def);
  return list;
}

bool has_kind(const std::vector<EditProposal>& props, EditKind kind) {
  for (const auto& p : props) {
    if (p.kind == kind) {
      return true;
    }
  }
  return false;
}

}  // namespace

TEST(ExpandEditor, ProposesSpecializeReorderAndPrune) {
  const auto list = sample_list();
  ExpandEditor editor(/*n_atoms=*/3, /*max_body_literals=*/3);
  const auto props = editor.propose(list);
  ASSERT_GT(props.size(), 0u);
  EXPECT_TRUE(has_kind(props, EditKind::Specialize));
  EXPECT_TRUE(has_kind(props, EditKind::Reorder));
  EXPECT_TRUE(has_kind(props, EditKind::Prune));
}

TEST(ExpandEditor, SpecializeAddsLiteralAtClausePosition) {
  const auto list = sample_list();
  ExpandEditor editor(3, 3);
  const auto props = editor.propose(list);
  bool found = false;
  for (const auto& p : props) {
    if (p.kind != EditKind::Specialize || p.clause_index != 0) {
      continue;
    }
    EXPECT_TRUE(p.has_added_literal);
    EXPECT_EQ(p.list.clauses[0].body.size(), 2u);
    found = true;
    break;
  }
  EXPECT_TRUE(found);
}

TEST(ExpandEditor, ReorderSwapsNonDefaultClauses) {
  const auto list = sample_list();
  ExpandEditor editor(3, 3);
  const auto props = editor.propose(list);
  for (const auto& p : props) {
    if (p.kind != EditKind::Reorder) {
      continue;
    }
    EXPECT_EQ(p.list.clauses[0].head_action, 2);
    EXPECT_EQ(p.list.clauses[1].head_action, 1);
    EXPECT_TRUE(p.list.clauses.back().is_default);
    return;
  }
  FAIL() << "missing reorder proposal";
}

TEST(ExpandEditor, PruneRemovesConditionalClause) {
  const auto list = sample_list();
  ExpandEditor editor(3, 3);
  const auto props = editor.propose(list);
  for (const auto& p : props) {
    if (p.kind != EditKind::Prune || p.clause_index != 0) {
      continue;
    }
    EXPECT_EQ(p.list.clauses.size(), 2u);
    EXPECT_TRUE(p.list.clauses.back().is_default);
    return;
  }
  FAIL() << "missing prune proposal";
}

TEST(ExpandEditor, RolloutPolicyUsesFirstMatchSemantics) {
  const auto list = sample_list();
  Example ex0;
  ex0.atoms = {true, false, false};
  Example ex1;
  ex1.atoms = {false, true, false};
  Example ex2;
  ex2.atoms = {false, false, true};

  const auto policy = rollout_policy(list, {ex0, ex1, ex2});
  ASSERT_EQ(policy.size(), 3u);
  EXPECT_EQ(policy[0], 1);
  EXPECT_EQ(policy[1], 2);
  EXPECT_EQ(policy[2], 0);
}

TEST(ExpansionLoop, AcceptsEditWhenDeltaJMeetsTau) {
  const int n = 2;
  const int a = 2;
  std::vector<double> P = {0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0};
  std::vector<double> r = {1.0, 0.0, 0.0, 0.0};
  std::vector<double> mu0 = {1.0, 0.0};
  solverrl::ExactEvaluator eval(n, a, std::move(P), std::move(r), std::move(mu0), 0.5, 1, 10);

  solverrl::Example s0;
  s0.atoms = {true};
  s0.action = 0;
  solverrl::Example s1;
  s1.atoms = {false};
  s1.action = 0;

  solverrl::ExpandEditor editor(1, 3);
  solverrl::ExpansionLoop loop(eval, {s0, s1}, editor, 1e-9, 10);

  solverrl::DecisionList list;
  solverrl::Clause bad;
  bad.body = {solverrl::Literal{0, false}};
  bad.head_action = 1;
  list.clauses.push_back(bad);
  solverrl::Clause def;
  def.is_default = true;
  def.head_action = 0;
  list.clauses.push_back(def);

  const auto result = loop.run(list);
  EXPECT_GT(result.iterations, 0);
  EXPECT_NEAR(result.final_return, 1.0, 1e-9);
  EXPECT_GE(result.return_curve.size(), 2u);
  for (std::size_t i = 1; i < result.return_curve.size(); ++i) {
    EXPECT_GE(result.return_curve[i], result.return_curve[i - 1] - 1e-12);
  }
}

TEST(ExpansionLoop, StopsWhenTauTooLarge) {
  const int n = 2;
  const int a = 2;
  std::vector<double> P = {0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0};
  std::vector<double> r = {1.0, 0.0, 0.0, 0.0};
  std::vector<double> mu0 = {1.0, 0.0};
  solverrl::ExactEvaluator eval(n, a, std::move(P), std::move(r), std::move(mu0), 0.5, 1, 10);

  solverrl::Example s0;
  s0.atoms = {true};
  solverrl::Example s1;
  s1.atoms = {false};

  solverrl::ExpandEditor editor(1, 3);
  solverrl::ExpansionLoop loop(eval, {s0, s1}, editor, 1e9, 10);

  solverrl::DecisionList list;
  solverrl::Clause bad;
  bad.body = {solverrl::Literal{0, false}};
  bad.head_action = 1;
  list.clauses.push_back(bad);
  solverrl::Clause def;
  def.is_default = true;
  def.head_action = 0;
  list.clauses.push_back(def);

  const auto result = loop.run(list);
  EXPECT_EQ(result.iterations, 0);
  EXPECT_EQ(result.return_curve.size(), 1u);
}
