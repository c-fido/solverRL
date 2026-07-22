#include <gtest/gtest.h>

#include "solverrl/foil.hpp"

using solverrl::Clause;
using solverrl::Example;
using solverrl::Literal;
using solverrl::OrderedCovering;
using solverrl::coverage_precision_score;

TEST(FoilScore, IsCoverageTimesPrecision) {
  // coverage = 4/8 = 0.5, precision = 4/5 = 0.8 → 0.4
  EXPECT_DOUBLE_EQ(coverage_precision_score(/*w_pos_cov=*/4, /*w_all_cov=*/5,
                                            /*w_pos_rem=*/8),
                   0.4);
}

TEST(FoilScore, ZeroWhenNothingCoveredOrNoPositivesRemain) {
  EXPECT_DOUBLE_EQ(coverage_precision_score(0, 0, 10), 0.0);
  EXPECT_DOUBLE_EQ(coverage_precision_score(1, 1, 0), 0.0);
}

TEST(OrderedCovering, SeparatesTwoActionsWithSingletonLiterals) {
  std::vector<Example> data;
  for (int i = 0; i < 4; ++i) {
    data.push_back(Example{/*atoms=*/{true}, /*action=*/0, /*weight=*/1.0});
  }
  for (int i = 0; i < 4; ++i) {
    data.push_back(Example{/*atoms=*/{false}, /*action=*/1, /*weight=*/1.0});
  }

  OrderedCovering learner(/*n_atoms=*/1, /*n_actions=*/2);
  const auto list = learner.fit(data);

  ASSERT_FALSE(list.clauses.empty());
  EXPECT_TRUE(list.clauses.back().is_default);
  EXPECT_TRUE(list.clauses.back().body.empty());

  for (const auto& ex : data) {
    EXPECT_EQ(list.predict(ex), ex.action);
  }
}

TEST(OrderedCovering, PrefersHighWeightPositivesInScore) {
  std::vector<Example> data;
  data.push_back(Example{{true}, 0, 10.0});
  for (int i = 0; i < 9; ++i) {
    data.push_back(Example{{true}, 1, 0.1});
  }
  data.push_back(Example{{false}, 1, 1.0});

  const double score_atom0 =
      OrderedCovering::score_literal(data, /*target_action=*/0, Literal{0, false});
  const double score_not_atom0 =
      OrderedCovering::score_literal(data, /*target_action=*/0, Literal{0, true});
  EXPECT_GT(score_atom0, 0.0);
  EXPECT_GT(score_atom0, score_not_atom0);
}

TEST(OrderedCovering, GrowsConjunctionUpToBound) {
  // action 0 only when atom0 ∧ atom1; otherwise action 1.
  std::vector<Example> data;
  data.push_back(Example{{true, true}, 0, 5.0});
  data.push_back(Example{{true, false}, 1, 1.0});
  data.push_back(Example{{false, true}, 1, 1.0});
  data.push_back(Example{{false, false}, 1, 1.0});

  OrderedCovering learner(/*n_atoms=*/2, /*n_actions=*/2, /*min_score=*/1e-12,
                          /*max_body_literals=*/3);
  const auto list = learner.fit(data);

  // Some non-default clause should use 2 literals for the pure conjunction.
  bool found_big = false;
  for (const auto& c : list.clauses) {
    if (!c.is_default && c.head_action == 0 && c.body.size() >= 2) {
      found_big = true;
      EXPECT_TRUE(c.covers(data[0]));
      EXPECT_FALSE(c.covers(data[1]));
      EXPECT_FALSE(c.covers(data[2]));
    }
  }
  EXPECT_TRUE(found_big);
  for (const auto& ex : data) {
    EXPECT_EQ(list.predict(ex), ex.action);
  }
}

TEST(OrderedCovering, DefaultIsResidualMajority) {
  // Only action 1 remains after a perfect first rule on atom0→action0.
  std::vector<Example> data;
  data.push_back(Example{{true}, 0, 1.0});
  data.push_back(Example{{false}, 1, 3.0});
  data.push_back(Example{{false}, 1, 3.0});

  OrderedCovering learner(1, 2);
  const auto list = learner.fit(data);
  ASSERT_TRUE(list.clauses.back().is_default);
  EXPECT_EQ(list.clauses.back().head_action, 1);
  EXPECT_EQ(OrderedCovering::weighted_majority_action(data, 2), 1);
}
