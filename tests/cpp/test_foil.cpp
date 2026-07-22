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
  // atom0 true ⇒ action 0; atom0 false ⇒ action 1. Equal visitation weights.
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
  // First clause should pick up one action cleanly via atom0 or ¬atom0.
  EXPECT_TRUE(list.clauses[0].covers(data[0]) || list.clauses[0].covers(data[4]));

  // Every training example must be covered by some clause in order
  // (no default yet — covering continues until residual empty or gain stops).
  for (const auto& ex : data) {
    bool hit = false;
    for (const auto& c : list.clauses) {
      if (c.covers(ex)) {
        EXPECT_EQ(c.head_action, ex.action);
        hit = true;
        break;
      }
    }
    EXPECT_TRUE(hit) << "example action=" << ex.action;
  }
}

TEST(OrderedCovering, PrefersHighWeightPositivesInScore) {
  // One heavy positive with atom0, many light negatives with atom0.
  // Singleton atom0 should still score via weighted coverage×precision.
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
