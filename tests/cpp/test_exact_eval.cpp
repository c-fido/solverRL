#include <gtest/gtest.h>

#include "solverrl/exact_eval.hpp"

using solverrl::AdvantageGapCert;
using solverrl::ExactEvaluator;

namespace {

ExactEvaluator tiny_evaluator() {
  // Two states: 0 (start) --action 0--> 1 (done). gamma=0.5, mu0=[1,0].
  const int n = 2;
  const int a = 1;
  std::vector<double> P = {
      0.0, 1.0,  // action 0 from s0 -> s1
      0.0, 1.0,  // action 0 from s1 -> s1
  };
  std::vector<double> r = {1.0, 0.0};
  std::vector<double> mu0 = {1.0, 0.0};
  return ExactEvaluator(n, a, std::move(P), std::move(r), std::move(mu0), 0.5, 1, 10);
}

}  // namespace

TEST(ExactEvaluator, ReturnAndSuccessOnTinyChain) {
  auto eval = tiny_evaluator();
  const std::vector<int> policy = {0, 0};
  EXPECT_NEAR(eval.exact_return(policy), 1.0, 1e-9);
  EXPECT_NEAR(eval.success_probability(policy), 1.0, 1e-9);
}

TEST(ExactEvaluator, AdvantageGapCertOnDisagreement) {
  // Add a second action that loops at s0 with zero reward.
  const int n = 2;
  const int a = 2;
  std::vector<double> P = {
      0.0, 1.0,  // a0 s0 -> s1
      0.0, 1.0,  // a0 s1 -> s1
      1.0, 0.0,  // a1 s0 -> s0
      0.0, 1.0,  // a1 s1 -> s1
  };
  std::vector<double> r = {1.0, 0.0, 0.0, 0.0};
  std::vector<double> mu0 = {1.0, 0.0};
  ExactEvaluator eval(n, a, std::move(P), std::move(r), std::move(mu0), 0.5, 1, 10);

  const std::vector<int> teacher = {0, 0};
  const std::vector<int> student = {1, 0};
  const AdvantageGapCert cert = eval.advantage_gap_cert(student, teacher);

  EXPECT_EQ(cert.n_disagree, 1u);
  EXPECT_GT(cert.return_gap, 0.0);
  EXPECT_GT(cert.weighted_gap, 0.0);
  EXPECT_GE(cert.max_gap, cert.weighted_gap);
}
