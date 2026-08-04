#pragma once

#include <cstddef>
#include <vector>

namespace solverrl {

struct AdvantageGapCert {
  // Sum over disagreeing states s of mu0(s) * (Q_T(s, a_T) - Q_T(s, a_S)).
  double weighted_gap = 0.0;
  // J(pi_T) - J(pi_S) under the exact evaluator.
  double return_gap = 0.0;
  std::size_t n_disagree = 0;
  double max_gap = 0.0;
};

class ExactEvaluator {
 public:
  ExactEvaluator(int n_states, int n_actions, std::vector<double> transition,
                 std::vector<double> reward, std::vector<double> initial, double gamma,
                 int done_index, int horizon);

  int n_states() const { return n_; }
  int n_actions() const { return a_; }
  int done_index() const { return done_index_; }
  int horizon() const { return horizon_; }
  double gamma() const { return gamma_; }

  double exact_return(const std::vector<int>& policy) const;
  double success_probability(const std::vector<int>& policy) const;
  AdvantageGapCert advantage_gap_cert(const std::vector<int>& student,
                                      const std::vector<int>& teacher) const;

 private:
  int n_;
  int a_;
  std::vector<double> P_;
  std::vector<double> r_;
  std::vector<double> mu0_;
  double gamma_;
  int done_index_;
  int horizon_;

  std::vector<double> exact_value(const std::vector<int>& policy) const;
  double q_value(const std::vector<double>& value, int state, int action) const;
};

}  // namespace solverrl
