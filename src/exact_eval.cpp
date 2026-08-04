#include "solverrl/exact_eval.hpp"

#include <cmath>
#include <stdexcept>

namespace solverrl {
namespace {

constexpr double kEps = 1e-15;

std::size_t p_index(int a, int s, int sp, int n) {
  return static_cast<std::size_t>(a * n * n + s * n + sp);
}

std::vector<double> solve_linear_system(std::vector<double> mat, std::vector<double> rhs) {
  const int n = static_cast<int>(rhs.size());
  if (n == 0) {
    return {};
  }

  for (int col = 0; col < n; ++col) {
    int pivot = col;
    double best = std::abs(mat[static_cast<std::size_t>(col * n + col)]);
    for (int row = col + 1; row < n; ++row) {
      const double v = std::abs(mat[static_cast<std::size_t>(row * n + col)]);
      if (v > best) {
        best = v;
        pivot = row;
      }
    }
    if (best < kEps) {
      throw std::runtime_error("exact_eval: singular Bellman system");
    }
    if (pivot != col) {
      for (int j = 0; j < n; ++j) {
        std::swap(mat[static_cast<std::size_t>(pivot * n + j)],
                  mat[static_cast<std::size_t>(col * n + j)]);
      }
      std::swap(rhs[static_cast<std::size_t>(pivot)], rhs[static_cast<std::size_t>(col)]);
    }

    const double diag = mat[static_cast<std::size_t>(col * n + col)];
    for (int j = col; j < n; ++j) {
      mat[static_cast<std::size_t>(col * n + j)] /= diag;
    }
    rhs[static_cast<std::size_t>(col)] /= diag;

    for (int row = 0; row < n; ++row) {
      if (row == col) {
        continue;
      }
      const double factor = mat[static_cast<std::size_t>(row * n + col)];
      if (std::abs(factor) < kEps) {
        continue;
      }
      for (int j = col; j < n; ++j) {
        mat[static_cast<std::size_t>(row * n + j)] -=
            factor * mat[static_cast<std::size_t>(col * n + j)];
      }
      rhs[static_cast<std::size_t>(row)] -= factor * rhs[static_cast<std::size_t>(col)];
    }
  }
  return rhs;
}

}  // namespace

ExactEvaluator::ExactEvaluator(int n_states, int n_actions, std::vector<double> transition,
                               std::vector<double> reward, std::vector<double> initial,
                               double gamma, int done_index, int horizon)
    : n_(n_states),
      a_(n_actions),
      P_(std::move(transition)),
      r_(std::move(reward)),
      mu0_(std::move(initial)),
      gamma_(gamma),
      done_index_(done_index),
      horizon_(horizon) {
  if (n_ <= 0 || a_ <= 0) {
    throw std::invalid_argument("ExactEvaluator: invalid state/action counts");
  }
  if (static_cast<int>(P_.size()) != a_ * n_ * n_) {
    throw std::invalid_argument("ExactEvaluator: transition shape mismatch");
  }
  if (static_cast<int>(r_.size()) != a_ * n_) {
    throw std::invalid_argument("ExactEvaluator: reward shape mismatch");
  }
  if (static_cast<int>(mu0_.size()) != n_) {
    throw std::invalid_argument("ExactEvaluator: initial distribution shape mismatch");
  }
  if (done_index_ < 0 || done_index_ >= n_) {
    throw std::invalid_argument("ExactEvaluator: done_index out of range");
  }
  if (horizon_ <= 0) {
    throw std::invalid_argument("ExactEvaluator: horizon must be positive");
  }
}

std::vector<double> ExactEvaluator::exact_value(const std::vector<int>& policy) const {
  if (static_cast<int>(policy.size()) != n_) {
    throw std::invalid_argument("ExactEvaluator: policy length mismatch");
  }

  std::vector<double> system(static_cast<std::size_t>(n_ * n_), 0.0);
  std::vector<double> rhs(static_cast<std::size_t>(n_), 0.0);

  for (int i = 0; i < n_; ++i) {
    const int action = policy[static_cast<std::size_t>(i)];
    if (action < 0 || action >= a_) {
      throw std::invalid_argument("ExactEvaluator: policy action out of range");
    }
    for (int j = 0; j < n_; ++j) {
      double entry = (i == j ? 1.0 : 0.0);
      entry -= gamma_ * P_[p_index(action, i, j, n_)];
      system[static_cast<std::size_t>(i * n_ + j)] = entry;
    }
    rhs[static_cast<std::size_t>(i)] = r_[static_cast<std::size_t>(action * n_ + i)];
  }
  return solve_linear_system(std::move(system), std::move(rhs));
}

double ExactEvaluator::q_value(const std::vector<double>& value, int state,
                               int action) const {
  double q = r_[static_cast<std::size_t>(action * n_ + state)];
  for (int j = 0; j < n_; ++j) {
    q += gamma_ * P_[p_index(action, state, j, n_)] * value[static_cast<std::size_t>(j)];
  }
  return q;
}

double ExactEvaluator::exact_return(const std::vector<int>& policy) const {
  const auto value = exact_value(policy);
  double total = 0.0;
  for (int i = 0; i < n_; ++i) {
    total += mu0_[static_cast<std::size_t>(i)] * value[static_cast<std::size_t>(i)];
  }
  return total;
}

double ExactEvaluator::success_probability(const std::vector<int>& policy) const {
  if (static_cast<int>(policy.size()) != n_) {
    throw std::invalid_argument("ExactEvaluator: policy length mismatch");
  }

  std::vector<double> mass = mu0_;
  double success = 0.0;
  for (int t = 0; t < horizon_; ++t) {
    std::vector<double> next(static_cast<std::size_t>(n_), 0.0);
    for (int i = 0; i < n_; ++i) {
      const double m = mass[static_cast<std::size_t>(i)];
      if (m < kEps || i == done_index_) {
        continue;
      }
      const int action = policy[static_cast<std::size_t>(i)];
      for (int j = 0; j < n_; ++j) {
        next[static_cast<std::size_t>(j)] += m * P_[p_index(action, i, j, n_)];
      }
    }
    success += next[static_cast<std::size_t>(done_index_)];
    next[static_cast<std::size_t>(done_index_)] = 0.0;
    mass.swap(next);
    double sum = 0.0;
    for (double v : mass) {
      sum += v;
    }
    if (sum < kEps) {
      break;
    }
  }
  return success;
}

AdvantageGapCert ExactEvaluator::advantage_gap_cert(const std::vector<int>& student,
                                                    const std::vector<int>& teacher) const {
  if (static_cast<int>(student.size()) != n_ || static_cast<int>(teacher.size()) != n_) {
    throw std::invalid_argument("ExactEvaluator: policy length mismatch");
  }

  const auto value = exact_value(teacher);
  AdvantageGapCert cert;
  cert.return_gap = exact_return(teacher) - exact_return(student);

  for (int i = 0; i < n_; ++i) {
    const int a_s = student[static_cast<std::size_t>(i)];
    const int a_t = teacher[static_cast<std::size_t>(i)];
    if (a_s == a_t) {
      continue;
    }
    ++cert.n_disagree;
    const double q_t = q_value(value, i, a_t);
    const double q_s = q_value(value, i, a_s);
    const double gap = q_t - q_s;
    cert.weighted_gap += mu0_[static_cast<std::size_t>(i)] * gap;
    if (gap > cert.max_gap) {
      cert.max_gap = gap;
    }
  }
  return cert;
}

}  // namespace solverrl
