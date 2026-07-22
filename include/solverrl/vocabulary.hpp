#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace solverrl {

enum class ArgSort { State, Object, Direction };

struct PredicateSpec {
  std::string name;
  std::vector<ArgSort> arg_sorts;
  bool allows_negation = true;
  // True for dir_to/3: the Direction argument is a shared variable with move(D).
  bool binds_direction_var = false;

  std::size_t arity() const { return arg_sorts.size(); }
};

struct Vocabulary {
  std::string name;
  std::vector<std::string> objects;
  std::vector<std::string> directions;
  std::vector<PredicateSpec> predicates;

  static Vocabulary KeyDoor();

  const PredicateSpec* find(const std::string& pred_name) const;
  std::size_t size() const { return predicates.size(); }
};

}  // namespace solverrl
