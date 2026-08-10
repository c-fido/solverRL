#pragma once

#include <string>
#include <vector>

#include "solverrl/foil.hpp"

namespace solverrl {

// Ground-atom naming for Prolog emission (FOIL atom_id → literal text).
struct AtomSchema {
  std::string predicate;
  std::vector<std::string> object_args;
  bool binds_direction = false;  // emit shared variable D (e.g. dir_to(S, key, D))
  std::string fixed_direction;   // emit a ground direction (e.g. up)
};

struct EmitConfig {
  std::vector<std::string> action_names;
  std::vector<AtomSchema> atoms;
  // Object names for shared-D clauses: dir_to(S, object_names[dir_object], D).
  std::vector<std::string> dir_objects;
};

// Emit a SWI-Prolog program: perception/path stubs + decision list with cuts.
// Non-default clauses get a trailing cut (!); the default (empty body) is last.
std::string emit_prolog(const DecisionList& list, const EmitConfig& cfg);

}  // namespace solverrl
