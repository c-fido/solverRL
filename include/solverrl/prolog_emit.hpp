#pragma once

#include <string>
#include <vector>

#include "solverrl/foil.hpp"

namespace solverrl {

// Ground-atom naming for emission (FOIL atom_id → Prolog literal).
struct AtomSchema {
  std::string predicate;           // e.g. on_key, dir_to
  std::vector<std::string> object_args;  // e.g. {"key"} for dir_to(S,key,D)
  bool binds_direction = false;    // append Direction var D
};

struct EmitConfig {
  std::vector<std::string> action_names;  // index → Prolog action term
  std::vector<AtomSchema> atoms;          // index → schema
};

// Emit a SWI-Prolog program: perception/path stubs + decision list with cuts.
// Non-default clauses get a trailing cut (!); the default (empty body) is last.
std::string emit_prolog(const DecisionList& list, const EmitConfig& cfg);

}  // namespace solverrl
