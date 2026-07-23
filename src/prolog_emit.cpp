#include "solverrl/prolog_emit.hpp"

#include <sstream>
#include <stdexcept>

namespace solverrl {
namespace {

std::string emit_literal(const Literal& lit, const AtomSchema& schema) {
  std::ostringstream oss;
  if (lit.negated) {
    oss << "\\+ ";
  }
  oss << schema.predicate << "(S";
  for (const auto& arg : schema.object_args) {
    oss << ", " << arg;
  }
  if (schema.binds_direction) {
    oss << ", D";
  }
  oss << ")";
  return oss.str();
}

std::string emit_perception_layer(const EmitConfig& cfg) {
  std::ostringstream oss;
  oss << "% === perception_and_path_layer ===\n";
  oss << "% Perception predicates are interpreted from the KeyDoor state.\n";
  oss << "% dir_to/3 is the path-aware direction relation (tabled BFS over the grid).\n";
  oss << "% Stubs below keep the file loadable; the runtime supplies real clauses.\n";

  bool saw_dir = false;
  for (const auto& atom : cfg.atoms) {
    if (atom.binds_direction) {
      saw_dir = true;
      break;
    }
  }
  if (saw_dir) {
    oss << ":- dynamic dir_to/3.\n";
    oss << "% dir_to(S, Obj, D) :- ... tabled shortest-path step (wired by runtime).\n";
  }
  oss << "\n";
  return oss.str();
}

}  // namespace

std::string emit_prolog(const DecisionList& list, const EmitConfig& cfg) {
  if (list.clauses.empty()) {
    throw std::invalid_argument("emit_prolog: empty decision list");
  }

  std::ostringstream oss;
  oss << emit_perception_layer(cfg);
  oss << "% === decision_list ===\n";

  for (std::size_t i = 0; i < list.clauses.size(); ++i) {
    const Clause& c = list.clauses[i];
    if (c.head_action < 0 ||
        static_cast<std::size_t>(c.head_action) >= cfg.action_names.size()) {
      throw std::out_of_range("emit_prolog: head_action out of range");
    }
    const std::string& action = cfg.action_names[static_cast<std::size_t>(c.head_action)];
    const bool is_last = (i + 1 == list.clauses.size());
    const bool is_default = c.is_default || c.body.empty();

    if (is_default) {
      if (!is_last) {
        throw std::invalid_argument("emit_prolog: default/empty-body clause must be last");
      }
      oss << "act(_S, " << action << ").\n";
      continue;
    }

    oss << "act(S, " << action << ") :- ";
    for (std::size_t j = 0; j < c.body.size(); ++j) {
      const Literal& lit = c.body[j];
      if (lit.atom_id < 0 ||
          static_cast<std::size_t>(lit.atom_id) >= cfg.atoms.size()) {
        throw std::out_of_range("emit_prolog: atom_id out of range");
      }
      if (j > 0) {
        oss << ", ";
      }
      oss << emit_literal(lit, cfg.atoms[static_cast<std::size_t>(lit.atom_id)]);
    }
    oss << ", !.\n";
  }

  return oss.str();
}

}  // namespace solverrl
