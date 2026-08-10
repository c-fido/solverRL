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
  if (!schema.fixed_direction.empty()) {
    oss << ", " << schema.fixed_direction;
  } else if (schema.binds_direction) {
    oss << ", D";
  }
  oss << ")";
  return oss.str();
}

std::string emit_perception_layer(const EmitConfig& cfg) {
  std::ostringstream oss;
  oss << "% === perception_and_path_layer ===\n";
  oss << "% Perception predicates are supplied at runtime from the KeyDoor state.\n";
  oss << "% dir_to/3 gives the first step of a shortest path toward an object.\n";
  oss << "% The header below keeps the file loadable in SWI-Prolog.\n";

  bool saw_dir = !cfg.dir_objects.empty();
  for (const auto& atom : cfg.atoms) {
    if (atom.binds_direction || atom.predicate == "dir_to") {
      saw_dir = true;
      break;
    }
  }
  if (saw_dir) {
    oss << ":- dynamic dir_to/3.\n";
    oss << "% dir_to(S, Obj, D) is defined by the runtime perception layer.\n";
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
    const bool is_last = (i + 1 == list.clauses.size());
    const bool is_default = c.is_default || (!c.binds_direction && c.body.empty());

    if (is_default) {
      if (!is_last) {
        throw std::invalid_argument("emit_prolog: default/empty-body clause must be last");
      }
      if (c.head_action < 0 ||
          static_cast<std::size_t>(c.head_action) >= cfg.action_names.size()) {
        throw std::out_of_range("emit_prolog: head_action out of range");
      }
      oss << "act(_S, " << cfg.action_names[static_cast<std::size_t>(c.head_action)]
          << ").\n";
      continue;
    }

    if (c.binds_direction) {
      if (c.dir_object < 0 ||
          static_cast<std::size_t>(c.dir_object) >= cfg.dir_objects.size()) {
        throw std::out_of_range("emit_prolog: dir_object out of range");
      }
      oss << "act(S, move(D)) :- dir_to(S, "
          << cfg.dir_objects[static_cast<std::size_t>(c.dir_object)] << ", D)";
      for (const auto& lit : c.body) {
        if (lit.atom_id < 0 ||
            static_cast<std::size_t>(lit.atom_id) >= cfg.atoms.size()) {
          throw std::out_of_range("emit_prolog: atom_id out of range");
        }
        oss << ", " << emit_literal(lit, cfg.atoms[static_cast<std::size_t>(lit.atom_id)]);
      }
      oss << ", !.\n";
      continue;
    }

    if (c.head_action < 0 ||
        static_cast<std::size_t>(c.head_action) >= cfg.action_names.size()) {
      throw std::out_of_range("emit_prolog: head_action out of range");
    }
    const std::string& action = cfg.action_names[static_cast<std::size_t>(c.head_action)];

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
