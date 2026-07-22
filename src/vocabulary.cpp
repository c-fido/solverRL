#include "solverrl/vocabulary.hpp"

namespace solverrl {

Vocabulary Vocabulary::KeyDoor() {
  Vocabulary v;
  v.name = "keydoor";
  v.objects = {"key", "door", "goal"};
  v.directions = {"up", "down", "left", "right"};

  // Nullary-in-spirit perception predicates over the state (paper Listing 1).
  auto unary = [](const char* name) {
    PredicateSpec p;
    p.name = name;
    p.arg_sorts = {ArgSort::State};
    p.allows_negation = true;
    p.binds_direction_var = false;
    return p;
  };
  v.predicates.push_back(unary("on_key"));
  v.predicates.push_back(unary("door_open"));
  v.predicates.push_back(unary("same_room_goal"));
  v.predicates.push_back(unary("adj_door"));
  v.predicates.push_back(unary("carrying"));

  // Path-aware direction: dir_to(S, Obj, D) shares D with act(S, move(D)).
  PredicateSpec dir_to;
  dir_to.name = "dir_to";
  dir_to.arg_sorts = {ArgSort::State, ArgSort::Object, ArgSort::Direction};
  dir_to.allows_negation = false;
  dir_to.binds_direction_var = true;
  v.predicates.push_back(dir_to);

  return v;
}

const PredicateSpec* Vocabulary::find(const std::string& pred_name) const {
  for (const auto& p : predicates) {
    if (p.name == pred_name) {
      return &p;
    }
  }
  return nullptr;
}

}  // namespace solverrl
