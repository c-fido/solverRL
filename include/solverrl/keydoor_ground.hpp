#pragma once

#include <cstddef>
#include <vector>

#include "solverrl/prolog_emit.hpp"

namespace solverrl {

constexpr int kKeyDoorNumAtoms = 17;
constexpr int kKeyDoorNumActions = 6;
constexpr int kKeyDoorStateFields = 9;

struct KeyDoorState {
  int door_row = -1;
  int agent_r = -1;
  int agent_c = -1;
  int key_r = -1;
  int key_c = -1;
  int goal_r = -1;
  int goal_c = -1;
  bool door_open = false;
  bool carrying = false;

  bool is_done() const { return door_row < 0; }
};

KeyDoorState decode_keydoor_state(const int* fields);

std::vector<bool> ground_atoms(const KeyDoorState& state);

EmitConfig keydoor_emit_config();

}  // namespace solverrl
