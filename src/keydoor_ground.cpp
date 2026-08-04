#include "solverrl/keydoor_ground.hpp"

#include <cmath>
#include <queue>
#include <tuple>
#include <vector>

namespace solverrl {
namespace {

constexpr int kHeight = 4;
constexpr int kWidth = 6;
constexpr int kWallCol = 3;
constexpr int kLeftCols[] = {0, 1, 2};
constexpr int kRightCols[] = {4, 5};
constexpr int kNumDirs = 4;
constexpr int kDr[] = {-1, 1, 0, 0};
constexpr int kDc[] = {0, 0, -1, 1};
constexpr const char* kDirections[] = {"up", "down", "left", "right"};
constexpr const char* kObjects[] = {"key", "door", "goal"};

using Pos = std::pair<int, int>;

bool in_left(int c) {
  for (int lc : kLeftCols) {
    if (c == lc) {
      return true;
    }
  }
  return false;
}

bool in_right(int c) {
  for (int rc : kRightCols) {
    if (c == rc) {
      return true;
    }
  }
  return false;
}

Pos door_cell(int door_row) { return {door_row, kWallCol}; }

bool is_walkable(Pos pos, int door_row, bool door_open) {
  const int r = pos.first;
  const int c = pos.second;
  if (r < 0 || r >= kHeight || c < 0 || c >= kWidth) {
    return false;
  }
  if (in_left(c)) {
    return true;
  }
  if (in_right(c)) {
    return true;
  }
  if (c == kWallCol) {
    return r == door_row && door_open;
  }
  return false;
}

bool adjacent(Pos a, Pos b) {
  return std::abs(a.first - b.first) + std::abs(a.second - b.second) == 1;
}

bool same_room(Pos a, Pos b) {
  return (in_left(a.second) && in_left(b.second)) ||
         (in_right(a.second) && in_right(b.second));
}

// BFS distances from start; unreachable cells stay -1.
std::vector<std::vector<int>> bfs_dist(Pos start, int door_row, bool door_open) {
  std::vector<std::vector<int>> dist(kHeight, std::vector<int>(kWidth, -1));
  if (!is_walkable(start, door_row, door_open)) {
    return dist;
  }
  std::queue<Pos> q;
  dist[start.first][start.second] = 0;
  q.push(start);
  while (!q.empty()) {
    const Pos cur = q.front();
    q.pop();
    for (int d = 0; d < kNumDirs; ++d) {
      const Pos nxt{cur.first + kDr[d], cur.second + kDc[d]};
      if (!is_walkable(nxt, door_row, door_open)) {
        continue;
      }
      if (dist[nxt.first][nxt.second] >= 0) {
        continue;
      }
      dist[nxt.first][nxt.second] = dist[cur.first][cur.second] + 1;
      q.push(nxt);
    }
  }
  return dist;
}

void set_dir_to_atoms(std::vector<bool>& atoms, int base_idx, Pos agent, Pos target,
                      int door_row, bool door_open) {
  const auto dist = bfs_dist(agent, door_row, door_open);
  if (target.first < 0 || target.second < 0 ||
      dist[target.first][target.second] < 0) {
    return;
  }
  const int target_dist = dist[target.first][target.second];
  if (target_dist == 0) {
    return;
  }
  for (int d = 0; d < kNumDirs; ++d) {
    const Pos nxt{agent.first + kDr[d], agent.second + kDc[d]};
    if (!is_walkable(nxt, door_row, door_open)) {
      continue;
    }
    if (dist[nxt.first][nxt.second] == target_dist - 1) {
      atoms[static_cast<std::size_t>(base_idx + d)] = true;
    }
  }
}

}  // namespace

KeyDoorState decode_keydoor_state(const int* fields) {
  KeyDoorState s;
  s.door_row = fields[0];
  s.agent_r = fields[1];
  s.agent_c = fields[2];
  s.key_r = fields[3];
  s.key_c = fields[4];
  s.goal_r = fields[5];
  s.goal_c = fields[6];
  s.door_open = fields[7] != 0;
  s.carrying = fields[8] != 0;
  return s;
}

std::vector<bool> ground_atoms(const KeyDoorState& state) {
  std::vector<bool> atoms(static_cast<std::size_t>(kKeyDoorNumAtoms), false);
  if (state.is_done()) {
    return atoms;
  }

  const Pos agent{state.agent_r, state.agent_c};
  const Pos goal{state.goal_r, state.goal_c};
  const Pos door = door_cell(state.door_row);

  if (state.key_r >= 0 && state.key_c >= 0) {
    atoms[0] = (agent == Pos{state.key_r, state.key_c});
  }
  atoms[1] = state.door_open;
  atoms[2] = same_room(agent, goal);
  atoms[3] = adjacent(agent, door);
  atoms[4] = state.carrying;

  // dir_to blocks: key (5-8), door (9-12), goal (13-16)
  if (state.key_r >= 0 && state.key_c >= 0) {
    set_dir_to_atoms(atoms, 5, agent, Pos{state.key_r, state.key_c}, state.door_row,
                     state.door_open);
  }
  set_dir_to_atoms(atoms, 9, agent, door, state.door_row, state.door_open);
  set_dir_to_atoms(atoms, 13, agent, goal, state.door_row, state.door_open);

  return atoms;
}

EmitConfig keydoor_emit_config() {
  EmitConfig cfg;
  cfg.action_names = {"move(up)", "move(down)", "move(left)", "move(right)", "pickup",
                      "toggle"};
  cfg.atoms = {
      AtomSchema{"on_key", {}, false, ""},
      AtomSchema{"door_open", {}, false, ""},
      AtomSchema{"same_room_goal", {}, false, ""},
      AtomSchema{"adj_door", {}, false, ""},
      AtomSchema{"carrying", {}, false, ""},
  };
  for (const char* obj : kObjects) {
    for (const char* dir : kDirections) {
      cfg.atoms.push_back(AtomSchema{"dir_to", {obj}, false, dir});
    }
  }
  return cfg;
}

}  // namespace solverrl
