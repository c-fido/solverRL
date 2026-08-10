#include <gtest/gtest.h>

#include "solverrl/keydoor_ground.hpp"

using solverrl::KeyDoorState;
using solverrl::decode_keydoor_state;
using solverrl::ground_atoms;
using solverrl::kKeyDoorNumAtoms;

namespace {

KeyDoorState make_state(int door_row, int ar, int ac, int kr, int kc, int gr, int gc,
                        bool door_open, bool carrying) {
  KeyDoorState s;
  s.door_row = door_row;
  s.agent_r = ar;
  s.agent_c = ac;
  s.key_r = kr;
  s.key_c = kc;
  s.goal_r = gr;
  s.goal_c = gc;
  s.door_open = door_open;
  s.carrying = carrying;
  return s;
}

}  // namespace

TEST(KeyDoorGround, DoneStateHasAllFalseAtoms) {
  const int fields[] = {-1, -1, -1, -1, -1, -1, -1, 0, 0};
  const auto s = decode_keydoor_state(fields);
  const auto atoms = ground_atoms(s);
  ASSERT_EQ(atoms.size(), static_cast<std::size_t>(kKeyDoorNumAtoms));
  for (bool v : atoms) {
    EXPECT_FALSE(v);
  }
}

TEST(KeyDoorGround, OnKeyAndCarryingPerception) {
  // door_row=0, agent on key at (0,0), goal far, door closed, not carrying
  auto s = make_state(0, 0, 0, 0, 0, 0, 4, false, false);
  auto atoms = ground_atoms(s);
  EXPECT_TRUE(atoms[0]);   // on_key
  EXPECT_FALSE(atoms[4]);  // carrying

  s.carrying = true;
  atoms = ground_atoms(s);
  EXPECT_TRUE(atoms[4]);
}

TEST(KeyDoorGround, AdjDoorWhenBesideWallDoor) {
  // door at (1,3); agent at (1,2) adjacent
  auto s = make_state(1, 1, 2, 0, 0, 0, 4, false, true);
  const auto atoms = ground_atoms(s);
  EXPECT_TRUE(atoms[3]);  // adj_door
}

TEST(KeyDoorGround, DirToGoalIncludesShortestStep) {
  // agent (0,0), goal (0,4) in right room — need door open to reach; with door open,
  // right is a valid first step toward goal once path exists through door.
  auto s = make_state(0, 0, 0, 0, 1, 0, 4, true, true);
  const auto atoms = ground_atoms(s);
  // At least one dir_to(goal, *) should be true when goal is reachable.
  const bool any_goal_dir = atoms[13] || atoms[14] || atoms[15] || atoms[16];
  EXPECT_TRUE(any_goal_dir);
}

TEST(KeyDoorGround, DirToIsFunctionalSingleDirection) {
  auto s = make_state(0, 0, 0, 2, 1, 0, 4, false, false);
  const auto atoms = ground_atoms(s);
  int key_dirs = 0;
  for (int d = 0; d < 4; ++d) {
    if (atoms[static_cast<std::size_t>(5 + d)]) {
      ++key_dirs;
    }
  }
  EXPECT_EQ(key_dirs, 1);
}

TEST(KeyDoorGround, DirToDoorWorksWhenDoorClosed) {
  // Agent in left room with key; closed door must still yield a navigation dir.
  auto s = make_state(/*door_row=*/1, /*ar=*/0, /*ac=*/0, /*kr=*/-1, /*kc=*/-1,
                      /*gr=*/0, /*gc=*/4, /*door_open=*/false, /*carrying=*/true);
  const auto atoms = ground_atoms(s);
  EXPECT_FALSE(atoms[3]);  // not yet adjacent
  const int door_dirs =
      static_cast<int>(atoms[9]) + atoms[10] + atoms[11] + atoms[12];
  EXPECT_EQ(door_dirs, 1);
}

TEST(KeyDoorGround, DirToKeyPrefersShortestStepNotArbitraryNeighbor) {
  // agent (0,0), key (0,1) → must be right (index 3 within key block 5..8).
  auto s = make_state(0, 0, 0, 0, 1, 0, 4, false, false);
  const auto atoms = ground_atoms(s);
  EXPECT_FALSE(atoms[5]);  // up
  EXPECT_FALSE(atoms[6]);  // down
  EXPECT_FALSE(atoms[7]);  // left
  EXPECT_TRUE(atoms[8]);   // right
}
