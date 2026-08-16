"""MiniGrid DoorKey: env factory, relational grounding, scripted expert."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# MiniGrid action ids
ACT_LEFT = 0
ACT_RIGHT = 1
ACT_FORWARD = 2
ACT_PICKUP = 3
ACT_DROP = 4
ACT_TOGGLE = 5
ACT_DONE = 6

ACTION_NAMES = ("left", "right", "forward", "pickup", "drop", "toggle", "done")

ATOM_NAMES = (
    "carrying",
    "door_open",
    "door_locked",
    "front_key",
    "front_door",
    "front_goal",
    "front_wall",
    # Active-subgoal nav only (matches expert phase priority). Emitting
    # key/door/goal nav simultaneously lets FOIL latch onto spurious
    # \+nav_key_* correlations that break closed-loop control.
    "nav_left",
    "nav_right",
    "nav_forward",
)

NUM_ATOMS = len(ATOM_NAMES)
NUM_ACTIONS = len(ACTION_NAMES)

DIR_TO_VEC = ((1, 0), (0, 1), (-1, 0), (0, -1))  # right, down, left, up


def make_doorkey_env(size: int = 8, *, seed: Optional[int] = None, fully_obs: bool = True):
    """Create MiniGrid-DoorKey-{size}x{size}-v0 (optionally fully observable)."""
    import gymnasium as gym
    import minigrid  # noqa: F401 — registers envs
    from minigrid.wrappers import FullyObsWrapper

    if size not in (5, 6, 8, 16):
        raise ValueError(f"unsupported DoorKey size {size}")
    env = gym.make(f"MiniGrid-DoorKey-{size}x{size}-v0")
    if fully_obs:
        env = FullyObsWrapper(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


@dataclass(frozen=True)
class DoorKeyWorld:
    width: int
    height: int
    agent_pos: Tuple[int, int]
    agent_dir: int
    carrying_key: bool
    key_pos: Optional[Tuple[int, int]]
    door_pos: Optional[Tuple[int, int]]
    door_state: int  # 0 open, 1 closed, 2 locked
    goal_pos: Optional[Tuple[int, int]]
    walls: Tuple[Tuple[int, int], ...]

    @property
    def door_open(self) -> bool:
        return self.door_state == 0

    @property
    def door_locked(self) -> bool:
        return self.door_state == 2


def extract_world(env) -> DoorKeyWorld:
    """Read relational world state from a (FullyObs) MiniGrid DoorKey env."""
    u = env.unwrapped
    grid = u.grid
    walls: List[Tuple[int, int]] = []
    key_pos = None
    door_pos = None
    door_state = 1
    goal_pos = None

    for x in range(u.width):
        for y in range(u.height):
            cell = grid.get(x, y)
            if cell is None:
                continue
            typ = cell.type
            if typ == "wall":
                walls.append((x, y))
            elif typ == "key":
                key_pos = (x, y)
            elif typ == "door":
                door_pos = (x, y)
                if cell.is_open:
                    door_state = 0
                elif cell.is_locked:
                    door_state = 2
                else:
                    door_state = 1
            elif typ == "goal":
                goal_pos = (x, y)

    carrying = u.carrying is not None and getattr(u.carrying, "type", None) == "key"
    ax, ay = int(u.agent_pos[0]), int(u.agent_pos[1])
    return DoorKeyWorld(
        width=int(u.width),
        height=int(u.height),
        agent_pos=(ax, ay),
        agent_dir=int(u.agent_dir),
        carrying_key=bool(carrying),
        key_pos=key_pos,
        door_pos=door_pos,
        door_state=door_state,
        goal_pos=goal_pos,
        walls=tuple(walls),
    )


def _front_cell(world: DoorKeyWorld) -> Tuple[int, int]:
    dx, dy = DIR_TO_VEC[world.agent_dir]
    x, y = world.agent_pos
    return x + dx, y + dy


def _in_bounds(world: DoorKeyWorld, x: int, y: int) -> bool:
    return 0 <= x < world.width and 0 <= y < world.height


def _is_blocked(world: DoorKeyWorld, x: int, y: int) -> bool:
    if not _in_bounds(world, x, y):
        return True
    if (x, y) in world.walls:
        return True
    if world.door_pos == (x, y) and world.door_state != 0:
        return True
    return False


def _walkable(world: DoorKeyWorld, x: int, y: int) -> bool:
    return not _is_blocked(world, x, y)


def _successors(world: DoorKeyWorld, x: int, y: int, d: int):
    # turn left / right / forward
    yield ACT_LEFT, (x, y, (d - 1) % 4)
    yield ACT_RIGHT, (x, y, (d + 1) % 4)
    dx, dy = DIR_TO_VEC[d]
    nx, ny = x + dx, y + dy
    if _walkable(world, nx, ny):
        yield ACT_FORWARD, (nx, ny, d)


def nav_action_toward(
    world: DoorKeyWorld,
    target: Optional[Tuple[int, int]],
    *,
    face_only: bool = False,
) -> Optional[int]:
    """First MiniGrid action on a shortest path toward target (or facing it)."""
    if target is None:
        return None

    start = (world.agent_pos[0], world.agent_pos[1], world.agent_dir)

    def is_goal(state: Tuple[int, int, int]) -> bool:
        x, y, d = state
        if face_only:
            fx, fy = x + DIR_TO_VEC[d][0], y + DIR_TO_VEC[d][1]
            return (fx, fy) == target
        return (x, y) == target

    if is_goal(start):
        return None

    q = deque([start])
    came_from: Dict[Tuple[int, int, int], Tuple[Optional[Tuple[int, int, int]], Optional[int]]] = {
        start: (None, None)
    }
    while q:
        cur = q.popleft()
        x, y, d = cur
        for action, nxt in _successors(world, x, y, d):
            if nxt in came_from:
                continue
            came_from[nxt] = (cur, action)
            if is_goal(nxt):
                step = nxt
                while True:
                    prev, act = came_from[step]
                    if prev == start:
                        return act
                    if prev is None:
                        return None
                    step = prev
            q.append(nxt)
    return None


def active_nav_action(world: DoorKeyWorld) -> Optional[int]:
    """First nav action for the expert's current subgoal (or None if at face/target)."""
    if not world.carrying_key and world.key_pos is not None:
        return nav_action_toward(world, world.key_pos, face_only=True)
    if world.door_locked and world.door_pos is not None:
        return nav_action_toward(world, world.door_pos, face_only=True)
    if world.goal_pos is not None:
        return nav_action_toward(world, world.goal_pos, face_only=False)
    return None


def ground_atoms(world: DoorKeyWorld) -> np.ndarray:
    """Return (NUM_ATOMS,) uint8 relational feature vector."""
    atoms = np.zeros(NUM_ATOMS, dtype=np.uint8)
    atoms[0] = int(world.carrying_key)
    atoms[1] = int(world.door_open)
    atoms[2] = int(world.door_locked)

    fx, fy = _front_cell(world)
    atoms[3] = int(world.key_pos == (fx, fy))
    atoms[4] = int(world.door_pos == (fx, fy))
    atoms[5] = int(world.goal_pos == (fx, fy))
    atoms[6] = int(
        _is_blocked(world, fx, fy)
        and world.door_pos != (fx, fy)
        and world.key_pos != (fx, fy)
        and world.goal_pos != (fx, fy)
    )

    action = active_nav_action(world)
    if action == ACT_LEFT:
        atoms[7] = 1
    elif action == ACT_RIGHT:
        atoms[8] = 1
    elif action == ACT_FORWARD:
        atoms[9] = 1
    return atoms


def absolute_features(world: DoorKeyWorld) -> np.ndarray:
    """Absolute-coordinate features for the tree baseline (size-sensitive)."""
    kx, ky = world.key_pos if world.key_pos else (-1, -1)
    dx, dy = world.door_pos if world.door_pos else (-1, -1)
    gx, gy = world.goal_pos if world.goal_pos else (-1, -1)
    return np.asarray(
        [
            world.width,
            world.height,
            world.agent_pos[0],
            world.agent_pos[1],
            world.agent_dir,
            int(world.carrying_key),
            kx,
            ky,
            dx,
            dy,
            world.door_state,
            gx,
            gy,
        ],
        dtype=np.float64,
    )


def expert_action(world: DoorKeyWorld) -> int:
    """Scripted DoorKey expert using the same relational nav atoms."""
    fx, fy = _front_cell(world)

    if world.carrying_key and world.door_pos == (fx, fy) and world.door_locked:
        return ACT_TOGGLE
    if (not world.carrying_key) and world.key_pos == (fx, fy):
        return ACT_PICKUP

    a = active_nav_action(world)
    if a is not None:
        return a
    return ACT_LEFT
