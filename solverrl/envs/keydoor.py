"""KeyDoor finite MDP (Garrido-Merchán, arXiv:2607.15459v2).

Four-row grid: left room 3 cols, wall with one door cell, right room 2 cols.
Actions: cardinal moves, pickup, toggle. Exact model access for census / Bellman solves.

Note: the paper reports |S_r|=16,944. A prose-faithful reimplementation yields a smaller
closed reachable set (~5.8k including DONE). Dynamics/layout details beyond the paper text
may differ from the authors' private env; we keep exact eval correct for *this* model and
track the cardinality gap explicitly.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

GAMMA = 0.99
HORIZON = 120
STEP_PENALTY = -0.01
SUCCESS_REWARD = 1.0
R_MAX = abs(SUCCESS_REWARD + STEP_PENALTY)  # 0.99

HEIGHT = 4
LEFT_COLS = (0, 1, 2)
WALL_COL = 3
RIGHT_COLS = (4, 5)
WIDTH = 6

ACTIONS: Tuple[str, ...] = ("up", "down", "left", "right", "pickup", "toggle")
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}
MOVE_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

Pos = Tuple[int, int]
# door_row, agent, key_or_none, goal, door_open, carrying
State = Tuple[int, Pos, Optional[Pos], Pos, bool, bool]
DONE: State = (-1, (-1, -1), None, (-1, -1), False, False)


def left_cells() -> List[Pos]:
    return [(r, c) for r in range(HEIGHT) for c in LEFT_COLS]


def right_cells() -> List[Pos]:
    return [(r, c) for r in range(HEIGHT) for c in RIGHT_COLS]


def door_cell(door_row: int) -> Pos:
    return (door_row, WALL_COL)


def is_walkable(pos: Pos, door_row: int, door_open: bool) -> bool:
    r, c = pos
    if not (0 <= r < HEIGHT and 0 <= c < WIDTH):
        return False
    if c in LEFT_COLS:
        return True
    if c in RIGHT_COLS:
        # Reachable only after opening; once there, movement stays valid even if closed again.
        return True
    if c == WALL_COL:
        return r == door_row and door_open
    return False


def adjacent(a: Pos, b: Pos) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


@dataclass(frozen=True)
class Transition:
    next_state: State
    reward: float
    terminated: bool


def initial_states() -> List[State]:
    states: List[State] = []
    for door_row in range(HEIGHT):
        for agent in left_cells():
            for key in left_cells():
                for goal in right_cells():
                    states.append((door_row, agent, key, goal, False, False))
    return states


def transition(state: State, action: int | str) -> Transition:
    if state == DONE:
        return Transition(DONE, 0.0, True)

    if isinstance(action, str):
        action = ACTION_TO_ID[action]
    name = ACTIONS[action]

    door_row, agent, key, goal, door_open, carrying = state
    door = door_cell(door_row)

    if name in MOVE_DELTA:
        dr, dc = MOVE_DELTA[name]
        nxt = (agent[0] + dr, agent[1] + dc)
        if not is_walkable(nxt, door_row, door_open):
            return Transition(state, STEP_PENALTY, False)
        if nxt == goal:
            return Transition(DONE, SUCCESS_REWARD + STEP_PENALTY, True)
        return Transition(
            (door_row, nxt, key, goal, door_open, carrying),
            STEP_PENALTY,
            False,
        )

    if name == "pickup":
        if (not carrying) and key is not None and agent == key:
            return Transition(
                (door_row, agent, None, goal, door_open, True),
                STEP_PENALTY,
                False,
            )
        return Transition(state, STEP_PENALTY, False)

    if name == "toggle":
        if carrying and adjacent(agent, door):
            return Transition(
                (door_row, agent, key, goal, not door_open, carrying),
                STEP_PENALTY,
                False,
            )
        return Transition(state, STEP_PENALTY, False)

    raise ValueError(f"unknown action {action}")


def successors(state: State) -> Iterable[State]:
    for a in range(len(ACTIONS)):
        yield transition(state, a).next_state


def enumerate_reachable(include_done: bool = True) -> List[State]:
    """BFS from μ0 support; S_r closed under all actions (Assumption 3)."""
    start = initial_states()
    seen = set(start)
    q = deque(start)
    while q:
        s = q.popleft()
        for nxt in successors(s):
            if nxt not in seen:
                seen.add(nxt)
                if nxt != DONE:
                    q.append(nxt)
    if include_done:
        seen.add(DONE)
    else:
        seen.discard(DONE)

    def _sort_key(s: State):
        door_row, agent, key, goal, door_open, carrying = s
        return (door_row, agent, key is None, key or (-1, -1), goal, door_open, carrying)

    ordered = sorted((s for s in seen if s != DONE), key=_sort_key)
    if include_done and DONE in seen:
        ordered.append(DONE)
    return ordered


def env_source_sha256() -> str:
    path = Path(__file__).resolve()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_to_obs(state: State, n_states: int, index: Dict[State, int]) -> np.ndarray:
    """Flat one-hot over reachable index (paper: flat one-hot of the state)."""
    obs = np.zeros(n_states, dtype=np.float32)
    obs[index[state]] = 1.0
    return obs


# Shared across VecEnv workers so we do not re-BFS the census per env instance.
_REACHABLE_CACHE: Optional[List[State]] = None
_INITIALS_CACHE: Optional[List[State]] = None


def _cached_reachable() -> List[State]:
    global _REACHABLE_CACHE
    if _REACHABLE_CACHE is None:
        _REACHABLE_CACHE = enumerate_reachable(include_done=True)
    return _REACHABLE_CACHE


def _cached_initials() -> List[State]:
    global _INITIALS_CACHE
    if _INITIALS_CACHE is None:
        _INITIALS_CACHE = initial_states()
    return _INITIALS_CACHE


class KeyDoorEnv(gym.Env):
    """Gymnasium KeyDoor with exact-model helpers."""

    metadata = {"render_modes": []}

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.action_space = spaces.Discrete(len(ACTIONS))
        self._reachable = _cached_reachable()
        self._index = {s: i for i, s in enumerate(self._reachable)}
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(len(self._reachable),), dtype=np.float32
        )
        self._state: State = _cached_initials()[0]
        self._steps = 0
        self._rng = np.random.default_rng(seed)
        self._initials = _cached_initials()

    @property
    def reachable_states(self) -> Sequence[State]:
        return self._reachable

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = self._initials[int(self._rng.integers(0, len(self._initials)))]
        self._steps = 0
        obs = state_to_obs(self._state, len(self._reachable), self._index)
        return obs, {"state": self._state}

    def step(self, action: int):
        tr = transition(self._state, int(action))
        self._state = tr.next_state
        self._steps += 1
        terminated = tr.terminated
        truncated = (not terminated) and (self._steps >= HORIZON)
        obs = state_to_obs(self._state, len(self._reachable), self._index)
        return obs, float(tr.reward), terminated, truncated, {"state": self._state}

    def transition_model(self, state: State, action: int) -> Transition:
        return transition(state, action)
