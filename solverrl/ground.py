"""Encode KeyDoor states for C++ grounding."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from solverrl.envs.keydoor import DONE, State

KEYDOOR_STATE_FIELDS = 9


def encode_state(state: State) -> np.ndarray:
    """Row vector: door_row, agent_r/c, key_r/c, goal_r/c, door_open, carrying."""
    if state == DONE:
        return np.array([-1, -1, -1, -1, -1, -1, -1, 0, 0], dtype=np.int32)
    door_row, agent, key, goal, door_open, carrying = state
    kr, kc = key if key is not None else (-1, -1)
    return np.array(
        [
            door_row,
            agent[0],
            agent[1],
            kr,
            kc,
            goal[0],
            goal[1],
            int(door_open),
            int(carrying),
        ],
        dtype=np.int32,
    )


def encode_states(states: Sequence[State]) -> np.ndarray:
    """Shape (n, 9) int32."""
    rows = [encode_state(s) for s in states]
    return np.stack(rows, axis=0)


def ground_atoms(states: Sequence[State]) -> np.ndarray:
    """Ground states to atom matrix (n, KEYDOOR_NUM_ATOMS) uint8 via C++."""
    import solverrl_core

    encoded = encode_states(states)
    return np.asarray(solverrl_core.ground_keydoor_states(encoded), dtype=np.uint8)
