"""Exact finite-MDP evaluation for KeyDoor (Proposition 2 style)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from solverrl.envs.keydoor import (
    ACTIONS,
    DONE,
    GAMMA,
    HORIZON,
    State,
    enumerate_reachable,
    initial_states,
    transition,
)


@dataclass
class ExactMDP:
    states: List[State]
    index: Dict[State, int]
    # P[a, s, s'] and r[a, s]
    P: np.ndarray
    r: np.ndarray
    mu0: np.ndarray
    gamma: float = GAMMA

    @property
    def n(self) -> int:
        return len(self.states)

    @classmethod
    def from_keydoor(cls, gamma: float = GAMMA) -> "ExactMDP":
        states = enumerate_reachable(include_done=True)
        index = {s: i for i, s in enumerate(states)}
        n = len(states)
        A = len(ACTIONS)
        P = np.zeros((A, n, n), dtype=np.float64)
        r = np.zeros((A, n), dtype=np.float64)
        for s in states:
            i = index[s]
            for a in range(A):
                tr = transition(s, a)
                j = index[tr.next_state]
                P[a, i, j] = 1.0
                r[a, i] = tr.reward

        mu0 = np.zeros(n, dtype=np.float64)
        initials = initial_states()
        for s in initials:
            mu0[index[s]] += 1.0
        mu0 /= mu0.sum()
        return cls(states=states, index=index, P=P, r=r, mu0=mu0, gamma=gamma)

    def policy_matrices(self, policy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """policy: shape (n,) int actions. Returns P_pi (n,n), r_pi (n,)."""
        policy = np.asarray(policy, dtype=np.int64)
        assert policy.shape == (self.n,)
        P_pi = np.zeros((self.n, self.n), dtype=np.float64)
        r_pi = np.zeros(self.n, dtype=np.float64)
        for i, a in enumerate(policy):
            P_pi[i] = self.P[a, i]
            r_pi[i] = self.r[a, i]
        return P_pi, r_pi

    def exact_value(self, policy: np.ndarray) -> np.ndarray:
        P_pi, r_pi = self.policy_matrices(policy)
        a = np.eye(self.n) - self.gamma * P_pi
        return np.linalg.solve(a, r_pi)

    def exact_return(self, policy: np.ndarray) -> float:
        v = self.exact_value(policy)
        return float(self.mu0 @ v)

    def cpp_evaluator(self, horizon: int = HORIZON):
        """C++ ExactEvaluator over this MDP (for EXPAND / certificate)."""
        import solverrl_core

        return solverrl_core.ExactEvaluator(
            np.ascontiguousarray(self.P, dtype=np.float64),
            np.ascontiguousarray(self.r, dtype=np.float64),
            np.ascontiguousarray(self.mu0, dtype=np.float64),
            float(self.gamma),
            int(self.index[DONE]),
            int(horizon),
        )

    def advantage_gap_cert(
        self, student: np.ndarray, teacher: np.ndarray, horizon: int = HORIZON
    ) -> dict:
        """Teacher-Q advantage gap certificate on disagreeing states."""
        evalr = self.cpp_evaluator(horizon=horizon)
        cert = evalr.advantage_gap_cert(
            np.asarray(student, dtype=np.int64),
            np.asarray(teacher, dtype=np.int64),
        )
        return {
            "weighted_gap": float(cert.weighted_gap),
            "return_gap": float(cert.return_gap),
            "n_disagree": int(cert.n_disagree),
            "max_gap": float(cert.max_gap),
        }


def success_within_horizon(mdp: ExactMDP, policy: np.ndarray, horizon: int = HORIZON) -> float:
    """Probability of hitting DONE within `horizon` steps from μ0 (finite-horizon reachability)."""
    policy = np.asarray(policy, dtype=np.int64)
    done_i = mdp.index[DONE]
    # p[t, s] unused; iterate distribution mass on non-done
    mass = mdp.mu0.copy()
    success = 0.0
    for _ in range(horizon):
        next_mass = np.zeros_like(mass)
        for i, a in enumerate(policy):
            if mass[i] == 0.0:
                continue
            if i == done_i:
                continue
            next_mass += mass[i] * mdp.P[a, i]
        success += next_mass[done_i]
        next_mass[done_i] = 0.0
        mass = next_mass
        if mass.sum() < 1e-15:
            break
    return float(success)
