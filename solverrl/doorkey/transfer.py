"""DoorKey dataset collection, relational distill, and size-transfer evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from solverrl.doorkey import (
    ACTION_NAMES,
    ATOM_NAMES,
    NUM_ACTIONS,
    NUM_ATOMS,
    absolute_features,
    expert_action,
    extract_world,
    ground_atoms,
    make_doorkey_env,
)


@dataclass(frozen=True)
class TransferMetrics:
    size: int
    n_episodes: int
    mean_return: float
    success_rate: float
    mean_steps: float


@dataclass(frozen=True)
class DoorKeyTransferReport:
    train_size: int
    n_train_transitions: int
    distill_fidelity: float
    n_clauses: int
    prolog: str
    relational: Dict[int, TransferMetrics]
    tree: Dict[int, TransferMetrics]


def collect_expert_dataset(
    size: int = 8,
    *,
    n_episodes: int = 80,
    seed: int = 0,
    max_steps: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll out the scripted expert; return atoms, actions, absolute features."""
    env = make_doorkey_env(size, seed=seed)
    atoms_rows: List[np.ndarray] = []
    action_rows: List[int] = []
    abs_rows: List[np.ndarray] = []

    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        for _ in range(max_steps):
            world = extract_world(env)
            atoms = ground_atoms(world)
            action = int(expert_action(world))
            atoms_rows.append(atoms)
            action_rows.append(action)
            abs_rows.append(absolute_features(world))
            _obs, _r, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
    env.close()
    return (
        np.stack(atoms_rows, axis=0),
        np.asarray(action_rows, dtype=np.int64),
        np.stack(abs_rows, axis=0),
    )


def distill_relational(
    atoms: np.ndarray,
    actions: np.ndarray,
    *,
    min_score: float = 1e-12,
    max_body_literals: int = 3,
):
    import solverrl_core

    learner = solverrl_core.RuleLearner.from_schema(
        NUM_ATOMS,
        NUM_ACTIONS,
        list(ATOM_NAMES),
        list(ACTION_NAMES),
        min_score,
        max_body_literals,
    )
    learner.fit(np.ascontiguousarray(atoms, dtype=np.uint8), np.ascontiguousarray(actions, dtype=np.int64))
    fidelity = float(learner.fidelity(atoms, actions))
    return learner, fidelity


def _predict_relational(learner, env) -> int:
    world = extract_world(env)
    atoms = ground_atoms(world)[None, :]
    return int(learner.predict(np.ascontiguousarray(atoms, dtype=np.uint8))[0])


def evaluate_relational_policy(
    learner,
    size: int,
    *,
    n_episodes: int = 40,
    seed: int = 1000,
    max_steps: int = 256,
) -> TransferMetrics:
    env = make_doorkey_env(size, seed=seed)
    returns: List[float] = []
    successes = 0
    steps_list: List[int] = []

    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        total = 0.0
        steps = 0
        success = False
        for _ in range(max_steps):
            action = _predict_relational(learner, env)
            _obs, reward, terminated, truncated, _info = env.step(action)
            total += float(reward)
            steps += 1
            if terminated:
                success = True
                break
            if truncated:
                break
        returns.append(total)
        steps_list.append(steps)
        successes += int(success)
    env.close()
    return TransferMetrics(
        size=size,
        n_episodes=n_episodes,
        mean_return=float(np.mean(returns)),
        success_rate=successes / max(n_episodes, 1),
        mean_steps=float(np.mean(steps_list)),
    )


def fit_coordinate_tree(features: np.ndarray, actions: np.ndarray, *, max_depth: int = 12):
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    clf.fit(features, actions)
    return clf


def evaluate_tree_policy(
    clf,
    size: int,
    *,
    n_episodes: int = 40,
    seed: int = 1000,
    max_steps: int = 256,
) -> TransferMetrics:
    env = make_doorkey_env(size, seed=seed)
    returns: List[float] = []
    successes = 0
    steps_list: List[int] = []

    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        total = 0.0
        steps = 0
        success = False
        for _ in range(max_steps):
            world = extract_world(env)
            feat = absolute_features(world)[None, :]
            action = int(clf.predict(feat)[0])
            _obs, reward, terminated, truncated, _info = env.step(action)
            total += float(reward)
            steps += 1
            if terminated:
                success = True
                break
            if truncated:
                break
        returns.append(total)
        steps_list.append(steps)
        successes += int(success)
    env.close()
    return TransferMetrics(
        size=size,
        n_episodes=n_episodes,
        mean_return=float(np.mean(returns)),
        success_rate=successes / max(n_episodes, 1),
        mean_steps=float(np.mean(steps_list)),
    )


def run_doorkey_transfer(
    *,
    train_size: int = 8,
    eval_sizes: Sequence[int] = (6, 8, 16),
    n_train_episodes: int = 80,
    n_eval_episodes: int = 40,
    seed: int = 0,
    max_body_literals: int = 3,
    out_prolog: Optional[Path] = None,
) -> DoorKeyTransferReport:
    atoms, actions, abs_feats = collect_expert_dataset(
        train_size, n_episodes=n_train_episodes, seed=seed
    )
    learner, fidelity = distill_relational(atoms, actions, max_body_literals=max_body_literals)
    prolog = learner.to_prolog()
    if out_prolog is not None:
        out_prolog = Path(out_prolog)
        out_prolog.parent.mkdir(parents=True, exist_ok=True)
        out_prolog.write_text(prolog, encoding="utf-8")

    tree = fit_coordinate_tree(abs_feats, actions)

    relational: Dict[int, TransferMetrics] = {}
    tree_metrics: Dict[int, TransferMetrics] = {}
    for size in eval_sizes:
        relational[size] = evaluate_relational_policy(
            learner, size, n_episodes=n_eval_episodes, seed=seed + 10_000
        )
        tree_metrics[size] = evaluate_tree_policy(
            tree, size, n_episodes=n_eval_episodes, seed=seed + 10_000
        )

    return DoorKeyTransferReport(
        train_size=train_size,
        n_train_transitions=int(len(actions)),
        distill_fidelity=fidelity,
        n_clauses=int(learner.n_clauses),
        prolog=prolog,
        relational=relational,
        tree=tree_metrics,
    )


def format_transfer_table(report: DoorKeyTransferReport) -> str:
    lines = [
        f"DoorKey transfer (train {report.train_size}x{report.train_size}, "
        f"transitions={report.n_train_transitions}, fidelity={report.distill_fidelity:.3f}, "
        f"clauses={report.n_clauses})",
        "",
        f"{'size':>6}  {'rel_succ':>8}  {'rel_ret':>8}  {'tree_succ':>9}  {'tree_ret':>8}",
    ]
    for size in sorted(report.relational.keys()):
        r = report.relational[size]
        t = report.tree[size]
        lines.append(
            f"{size:>6}  {r.success_rate:8.3f}  {r.mean_return:8.3f}  "
            f"{t.success_rate:9.3f}  {t.mean_return:8.3f}"
        )
    return "\n".join(lines)
