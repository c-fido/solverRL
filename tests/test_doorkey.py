"""DoorKey relational grounding and expert tests."""

from __future__ import annotations

import numpy as np
import pytest

from solverrl.doorkey import (
    NUM_ATOMS,
    expert_action,
    extract_world,
    ground_atoms,
    make_doorkey_env,
)


def test_make_doorkey_sizes():
    for size in (6, 8):
        env = make_doorkey_env(size, seed=0)
        assert env.unwrapped.width == size
        env.close()


def test_expert_solves_8x8():
    env = make_doorkey_env(8, seed=0)
    successes = 0
    for ep in range(20):
        env.reset(seed=ep)
        for _ in range(256):
            world = extract_world(env)
            action = expert_action(world)
            _o, _r, term, trunc, _i = env.step(action)
            if term:
                successes += 1
                break
            if trunc:
                break
    env.close()
    assert successes >= 18


def test_ground_atoms_shape_and_range():
    env = make_doorkey_env(8, seed=1)
    env.reset(seed=1)
    atoms = ground_atoms(extract_world(env))
    env.close()
    assert atoms.shape == (NUM_ATOMS,)
    assert atoms.dtype == np.uint8
    assert set(np.unique(atoms)).issubset({0, 1})


def test_active_nav_is_exclusive():
    from solverrl.doorkey import ATOM_NAMES

    env = make_doorkey_env(8, seed=2)
    env.reset(seed=2)
    atoms = ground_atoms(extract_world(env))
    env.close()
    nav = [atoms[ATOM_NAMES.index(n)] for n in ("nav_left", "nav_right", "nav_forward")]
    assert sum(nav) <= 1


@pytest.mark.slow
def test_doorkey_transfer_relational_beats_tree_offsize():
    from solverrl.doorkey.transfer import run_doorkey_transfer

    report = run_doorkey_transfer(
        train_size=8,
        eval_sizes=(6, 8, 16),
        n_train_episodes=40,
        n_eval_episodes=20,
        seed=0,
    )
    assert report.distill_fidelity >= 0.85
    assert report.relational[8].success_rate >= 0.7
    # Relational should transfer; absolute trees should collapse off training size.
    assert report.relational[6].success_rate >= 0.5
    assert report.relational[16].success_rate >= 0.4
    assert report.tree[6].success_rate <= 0.2
    assert report.tree[16].success_rate <= report.relational[16].success_rate
