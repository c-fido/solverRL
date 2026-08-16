# SolverRL

Distill a frozen PPO teacher into an executable Prolog decision list, then expand it under an **exact** return oracle on KeyDoor (Garrido-Merchán, *From Black Box to Executable Logic*, [arXiv:2607.15459v2](https://arxiv.org/abs/2607.15459)).

C++ (`solverrl_core`) owns induction, exact evaluation, EXPAND, and the advantage-gap certificate. Python owns the env, PPO, census, plotting, and CLI.

---

## Proof sketch

### Exact return (Proposition 2 style)

KeyDoor is a finite discounted MDP \((S_r, A, P, r, \mu_0, \gamma)\) with known model access. For a deterministic policy \(\pi: S_r \to A\),

\[
V^\pi = (I - \gamma P^\pi)^{-1} r^\pi, \qquad
J(\pi) = \mu_0^\top V^\pi.
\]

We also report finite-horizon success (probability of hitting absorbing `DONE` within horizon \(H=120\)). Both quantities are computed by the C++ `ExactEvaluator` (linear solve + mass propagation)—not Monte Carlo. That is the KeyDoor certificate backbone.

### EXPAND termination (Proposition 1)

**Loop.** From a decision list \(\Pi_0\), at each iteration propose local edits (specialize, reorder, prune, retarget-head, add-clause). Accept the edit that maximizes exact \(\Delta J = J(\Pi') - J(\Pi)\) among proposals with \(\Delta J \ge \tau > 0\). Stop when no such edit exists, or after a hard iteration cap.

**Invariant.** After every accepted edit,

\[
J(\Pi_{t+1}) \ge J(\Pi_t) + \tau.
\]

So the certified-return curve is **monotone non-decreasing**.

**Termination.** Rewards are bounded, so \(J(\pi)\) is bounded above (for KeyDoor, \(J \le O(1)\) under \(\gamma < 1\)). A strictly increasing sequence with steps \(\ge \tau\) cannot be infinite. Hence EXPAND terminates after finitely many accepted edits (the iteration cap is only a practical backstop).

**What we do *not* claim.** Termination does not imply global optimality over all Prolog programs—only that hill-climbing under the exact oracle stops. R1 may accept zero edits when distill already solves; R2 shows the interesting case where one accepted edit lifts success \(0 \to 1\).

### Advantage-gap certificate (Proposition 3 style)

Compare student \(\pi_S\) to teacher \(\pi_T\) (census greedy labels / checkpoint):

| Field | Meaning |
|--------|---------|
| `return_gap` | \(J(\pi_T) - J(\pi_S)\) under the exact evaluator |
| `n_disagree` | \(\lvert\{s : \pi_S(s) \ne \pi_T(s)\}\rvert\) |
| `weighted_gap` | \(\sum_{s:\,\pi_S(s)\ne\pi_T(s)} \mu_0(s)\,(Q^{\pi_T}(s,\pi_T(s)) - Q^{\pi_T}(s,\pi_S(s)))\) |
| `max_gap` | max per-state teacher-Q gap on disagreements |
| `non_vacuous` | `n_disagree > 0` |

**Reading the numbers.**

- `return_gap < 0` means the student has **higher** exact return than the teacher (common on R2 after EXPAND; also possible on R1 when shared-`D` navigation beats PPO tie-breaks).
- `weighted_gap` / `max_gap` are machine-checkable summaries of teacher-advantage on states where the policies differ; they are **not** a substitute for \(J\), but they make disagreement interpretable.
- A vacuous cert (`n_disagree = 0`) would mean identical action maps; our R1/R2 runs are non-vacuous because canonical `dir_to` need not match PPO’s choice among shortest-path ties.

Implemented in `ExactEvaluator::advantage_gap_cert` and printed by `python -m solverrl.run_r1` (optional `--out-cert`).

---

## Seed0 KeyDoor results (ship gate)

| Regime | Distill fidelity | EXPAND iters | Exact \(J\) | Exact success | Teacher success |
|--------|------------------|--------------|-------------|---------------|-----------------|
| R1 | 0.62 | 0 | 0.82 → 0.82 | 1.0 → 1.0 | 0.956 |
| R2 | 0.33 | 1 | −1.0 → 0.82 | 0 → 1.0 | 0.222 |

R1: distill already at exact success 1; EXPAND no-ops (paper-compatible).  
R2: undertrained teacher; one accepted edit recovers exact success 1 and beats the teacher.

Certified-return plot: `data/plots/certified_return_r1_r2.png`.

---

## DoorKey relational transfer (core #2)

Distill a Prolog decision list from a scripted expert on MiniGrid **DoorKey-8×8**, then evaluate the **unchanged** program on 6×6 and 16×16. Perception is size-invariant: phase-gated `nav_{left,right,forward}` (shortest path to the active subgoal) plus `front_*` / `carrying` / door state. Baseline: a sklearn tree over **absolute coordinates** (agent/key/door/goal xy)—strong on the train size, collapses elsewhere.

Seed0 (`--train-episodes 80 --eval-episodes 40`):

| Size | Relational success | Relational return | Tree success | Tree return |
|------|--------------------|-------------------|--------------|-------------|
| 6×6 | 1.000 | 0.970 | 0.000 | 0.000 |
| 8×8 (train) | 1.000 | 0.975 | 0.425 | 0.415 |
| 16×16 | 1.000 | 0.988 | 0.000 | 0.000 |

Fidelity 1.0, 6 clauses (`data/rules/doorkey_8x8.pl`). Run: `python -m solverrl.doorkey.run_transfer`.

---

## Quick start

CI runs on **Ubuntu** (`ubuntu-latest`): CMake + GoogleTest, then editable install + `pytest` (excludes `@pytest.mark.slow`). Local Windows needs VS C++ build tools + Python headers; if a rebuild fails with “Access is denied” on `solverrl_core*.pyd`, close processes holding the DLL (or rename it) and retry.

```powershell
# venv + editable install
.venv\Scripts\Activate.ps1
pip install -e . --no-build-isolation

# Tests: default skips slow DoorKey transfer; include everything with: pytest -m ""
pytest -q
# Optional long transfer only:
pytest -q -m slow

# One command: load (or train) → census → distill → EXPAND → cert/plot
# Reuses data/census/* if present; add --train to retrain teachers.
python -m solverrl.pipeline --seed 0 --regimes r1,r2 --max-iterations 50

# Single regime only
python -m solverrl.pipeline --regimes r2 --max-iterations 50

# Force retrain both teachers + censuses, then expand
python -m solverrl.pipeline --train-both --seed 0 --max-iterations 50

# DoorKey transfer: distill on 8x8, eval on 6/8/16 vs coordinate trees
python -m solverrl.doorkey.run_transfer --train-episodes 80 --eval-episodes 40
```

Linux / macOS (same flags CI uses):

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional CPU-only
pip install -r requirements.txt
pip install -e . --no-build-isolation
cmake -S . -B build && cmake --build build -j && ctest --test-dir build --output-on-failure
pytest -q
```

Outputs under `data/`:

| Path | Content |
|------|---------|
| `census/` | checkpoints + census `.npz` |
| `rules/*_expanded.pl` | Prolog decision lists |
| `certs/*_advantage_gap.json` | advantage-gap certificate |
| `plots/*_certified.png` | per-regime curves |
| `plots/certified_return_seed0.png` | R1 vs R2 comparison |
| `plots/pipeline_summary_seed0.json` | metrics table |
| `rules/doorkey_8x8.pl` | DoorKey relational program |
| `plots/doorkey_transfer.json` | DoorKey transfer table |

```powershell
# Or run a single census expand (lower-level CLI)
python -m solverrl.run_r1 data/census/census_r1_seed0.npz `
  --teacher-ckpt data/census/checkpoints/ppo_keydoor_r1_seed0.zip `
  --out-prolog data/rules/r1_expanded.pl `
  --out-cert data/certs/r1_advantage_gap.json `
  --out-plot data/plots/r1.png `
  --max-iterations 50

# Re-plot R1 vs R2 without re-running EXPAND
python -m solverrl.plot_expand `
  --out data/plots/certified_return_r1_r2.png `
  --series "R1:0.815778:1.0:0.956163" `
  --series "R2:-1.0,0.815778:0.0,1.0:0.221788"
```

Regenerate teachers/censuses only: `python -m solverrl.teachers.census --out-dir data/census --seed 0`.

---

## Layout

| Path | Role |
|------|------|
| `src/foil.cpp`, `rule_learner.cpp` | FOIL covering + shared-`D` clauses |
| `src/exact_eval.cpp` | Exact \(J\), success, advantage-gap cert |
| `src/expand.cpp` | Edit proposals + acceptance loop |
| `src/keydoor_ground.cpp` | Relational atom grounding (`dir_to`, …) |
| `solverrl/run_r1.py` | Distill → EXPAND → cert / plot CLI |
| `solverrl/pipeline.py` | One-command: census → distill → EXPAND → cert/plot |
| `solverrl/plot_expand.py` | Monotone certified-return curves |
| `solverrl/doorkey/` | MiniGrid DoorKey grounding + transfer CLI |
| `TASKS.md` | Build checklist |

**Boundary.** Fast / certified logic stays in C++. Env I/O, PPO, DoorKey rollouts, and plots stay in Python. CartPole (if added) is an honesty check only—not the core claim.
