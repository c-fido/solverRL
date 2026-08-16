"""CLI: DoorKey 8x8 distill → transfer eval on 6x6 / 16x16 vs coordinate trees."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from solverrl.doorkey.transfer import format_transfer_table, run_doorkey_transfer


def main() -> None:
    p = argparse.ArgumentParser(description="MiniGrid DoorKey relational transfer")
    p.add_argument("--train-size", type=int, default=8)
    p.add_argument("--eval-sizes", type=str, default="6,8,16")
    p.add_argument("--train-episodes", type=int, default=80)
    p.add_argument("--eval-episodes", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-literals", type=int, default=3)
    p.add_argument("--out-prolog", type=str, default="data/rules/doorkey_8x8.pl")
    p.add_argument("--out-json", type=str, default="data/plots/doorkey_transfer.json")
    args = p.parse_args()

    eval_sizes = tuple(int(x) for x in args.eval_sizes.split(",") if x.strip())
    report = run_doorkey_transfer(
        train_size=args.train_size,
        eval_sizes=eval_sizes,
        n_train_episodes=args.train_episodes,
        n_eval_episodes=args.eval_episodes,
        seed=args.seed,
        max_body_literals=args.max_literals,
        out_prolog=Path(args.out_prolog),
    )
    print(format_transfer_table(report))

    payload = {
        "train_size": report.train_size,
        "n_train_transitions": report.n_train_transitions,
        "distill_fidelity": report.distill_fidelity,
        "n_clauses": report.n_clauses,
        "relational": {
            str(k): {
                "success_rate": v.success_rate,
                "mean_return": v.mean_return,
                "mean_steps": v.mean_steps,
                "n_episodes": v.n_episodes,
            }
            for k, v in report.relational.items()
        },
        "tree": {
            str(k): {
                "success_rate": v.success_rate,
                "mean_return": v.mean_return,
                "mean_steps": v.mean_steps,
                "n_episodes": v.n_episodes,
            }
            for k, v in report.tree.items()
        },
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote_json={out}")
    print(f"wrote_prolog={args.out_prolog}")


if __name__ == "__main__":
    main()
