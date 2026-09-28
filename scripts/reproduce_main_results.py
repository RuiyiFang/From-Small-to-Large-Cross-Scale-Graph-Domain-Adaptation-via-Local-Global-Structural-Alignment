r"""Reproduce the main CSGDA results (Tables 2 and 3).

Sweeps every transfer task x every sampling ratio, averaging over the sampling
seeds, and writes both the raw per-run records and a pivoted accuracy table that
matches the layout of the paper.

Examples
--------
::

    python scripts/reproduce_main_results.py --data-root /path/to/data
    python scripts/reproduce_main_results.py --family Citation --seeds 42 123 456
    python scripts/reproduce_main_results.py --tasks A->D D->A --ratios 0.01

The run is resumable: existing rows in ``--out`` are skipped, so an interrupted
sweep can simply be restarted with the same arguments.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lgsa.config import LGSAConfig, loss_weights_for_task  # noqa: E402
from lgsa.cross_scale import SOURCE_RATIOS, CrossScaleConfig, sample_cross_scale_source  # noqa: E402
from lgsa.datasets import TASKS, load_task  # noqa: E402
from lgsa.trainer import train_lgsa  # noqa: E402
from lgsa.utils import aggregate, write_csv  # noqa: E402

DEFAULT_SEEDS = (42, 123, 456, 789, 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--family", default=None, choices=["Citation", "Airport", "Social", "ACM"])
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--ratios", nargs="*", type=float, default=list(SOURCE_RATIOS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None, help="CSV of raw per-run results")
    return parser.parse_args()


def load_done(path: str) -> set:
    if not path or not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return {
            (row["task"], row["ratio"], row["seed"]) for row in csv.DictReader(handle)
        }


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f"results_lgsa_{timestamp}.csv"

    tasks = list(TASKS)
    if args.family:
        tasks = [t for t in tasks if t.family == args.family]
    if args.tasks:
        wanted = set(args.tasks)
        tasks = [t for t in tasks if t.short_name in wanted or t.name in wanted]

    done = load_done(out_path)
    rows = []
    if done:
        with open(out_path, newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        print(f"Resuming: {len(done)} runs already present in {out_path}")

    for task in tasks:
        try:
            full_source, target, task = load_task(task, args.data_root)
        except (FileNotFoundError, ValueError) as error:
            print(f"[skip] {task.short_name}: {error}")
            continue

        num_classes = int(max(full_source.y.max(), target.y.max()).item()) + 1
        mu, eta = loss_weights_for_task(task.short_name)

        for ratio in args.ratios:
            for seed in args.seeds:
                key = (task.short_name, str(ratio), str(seed))
                if key in done:
                    continue
                print(f"\n>>> {task.short_name} | rho={ratio:.0%} | seed={seed}")
                try:
                    sampled, info = sample_cross_scale_source(
                        full_source, CrossScaleConfig(ratio=ratio, seed=seed)
                    )
                    config = LGSAConfig(
                        epochs=args.epochs, mu=mu, eta=eta, seed=seed, device=args.device
                    )
                    metrics = train_lgsa(sampled, target, num_classes, config, verbose=False)
                except Exception:  # keep the sweep alive, record the failure
                    traceback.print_exc()
                    continue

                row = {
                    "task": task.short_name,
                    "family": task.family,
                    "ratio": ratio,
                    "seed": seed,
                    "source_nodes": info["num_nodes"],
                    "source_edges": info["num_edges"],
                    "mu": mu,
                    "eta": eta,
                    **metrics,
                }
                rows.append(row)
                write_csv(out_path, rows)
                print(
                    f"    acc {metrics['accuracy']:.4f} | macro-F1 {metrics['macro_f1']:.4f} "
                    f"| {metrics['train_seconds']:.1f}s"
                )

    if not rows:
        print("No results produced.")
        return

    # ---- pivot into the layout of Tables 2 and 3 ----
    grouped = defaultdict(list)
    for row in rows:
        grouped[(str(row["ratio"]), row["task"])].append(float(row["accuracy"]))

    summary = []
    for (ratio, task), values in sorted(grouped.items()):
        statistics = aggregate(values)
        summary.append(
            {
                "ratio": ratio,
                "task": task,
                "acc_mean": round(statistics["mean"], 4),
                "acc_std": round(statistics["std"], 4),
                "num_seeds": statistics["n"],
            }
        )
    summary_path = out_path.replace(".csv", "_summary.csv")
    write_csv(summary_path, summary)

    print(f"\nRaw results : {out_path}")
    print(f"Summary     : {summary_path}")


if __name__ == "__main__":
    main()
