r"""Ablation study (Sec. 4.2, Figures 3, 6 and 7).

Variants
--------
======  ===================================================================
LGSA    the full model
LGSA1   w/o cross-domain node augmentation and label-signal injection
        (the local branch degenerates to a vanilla GCN over the truncated
        :math:`G^S`)
LGSA2   w/o the global branch (prediction is read off the local logits)
LGSA3   uniform 0.5 fusion instead of the entropy gate, and no
        :math:`\mathcal L_{cons}`
======  ===================================================================

Example
-------
::

    python scripts/run_ablation.py --family Airport --data-root /path/to/data
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lgsa.config import LGSAConfig, loss_weights_for_task  # noqa: E402
from lgsa.cross_scale import SOURCE_RATIOS, CrossScaleConfig, sample_cross_scale_source  # noqa: E402
from lgsa.datasets import TASKS, load_task  # noqa: E402
from lgsa.trainer import train_lgsa  # noqa: E402
from lgsa.utils import aggregate, write_csv  # noqa: E402

VARIANTS = {
    "LGSA": {},
    "LGSA1": {"use_cross_domain_augmentation": False},
    "LGSA2": {"use_global_branch": False},
    "LGSA3": {"use_adaptive_fusion": False},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--family", default="Airport", choices=["Citation", "Airport", "Social", "ACM"])
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--ratios", nargs="*", type=float, default=list(SOURCE_RATIOS))
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 456])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="results_ablation.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tasks = [t for t in TASKS if t.family == args.family]
    if args.tasks:
        wanted = set(args.tasks)
        tasks = [t for t in TASKS if t.short_name in wanted]

    rows = []
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
                sampled, _ = sample_cross_scale_source(
                    full_source, CrossScaleConfig(ratio=ratio, seed=seed)
                )
                for variant, overrides in VARIANTS.items():
                    config = LGSAConfig(
                        epochs=args.epochs,
                        mu=mu,
                        eta=eta,
                        seed=seed,
                        device=args.device,
                        **overrides,
                    )
                    metrics = train_lgsa(sampled, target, num_classes, config, verbose=False)
                    rows.append(
                        {
                            "task": task.short_name,
                            "family": task.family,
                            "ratio": ratio,
                            "seed": seed,
                            "variant": variant,
                            **metrics,
                        }
                    )
                    print(
                        f"{task.short_name:>8s} rho={ratio:<5.0%} seed={seed:<5d} "
                        f"{variant:<6s} acc {metrics['accuracy']:.4f}"
                    )

    if not rows:
        print("No results produced.")
        return

    write_csv(args.out, rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["ratio"], row["task"], row["variant"])].append(row["accuracy"])
    summary = [
        {
            "ratio": ratio,
            "task": task,
            "variant": variant,
            "acc_mean": round(aggregate(values)["mean"], 4),
            "acc_std": round(aggregate(values)["std"], 4),
        }
        for (ratio, task, variant), values in sorted(grouped.items())
    ]
    summary_path = args.out.replace(".csv", "_summary.csv")
    write_csv(summary_path, summary)

    print(f"\nRaw results : {args.out}")
    print(f"Summary     : {summary_path}")


if __name__ == "__main__":
    main()
