r"""Run LGSA on a single CSGDA transfer task.

Examples
--------
::

    python scripts/run_lgsa.py --task A->D --ratio 0.05 --data-root /path/to/data
    python scripts/run_lgsa.py --task E->B --ratio 0.01 --seeds 42 123 456
    python scripts/run_lgsa.py --task A->C --ratio 0.05 --mu 0.1 --eta 0.5

By default the per-task ``(mu, eta)`` of Table 12 are used; passing ``--mu`` /
``--eta`` overrides them.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lgsa.config import LGSAConfig, loss_weights_for_task  # noqa: E402
from lgsa.cross_scale import CrossScaleConfig, sample_cross_scale_source  # noqa: E402
from lgsa.datasets import load_task  # noqa: E402
from lgsa.trainer import train_lgsa  # noqa: E402
from lgsa.utils import aggregate, write_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", required=True, help="e.g. A->D, U->B, B1->B2, A3->A4")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--ratio", type=float, default=0.05)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42])
    parser.add_argument("--alpha", type=float, default=0.5, help="BFS share of the node budget")
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-neighbours", type=int, default=5, help="M in Eq. 2")
    parser.add_argument("--pseudo-threshold", type=float, default=0.99, help="tau_p in Eq. 9")
    parser.add_argument(
        "--pseudo-warmup", type=int, default=100, help="epochs before V^T_conf is populated"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None, help="optional CSV path for the per-seed results")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    full_source, target, task = load_task(args.task, args.data_root)
    num_classes = int(max(full_source.y.max(), target.y.max()).item()) + 1
    mu, eta = loss_weights_for_task(task.short_name)

    print(f"Task      : {task.short_name}  ({task.source} -> {task.target}, {task.family})")
    print(f"Full src  : {full_source.x.size(0)} nodes, {full_source.edge_index.size(1) // 2} edges")
    print(f"Target    : {target.x.size(0)} nodes, {target.edge_index.size(1) // 2} edges")
    print(f"Classes   : {num_classes} | rho = {args.ratio:.0%} | mu = {mu}, eta = {eta}")

    rows = []
    for seed in args.seeds:
        sampled, info = sample_cross_scale_source(
            full_source, CrossScaleConfig(ratio=args.ratio, seed=seed, alpha=args.alpha)
        )
        print(
            f"\n[seed {seed}] G^S: {info['num_nodes']} nodes "
            f"({info['num_bfs_nodes']} BFS + {info['num_dfs_nodes']} DFS, K={info['num_bfs_clusters']}), "
            f"{info['num_edges']} edges, {info['num_classes_covered']}/{num_classes} classes"
        )

        config = LGSAConfig(
            hidden_dim=args.hidden_dim,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            num_augmented_neighbours=args.num_neighbours,
            pseudo_threshold=args.pseudo_threshold,
            pseudo_warmup_epochs=args.pseudo_warmup,
            mu=args.mu if args.mu is not None else mu,
            eta=args.eta if args.eta is not None else eta,
            seed=seed,
            device=args.device,
        )
        metrics = train_lgsa(sampled, target, num_classes, config, verbose=not args.quiet)
        print(
            f"[seed {seed}] acc {metrics['accuracy']:.4f} | "
            f"macro-F1 {metrics['macro_f1']:.4f} | "
            f"local {metrics['local_accuracy']:.4f} / global {metrics['global_accuracy']:.4f} | "
            f"gate {metrics['mean_gate']:.3f} | {metrics['train_seconds']:.1f}s"
        )
        rows.append({"task": task.short_name, "ratio": args.ratio, "seed": seed, **metrics})

    accuracy = aggregate([row["accuracy"] for row in rows])
    macro = aggregate([row["macro_f1"] for row in rows])
    print(
        f"\n=== {task.short_name} @ rho={args.ratio:.0%} over {accuracy['n']} seeds ===\n"
        f"    ACC      {accuracy['mean']:.4f} +/- {accuracy['std']:.4f}\n"
        f"    Macro-F1 {macro['mean']:.4f} +/- {macro['std']:.4f}"
    )

    if args.out:
        write_csv(args.out, rows)
        print(f"    saved -> {args.out}")


if __name__ == "__main__":
    main()
