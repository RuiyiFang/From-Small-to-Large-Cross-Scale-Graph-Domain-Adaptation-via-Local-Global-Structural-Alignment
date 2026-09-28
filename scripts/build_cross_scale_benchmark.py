r"""Build and export the cross-scale (CSGDA) benchmark.

This script materialises the sampled source graphs :math:`G^S = S_\rho(G^S_{full})`
to disk so that **every method -- LGSA and all baselines -- trains on byte-identical
source graphs**.  Exporting the benchmark once is the recommended workflow: it
removes any doubt about whether a comparison used the same sampled source, and it
lets a follower reproduce the setting without re-implementing the sampler.

Output layout::

    <out-dir>/
      manifest.csv                     # one row per (task, ratio, seed) + statistics
      statistics_table1.csv            # Table 1 / Table 4 of the paper
      A->D/rho0.01/seed42.pt           # torch-saved payload (see below)
      A->D/rho0.05/seed42.pt
      ...

Each ``.pt`` payload is a dict::

    {
      "task":        "A->D",
      "source":      "ACMv9",  "target": "DBLPv7",
      "ratio":       0.01,     "seed":   42,
      "node_idx":    LongTensor[N^S],   # indices into the FULL source graph
      "edge_index":  LongTensor[2, |E^S|],  # relabelled to 0..N^S-1
      "y":           LongTensor[N^S],
      "sampler":     {...}              # the full CrossScaleConfig + statistics
    }

Node features are intentionally **not** stored (they are simply
``full_source.x[node_idx]``), which keeps the export small and guarantees that
everybody uses the same feature matrix.  Use
``scripts/load_exported_source.py``-style code, or::

    from lgsa.cross_scale import induced_subgraph
    payload = torch.load("A->D/rho0.01/seed42.pt")
    source  = induced_subgraph(full_source, payload["node_idx"].tolist())

Examples
--------
::

    # the full benchmark: 16 tasks x 3 ratios x 10 seeds
    python scripts/build_cross_scale_benchmark.py --data-root /path/to/data --out-dir cross_scale_benchmark

    # only the citation family, statistics table only (no .pt export)
    python scripts/build_cross_scale_benchmark.py --family Citation --stats-only
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lgsa.cross_scale import (  # noqa: E402
    SOURCE_RATIOS,
    CrossScaleConfig,
    cross_scale_statistics,
    sample_cross_scale_source,
)
from lgsa.datasets import TASKS, load_task  # noqa: E402
from lgsa.utils import write_csv  # noqa: E402

#: The ten sampling seeds averaged in the paper (Appendix D).
DEFAULT_SEEDS = (42, 123, 456, 789, 1024, 2048, 3407, 4096, 5555, 6789)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", default=None, help="root directory of the raw datasets")
    parser.add_argument("--out-dir", default="cross_scale_benchmark")
    parser.add_argument("--family", default=None, choices=["Citation", "Airport", "Social", "ACM"])
    parser.add_argument("--tasks", nargs="*", default=None, help="e.g. A->D D->A")
    parser.add_argument("--ratios", nargs="*", type=float, default=list(SOURCE_RATIOS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--alpha", type=float, default=0.5, help="BFS share of the node budget")
    parser.add_argument("--num-bfs-clusters", type=int, default=None, help="K; default is Eq. (27)")
    parser.add_argument("--stats-only", action="store_true", help="do not write the .pt files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tasks = list(TASKS)
    if args.family:
        tasks = [t for t in tasks if t.family == args.family]
    if args.tasks:
        wanted = set(args.tasks)
        tasks = [t for t in tasks if t.short_name in wanted or t.name in wanted]
    if not tasks:
        raise SystemExit("no task selected")

    os.makedirs(args.out_dir, exist_ok=True)
    manifest, table1 = [], []

    for task in tasks:
        try:
            full_source, target, task = load_task(task, args.data_root)
        except (FileNotFoundError, ValueError) as error:
            print(f"[skip] {task.short_name}: {error}")
            continue

        print(f"\n=== {task.short_name}  ({task.source} -> {task.target}, {task.family}) ===")
        row = {
            "task": task.short_name,
            "family": task.family,
            "full_source_nodes": int(full_source.x.size(0)),
            "full_source_edges": int(full_source.edge_index.size(1) // 2),
            "target_nodes": int(target.x.size(0)),
            "target_edges": int(target.edge_index.size(1) // 2),
        }

        for ratio in args.ratios:
            nodes_per_seed, edges_per_seed = [], []
            for seed in args.seeds:
                config = CrossScaleConfig(
                    ratio=ratio,
                    seed=seed,
                    alpha=args.alpha,
                    num_bfs_clusters=args.num_bfs_clusters,
                )
                sampled, info = sample_cross_scale_source(full_source, config)
                statistics = cross_scale_statistics(sampled, full_source, target)

                nodes_per_seed.append(statistics["sampled_source_nodes"])
                edges_per_seed.append(statistics["sampled_source_edges"])

                manifest.append({"task": task.short_name, "family": task.family, **info, **statistics})

                if not args.stats_only:
                    # "A->D" is not a legal directory name on Windows
                    directory = os.path.join(
                        args.out_dir, task.short_name.replace("->", "_to_"), f"rho{ratio:g}"
                    )
                    os.makedirs(directory, exist_ok=True)
                    torch.save(
                        {
                            "task": task.short_name,
                            "source": task.source,
                            "target": task.target,
                            "ratio": ratio,
                            "seed": seed,
                            "node_idx": sampled.node_idx,
                            "edge_index": sampled.edge_index,
                            "y": sampled.y,
                            "sampler": {**config.as_dict(), **info, **statistics},
                        },
                        os.path.join(directory, f"seed{seed}.pt"),
                    )

            mean_nodes = sum(nodes_per_seed) / len(nodes_per_seed)
            mean_edges = sum(edges_per_seed) / len(edges_per_seed)
            row[f"rho{ratio:g}_nodes"] = round(mean_nodes, 1)
            row[f"rho{ratio:g}_edges"] = round(mean_edges, 1)
            print(
                f"  rho={ratio:>5.0%}  #N={mean_nodes:8.1f}  #E={mean_edges:9.1f}"
                f"  (mean over {len(args.seeds)} sampling seeds)"
            )

        table1.append(row)

    if manifest:
        write_csv(os.path.join(args.out_dir, "manifest.csv"), manifest)
        write_csv(os.path.join(args.out_dir, "statistics_table1.csv"), table1)
        print(f"\nWrote {len(manifest)} sampled source graphs to '{args.out_dir}'.")
        print(f"  manifest           : {os.path.join(args.out_dir, 'manifest.csv')}")
        print(f"  Table 1 statistics : {os.path.join(args.out_dir, 'statistics_table1.csv')}")
    else:
        print("\nNothing was produced -- check --data-root.")


if __name__ == "__main__":
    main()
