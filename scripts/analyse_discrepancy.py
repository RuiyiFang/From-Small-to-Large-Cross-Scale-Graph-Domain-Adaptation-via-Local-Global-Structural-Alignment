r"""Local / global discrepancy diagnostics of the cross-scale setting (Appendix A).

Reproduces the motivation experiment of Fig. 1(b) and Fig. 5.  It measures how far
the *sampled* source graph drifts from the target graph **without using any target
label**, via two node-level scores and their 1-D Wasserstein distances.

Global semantic score (Eq. 14-15)
    :math:`\psi^D(v) = [x^D_v \| \phi^D(v)]` is standardised over the union of the
    two domains and projected onto the first principal direction, then rescaled to
    :math:`[0, 1]`.  Its distribution shows how much of the feature-structure space
    each graph covers.

Local neighbour-degree score (Eq. 16)
    .. math::
        s^D_{deg}(v) = \frac{1}{|N^D(v)|}\sum_{u \in N^D(v)}
                       \frac{\min\{\deg(v), \deg(u)\}}{\max\{\deg(v), \deg(u)\}+\epsilon}

    which measures whether a node connects to neighbours with similar degree
    profiles, and therefore fragments when induced edges are removed.

Discrepancies (Eq. 18)
    :math:`\Delta_{glo} = W_1(P^S_{glo}, P^T_{glo})`,
    :math:`\Delta_{loc} = W_1(P^S_{deg}, P^T_{deg})`.

The full source graph is reported alongside the sampled ones, which is the
comparison that makes the *cross-scale* gap visible: CSGDA is not merely "fewer
labels", it is a simultaneous global-semantic and local-structural distortion.

Example
-------
::

    python scripts/analyse_discrepancy.py --data-root /path/to/data --family Citation
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from scipy.stats import wasserstein_distance

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lgsa.cross_scale import SOURCE_RATIOS, CrossScaleConfig, sample_cross_scale_source  # noqa: E402
from lgsa.datasets import TASKS, load_task  # noqa: E402
from lgsa.descriptors import structural_descriptor  # noqa: E402
from lgsa.utils import write_csv  # noqa: E402

EPSILON = 1e-8


def _first_principal_direction(matrix: np.ndarray, num_iterations: int = 20) -> np.ndarray:
    """Top right-singular vector via power iteration (avoids a full SVD)."""
    generator = np.random.default_rng(0)
    direction = generator.normal(size=matrix.shape[1]).astype(matrix.dtype)
    direction /= np.linalg.norm(direction)
    for _ in range(num_iterations):
        direction = matrix.T @ (matrix @ direction)
        norm = np.linalg.norm(direction)
        if norm < EPSILON:
            break
        direction /= norm
    return direction


def global_semantic_scores(graphs) -> list:
    """Eq. 14-15: PCA projection of ``[x || phi]`` onto the first component."""
    features = [
        torch.cat([graph.x, structural_descriptor(graph)], dim=1).numpy().astype(np.float32)
        for graph in graphs
    ]
    joint = np.concatenate(features, axis=0)
    mean = joint.mean(axis=0, keepdims=True)
    std = joint.std(axis=0, keepdims=True)
    std[std < EPSILON] = 1.0
    joint -= mean
    joint /= std

    direction = _first_principal_direction(joint)
    del joint

    projected = [(((block - mean) / std) @ direction) for block in features]
    low = min(float(p.min()) for p in projected)
    high = max(float(p.max()) for p in projected)
    span = max(high - low, EPSILON)
    return [(p - low) / span for p in projected]


def neighbour_degree_scores(graph) -> np.ndarray:
    """Eq. 16: degree-profile consistency of each node's one-hop neighbourhood."""
    num_nodes = int(graph.x.size(0))
    scores = np.zeros(num_nodes)
    edge_index = graph.edge_index
    if edge_index.numel() == 0:
        return scores

    degree = np.zeros(num_nodes)
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    np.add.at(degree, src, 1.0)

    ratio = np.minimum(degree[src], degree[dst]) / (np.maximum(degree[src], degree[dst]) + EPSILON)
    np.add.at(scores, src, ratio)
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where(degree > 0, scores / np.maximum(degree, 1.0), 0.0)
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--family", default=None, choices=["Citation", "Airport", "Social", "ACM"])
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--ratios", nargs="*", type=float, default=list(SOURCE_RATIOS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="discrepancy_analysis.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tasks = list(TASKS)
    if args.family:
        tasks = [t for t in tasks if t.family == args.family]
    if args.tasks:
        wanted = set(args.tasks)
        tasks = [t for t in tasks if t.short_name in wanted]

    rows = []
    for task in tasks:
        try:
            full_source, target, task = load_task(task, args.data_root)
        except (FileNotFoundError, ValueError) as error:
            print(f"[skip] {task.short_name}: {error}")
            continue

        sampled = {
            ratio: sample_cross_scale_source(
                full_source, CrossScaleConfig(ratio=ratio, seed=args.seed)
            )[0]
            for ratio in args.ratios
        }
        graphs = [target, full_source] + [sampled[r] for r in args.ratios]
        labels = ["target", "full_source"] + [f"rho={r:g}" for r in args.ratios]

        global_scores = global_semantic_scores(graphs)
        local_scores = [neighbour_degree_scores(graph) for graph in graphs]

        target_global, target_local = global_scores[0], local_scores[0]
        print(f"\n=== {task.short_name} ===")
        for name, gscore, lscore in list(zip(labels, global_scores, local_scores))[1:]:
            delta_global = float(wasserstein_distance(gscore, target_global))
            delta_local = float(wasserstein_distance(lscore, target_local))
            rows.append(
                {
                    "task": task.short_name,
                    "family": task.family,
                    "source_view": name,
                    "delta_global": round(delta_global, 5),
                    "delta_local": round(delta_local, 5),
                }
            )
            print(
                f"  {name:<12s} Delta_glo = {delta_global:.4f}   Delta_loc = {delta_local:.4f}"
            )

    if rows:
        write_csv(args.out, rows)
        print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
