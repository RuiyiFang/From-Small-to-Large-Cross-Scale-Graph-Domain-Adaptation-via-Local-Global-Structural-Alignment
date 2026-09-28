r"""
Cross-Scale Source-Graph Construction -- the sampling operator :math:`S_\rho`.
=============================================================================

This module is the **reference implementation of the CSGDA benchmark setting**
introduced in

    "From Small to Large: Cross-Scale Graph Domain Adaptation via
     Local-Global Structural Alignment"  (Sec. 3.1, Appendix B & C)

and it is deliberately kept **self-contained**: it depends only on the Python
standard library and (optionally) ``torch`` / ``torch_geometric``.  Anybody who
wants to reproduce or build on the cross-scale setting should only need this
single file -- see :func:`select_cross_scale_nodes`, which is pure Python and
can be copy-pasted into any codebase.

Problem setting
---------------
In Cross-Scale Graph Domain Adaptation (CSGDA) the training-time source graph is
*not* the full labelled source graph :math:`G^S_{full}`.  Instead it is a
node-induced subgraph produced by a sampling operator :math:`S_\rho`

.. math::
    G^S = S_\rho(G^S_{full}),\qquad
    |V^S| = N^S = \lfloor \rho\,|V^S_{full}| \rfloor ,\qquad
    \rho \in \{1\%, 5\%, 10\%\}

while the target graph :math:`G^T` stays **fully observed but unlabelled**.

The operator :math:`S_\rho` (Appendix B)
----------------------------------------
A purely uniform sampler destroys local neighbourhoods; a single BFS / random
walk over-concentrates on one dense community.  :math:`S_\rho` therefore splits
the node budget into two halves (``alpha = 0.5``, validated in Appendix C.2/C.4):

======  =========================================================  ============
Stage   Rule                                                       Purpose
======  =========================================================  ============
1 BFS   :math:`N^S_{BFS}=\lfloor \alpha N^S \rfloor` nodes grown    local
        as :math:`K` BFS clusters from uniformly random seeds,      cohesion
        per-cluster budget :math:`q_k=\lceil N^S_{BFS}/K \rceil`;
        neighbours are shuffled before expansion, and an exhausted
        cluster restarts from another unsampled node.
2 DFS   the remaining :math:`N^S_{DFS}=N^S-N^S_{BFS}` nodes are     global
        collected by random-restart DFS probes: follow one random   coverage
        unvisited neighbour, restart from a random unsampled node
        when the walk is stuck.
3 IND   :math:`V^S=V^S_{BFS}\cup V^S_{DFS}` and                     induced
        :math:`E^S=\{(i,j)\in E^S_{full}\mid i,j\in V^S\}`;         subgraph
        :math:`A^S,X^S,Y^S` are the corresponding restrictions.
======  =========================================================  ============

Number of BFS clusters (Appendix C.3, Eq. 27)

.. math::
    K = \left\lceil \sqrt{b_{BFS}} \right\rceil
      = \left\lceil \sqrt{\rho |V^S_{full}| / 2} \right\rceil

so that both the number of clusters and the per-cluster size scale as
:math:`\Theta(\sqrt{b})`.

Guarantees / conventions
------------------------
* **No target labels are ever touched.**  :math:`S_\rho` is applied to the
  source graph only.
* **No edge outside** :math:`V^S` is retained -- the result is a strict
  node-induced subgraph, so the sampled source is incomplete in *both* nodes and
  edges (this is what makes CSGDA harder than "fewer labels").
* **Degenerate budgets are floored to 2 nodes** so that at least one candidate
  edge can exist (needed for Brazil at :math:`\rho=1\%`, cf. Appendix C.1).
* Sampling is **fully determined by** ``(num_nodes, edge_index, ratio, seed,
  alpha, K)``.  The same seed therefore yields the same source graph for *every*
  method, which is what makes the baseline comparison fair.

Command line
------------
::

    python -m lgsa.cross_scale --source ACMv9 --ratio 0.05 --seed 42
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "CrossScaleConfig",
    "build_adjacency",
    "select_cross_scale_nodes",
    "sample_cross_scale_source",
    "induced_subgraph",
    "cross_scale_statistics",
    "SOURCE_RATIOS",
]

#: Sampling ratios used in the paper (Sec. 3.1).
SOURCE_RATIOS: Tuple[float, ...] = (0.01, 0.05, 0.10)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class CrossScaleConfig:
    """Configuration of the sampling operator :math:`S_\\rho`.

    Attributes
    ----------
    ratio:
        Source sampling ratio :math:`\\rho \\in (0, 1]`.
    seed:
        Sampling seed.  Ten sampling seeds are averaged in the paper.
    alpha:
        Fraction of the node budget assigned to the BFS stage.  ``0.5``
        (50:50 BFS:DFS) is the default validated in Appendix C.2/C.4.
    num_bfs_clusters:
        Number of BFS clusters :math:`K`.  ``None`` uses Eq. (27),
        :math:`K=\\lceil\\sqrt{b_{BFS}}\\rceil`.
    min_nodes:
        Lower bound on the node budget, used to keep :math:`\\rho=1\\%` on very
        small graphs (e.g. Brazil) non-degenerate.
    """

    ratio: float = 0.05
    seed: int = 42
    alpha: float = 0.5
    num_bfs_clusters: Optional[int] = None
    min_nodes: int = 2

    def __post_init__(self) -> None:
        if not 0.0 < self.ratio <= 1.0:
            raise ValueError(f"ratio must lie in (0, 1], got {self.ratio}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must lie in [0, 1], got {self.alpha}")
        if self.num_bfs_clusters is not None and self.num_bfs_clusters < 1:
            raise ValueError("num_bfs_clusters must be >= 1")

    def as_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# core sampler -- pure Python, no torch required
# ---------------------------------------------------------------------------
def build_adjacency(edge_list: Sequence[Tuple[int, int]], num_nodes: int) -> List[List[int]]:
    """Build an undirected adjacency list from a list of ``(src, dst)`` pairs."""
    adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
    for src, dst in edge_list:
        if src == dst:
            continue
        adjacency[src].append(dst)
        adjacency[dst].append(src)
    return [sorted(set(neighbours)) for neighbours in adjacency]


def _bfs_clusters(
    adjacency: List[List[int]],
    budget: int,
    num_clusters: int,
    rng: random.Random,
) -> Tuple[List[int], List[float]]:
    """Stage 1 -- grow ``num_clusters`` BFS balls until ``budget`` nodes are taken.

    Returns the selected nodes and the per-cluster BFS radius (the largest hop
    distance from the seed that the cluster budget reaches, reported as
    :math:`\\bar R` in Appendix C.3).
    """
    num_nodes = len(adjacency)
    selected: set[int] = set()
    radii: List[float] = []
    if budget <= 0 or num_nodes == 0:
        return [], radii

    per_cluster = max(1, math.ceil(budget / max(1, num_clusters)))
    unsampled = list(range(num_nodes))
    rng.shuffle(unsampled)
    cursor = 0

    for _ in range(num_clusters):
        if len(selected) >= budget:
            break
        # pick a seed that has not been sampled yet
        while cursor < num_nodes and unsampled[cursor] in selected:
            cursor += 1
        if cursor >= num_nodes:
            break
        seed_node = unsampled[cursor]
        cursor += 1

        cluster = {seed_node}
        selected.add(seed_node)
        queue: List[Tuple[int, int]] = [(seed_node, 0)]
        head, radius = 0, 0
        while head < len(queue) and len(cluster) < per_cluster and len(selected) < budget:
            node, depth = queue[head]
            head += 1
            neighbours = list(adjacency[node])
            rng.shuffle(neighbours)
            for neighbour in neighbours:
                if neighbour in selected:
                    continue
                cluster.add(neighbour)
                selected.add(neighbour)
                queue.append((neighbour, depth + 1))
                radius = max(radius, depth + 1)
                if len(cluster) >= per_cluster or len(selected) >= budget:
                    break
        radii.append(float(radius))

    # a cluster may terminate early because its connected region is exhausted;
    # restart BFS from another unsampled node until the BFS budget is filled
    while len(selected) < budget and cursor < num_nodes:
        node = unsampled[cursor]
        cursor += 1
        if node in selected:
            continue
        selected.add(node)
        queue = [(node, 0)]
        head = 0
        while head < len(queue) and len(selected) < budget:
            current, depth = queue[head]
            head += 1
            neighbours = list(adjacency[current])
            rng.shuffle(neighbours)
            for neighbour in neighbours:
                if neighbour in selected:
                    continue
                selected.add(neighbour)
                queue.append((neighbour, depth + 1))
                if len(selected) >= budget:
                    break

    return sorted(selected), radii


def _dfs_probes(
    adjacency: List[List[int]],
    budget: int,
    already_selected: set[int],
    rng: random.Random,
) -> List[int]:
    """Stage 2 -- scattered random-restart DFS probes for global coverage."""
    num_nodes = len(adjacency)
    selected: set[int] = set()
    if budget <= 0:
        return []

    roots = [v for v in range(num_nodes) if v not in already_selected]
    rng.shuffle(roots)
    cursor = 0

    while len(selected) < budget and cursor < len(roots):
        node = roots[cursor]
        cursor += 1
        if node in already_selected or node in selected:
            continue
        selected.add(node)
        current = node
        # walk depth-first along unvisited neighbours
        while len(selected) < budget:
            candidates = [
                v
                for v in adjacency[current]
                if v not in already_selected and v not in selected
            ]
            if not candidates:
                break  # stuck -> restart from another unsampled root
            current = rng.choice(candidates)
            selected.add(current)

    return sorted(selected)


def select_cross_scale_nodes(
    edge_list: Sequence[Tuple[int, int]],
    num_nodes: int,
    config: Optional[CrossScaleConfig] = None,
    **kwargs,
) -> Tuple[List[int], Dict]:
    """Run the operator :math:`S_\\rho` and return the sampled source node set.

    This is the *only* function required to reproduce the CSGDA setting; it is
    framework-agnostic (plain Python lists in, plain Python lists out).

    Parameters
    ----------
    edge_list:
        Iterable of ``(src, dst)`` node-index pairs of :math:`G^S_{full}`.
        The graph is treated as undirected; duplicated/reversed pairs are fine.
    num_nodes:
        :math:`|V^S_{full}|`.
    config:
        :class:`CrossScaleConfig`.  Keyword arguments (``ratio=``, ``seed=``,
        ...) may be given instead.

    Returns
    -------
    nodes:
        Sorted list of sampled source node indices (into the *full* source graph).
    info:
        Bookkeeping dict: budgets, :math:`K`, per-stage node counts and the
        average BFS radius :math:`\\bar R`.
    """
    config = config or CrossScaleConfig(**kwargs)

    budget = int(math.floor(num_nodes * config.ratio))
    budget = max(config.min_nodes, min(budget, num_nodes))
    bfs_budget = int(math.floor(budget * config.alpha))
    dfs_budget = budget - bfs_budget

    num_clusters = config.num_bfs_clusters
    if num_clusters is None:  # Eq. (27)
        num_clusters = max(1, math.ceil(math.sqrt(max(1, bfs_budget))))

    rng = random.Random(config.seed)
    adjacency = build_adjacency(edge_list, num_nodes)

    bfs_nodes, radii = _bfs_clusters(adjacency, bfs_budget, num_clusters, rng)
    dfs_nodes = _dfs_probes(adjacency, dfs_budget, set(bfs_nodes), rng)

    selected = set(bfs_nodes) | set(dfs_nodes)

    # top up with uniform random nodes if the graph structure could not supply
    # enough nodes (tiny or extremely fragmented graphs)
    if len(selected) < budget:
        remainder = [v for v in range(num_nodes) if v not in selected]
        rng.shuffle(remainder)
        for node in remainder[: budget - len(selected)]:
            selected.add(node)

    info = {
        "ratio": config.ratio,
        "seed": config.seed,
        "alpha": config.alpha,
        "budget": budget,
        "bfs_budget": bfs_budget,
        "dfs_budget": dfs_budget,
        "num_bfs_clusters": num_clusters,
        "num_bfs_nodes": len(bfs_nodes),
        "num_dfs_nodes": len(dfs_nodes),
        "num_sampled_nodes": len(selected),
        "mean_bfs_radius": (sum(radii) / len(radii)) if radii else 0.0,
    }
    return sorted(selected), info


# ---------------------------------------------------------------------------
# torch_geometric bindings
# ---------------------------------------------------------------------------
def induced_subgraph(data, nodes: Sequence[int]):
    """Return the node-induced subgraph of a PyG ``Data`` object (Eq. 25-26)."""
    import torch
    from torch_geometric.data import Data

    node_tensor = torch.as_tensor(sorted(nodes), dtype=torch.long)
    remap = {int(old): new for new, old in enumerate(node_tensor.tolist())}

    edge_index = getattr(data, "edge_index", None)
    if edge_index is not None and edge_index.numel() > 0:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[node_tensor] = True
        keep = mask[edge_index[0]] & mask[edge_index[1]]
        kept = edge_index[:, keep]
        if kept.numel() > 0:
            relabelled = torch.tensor(
                [[remap[int(s)] for s in kept[0]], [remap[int(t)] for t in kept[1]]],
                dtype=torch.long,
            )
        else:
            relabelled = torch.zeros(2, 0, dtype=torch.long)
    else:
        relabelled = torch.zeros(2, 0, dtype=torch.long)

    sub = Data(
        x=data.x[node_tensor],
        y=data.y[node_tensor],
        edge_index=relabelled,
    )
    sub.node_idx = node_tensor  # indices into the full source graph
    return sub


def sample_cross_scale_source(
    data,
    config: Optional[CrossScaleConfig] = None,
    **kwargs,
) -> Tuple["object", Dict]:
    """Apply :math:`S_\\rho` to a PyG ``Data`` source graph.

    Parameters
    ----------
    data:
        The **full** source graph :math:`G^S_{full}` as a ``torch_geometric``
        ``Data`` object with ``x``, ``y`` and ``edge_index``.
    config / kwargs:
        See :class:`CrossScaleConfig`.

    Returns
    -------
    (sampled_source, info)
        ``sampled_source`` is a new ``Data`` holding :math:`G^S`; its
        ``node_idx`` attribute maps back to :math:`G^S_{full}`.
    """
    config = config or CrossScaleConfig(**kwargs)

    edge_index = getattr(data, "edge_index", None)
    if edge_index is None or edge_index.numel() == 0:
        edge_list: List[Tuple[int, int]] = []
    else:
        edge_list = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    num_nodes = int(data.x.size(0))
    nodes, info = select_cross_scale_nodes(edge_list, num_nodes, config)
    sub = induced_subgraph(data, nodes)

    info.update(
        {
            "num_full_nodes": num_nodes,
            "num_full_edges": int(edge_index.size(1) // 2) if edge_index is not None else 0,
            "num_nodes": int(sub.x.size(0)),
            "num_edges": int(sub.edge_index.size(1) // 2),
            "num_classes_covered": int(sub.y.unique().numel()),
        }
    )
    return sub, info


def cross_scale_statistics(sampled_source, full_source, target) -> Dict:
    """Per-task statistics used to reproduce Table 1 and Table 4 of the paper."""
    import torch

    def _counts(graph):
        edges = getattr(graph, "edge_index", None)
        num_edges = int(edges.size(1) // 2) if edges is not None and edges.numel() else 0
        return int(graph.x.size(0)), num_edges

    full_n, full_e = _counts(full_source)
    tgt_n, tgt_e = _counts(target)
    src_n, src_e = _counts(sampled_source)

    degrees = torch.zeros(src_n)
    if sampled_source.edge_index.numel() > 0:
        degrees.scatter_add_(
            0,
            sampled_source.edge_index[0],
            torch.ones(sampled_source.edge_index.size(1)),
        )

    num_classes = int(full_source.y.max().item()) + 1
    return {
        "full_source_nodes": full_n,
        "full_source_edges": full_e,
        "target_nodes": tgt_n,
        "target_edges": tgt_e,
        "sampled_source_nodes": src_n,
        "sampled_source_edges": src_e,
        "node_scale_gap": tgt_n / max(1, src_n),
        "edge_scale_gap": tgt_e / max(1, src_e),
        "class_coverage": float(sampled_source.y.unique().numel()) / max(1, num_classes),
        "isolated_node_ratio": float((degrees == 0).float().mean()),
        "mean_degree": float(degrees.mean()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main() -> None:
    from lgsa.datasets import load_graph

    parser = argparse.ArgumentParser(description="Inspect the cross-scale sampler S_rho.")
    parser.add_argument("--source", required=True, help="source dataset name, e.g. ACMv9")
    parser.add_argument("--data-root", default=None, help="dataset root directory")
    parser.add_argument("--ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--num-bfs-clusters", type=int, default=None)
    args = parser.parse_args()

    graph = load_graph(args.source, root=args.data_root)
    config = CrossScaleConfig(
        ratio=args.ratio,
        seed=args.seed,
        alpha=args.alpha,
        num_bfs_clusters=args.num_bfs_clusters,
    )
    _, info = sample_cross_scale_source(graph, config)
    width = max(len(k) for k in info)
    for key, value in info.items():
        print(f"{key:<{width}} : {value}")


if __name__ == "__main__":
    _main()
