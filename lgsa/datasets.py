"""Dataset loading and the CSGDA transfer-task registry.

Four benchmark families are used in the paper (Appendix J):

========  =====================================  =======  =======  =======
Family    Graphs                                 #Node    #Edge    #Label
========  =====================================  =======  =======  =======
Airport   USA / Brazil / Europe                  1190 /   13599 /  4
                                                 131 /    1038 /
                                                 399      5995
Citation  ACMv9 / Citationv1 / DBLPv7            9360 /   15556 /  5
                                                 8935 /   15098 /
                                                 5484     8117
Social    Blog1 / Blog2                          2300 /   33471 /  6
                                                 2896     53836
ACM       ACM3 (P-S-P) / ACM4 (P-A-P)            3025 /   2221699/ 3
                                                 4019     57853
========  =====================================  =======  =======  =======

Expected on-disk layout (the standard ``torch_geometric.InMemoryDataset``
convention, which is what ``pygda`` produces)::

    <data_root>/
      ACMv9/processed/data.pt
      Citationv1/processed/data.pt
      DBLPv7/processed/data.pt
      USA/processed/data.pt
      BRAZIL/processed/data.pt
      EUROPE/processed/data.pt
      Blog1/processed/data.pt
      Blog2/processed/data.pt
      ACM3/processed/data.pt
      ACM4/processed/data.pt

Each ``data.pt`` holds a PyG ``Data`` (or an ``(data, slices)`` tuple) with
``x`` (node features), ``y`` (node labels) and ``edge_index``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data

__all__ = [
    "TransferTask",
    "DATASET_DIRS",
    "TASKS",
    "load_graph",
    "load_task",
    "resolve_data_root",
]

#: Environment variable consulted when ``root`` is not given explicitly.
DATA_ROOT_ENV = "LGSA_DATA_ROOT"

#: Canonical dataset name -> candidate directory names on disk.
DATASET_DIRS: Dict[str, Tuple[str, ...]] = {
    "USA": ("USA", "usa"),
    "Brazil": ("BRAZIL", "Brazil", "brazil"),
    "Europe": ("EUROPE", "Europe", "europe"),
    "ACMv9": ("ACMv9", "acmv9", "ACMV9"),
    "Citationv1": ("Citationv1", "citationv1", "CITATIONV1"),
    "DBLPv7": ("DBLPv7", "dblpv7", "DBLPV7"),
    "Blog1": ("Blog1", "blog1", "BLOG1"),
    "Blog2": ("Blog2", "blog2", "BLOG2"),
    "ACM3": ("ACM3", "acm3"),
    "ACM4": ("ACM4", "acm4"),
}

#: Short aliases used in the paper's tables.
ALIASES: Dict[str, str] = {
    "U": "USA", "B": "Brazil", "E": "Europe",
    "A": "ACMv9", "C": "Citationv1", "D": "DBLPv7",
    "B1": "Blog1", "B2": "Blog2",
    "A3": "ACM3", "A4": "ACM4",
}


@dataclass(frozen=True)
class TransferTask:
    """A single CSGDA transfer task ``source -> target``."""

    source: str
    target: str
    family: str

    @property
    def name(self) -> str:
        return f"{self.source}->{self.target}"

    @property
    def short_name(self) -> str:
        inverse = {v: k for k, v in ALIASES.items()}
        return f"{inverse.get(self.source, self.source)}->{inverse.get(self.target, self.target)}"


def _tasks(pairs: List[Tuple[str, str]], family: str) -> List[TransferTask]:
    return [TransferTask(ALIASES[s], ALIASES[t], family) for s, t in pairs]


#: All 20 transfer tasks evaluated in Tables 2 and 3.
TASKS: List[TransferTask] = (
    _tasks(
        [("A", "D"), ("D", "A"), ("A", "C"), ("C", "A"), ("C", "D"), ("D", "C")],
        "Citation",
    )
    + _tasks([("A3", "A4"), ("A4", "A3")], "ACM")
    + _tasks(
        [("U", "B"), ("U", "E"), ("B", "U"), ("B", "E"), ("E", "U"), ("E", "B")],
        "Airport",
    )
    + _tasks([("B1", "B2"), ("B2", "B1")], "Social")
)

TASK_BY_NAME: Dict[str, TransferTask] = {t.short_name: t for t in TASKS}
TASK_BY_NAME.update({t.name: t for t in TASKS})


def resolve_data_root(root: Optional[str] = None) -> str:
    """Resolve the dataset root from the argument, the env var, or ``./data``."""
    if root:
        return root
    if os.environ.get(DATA_ROOT_ENV):
        return os.environ[DATA_ROOT_ENV]
    return os.path.join(os.getcwd(), "data")


def _load_processed(folder: str) -> Optional[Data]:
    path = os.path.join(folder, "processed", "data.pt")
    if not os.path.exists(path):
        path = os.path.join(folder, "data.pt")
        if not os.path.exists(path):
            return None
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, tuple) and len(obj) == 2:
        obj = obj[0]
    if isinstance(obj, list):
        obj = obj[0]
    return obj if isinstance(obj, Data) else None


def load_graph(name: str, root: Optional[str] = None) -> Data:
    """Load one benchmark graph by canonical name or paper alias."""
    name = ALIASES.get(name, name)
    root = resolve_data_root(root)
    candidates = DATASET_DIRS.get(name, (name,))

    tried: List[str] = []
    for candidate in candidates:
        folder = os.path.join(root, candidate)
        tried.append(folder)
        data = _load_processed(folder)
        if data is None:
            # some releases nest the dataset one level deeper
            data = _load_processed(os.path.join(folder, candidate))
        if data is not None:
            return _sanitise(data, name)

    raise FileNotFoundError(
        f"Could not load dataset '{name}'. Tried:\n  "
        + "\n  ".join(tried)
        + f"\nSet the dataset root with --data-root or the {DATA_ROOT_ENV} env var."
    )


def _sanitise(data: Data, name: str) -> Data:
    if getattr(data, "x", None) is None:
        raise ValueError(f"dataset '{name}' has no node features `x`")
    if getattr(data, "y", None) is None:
        raise ValueError(f"dataset '{name}' has no node labels `y`")
    if getattr(data, "edge_index", None) is None:
        data.edge_index = torch.zeros(2, 0, dtype=torch.long)

    data.x = data.x.float()
    data.y = data.y.view(-1).long()
    data.edge_index = data.edge_index.long()

    # make the graph undirected and drop self-loops / duplicates
    edge_index = data.edge_index
    if edge_index.numel() > 0:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        mask = edge_index[0] != edge_index[1]
        data.edge_index = _unique_columns(edge_index[:, mask])
    return data


def _unique_columns(edge_index: torch.Tensor) -> torch.Tensor:
    """Deduplicate edge columns while preserving the (i, j) / (j, i) pairing."""
    order = torch.argsort(edge_index[0] * (edge_index.max() + 1) + edge_index[1])
    sorted_edges = edge_index[:, order]
    keep = torch.ones(sorted_edges.size(1), dtype=torch.bool)
    if sorted_edges.size(1) > 1:
        same = (sorted_edges[0, 1:] == sorted_edges[0, :-1]) & (
            sorted_edges[1, 1:] == sorted_edges[1, :-1]
        )
        keep[1:] = ~same
    return sorted_edges[:, keep]


def load_task(task: "TransferTask | str", root: Optional[str] = None) -> Tuple[Data, Data, TransferTask]:
    """Load ``(full_source_graph, target_graph, task)`` for a CSGDA transfer task."""
    if isinstance(task, str):
        if task not in TASK_BY_NAME:
            raise KeyError(f"unknown task '{task}'. Known: {sorted(TASK_BY_NAME)}")
        task = TASK_BY_NAME[task]

    source = load_graph(task.source, root)
    target = load_graph(task.target, root)

    if source.x.size(1) != target.x.size(1):
        raise ValueError(
            f"feature dimensions differ for {task.name}: "
            f"{source.x.size(1)} vs {target.x.size(1)}. "
            "CSGDA assumes a shared input feature space."
        )
    return source, target, task
