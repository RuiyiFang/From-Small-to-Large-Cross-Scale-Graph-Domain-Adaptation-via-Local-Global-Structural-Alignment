r"""Global structural descriptors :math:`\phi(v)` (Sec. 3.4).

For every node the descriptor stacks four scalar topological statistics

.. math::
    \phi(v) = \big(\deg(v),\ c(v),\ n_2(v),\ n_3(v)\big)^\top

where :math:`c(v)` is the local clustering coefficient and :math:`n_k(v)` the
size of the :math:`k`-hop neighbourhood.  Source-side sampling systematically
shrinks all four statistics, and this first-moment shift is exactly what the
global branch of LGSA absorbs through mean-embedding alignment (Eq. 7).

Because the descriptors depend only on the input graphs they are computed
**once** before training (Appendix H), at a cost of
:math:`O(|E|) + O(N\bar d^k)`.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

__all__ = ["DESCRIPTOR_DIM", "structural_descriptor", "standardise_descriptors"]

#: Dimensionality of phi(v).
DESCRIPTOR_DIM = 4


def _to_scipy(edge_index: torch.Tensor, num_nodes: int) -> sp.csr_matrix:
    """Binary, symmetric, self-loop-free CSR adjacency."""
    if edge_index.numel() == 0:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.int8)
    rows = edge_index[0].cpu().numpy()
    cols = edge_index[1].cpu().numpy()
    keep = rows != cols  # drop self-loops before assembling, not after
    rows, cols = rows[keep], cols[keep]
    data = np.ones(rows.shape[0], dtype=np.int8)
    adjacency = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.data[:] = 1
    return adjacency


def _clustering_coefficient(adjacency: sp.csr_matrix) -> np.ndarray:
    """``c(v) = 2 * triangles(v) / (d(v) (d(v) - 1))``."""
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    if adjacency.nnz == 0:
        return np.zeros_like(degree, dtype=np.float64)
    paths = (adjacency @ adjacency).multiply(adjacency)
    triangles = np.asarray(paths.sum(axis=1)).ravel() / 2.0
    denominator = degree * (degree - 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        coefficient = np.where(denominator > 0, 2.0 * triangles / denominator, 0.0)
    return np.clip(coefficient, 0.0, 1.0)


def _khop_sizes(adjacency: sp.csr_matrix, max_hop: int = 3) -> np.ndarray:
    """``(N, max_hop)`` matrix with the size of each k-hop neighbourhood."""
    num_nodes = adjacency.shape[0]
    sizes = np.zeros((num_nodes, max_hop), dtype=np.float64)
    if adjacency.nnz == 0:
        return sizes

    reach = adjacency.copy()
    frontier = adjacency.copy()
    for hop in range(max_hop):
        if hop > 0:
            frontier = (frontier @ adjacency)
            frontier.data[:] = 1
            reach = reach.maximum(frontier)
            reach.data[:] = 1
        counts = np.asarray(reach.sum(axis=1)).ravel()
        # a node is not part of its own k-hop neighbourhood
        counts -= np.asarray(reach.diagonal()).ravel()
        sizes[:, hop] = counts
    return sizes


def structural_descriptor(data: Data, max_hop: int = 3) -> torch.Tensor:
    r"""Compute :math:`\Phi \in \mathbb{R}^{N \times 4}` for one graph.

    Columns are ``[degree, clustering coefficient, n_2(v), n_3(v)]``.
    """
    num_nodes = int(data.x.size(0))
    adjacency = _to_scipy(data.edge_index, num_nodes)

    degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float64)
    clustering = _clustering_coefficient(adjacency)
    khop = _khop_sizes(adjacency, max_hop=max_hop)

    phi = np.stack([degree, clustering, khop[:, 1], khop[:, 2]], axis=1)
    return torch.from_numpy(phi).float()


def standardise_descriptors(
    source_phi: torch.Tensor, target_phi: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Log-compress the count columns and z-score over the union of both domains.

    Standardising *jointly* (rather than per domain) is essential: it keeps the
    relative source/target shift in :math:`\Phi` visible to the mean-embedding
    alignment of Eq. (7) instead of normalising it away.
    """
    source = source_phi.clone()
    target = target_phi.clone()
    # columns 0, 2, 3 are unbounded counts; column 1 already lives in [0, 1]
    for column in (0, 2, 3):
        source[:, column] = torch.log1p(source[:, column])
        target[:, column] = torch.log1p(target[:, column])

    joint = torch.cat([source, target], dim=0)
    mean = joint.mean(dim=0, keepdim=True)
    std = joint.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (source - mean) / std, (target - mean) / std
