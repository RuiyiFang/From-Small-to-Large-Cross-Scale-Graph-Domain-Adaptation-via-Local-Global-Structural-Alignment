r"""Local information compensation (Sec. 3.3).

Two one-time pre-processing steps rebuild the neighbourhood context that source
sampling destroyed:

1. **Cross-domain node augmentation** (Eq. 2-3).  Every sampled source node
   :math:`v_s` is connected to its top-:math:`M` most similar target nodes in
   feature space,

   .. math::
       N_{aug}(v_s) = \mathrm{TopM}_{v_t \in V^T}
                      \cos\!\big(x^S_{v_s},\, x^T_{v_t}\big),

   giving the augmented source graph
   :math:`\tilde G^S = \{\tilde V^S,\, E^S \cup E_{aug},\, \tilde A^S,\, \tilde X^S\}`.

2. **Label signal propagation**.  Source supervision is diffused over
   :math:`\tilde A^S` for :math:`T` iterations with teleport coefficient
   :math:`\alpha`, starting from the one-hot source labels on the original
   source nodes and zeros on the recruited target-side nodes.  The resulting
   soft matrix :math:`\tilde Y^S` is used **only as an input-side signal** of
   the local GCN branch -- never as a training target.

Neither step uses target labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

__all__ = ["AugmentedSource", "cross_domain_augment", "propagate_label_signal"]


@dataclass
class AugmentedSource:
    r"""The augmented source graph :math:`\tilde G^S` and its label signal.

    Attributes
    ----------
    x:
        ``(|V~^S|, d)`` features; rows ``[:num_source]`` are the original source
        nodes, the remaining rows are the recruited target-side nodes.
    edge_index:
        ``(2, |E^S| + |E_aug|)`` undirected edges of :math:`\tilde G^S`.
    label_signal:
        ``(|V~^S|, C)`` propagated label signal :math:`\tilde Y^S`.
    num_source:
        :math:`N^S`; only these rows carry ground-truth labels and are read back
        as :math:`H^S_{loc}` after the source-side forward pass.
    y:
        ``(N^S,)`` ground-truth source labels.
    recruited:
        Target node indices that were pulled into :math:`\tilde G^S`.
    """

    x: torch.Tensor
    edge_index: torch.Tensor
    label_signal: torch.Tensor
    num_source: int
    y: torch.Tensor
    recruited: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))

    def to(self, device: torch.device) -> "AugmentedSource":
        return AugmentedSource(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            label_signal=self.label_signal.to(device),
            num_source=self.num_source,
            y=self.y.to(device),
            recruited=self.recruited.to(device),
        )


@torch.no_grad()
def cross_domain_augment(
    source: Data,
    target: Data,
    num_neighbours: int = 5,
    chunk_size: int = 512,
    device: Optional[torch.device] = None,
) -> AugmentedSource:
    r"""Build :math:`\tilde G^S` by attaching the top-:math:`M` target neighbours.

    The similarity search is a single one-shot :math:`O(N^S N^T d)` pass executed
    before training (Appendix F), chunked over source nodes to bound memory.

    Parameters
    ----------
    source:
        The sampled source graph :math:`G^S`.
    target:
        The fully observed (unlabelled) target graph :math:`G^T`.
    num_neighbours:
        :math:`M`, the number of cross-domain neighbours per source node.
    """
    device = device or torch.device("cpu")
    num_source = int(source.x.size(0))
    num_target = int(target.x.size(0))
    num_neighbours = max(0, min(num_neighbours, num_target))

    source_x = source.x.to(device)
    target_x = target.x.to(device)

    if num_neighbours == 0:
        recruited = torch.zeros(0, dtype=torch.long)
        cross_edges = torch.zeros(2, 0, dtype=torch.long)
    else:
        source_norm = F.normalize(source_x, dim=1, eps=1e-8)
        target_norm = F.normalize(target_x, dim=1, eps=1e-8)

        topk_indices = []
        for start in range(0, num_source, chunk_size):
            block = source_norm[start : start + chunk_size]
            similarity = block @ target_norm.t()  # (chunk, N^T)
            topk_indices.append(similarity.topk(num_neighbours, dim=1).indices.cpu())
        topk = torch.cat(topk_indices, dim=0)  # (N^S, M)

        # each distinct recruited target node becomes one node of G~^S
        recruited, inverse = torch.unique(topk.reshape(-1), return_inverse=True)
        local_target_id = inverse.view(num_source, num_neighbours) + num_source
        source_ids = (
            torch.arange(num_source).unsqueeze(1).expand(-1, num_neighbours).reshape(-1)
        )
        cross_edges = torch.stack([source_ids, local_target_id.reshape(-1)], dim=0)
        cross_edges = torch.cat([cross_edges, cross_edges.flip(0)], dim=1)

    x = torch.cat([source.x, target.x[recruited]], dim=0)
    edge_index = torch.cat([source.edge_index, cross_edges], dim=1)

    return AugmentedSource(
        x=x,
        edge_index=edge_index,
        label_signal=torch.zeros(x.size(0), 0),  # filled by propagate_label_signal
        num_source=num_source,
        y=source.y.clone(),
        recruited=recruited,
    )


@torch.no_grad()
def propagate_label_signal(
    augmented: AugmentedSource,
    num_classes: int,
    steps: int = 3,
    teleport: float = 0.5,
) -> AugmentedSource:
    r"""Diffuse source labels over :math:`\tilde A^S` to obtain :math:`\tilde Y^S`.

    .. math::
        \tilde Y^{(0)} = [\,Y^S ;\, \mathbf{0}\,], \qquad
        \tilde Y^{(t+1)} = (1-\alpha)\, \hat{\tilde A}^S \tilde Y^{(t)}
                           + \alpha\, \tilde Y^{(0)}

    The teleport coefficient :math:`\alpha` preserves the original source
    supervision during diffusion; after :math:`T` steps the rows over the
    recruited target-side nodes act as soft label signals.
    """
    num_nodes = augmented.num_nodes
    signal_0 = torch.zeros(num_nodes, num_classes)
    signal_0[: augmented.num_source] = F.one_hot(augmented.y, num_classes).float()

    edge_index = augmented.edge_index
    if edge_index.numel() == 0 or steps <= 0:
        augmented.label_signal = signal_0
        return augmented

    # symmetric normalisation  D^{-1/2} A D^{-1/2}
    degree = torch.zeros(num_nodes).scatter_add_(
        0, edge_index[0], torch.ones(edge_index.size(1))
    )
    deg_inv_sqrt = degree.clamp(min=1).pow(-0.5)
    weights = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]

    signal = signal_0.clone()
    for _ in range(steps):
        aggregated = torch.zeros_like(signal)
        aggregated.index_add_(0, edge_index[0], signal[edge_index[1]] * weights.unsqueeze(1))
        signal = (1.0 - teleport) * aggregated + teleport * signal_0

    augmented.label_signal = signal
    return augmented
