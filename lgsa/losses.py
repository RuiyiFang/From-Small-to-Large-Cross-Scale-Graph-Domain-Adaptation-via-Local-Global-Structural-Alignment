r"""Loss functions of LGSA (Eq. 5, 7, 10, 12).

Overall objective (Eq. 13)

.. math::
    \mathcal L = \mathcal L_{fuse}
               + \mu\,\mathcal L_{glo}
               + (1-\mu)\,\mathcal L_{loc}
               + \eta\,\mathcal L_{cons}
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

__all__ = [
    "local_loss",
    "global_loss",
    "fusion_loss",
    "consistency_loss",
    "class_prototypes",
]


def class_prototypes(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    mask: Optional[torch.Tensor] = None,
):
    """Class-wise mean embeddings and a mask of the classes that are present."""
    if mask is not None:
        embeddings = embeddings[mask]
        labels = labels[mask]

    prototypes = torch.zeros(num_classes, embeddings.size(1), device=embeddings.device)
    counts = torch.zeros(num_classes, device=embeddings.device)
    if labels.numel() > 0:
        prototypes.index_add_(0, labels, embeddings)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=embeddings.dtype))
    present = counts > 0
    prototypes[present] = prototypes[present] / counts[present].unsqueeze(1)
    return prototypes, present


def local_loss(
    source_logits: torch.Tensor,
    source_labels: torch.Tensor,
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    confident_mask: torch.Tensor,
    pseudo_labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    r"""Eq. 5 -- source cross-entropy plus class-prototype matching.

    .. math::
        \mathcal L_{loc} = \mathrm{CE}(Z^S_{loc}, Y^S)
            + \sum_{c=1}^{C} \big\| \bar h^S_{loc,c} - \bar h^T_{loc,c} \big\|_2^2

    Classes without confident target pseudo labels skip their prototype term.
    """
    loss = F.cross_entropy(source_logits, source_labels)

    if confident_mask.any():
        source_prototypes, source_present = class_prototypes(
            source_embeddings, source_labels, num_classes
        )
        target_prototypes, target_present = class_prototypes(
            target_embeddings, pseudo_labels, num_classes, mask=confident_mask
        )
        shared = source_present & target_present
        if shared.any():
            difference = source_prototypes[shared] - target_prototypes[shared]
            loss = loss + difference.pow(2).sum(dim=1).sum()
    return loss


def global_loss(
    source_logits: torch.Tensor,
    source_labels: torch.Tensor,
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
) -> torch.Tensor:
    r"""Eq. 7 -- source cross-entropy plus marginal mean-embedding alignment."""
    cross_entropy = F.cross_entropy(source_logits, source_labels)
    difference = source_embeddings.mean(dim=0) - target_embeddings.mean(dim=0)
    return cross_entropy + difference.pow(2).sum()


def fusion_loss(
    source_logits: torch.Tensor,
    source_labels: torch.Tensor,
    target_logits: torch.Tensor,
    confident_mask: torch.Tensor,
    pseudo_labels: torch.Tensor,
) -> torch.Tensor:
    r"""Eq. 10 -- source supervision plus confident target pseudo-label supervision.

    The second term is omitted when :math:`V^T_{conf} = \emptyset`.
    """
    loss = F.cross_entropy(source_logits, source_labels)
    if confident_mask.any():
        loss = loss + F.cross_entropy(
            target_logits[confident_mask], pseudo_labels[confident_mask]
        )
    return loss


def consistency_loss(
    local_projection: torch.Tensor,
    global_projection: torch.Tensor,
    temperature: float = 0.5,
    max_nodes: int = 4096,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    r"""Eq. 12 -- InfoNCE between the two target-side views.

    The two projections of the same target node form the positive pair; all other
    target nodes are negatives (the denominator includes :math:`u = v`).

    ``max_nodes`` sub-samples the target nodes used as the negative pool in each
    step, which bounds the :math:`O(|V^T|^2)` similarity matrix on the large
    target graphs.  Set it to ``0`` to use every target node.
    """
    num_nodes = local_projection.size(0)
    if num_nodes < 2:
        return local_projection.sum() * 0.0

    if max_nodes and num_nodes > max_nodes:
        index = torch.randperm(num_nodes, device=local_projection.device, generator=generator)
        index = index[:max_nodes]
        local_projection = local_projection[index]
        global_projection = global_projection[index]

    logits = local_projection @ global_projection.t() / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, targets)
