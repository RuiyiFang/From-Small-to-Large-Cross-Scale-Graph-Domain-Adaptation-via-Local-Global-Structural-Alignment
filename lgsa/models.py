r"""Network modules of LGSA (Sec. 3.3 - 3.5).

* :class:`LocalBranch`  -- :math:`L`-layer GCN over :math:`[\tilde X \| \tilde Y]`
  (Eq. 4), shared across domains.
* :class:`GlobalBranch` -- MLP over :math:`[X \| \Phi]` (Eq. 6), no message
  passing, therefore decoupled from the incomplete source adjacency.
* :class:`AdaptiveFusion` -- entropy-based branch reliability (Eq. 8) and the
  sigmoid gate :math:`\gamma_v` with a ``softplus`` slope, which keeps the gate
  monotonically increasing in the local-branch reliability advantage.
* :class:`ProjectionHead` -- 2-layer head used by the InfoNCE consistency loss
  (Eq. 11).
* :class:`LGSA` -- the full model, returning every quantity needed by the losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

__all__ = ["LocalBranch", "GlobalBranch", "AdaptiveFusion", "ProjectionHead", "LGSA", "BranchOutput"]


@dataclass
class BranchOutput:
    """Embeddings and logits of one forward pass."""

    local_embedding: torch.Tensor
    local_logits: torch.Tensor
    global_embedding: torch.Tensor
    global_logits: torch.Tensor
    fused_embedding: torch.Tensor
    fused_logits: torch.Tensor
    gate: torch.Tensor


class LocalBranch(nn.Module):
    r"""Shared :math:`L`-layer GCN encoder (Eq. 4).

    The input dimension is ``in_dim + num_classes``: the label-signal channels
    carry :math:`\tilde Y^S` on the source side and zeros on the target side, so
    that both forward passes share the same encoder weights.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        dims = [in_dim + num_classes] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(
            GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def encode(self, x: torch.Tensor, label_signal: torch.Tensor, edge_index: torch.Tensor):
        h = torch.cat([x, label_signal], dim=1)
        for index, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if index < len(self.convs) - 1:
                h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
            else:
                h = F.relu(h)
        return h

    def forward(self, x: torch.Tensor, label_signal: torch.Tensor, edge_index: torch.Tensor):
        h = self.encode(x, label_signal, edge_index)
        return h, self.classifier(h)


class GlobalBranch(nn.Module):
    r"""Shared MLP encoder over :math:`[X \| \Phi]` (Eq. 6)."""

    def __init__(
        self,
        in_dim: int,
        descriptor_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        dims = [in_dim + descriptor_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def encode(self, x: torch.Tensor, descriptor: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, descriptor], dim=1)
        for index, layer in enumerate(self.layers):
            h = layer(h)
            if index < len(self.layers) - 1:
                h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
            else:
                h = F.relu(h)
        return h

    def forward(self, x: torch.Tensor, descriptor: torch.Tensor):
        h = self.encode(x, descriptor)
        return h, self.classifier(h)


class AdaptiveFusion(nn.Module):
    r"""Entropy-gated fusion of the two branches (Sec. 3.5).

    .. math::
        r_{\cdot,v} = 1 - \frac{H(\mathrm{softmax}(z_{\cdot,v}))}{\log C}, \qquad
        \gamma_v = \sigma\!\big(\mathrm{softplus}(\omega_g)(r_{loc,v}-r_{glo,v}) + b_g\big)

    ``omega`` and ``bias`` are the only learnable parameters; the ``softplus``
    re-parameterisation guarantees a non-negative slope, i.e. the gate can never
    invert the meaning of "the local branch is more reliable".
    """

    def __init__(self, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.omega = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.log_num_classes = float(torch.log(torch.tensor(float(num_classes))))

    def reliability(self, logits: torch.Tensor) -> torch.Tensor:
        probabilities = F.softmax(logits, dim=1)
        entropy = -(probabilities * torch.log(probabilities.clamp(min=1e-12))).sum(dim=1)
        return 1.0 - entropy / self.log_num_classes

    def forward(
        self,
        local_embedding: torch.Tensor,
        local_logits: torch.Tensor,
        global_embedding: torch.Tensor,
        global_logits: torch.Tensor,
    ):
        gap = self.reliability(local_logits) - self.reliability(global_logits)
        gate = torch.sigmoid(F.softplus(self.omega) * gap + self.bias).unsqueeze(1)
        fused = gate * local_embedding + (1.0 - gate) * global_embedding
        return fused, self.classifier(fused), gate.squeeze(1)


class UniformFusion(AdaptiveFusion):
    """Ablation LGSA3: replace the learned gate by uniform 0.5 averaging."""

    def forward(self, local_embedding, local_logits, global_embedding, global_logits):
        fused = 0.5 * local_embedding + 0.5 * global_embedding
        gate = torch.full((fused.size(0),), 0.5, device=fused.device)
        return fused, self.classifier(fused), gate


class ProjectionHead(nn.Module):
    r"""2-layer projection head with :math:`\ell_2` normalisation (Eq. 11)."""

    def __init__(self, in_dim: int, projection_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, projection_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(h), dim=1, eps=1e-8)


class LGSA(nn.Module):
    """The complete LGSA model."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        descriptor_dim: int = 4,
        hidden_dim: int = 128,
        projection_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.5,
        use_global_branch: bool = True,
        use_adaptive_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_global_branch = use_global_branch

        self.local = LocalBranch(in_dim, num_classes, hidden_dim, num_layers, dropout)
        self.global_ = (
            GlobalBranch(in_dim, descriptor_dim, num_classes, hidden_dim, num_layers, dropout)
            if use_global_branch
            else None
        )

        fusion_class = AdaptiveFusion if use_adaptive_fusion else UniformFusion
        self.fusion = fusion_class(hidden_dim, num_classes)
        self.local_projection = ProjectionHead(hidden_dim, projection_dim)
        self.global_projection = ProjectionHead(hidden_dim, projection_dim)

    def forward(
        self,
        x: torch.Tensor,
        label_signal: torch.Tensor,
        edge_index: torch.Tensor,
        descriptor: Optional[torch.Tensor] = None,
        node_slice: Optional[torch.Tensor] = None,
    ) -> BranchOutput:
        """Run both branches on one graph.

        ``node_slice`` selects the rows to keep after the source-side forward
        pass over the augmented graph -- the recruited target-side nodes are
        discarded, as specified in Sec. 3.3.
        """
        local_embedding, local_logits = self.local(x, label_signal, edge_index)
        if node_slice is not None:
            local_embedding = local_embedding[node_slice]
            local_logits = local_logits[node_slice]
            x = x[node_slice]

        if self.global_ is None:
            # LGSA2 ablation: the "global" view degenerates to the local one, so
            # the fused prediction is read off the local logits.
            return BranchOutput(
                local_embedding=local_embedding,
                local_logits=local_logits,
                global_embedding=local_embedding,
                global_logits=local_logits,
                fused_embedding=local_embedding,
                fused_logits=local_logits,
                gate=torch.ones(local_embedding.size(0), device=local_embedding.device),
            )

        global_embedding, global_logits = self.global_(x, descriptor)
        fused_embedding, fused_logits, gate = self.fusion(
            local_embedding, local_logits, global_embedding, global_logits
        )
        return BranchOutput(
            local_embedding=local_embedding,
            local_logits=local_logits,
            global_embedding=global_embedding,
            global_logits=global_logits,
            fused_embedding=fused_embedding,
            fused_logits=fused_logits,
            gate=gate,
        )
