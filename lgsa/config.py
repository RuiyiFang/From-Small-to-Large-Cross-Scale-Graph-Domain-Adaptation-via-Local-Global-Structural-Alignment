r"""Hyper-parameters of LGSA.

Defaults follow Appendix D (experimental setup) and Appendix G / Table 12
(per-task loss weights).  Only two loss-level hyper-parameters need tuning:

* :math:`\mu \in [0,1]` -- the local-global balance (the local coefficient is
  fixed to :math:`1-\mu`, with the local branch acting as the anchor of Eq. 13);
* :math:`\eta` -- the weight of the target-side consistency term.

Both are searched over :math:`\{0.01, 0.05, 0.1, 0.5, 1\}`; the response
surfaces are nearly flat inside :math:`[0.05, 0.5]` (Sec. 4.3).

The paper leaves :math:`\tau_p`, :math:`\tau_c`, :math:`M`, :math:`T`,
:math:`\alpha` and :math:`E` unspecified.  The values below are the ones used by
this implementation; ``pseudo_threshold`` and ``pseudo_warmup_epochs`` are
deliberately conservative, so that the target pseudo-label term of Eq. 10 only
starts acting once the fused classifier has settled on the small source graph.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, Tuple

__all__ = ["LGSAConfig", "TASK_LOSS_WEIGHTS", "loss_weights_for_task"]


@dataclass
class LGSAConfig:
    """Full configuration of one LGSA run."""

    # ---- architecture (Appendix D) ----
    hidden_dim: int = 128
    num_layers: int = 2
    projection_dim: int = 16
    dropout: float = 0.5

    # ---- optimisation (Appendix D) ----
    learning_rate: float = 5e-4
    weight_decay: float = 5e-4
    epochs: int = 200
    grad_clip: float = 1.0

    # ---- local information compensation (Sec. 3.3) ----
    num_augmented_neighbours: int = 5        # M in Eq. 2
    propagation_steps: int = 3               # T
    teleport: float = 0.5                    # alpha

    # ---- adaptive alignment (Sec. 3.5) ----
    pseudo_threshold: float = 0.99           # tau_p in Eq. 9
    pseudo_warmup_epochs: int = 100          # keep V^T_conf empty until then
    contrastive_temperature: float = 0.5     # tau_c in Eq. 12
    consistency_max_nodes: int = 4096        # negative-pool size for InfoNCE

    # ---- loss trade-offs (Eq. 13) ----
    mu: float = 0.1                          # global-branch weight
    eta: float = 0.1                         # consistency weight

    # ---- ablation switches (Sec. 4.2) ----
    use_cross_domain_augmentation: bool = True   # LGSA1 removes this
    use_global_branch: bool = True               # LGSA2 removes this
    use_adaptive_fusion: bool = True             # LGSA3 removes this (and L_cons)

    # ---- bookkeeping ----
    seed: int = 42
    device: str = "auto"
    log_every: int = 50

    def as_dict(self) -> Dict:
        return asdict(self)

    def with_task_weights(self, task_short_name: str) -> "LGSAConfig":
        """Return a copy using the per-task ``(mu, eta)`` of Table 12."""
        mu, eta = loss_weights_for_task(task_short_name, (self.mu, self.eta))
        return LGSAConfig(**{**self.as_dict(), "mu": mu, "eta": eta})


#: Table 12 -- per-task (mu, eta) at rho = 5%.
TASK_LOSS_WEIGHTS: Dict[str, Tuple[float, float]] = {
    # Airport
    "U->B": (0.5, 0.5),
    "U->E": (0.5, 0.1),
    "B->U": (0.5, 0.5),
    "B->E": (0.5, 0.5),
    "E->U": (0.5, 0.1),
    "E->B": (0.5, 0.5),
    # Citation
    "A->C": (0.1, 0.1),
    "A->D": (0.1, 0.1),
    "C->A": (0.1, 0.5),
    "C->D": (0.1, 0.1),
    "D->A": (0.1, 0.1),
    "D->C": (0.1, 0.1),
    # Social
    "B1->B2": (0.05, 0.1),
    "B2->B1": (0.05, 0.1),
    # ACM
    "A3->A4": (0.5, 0.1),
    "A4->A3": (0.1, 0.5),
}

#: Grid used for the sensitivity study of Sec. 4.3.
LOSS_WEIGHT_GRID: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.5, 1.0)


def loss_weights_for_task(
    task_short_name: str, default: Tuple[float, float] = (0.1, 0.1)
) -> Tuple[float, float]:
    """Look up ``(mu, eta)`` for a task, falling back to ``default``."""
    return TASK_LOSS_WEIGHTS.get(task_short_name, default)
