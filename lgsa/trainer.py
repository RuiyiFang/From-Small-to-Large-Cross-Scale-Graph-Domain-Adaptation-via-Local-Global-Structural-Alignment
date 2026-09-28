r"""Training procedure of LGSA -- the implementation of Algorithm 1 (Appendix H).

One-time pre-processing (independent of the model parameters):

1. cross-domain node augmentation :math:`\tilde G^S` (Eq. 2-3);
2. label-signal propagation :math:`\tilde Y^S`;
3. structural descriptors :math:`\Phi^S, \Phi^T`.

Then, for :math:`E` epochs, both branches and the fusion module are optimised
jointly.  The confident target set :math:`V^T_{conf}` and its pseudo labels are
refreshed **once per epoch from detached predictions**, so no gradient is
back-propagated through the pseudo-label selection.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import torch
from torch_geometric.data import Data

from lgsa.augment import cross_domain_augment, propagate_label_signal, AugmentedSource
from lgsa.config import LGSAConfig
from lgsa.descriptors import DESCRIPTOR_DIM, standardise_descriptors, structural_descriptor
from lgsa.losses import consistency_loss, fusion_loss, global_loss, local_loss
from lgsa.models import LGSA
from lgsa.utils import evaluate, resolve_device, set_seed

__all__ = ["train_lgsa", "prepare_inputs"]


def _identity_augmentation(source: Data, num_classes: int) -> AugmentedSource:
    """LGSA1 ablation: keep the truncated source graph and a zero label signal."""
    num_source = int(source.x.size(0))
    return AugmentedSource(
        x=source.x.clone(),
        edge_index=source.edge_index.clone(),
        label_signal=torch.zeros(num_source, num_classes),
        num_source=num_source,
        y=source.y.clone(),
        recruited=torch.zeros(0, dtype=torch.long),
    )


def prepare_inputs(
    source: Data,
    target: Data,
    num_classes: int,
    config: LGSAConfig,
    device: torch.device,
) -> Dict:
    """Run the one-time pre-processing of Algorithm 1."""
    if config.use_cross_domain_augmentation:
        augmented = cross_domain_augment(
            source,
            target,
            num_neighbours=config.num_augmented_neighbours,
            device=device,
        )
        augmented = propagate_label_signal(
            augmented,
            num_classes=num_classes,
            steps=config.propagation_steps,
            teleport=config.teleport,
        )
    else:
        augmented = _identity_augmentation(source, num_classes)

    source_phi = structural_descriptor(source)
    target_phi = structural_descriptor(target)
    source_phi, target_phi = standardise_descriptors(source_phi, target_phi)

    return {
        "augmented": augmented.to(device),
        "source_labels": source.y.to(device),
        "source_phi": source_phi.to(device),
        "target_x": target.x.to(device),
        "target_edge_index": target.edge_index.to(device),
        "target_phi": target_phi.to(device),
        "target_zero_signal": torch.zeros(target.x.size(0), num_classes, device=device),
        "num_recruited": int(augmented.recruited.numel()),
    }


def train_lgsa(
    source: Data,
    target: Data,
    num_classes: int,
    config: Optional[LGSAConfig] = None,
    verbose: bool = True,
) -> Dict:
    """Train LGSA on one CSGDA task and return the target-side metrics.

    Parameters
    ----------
    source:
        The **sampled** source graph :math:`G^S` produced by
        :func:`lgsa.cross_scale.sample_cross_scale_source`.
    target:
        The fully observed target graph.  Its labels are used **only** for the
        final evaluation, never during training.
    num_classes:
        Size of the shared label space :math:`C`.
    """
    config = config or LGSAConfig()
    set_seed(config.seed)
    device = resolve_device(config.device)

    inputs = prepare_inputs(source, target, num_classes, config, device)
    augmented: AugmentedSource = inputs["augmented"]
    source_labels = inputs["source_labels"]
    target_labels = target.y.to(device)
    source_slice = torch.arange(augmented.num_source, device=device)

    model = LGSA(
        in_dim=int(source.x.size(1)),
        num_classes=num_classes,
        descriptor_dim=DESCRIPTOR_DIM,
        hidden_dim=config.hidden_dim,
        projection_dim=config.projection_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        use_global_branch=config.use_global_branch,
        use_adaptive_fusion=config.use_adaptive_fusion,
    ).to(device)

    optimiser = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # V^T_conf starts empty (Algorithm 1)
    confident_mask = torch.zeros(target.x.size(0), dtype=torch.bool, device=device)
    pseudo_labels = torch.zeros(target.x.size(0), dtype=torch.long, device=device)

    use_consistency = config.use_global_branch and config.use_adaptive_fusion
    start = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimiser.zero_grad()

        source_out = model(
            augmented.x,
            augmented.label_signal,
            augmented.edge_index,
            descriptor=inputs["source_phi"],
            node_slice=source_slice,
        )
        target_out = model(
            inputs["target_x"],
            inputs["target_zero_signal"],
            inputs["target_edge_index"],
            descriptor=inputs["target_phi"],
        )

        loss_fuse = fusion_loss(
            source_out.fused_logits,
            source_labels,
            target_out.fused_logits,
            confident_mask,
            pseudo_labels,
        )
        loss_loc = local_loss(
            source_out.local_logits,
            source_labels,
            source_out.local_embedding,
            target_out.local_embedding,
            confident_mask,
            pseudo_labels,
            num_classes,
        )
        if config.use_global_branch:
            loss_glo = global_loss(
                source_out.global_logits,
                source_labels,
                source_out.global_embedding,
                target_out.global_embedding,
            )
        else:
            loss_glo = torch.zeros((), device=device)

        if use_consistency:
            loss_cons = consistency_loss(
                model.local_projection(target_out.local_embedding),
                model.global_projection(target_out.global_embedding),
                temperature=config.contrastive_temperature,
                max_nodes=config.consistency_max_nodes,
            )
        else:
            loss_cons = torch.zeros((), device=device)

        loss = (
            loss_fuse
            + config.mu * loss_glo
            + (1.0 - config.mu) * loss_loc
            + config.eta * loss_cons
        )
        loss.backward()
        if config.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimiser.step()

        # ---- pseudo-label refresh from detached predictions (Eq. 9) ----
        with torch.no_grad():
            model.eval()
            evaluation_out = model(
                inputs["target_x"],
                inputs["target_zero_signal"],
                inputs["target_edge_index"],
                descriptor=inputs["target_phi"],
            )
            probabilities = torch.softmax(evaluation_out.fused_logits, dim=1)
            confidence, prediction = probabilities.max(dim=1)
            if epoch >= config.pseudo_warmup_epochs:
                confident_mask = confidence >= config.pseudo_threshold
                pseudo_labels = prediction

        if verbose and (epoch % config.log_every == 0 or epoch == config.epochs):
            metrics = evaluate(evaluation_out.fused_logits, target_labels)
            print(
                f"    epoch {epoch:>4d} | loss {loss.item():.4f} "
                f"(fuse {loss_fuse.item():.3f} loc {loss_loc.item():.3f} "
                f"glo {loss_glo.item():.3f} cons {loss_cons.item():.3f}) "
                f"| |V_conf| {int(confident_mask.sum())} "
                f"| acc {metrics['accuracy']:.4f}"
            )

    # ---- final target prediction (no target labels were used for selection) ----
    model.eval()
    with torch.no_grad():
        final_out = model(
            inputs["target_x"],
            inputs["target_zero_signal"],
            inputs["target_edge_index"],
            descriptor=inputs["target_phi"],
        )
        metrics = evaluate(final_out.fused_logits, target_labels)
        metrics["local_accuracy"] = evaluate(final_out.local_logits, target_labels)["accuracy"]
        metrics["global_accuracy"] = evaluate(final_out.global_logits, target_labels)["accuracy"]
        metrics["mean_gate"] = float(final_out.gate.mean())

    metrics["train_seconds"] = time.time() - start
    metrics["num_recruited_nodes"] = inputs["num_recruited"]
    return metrics
