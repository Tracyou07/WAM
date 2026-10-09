"""Two-component weighted regression evidence, independent of policy owners."""
from __future__ import annotations

import torch


def route_objective_weighted(
    pred_private: torch.Tensor,
    pred_shared: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor | None,
    timestep_weight: torch.Tensor,
    prior_shared: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Exact evidence using native float32 token and batch denominators.

    The caller obtains fixed [B,T] weights from the native scheduler once.
    Empty tokens/samples retain the native T/B denominators.
    """
    if target.ndim != 3 or pred_private.shape != target.shape or pred_shared.shape != target.shape:
        raise ValueError("Predictions and target must have matching [B,T,D] shape")
    batch, tokens, dims = target.shape
    if not batch or not tokens or not dims or timestep_weight.shape != (batch, tokens) or prior_shared.shape != (batch,):
        raise ValueError("Invalid weighted evidence geometry")
    tensors = (pred_private, pred_shared, target, timestep_weight, prior_shared)
    if any(t.device != target.device for t in tensors):
        raise ValueError("Weighted evidence tensors must share a device")
    if timestep_weight.requires_grad or (action_mask is not None and action_mask.requires_grad):
        raise ValueError("Mask and scheduler weight must be parameter/route independent")
    if action_mask is not None and (action_mask.shape != target.shape or action_mask.device != target.device):
        raise ValueError("action_mask must match native [B,T,D] shape and device")
    mask = torch.ones_like(target, dtype=torch.float32) if action_mask is None else action_mask.float()
    weight = timestep_weight.float()
    if not torch.isfinite(mask).all() or not torch.isfinite(weight).all() or (mask < 0).any() or (weight < 0).any():
        raise ValueError("Mask and scheduler weight must be finite and nonnegative")
    if not torch.isfinite(prior_shared).all() or ((prior_shared < .1) | (prior_shared > .9)).any():
        raise ValueError("prior_shared must be within the frozen [0.1,0.9] range")
    denominator = mask.sum(-1).clamp_min(1.0)
    branch = []
    for prediction in (pred_private, pred_shared):
        error = torch.nn.functional.mse_loss(prediction.float(), target.detach().float(), reduction="none")
        branch.append(((error * weight[:, :, None] * mask).sum(-1) / denominator).mean(-1))
    energy = torch.stack(branch, dim=-1)
    if not torch.isfinite(energy).all():
        raise ValueError("Nonfinite branch energy")
    prior = torch.stack((1 - prior_shared, prior_shared), dim=-1).float()
    log_joint = prior.log() - energy
    log_posterior = log_joint.log_softmax(-1)
    posterior = log_posterior.exp()
    nll = -log_joint.logsumexp(-1)
    return {
        "loss": nll.mean(),
        "per_sample_nll": nll,
        "branch_weighted_mse": energy,
        "posterior": posterior,
        "kl": (posterior * (log_posterior - prior.log())).sum(-1),
        "coefficient": weight[:, :, None] * mask / (tokens * denominator[:, :, None]),
    }
