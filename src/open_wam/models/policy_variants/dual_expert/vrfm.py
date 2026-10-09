"""Continuous variational conditioning of the two native flow streams.

The posterior observes paired training inputs; the generator only receives z.
No learned prior, private projection path, or routing expert is introduced.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def diagonal_gaussian_kl(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    """KL(q || N(0,I)), summed over latent dimensions, averaged over samples."""
    mean, log_variance = mean.float(), log_variance.float()
    return .5 * (mean.square() + log_variance.exp() - 1 - log_variance).sum(-1).mean()


def _moments(value: torch.Tensor, *, dim: tuple[int, ...], mask: torch.Tensor | None = None):
    value = value.detach().float()
    if mask is None:
        mean = value.mean(dim=dim)
        variance = value.var(dim=dim, correction=0)
    else:
        weight = torch.broadcast_to(mask.detach().float(), value.shape)
        selected = torch.where(weight > 0, value, 0.)
        count = weight.sum(dim=dim).clamp_min(1.)
        mean = (selected * weight).sum(dim=dim) / count
        centered = torch.where(weight > 0, value - _expand_mean(mean, value.ndim, dim), 0.)
        variance = (centered.square() * weight).sum(dim=dim) / count
    deviation = (variance.clamp_min(0.) + 1e-8).sqrt()
    return torch.stack((mean, deviation), dim=-1) if mean.ndim == 1 else torch.cat((mean, deviation), dim=-1)


def _expand_mean(mean: torch.Tensor, ndim: int, dim: tuple[int, ...]):
    for axis in sorted(axis % ndim for axis in dim):
        mean = mean.unsqueeze(axis)
    return mean


class VariationalFlowConditioning(nn.Module):
    """Gaussian encoder plus generator-side additive latent projections."""

    def __init__(self, *, video_channels: int, action_dim: int, text_dim: int,
                 video_hidden_size: int, action_hidden_size: int,
                 latent_dim: int, kl_weight: float):
        super().__init__()
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        features = 4 * video_channels + 4 * action_dim + 4 + text_dim + 2
        self.posterior = nn.Sequential(nn.Linear(features, 128), nn.SiLU(),
                                       nn.Linear(128, 2 * latent_dim))
        self.video_projection = nn.Linear(latent_dim, video_hidden_size, bias=False)
        self.action_projection = nn.Linear(latent_dim, action_hidden_size, bias=False)

    def posterior_sample(self, *, clean_video: torch.Tensor, noisy_video: torch.Tensor,
                         clean_action: torch.Tensor, noisy_action: torch.Tensor,
                         video_timesteps: torch.Tensor, action_timesteps: torch.Tensor,
                         text_context: torch.Tensor, proprio_state: torch.Tensor | None,
                         action_mask: torch.Tensor | None = None):
        if clean_video.shape != noisy_video.shape or clean_action.shape != noisy_action.shape:
            raise ValueError('VRFM requires paired clean/noisy video and action shapes')
        batch = clean_video.shape[0]
        if clean_action.shape[0] != batch or text_context.shape[0] != batch:
            raise ValueError('VRFM paired inputs must have the same batch size')
        proprio = clean_video.new_zeros((batch, 2), dtype=torch.float32)
        if proprio_state is not None:
            proprio = _moments(proprio_state.reshape(batch, -1), dim=(1,)).reshape(batch, 2)
        features = torch.cat((
            _moments(clean_video, dim=(2, 3, 4)),
            _moments(noisy_video, dim=(2, 3, 4)),
            _moments(clean_action, dim=(1,), mask=action_mask),
            _moments(noisy_action, dim=(1,), mask=action_mask),
            _moments(video_timesteps / 1000., dim=(1,)).reshape(batch, 2),
            _moments(action_timesteps / 1000., dim=(1,)).reshape(batch, 2),
            text_context.detach().float().mean(1), proprio,
        ), dim=-1)
        if not torch.isfinite(features).all():
            raise ValueError('VRFM posterior features must be finite')
        parameters = self.posterior(features.to(dtype=self.posterior[0].weight.dtype)).float()
        mean, log_variance = parameters.chunk(2, dim=-1)
        # A finite variance range protects mixed-precision exponentiation.
        log_variance = log_variance.clamp(-20., 10.)
        z = mean + (.5 * log_variance).exp() * torch.randn_like(mean)
        return z, diagonal_gaussian_kl(mean, log_variance) * self.kl_weight

    def prior_sample(self, batch_size: int, *, device: torch.device,
                     generator: torch.Generator | None = None) -> torch.Tensor:
        """Fixed unit Gaussian. This boundary takes no targets or context."""
        return torch.randn(batch_size, self.latent_dim, device=device, generator=generator)

    def condition(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = z.to(dtype=self.video_projection.weight.dtype)
        return self.video_projection(z)[:, None], self.action_projection(z)[:, None]


def configure_vrfm(pipeline, latent_dim: int = 32, kl_weight: float = .001) -> dict:
    """Attach after native trainability selection, before optimizer/wrapping.

    All existing trainability flags and parameter identities are preserved.
    Only the encoder and two latent projections are added, trainable by default.
    """
    from .variant import DualExpertPolicyVariant
    if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim <= 0:
        raise ValueError('VRFM latent_dim must be a positive integer')
    if isinstance(kl_weight, bool) or not isinstance(kl_weight, (float, int)) or not math.isfinite(kl_weight) or kl_weight < 0:
        raise ValueError('VRFM kl_weight must be finite and nonnegative')
    policy = pipeline.policy_variant
    if not isinstance(policy, DualExpertPolicyVariant) or policy.packed_block_stack is None:
        raise ValueError('VRFM requires an assembled native dual-expert pipeline')
    if hasattr(policy, 'vrfm'):
        raise ValueError('VRFM is already configured')
    if getattr(policy, 'sharing_arm', None) is not None or hasattr(policy, 'routing_controller') or any(
        hasattr(block, 'private_video_to_k') for block in policy.packed_block_stack.packed_blocks
    ):
        raise ValueError('VRFM cannot be combined with legacy private K/V sharing')
    previous = {id(p): p.requires_grad for p in pipeline.parameters()}
    reference = next(policy.action_expert.parameters())
    policy.vrfm = VariationalFlowConditioning(
        video_channels=pipeline.visual_tower.config.latent_channels,
        action_dim=policy.action_dim, text_dim=pipeline.visual_tower.config.text_dim,
        video_hidden_size=pipeline.visual_tower.config.hidden_size,
        action_hidden_size=policy.action_expert.hidden_size,
        latent_dim=latent_dim, kl_weight=float(kl_weight),
    ).to(device=reference.device, dtype=reference.dtype)
    policy.vrfm.train(policy.training)
    return {
        'latent_dim': latent_dim, 'kl_weight': float(kl_weight), 'prior': 'N(0,I)',
        'existing_trainability_preserved': all(p.requires_grad == previous[id(p)]
            for p in pipeline.parameters() if id(p) in previous),
        'posterior_parameter_count': sum(p.numel() for p in policy.vrfm.posterior.parameters()),
        'conditioning_parameter_count': sum(p.numel() for layer in
            (policy.vrfm.video_projection, policy.vrfm.action_projection) for p in layer.parameters()),
    }


def vrfm_posterior_parameters(pipeline) -> tuple[nn.Parameter, ...]:
    """Identify q parameters to exclude from CAGrad; baseline returns empty."""
    vrfm = getattr(pipeline.policy_variant, 'vrfm', None)
    return () if vrfm is None else tuple(vrfm.posterior.parameters())
