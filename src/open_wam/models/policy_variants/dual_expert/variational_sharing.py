"""Weighted regression evidence, observed-prefix prior and chunk routing (v0.2)."""
from __future__ import annotations

import torch
from torch import nn
from open_wam.models.common.route_evidence import route_objective_weighted
from open_wam.models.visual_tower.public_pretraining import load_public_video_checkpoint_into_tower


def _validate_prior(prior_shared: torch.Tensor) -> None:
    if prior_shared.ndim != 1 or prior_shared.numel() == 0:
        raise ValueError("prior_shared must be a nonempty [B] tensor")
    if not torch.isfinite(prior_shared).all() or ((prior_shared <= 0) | (prior_shared >= 1)).any():
        raise ValueError("prior_shared must be finite and strictly between zero and one")


class ObservedPrefixPrior(nn.Module):
    """One zero-init linear prior; only explicitly observed latent pixels enter it."""

    def __init__(self, latent_channels: int) -> None:
        super().__init__()
        if latent_channels <= 0:
            raise ValueError("latent_channels must be positive")
        self.linear = nn.Linear(latent_channels, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, prefix_latents: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        if prefix_latents.ndim < 3 or prefix_latents.shape[1] != self.linear.in_features:
            raise ValueError("prefix_latents must have shape [B,C,...] with matching channels")
        if observed_mask is None or observed_mask.dtype != torch.bool:
            raise ValueError("An explicit boolean observed-prefix mask is required")
        if observed_mask.ndim != prefix_latents.ndim - 1 or observed_mask.shape[0] != prefix_latents.shape[0]:
            raise ValueError("observed_mask must have shape [B,...] matching latent geometry")
        try:
            mask = torch.broadcast_to(observed_mask, (prefix_latents.shape[0], *prefix_latents.shape[2:]))
        except RuntimeError as error:
            raise ValueError("observed_mask is incompatible with latent geometry") from error
        spatial_dims = tuple(range(1, mask.ndim))
        count = mask.sum(spatial_dims)
        if count.numel() == 0 or (count <= 0).any():
            raise ValueError("Every sample needs at least one observed latent coordinate")
        frozen = prefix_latents.detach()
        observed = torch.where(mask[:, None], frozen, torch.zeros_like(frozen))
        if not torch.isfinite(observed).all():
            raise ValueError("Observed prefix latents must be finite")
        feature = observed.sum(tuple(range(2, observed.ndim))) / count[:, None]
        return .1 + .8 * self.linear(feature.to(self.linear.weight.dtype)).squeeze(-1).sigmoid()

    prior_prob = forward


def select_route(prior_shared: torch.Tensor, uniform: torch.Tensor) -> torch.Tensor:
    """Select z once from an externally owned uniform; 0=private, 1=shared."""
    _validate_prior(prior_shared)
    if uniform.shape != prior_shared.shape or uniform.device != prior_shared.device:
        raise ValueError("Route uniform and prior must share [B] shape and device")
    if not torch.isfinite(uniform).all() or ((uniform < 0) | (uniform >= 1)).any():
        raise ValueError("Route uniform must be finite and in [0,1)")
    return (uniform < prior_shared).long()


class ChunkRouteSampler:
    """An explicit CPU RNG and chunk route; call begin_chunk once before the ODE."""

    def __init__(self, seed: int) -> None:
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self._current_route: torch.Tensor | None = None

    def begin_chunk(self, prior_shared: torch.Tensor, uniform: torch.Tensor | None = None) -> torch.Tensor:
        _validate_prior(prior_shared)
        if uniform is None:
            uniform = torch.rand(prior_shared.shape, generator=self.generator).to(prior_shared.device)
        self._current_route = select_route(prior_shared.detach(), uniform).detach().clone()
        return self.current_route

    @property
    def current_route(self) -> torch.Tensor:
        if self._current_route is None:
            raise RuntimeError("No route selected; begin_chunk must run before denoising")
        return self._current_route.clone()

    def get_state(self) -> dict:
        return {"generator_state": self.generator.get_state().clone(),
                "current_route": None if self._current_route is None else self._current_route.clone()}

    def set_state(self, state: dict) -> None:
        self.generator.set_state(state["generator_state"].cpu())
        route = state["current_route"]
        if route is not None and (route.ndim != 1 or route.dtype != torch.long or ((route < 0) | (route > 1)).any()):
            raise ValueError("Restored route must be a [B] long tensor of binary choices")
        self._current_route = None if route is None else route.detach().clone()


class PrefixRoutingController(nn.Module):
    """Registered prior and checkpointed route RNG, with no target access."""

    def __init__(self, latent_channels: int, arm: str, route_seed: int) -> None:
        super().__init__()
        from open_wam.configs.enums import VariationalSharingArm
        self.arm = VariationalSharingArm(arm)
        self.prior = (ObservedPrefixPrior(latent_channels)
            if self.arm in (VariationalSharingArm.VARIATIONAL_SHARING, VariationalSharingArm.DETERMINISTIC_KV_BLEND) else None)
        self.sampler = ChunkRouteSampler(route_seed) if self.arm is VariationalSharingArm.VARIATIONAL_SHARING else None
        if self.sampler is not None:
            self.register_buffer("route_rng_state", self.sampler.generator.get_state())
            self.register_buffer("chunk_route", torch.empty(0, dtype=torch.long))

    def training_kwargs(self, latents: torch.Tensor, observed_mask: torch.Tensor) -> dict:
        from open_wam.configs.enums import VariationalSharingArm, ActionVideoKvRouting
        if self.arm is VariationalSharingArm.FORCED_PRIVATE:
            return {"routing_mode": ActionVideoKvRouting.PRIVATE}
        probability = self.prior(latents, observed_mask)
        mode = ActionVideoKvRouting.TWO_ROUTES if self.arm is VariationalSharingArm.VARIATIONAL_SHARING else ActionVideoKvRouting.BLEND
        return {"routing_mode": mode, "prior_shared": probability}

    def begin_inference_chunk(self, observed_prefix: torch.Tensor, observed_mask: torch.Tensor,
                              uniform: torch.Tensor | None = None) -> dict:
        from open_wam.configs.enums import VariationalSharingArm, ActionVideoKvRouting
        if self.arm is VariationalSharingArm.FORCED_PRIVATE:
            return {"routing_mode": ActionVideoKvRouting.PRIVATE}
        probability = self.prior(observed_prefix, observed_mask)
        if self.arm is VariationalSharingArm.DETERMINISTIC_KV_BLEND:
            return {"routing_mode": ActionVideoKvRouting.BLEND, "prior_shared": probability}
        route = self.sampler.begin_chunk(probability, uniform)
        return {"routing_mode": ActionVideoKvRouting.SELECTED, "route_choices": route}

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self.sampler is not None:
            state = self.sampler.get_state()
            destination[prefix + "route_rng_state"] = state["generator_state"]
            route = state["current_route"]
            destination[prefix + "chunk_route"] = torch.empty(0, dtype=torch.long) if route is None else route

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if self.sampler is not None and prefix + "chunk_route" in state_dict:
            self.chunk_route = torch.empty_like(state_dict[prefix + "chunk_route"], device=self.route_rng_state.device)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        if self.sampler is not None:
            self.sampler.set_state({"generator_state": self.route_rng_state.cpu(),
                "current_route": None if not self.chunk_route.numel() else self.chunk_route})


def validate_route_profile(profile) -> None:
    """Check authored semantics and actual representation-independent visibility."""
    from open_wam.configs.enums import CurrentBlockCoupling, HistoryStreamVisibility
    from open_wam.models.common.packed_token_layout import PackedTokenStream
    metadata = profile.metadata
    if metadata.get("current_block_coupling") != CurrentBlockCoupling.DECOUPLED_SAME_STEP.value or metadata.get("history_stream_visibility") != HistoryStreamVisibility.VIDEO_ONLY:
        raise ValueError("Sharing requires decoupled_same_step and video_only history")
    if profile.token_layout is None or profile.self_attention_visibility is None:
        raise ValueError("Sharing requires explicit native token layout and visibility")
    streams = profile.token_layout.stream_id
    video = (streams == PackedTokenStream.VIDEO).nonzero().flatten()
    action = (streams == PackedTokenStream.ACTION).nonzero().flatten()
    for rows in video.split(128):
        if profile.self_attention_visibility(rows[:, None], action[None, :]).any():
            raise ValueError("Actual visibility lets video queries read action keys")


def configure_variational_sharing(pipeline, *, arm: str, expected_layers: int = 30,
                                   route_seed: int = 20261008) -> dict:
    """After native load/attach, before optimizer/FSDP: establish real owners."""
    from collections import Counter
    from open_wam.configs.enums import VariationalSharingArm, CurrentBlockCoupling, HistoryStreamVisibility
    policy = pipeline.policy_variant
    selected = VariationalSharingArm(arm)
    if getattr(policy, "sharing_arm", None) is not None:
        raise ValueError("Sharing is configured exactly once after native initialization")
    if policy.config.current_block_coupling != CurrentBlockCoupling.DECOUPLED_SAME_STEP or policy.config.history_stream_visibility != HistoryStreamVisibility.VIDEO_ONLY:
        raise ValueError("The frozen sharing contract requires decoupled_same_step/video_only")
    if not policy.conditioning.uses_legacy_prefix_contract():
        raise ValueError("Sharing currently requires the native condition_latents_prefix contract")
    if not policy._packed_block_stack_attached or policy.packed_block_stack is None:
        raise ValueError("Configure sharing after attach_visual_tower, before optimizer/FSDP")
    stack = policy.packed_block_stack
    if len(stack.packed_blocks) != expected_layers or expected_layers <= 0:
        raise ValueError("Actual paired layer count differs from the requested final layer")
    if len(pipeline.visual_tower.core.blocks) or len(policy.action_expert.blocks):
        raise ValueError("Original owning block containers must be empty after attachment")
    for objective in ("action", "latent"):
        if not policy.training_config.objective_enabled(objective) or policy.training_config.objective_weight(objective) != 1.:
            raise ValueError("The frozen contract requires world/action objectives with unit coefficients")
    pipeline.requires_grad_(False)
    policy.action_expert.requires_grad_(True)
    for pair in stack.packed_blocks:
        pair.action_block.requires_grad_(True)
    last = stack.packed_blocks[-1]
    last.video_block.attn1.to_k.requires_grad_(True)
    last.video_block.attn1.to_v.requires_grad_(True)
    if selected is not VariationalSharingArm.NATIVE_JOINT:
        stack.enable_private_video_kv(expected_layers=expected_layers)
        controller = PrefixRoutingController(pipeline.visual_tower.config.latent_channels, selected.value, route_seed)
        reference = last.video_block.attn1.to_k.weight
        policy.routing_controller = controller.to(device=reference.device, dtype=reference.dtype)
    else:
        policy.routing_controller = None
    policy.sharing_arm = selected
    all_parameters = list(pipeline.named_parameters(remove_duplicate=False))
    duplicate = sorted(id_ for id_,count in Counter(id(p) for _,p in all_parameters).items() if count > 1)
    if duplicate:
        raise ValueError("Duplicated registered Parameter identities after attachment")
    phi_ids = {id(p) for layer in (last.video_block.attn1.to_k,last.video_block.attn1.to_v) for p in layer.parameters()}
    actual_video = {id(p) for pair in stack.packed_blocks for p in pair.video_block.parameters() if p.requires_grad}
    if actual_video != phi_ids or any(p.requires_grad for p in pipeline.visual_tower.parameters()):
        raise ValueError("Actual world trainability does not match the final K/V-only contract")
    return {"arm": selected.value, "last_layer_index": expected_layers - 1,
            "duplicate_parameter_ids": duplicate,
            "trainable_parameters": {name: parameter.numel() for name,parameter in all_parameters if parameter.requires_grad},
            "phi_numel": sum(p.numel() for layer in (last.video_block.attn1.to_k,last.video_block.attn1.to_v) for p in layer.parameters()),
            "total_numel": sum(p.numel() for _,p in all_parameters)}
