"""Bounded native trainability and globally synchronized two-task gradients."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from open_wam.configs.enums import CurrentBlockCoupling, HistoryStreamVisibility

from .cagrad import cagrad_coefficients
from .settings import TRAINABILITY_SCOPE_ID
_GRADIENT_BUCKET_ELEMENTS = 4_000_000
_DOT_CHUNK_ELEMENTS = 1_000_000


def configure_native_trainability(
    pipeline: nn.Module, *, expected_layers: int
) -> dict[str, Any]:
    """Apply the shared, bounded native scope used by all four new methods."""
    policy = pipeline.policy_variant
    config = policy.config
    if config.current_block_coupling != CurrentBlockCoupling.DECOUPLED_SAME_STEP:
        raise ValueError("The bounded GradientWAM scope requires decoupled_same_step")
    if config.history_stream_visibility != HistoryStreamVisibility.VIDEO_ONLY:
        raise ValueError("The bounded GradientWAM scope requires video_only history")
    if not policy.conditioning.uses_legacy_prefix_contract():
        raise ValueError("The bounded scope requires the native condition_latents_prefix contract")
    stack = policy.packed_block_stack
    if not getattr(policy, "_packed_block_stack_attached", False) or stack is None:
        raise ValueError("Apply the bounded scope after native block attachment")
    if expected_layers <= 0 or len(stack.packed_blocks) != expected_layers:
        raise ValueError("Actual paired layer count differs from the configured final layer")
    if len(pipeline.visual_tower.core.blocks) or len(policy.action_expert.blocks):
        raise ValueError("Expected attached native video/action block ownership")
    if getattr(policy, "sharing_arm", None) is not None or getattr(
        policy, "routing_controller", None
    ) is not None:
        raise ValueError("The new-method scope cannot include legacy private routing")
    if any(
        hasattr(block, "private_video_to_k") or hasattr(block, "private_video_to_v")
        for block in stack.packed_blocks
    ):
        raise ValueError("The new-method scope cannot include private video K/V")
    training_config = policy.training_config
    for objective in ("action", "latent"):
        if not training_config.objective_enabled(objective) or training_config.objective_weight(objective) != 1.0:
            raise ValueError("The bounded scope requires unit-weight action and video objectives")

    pipeline.requires_grad_(False)
    policy.action_expert.requires_grad_(True)
    for pair in stack.packed_blocks:
        pair.action_block.requires_grad_(True)
    final_video = stack.packed_blocks[-1].video_block
    final_video.attn1.to_k.requires_grad_(True)
    final_video.attn1.to_v.requires_grad_(True)

    trainable = [
        (name, parameter)
        for name, parameter in pipeline.named_parameters(remove_duplicate=False)
        if parameter.requires_grad
    ]
    parameter_ids = [id(parameter) for _, parameter in trainable]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError("Duplicate trainable parameter owners after native attachment")
    video_ids = {
        id(parameter)
        for pair in stack.packed_blocks
        for parameter in pair.video_block.parameters()
        if parameter.requires_grad
    }
    expected_video_ids = {
        id(parameter)
        for module in (final_video.attn1.to_k, final_video.attn1.to_v)
        for parameter in module.parameters()
    }
    if video_ids != expected_video_ids or any(
        parameter.requires_grad for parameter in pipeline.visual_tower.parameters()
    ):
        raise ValueError("Trainable video parameters must be only final-block shared K/V")
    return {
        "scope_id": TRAINABILITY_SCOPE_ID,
        "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
        "trainable_parameters": {name: parameter.numel() for name, parameter in trainable},
        "video_shared_kv_parameter_count": sum(
            parameter.numel() for module in (final_video.attn1.to_k, final_video.attn1.to_v)
            for parameter in module.parameters()
        ),
    }


def cagrad_candidate_parameters(pipeline: nn.Module) -> tuple[nn.Parameter, ...]:
    """Return the small structural candidate set; autograd selects true overlap."""
    policy = pipeline.policy_variant
    stack = policy.packed_block_stack
    if stack is None or not stack.packed_blocks:
        raise ValueError("CAGrad requires an attached packed block stack")
    final_video = stack.packed_blocks[-1].video_block
    # In the decoupled/video-only contract, action queries can read video keys,
    # while video queries cannot read action keys. The action-side projection
    # and action expert therefore stay outside the bounded common candidate set.
    modules = [final_video.attn1.to_k, final_video.attn1.to_v]
    vrfm = getattr(policy, "vrfm", None)
    if vrfm is not None:
        modules.append(vrfm.video_projection)
    seen: set[int] = set()
    parameters: list[nn.Parameter] = []
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
    if not parameters:
        raise ValueError("CAGrad candidate set is empty")
    if vrfm is not None:
        posterior_ids = {id(parameter) for parameter in vrfm.posterior.parameters()}
        if any(id(parameter) in posterior_ids for parameter in parameters):
            raise ValueError("VRFM posterior parameters must be excluded from CAGrad")
    return tuple(parameters)


class CAGradGradientAccumulator:
    """Accumulate local task grads, average globally, then solve once per window."""

    def __init__(self, candidates: Iterable[nn.Parameter], *, c: float) -> None:
        self.candidates = tuple(candidates)
        if not self.candidates:
            raise ValueError("CAGrad requires at least one candidate parameter")
        if any(not parameter.requires_grad for parameter in self.candidates):
            raise ValueError("CAGrad candidates must be trainable")
        if len({id(parameter) for parameter in self.candidates}) != len(self.candidates):
            raise ValueError("CAGrad candidate parameters must be unique")
        self.c = float(c)
        self.video_grads = [torch.zeros_like(p, dtype=torch.float32) for p in self.candidates]
        self.action_grads = [torch.zeros_like(p, dtype=torch.float32) for p in self.candidates]
        self.connected = [[False, False] for _ in self.candidates]
        self.active = [False, False]
        self._active_task: int | None = None
        self._hook_handles = [
            parameter.register_hook(
                lambda _gradient, parameter_index=index: self._mark_connected(parameter_index)
            )
            for index, parameter in enumerate(self.candidates)
        ]

    def _mark_connected(self, parameter_index: int) -> None:
        if self._active_task is not None:
            self.connected[parameter_index][self._active_task] = True

    def close(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def accumulate(
        self,
        task_losses: Mapping[str, torch.Tensor],
        task_active: Mapping[str, bool],
        *,
        scale: float,
    ) -> None:
        for index, name in enumerate(("video", "action")):
            if not task_active[name]:
                continue
            loss = task_losses[name]
            if not loss.requires_grad:
                continue
            self.active[index] = True
            buffers = self.video_grads if index == 0 else self.action_grads
            saved_grads = [parameter.grad for parameter in self.candidates]
            self._active_task = index
            try:
                for parameter, buffer in zip(self.candidates, buffers, strict=True):
                    parameter.grad = buffer
                # Gradients accumulate directly into the two persistent FP32
                # task buffers. This avoids allocating a full extra gradient
                # tuple for models with billions of trainable parameters.
                torch.autograd.backward(
                    loss * float(scale), inputs=self.candidates, retain_graph=True
                )
            finally:
                for parameter, grad in zip(self.candidates, saved_grads, strict=True):
                    parameter.grad = grad
                self._active_task = None

    def finalize(self, ordinary_parameters: Iterable[nn.Parameter]) -> dict[str, Any]:
        flags = [int(value) for value in self.active]
        flags.extend(int(value) for pair in self.connected for value in pair)
        device = self.candidates[0].device
        flags_tensor = torch.tensor(flags, device=device, dtype=torch.int32)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size > 1:
            dist.all_reduce(flags_tensor, op=dist.ReduceOp.SUM)
        global_flags = flags_tensor.cpu().tolist()
        globally_active = [bool(global_flags[0]), bool(global_flags[1])]
        offset = 2
        globally_connected: list[tuple[bool, bool]] = []
        for _ in self.candidates:
            globally_connected.append(
                (bool(global_flags[offset]), bool(global_flags[offset + 1]))
            )
            offset += 2

        for buffer in (*self.video_grads, *self.action_grads):
            if world_size > 1:
                dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
                buffer.div_(world_size)

        common_indices = [
            index
            for index, (video_connected, action_connected) in enumerate(globally_connected)
            if globally_active == [True, True]
            and video_connected
            and action_connected
        ]
        common_ids = {id(self.candidates[index]) for index in common_indices}
        _synchronize_ordinary_gradients(
            ordinary_parameters,
            skip_parameter_ids=common_ids,
            world_size=world_size,
        )
        coefficients: tuple[float, float] | None = None
        applied = bool(common_indices)
        if applied:
            gram = _common_gram(
                [self.video_grads[index] for index in common_indices],
                [self.action_grads[index] for index in common_indices],
            )
            coefficients = cagrad_coefficients(gram, self.c)
            for index in common_indices:
                parameter = self.candidates[index]
                gradient = (
                    coefficients[0] * self.video_grads[index]
                    + coefficients[1] * self.action_grads[index]
                )
                parameter.grad = gradient.to(dtype=parameter.dtype)
        self.close()
        return {
            "applied": applied,
            "global_active_tasks": [
                name for name, active in zip(("video", "action"), globally_active, strict=True)
                if active
            ],
            "common_parameter_count": len(common_indices),
            "common_parameter_numel": sum(
                self.candidates[index].numel() for index in common_indices
            ),
            "coefficients": coefficients,
        }


def _common_gram(
    video_grads: list[torch.Tensor], action_grads: list[torch.Tensor]
) -> tuple[tuple[float, float], tuple[float, float]]:
    device = video_grads[0].device
    dots = torch.zeros(3, device=device, dtype=torch.float64)
    for video, action in zip(video_grads, action_grads, strict=True):
        video_flat, action_flat = video.reshape(-1), action.reshape(-1)
        for start in range(0, video_flat.numel(), _DOT_CHUNK_ELEMENTS):
            end = min(video_flat.numel(), start + _DOT_CHUNK_ELEMENTS)
            left = video_flat[start:end].double()
            right = action_flat[start:end].double()
            dots[0] += torch.dot(left, left)
            dots[1] += torch.dot(left, right)
            dots[2] += torch.dot(right, right)
    vv, va, aa = (float(value) for value in dots.cpu().tolist())
    return ((vv, va), (va, aa))


def _synchronize_ordinary_gradients(
    parameters: Iterable[nn.Parameter],
    *,
    skip_parameter_ids: set[int],
    world_size: int,
) -> None:
    selected = [
        parameter
        for parameter in parameters
        if parameter.requires_grad and id(parameter) not in skip_parameter_ids
    ]
    if not selected or world_size <= 1:
        return
    device = selected[0].device
    presence = torch.tensor(
        [parameter.grad is not None for parameter in selected],
        device=device,
        dtype=torch.int32,
    )
    dist.all_reduce(presence, op=dist.ReduceOp.SUM)
    active_parameters = [
        parameter for parameter, active in zip(selected, presence.tolist(), strict=True)
        if active > 0
    ]
    for parameter in active_parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        elif not parameter.grad.is_contiguous():
            parameter.grad = parameter.grad.contiguous()

    views: list[torch.Tensor] = []
    references: list[tuple[nn.Parameter, int, int]] = []
    bucket_numel = 0

    def reduce_bucket() -> None:
        nonlocal bucket_numel
        if not views:
            return
        bucket = torch.cat(views)
        dist.all_reduce(bucket, op=dist.ReduceOp.SUM)
        bucket.div_(world_size)
        cursor = 0
        for parameter, start, end in references:
            length = end - start
            parameter.grad.view(-1)[start:end].copy_(bucket[cursor : cursor + length])
            cursor += length
        views.clear()
        references.clear()
        bucket_numel = 0

    for parameter in active_parameters:
        flat = parameter.grad.view(-1)
        start = 0
        while start < flat.numel():
            take = min(_GRADIENT_BUCKET_ELEMENTS - bucket_numel, flat.numel() - start)
            end = start + take
            views.append(flat[start:end])
            references.append((parameter, start, end))
            bucket_numel += take
            start = end
            if bucket_numel == _GRADIENT_BUCKET_ELEMENTS:
                reduce_bucket()
    reduce_bucket()
