from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from open_wam.configs.enums import CurrentBlockCoupling, HistoryStreamVisibility

from gradientwam.cagrad_training import (
    CAGradGradientAccumulator,
    TRAINABILITY_SCOPE_ID,
    cagrad_candidate_parameters,
    configure_native_trainability,
)
from open_wam.training.step_executor import _native_dual_expert_task_activity


def _profile_fixture():
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_expert = nn.Linear(2, 2)
            self.action_expert.blocks = nn.ModuleList()
            self.v0 = nn.Linear(2, 2)
            self.v1 = nn.Linear(2, 2)
            self.a0 = nn.Linear(2, 2)
            self.a1 = nn.Linear(2, 2)
            self.packed_block_stack = SimpleNamespace(
                packed_blocks=[
                    SimpleNamespace(video_block=self.v0, action_block=self.a0),
                    SimpleNamespace(video_block=self.v1, action_block=self.a1),
                ]
            )
            self.config = SimpleNamespace(
                current_block_coupling=CurrentBlockCoupling.DECOUPLED_SAME_STEP,
                history_stream_visibility=HistoryStreamVisibility.VIDEO_ONLY,
            )
            self.conditioning = SimpleNamespace(
                uses_legacy_prefix_contract=lambda: True
            )
            self._packed_block_stack_attached = True
            self.routing_controller = None
            self.sharing_arm = None
            self.training_config = SimpleNamespace(
                objective_enabled=lambda _name: True,
                objective_weight=lambda _name: 1.0,
            )

    class Pipeline(nn.Module):
        def __init__(self):
            super().__init__()
            self.policy_variant = Policy()
            self.visual_tower = nn.Linear(2, 2)
            self.visual_tower.core = SimpleNamespace(blocks=nn.ModuleList())
            self.action_decoder = nn.Linear(2, 2)
            self.policy_variant.v1.attn1 = nn.Module()
            self.policy_variant.v1.attn1.to_k = nn.Linear(2, 2)
            self.policy_variant.v1.attn1.to_v = nn.Linear(2, 2)
            self.policy_variant.v0.attn1 = nn.Module()
            self.policy_variant.v0.attn1.to_k = nn.Linear(2, 2)
            self.policy_variant.v0.attn1.to_v = nn.Linear(2, 2)

    return Pipeline()


def test_native_profile_is_explicit_and_bounded():
    pipeline = _profile_fixture()
    report = configure_native_trainability(pipeline, expected_layers=2)
    policy = pipeline.policy_variant
    assert report["scope_id"] == TRAINABILITY_SCOPE_ID
    assert all(parameter.requires_grad for parameter in policy.a0.parameters())
    assert all(parameter.requires_grad for parameter in policy.a1.parameters())
    assert all(parameter.requires_grad for parameter in policy.action_expert.parameters())
    assert all(parameter.requires_grad for parameter in policy.v1.attn1.to_k.parameters())
    assert all(parameter.requires_grad for parameter in policy.v1.attn1.to_v.parameters())
    assert not any(parameter.requires_grad for parameter in policy.v0.parameters())
    assert not any(parameter.requires_grad for parameter in pipeline.visual_tower.parameters())
    assert not any(parameter.requires_grad for parameter in pipeline.action_decoder.parameters())
    common = cagrad_candidate_parameters(pipeline)
    expected = tuple(policy.v1.attn1.to_k.parameters()) + tuple(
        policy.v1.attn1.to_v.parameters()
    )
    assert {id(parameter) for parameter in common} == {
        id(parameter) for parameter in expected
    }


def _activity_leaf(action_mask, video_mask):
    return SimpleNamespace(
        policy_output=SimpleNamespace(
            decoder_artifacts=SimpleNamespace(
                contract="open_wam.dual_expert.decoder.v1",
                payload=SimpleNamespace(
                    action=SimpleNamespace(action_mask=action_mask),
                    video=SimpleNamespace(future_loss_mask=video_mask),
                ),
            )
        ),
        sample_outputs=(),
    )


def test_task_activity_uses_native_masks_across_variable_batch_samples():
    active_video = torch.ones(1, 1, 2, 2, 2)
    active_action = torch.ones(1, 3, 1)
    inactive_video = torch.zeros_like(active_video)
    inactive_action = torch.zeros_like(active_action)
    output = SimpleNamespace(
        policy_output=SimpleNamespace(decoder_artifacts=None),
        sample_outputs=(
            _activity_leaf(inactive_action, active_video),
            _activity_leaf(active_action, inactive_video),
        ),
    )
    assert _native_dual_expert_task_activity(output) == {
        "video": True,
        "action": True,
    }

    all_masked = SimpleNamespace(
        policy_output=SimpleNamespace(decoder_artifacts=None),
        sample_outputs=(
            _activity_leaf(inactive_action, inactive_video),
            _activity_leaf(inactive_action, inactive_video),
        ),
    )
    assert _native_dual_expert_task_activity(all_masked) == {
        "video": False,
        "action": False,
    }


def test_cagrad_accumulator_applies_one_combination_after_microbatch_sum():
    common = nn.Parameter(torch.tensor(0.0))
    ordinary = nn.Parameter(torch.tensor(0.0))
    params = (common, ordinary)
    accumulator = CAGradGradientAccumulator((common,), c=0.4)
    for video_grad, action_grad in ((2.0, -1.0), (4.0, 3.0)):
        video = common * video_grad
        action = common * action_grad
        accumulator.accumulate(
            {"video": video, "action": action},
            {"video": True, "action": True},
            scale=0.5,
        )
        (video + action + ordinary * 1.0).div(2).backward()
    report = accumulator.finalize(params)
    assert report["applied"] is True
    assert report["global_active_tasks"] == ["video", "action"]
    assert common.grad is not None
    # The solver runs once over the accumulated task means, not once per microbatch.
    from gradientwam.cagrad import cagrad_coefficients
    coefficients = cagrad_coefficients(((9.0, 3.0), (3.0, 1.0)), 0.4)
    assert common.grad.item() == pytest.approx(3 * coefficients[0] + coefficients[1], abs=1e-6)
    assert ordinary.grad.item() == pytest.approx(1.0, abs=1e-6)


def test_inactive_task_falls_back_to_ordinary_total_gradient():
    common = nn.Parameter(torch.tensor(0.0))
    accumulator = CAGradGradientAccumulator((common,), c=0.4)
    video = common * 10
    action = common * 2
    accumulator.accumulate(
        {"video": video, "action": action},
        {"video": False, "action": True},
        scale=1.0,
    )
    (video * 0 + action).backward()
    report = accumulator.finalize((common,))
    assert report["applied"] is False
    assert common.grad.item() == pytest.approx(2.0)


def test_zero_gradient_on_one_task_remains_in_structural_common_set():
    common = nn.Parameter(torch.tensor(0.0))
    accumulator = CAGradGradientAccumulator((common,), c=0.4)
    video = common * 0.0
    action = common * 2.0
    accumulator.accumulate(
        {"video": video, "action": action},
        {"video": True, "action": True},
        scale=1.0,
    )
    (video + action).backward()
    report = accumulator.finalize((common,))
    assert report["applied"] is True
    assert report["common_parameter_count"] == 1


def test_vrfm_candidates_are_filtered_by_real_task_graph_and_exclude_posterior():
    pipeline = _profile_fixture()
    configure_native_trainability(pipeline, expected_layers=2)
    policy = pipeline.policy_variant
    policy.vrfm = nn.Module()
    policy.vrfm.video_projection = nn.Linear(2, 2, bias=False)
    policy.vrfm.action_projection = nn.Linear(2, 2, bias=False)
    policy.vrfm.posterior = nn.Linear(2, 2)
    candidates = cagrad_candidate_parameters(pipeline)
    assert not {id(parameter) for parameter in policy.vrfm.posterior.parameters()} & {
        id(parameter) for parameter in candidates
    }

    inputs = torch.tensor([[1.0, 2.0]])
    video_bias = policy.vrfm.video_projection(inputs)
    action_bias = policy.vrfm.action_projection(inputs)
    shared = policy.v1.attn1.to_k(inputs) + policy.v1.attn1.to_v(inputs)
    video = (shared + video_bias).square().mean()
    action = (shared + video_bias + action_bias).square().mean()
    posterior_loss = policy.vrfm.posterior(inputs).square().mean()
    accumulator = CAGradGradientAccumulator(candidates, c=0.4)
    accumulator.accumulate(
        {"video": video, "action": action},
        {"video": True, "action": True},
        scale=1.0,
    )
    (video + action + posterior_loss).backward()
    report = accumulator.finalize(pipeline.parameters())
    expected_common = (
        tuple(policy.v1.attn1.to_k.parameters())
        + tuple(policy.v1.attn1.to_v.parameters())
        + tuple(policy.vrfm.video_projection.parameters())
    )
    assert report["applied"] is True
    assert report["common_parameter_numel"] == sum(p.numel() for p in expected_common)
    assert all(p.grad is not None for p in policy.vrfm.posterior.parameters())
