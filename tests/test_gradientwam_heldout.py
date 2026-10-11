"""CPU regression of the actual heldout runtime and native forward broadcasts."""
from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gradientwam.distributed_train import RankAwareTrainingRuntime
from open_wam.configs import TrainingConfig
from open_wam.configs.enums import DynamicsObjective
from open_wam.models.common.dynamics_objectives import sample_conditioning_mode
from open_wam.models.common.flow_supervision import build_video_frame_loss_mask
from open_wam.models.common.flow_training import (
    build_action_flow_match_train_artifacts,
    build_video_flow_match_train_artifacts,
)
from open_wam.models.decoder_artifacts import (
    DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT,
    DualExpertActionTrainArtifacts,
    DualExpertTrainArtifacts,
    DualExpertVideoTrainArtifacts,
)
from open_wam.models.policy_variants.contracts import DecoderArtifactEnvelope


class _TinyExecutor:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.batch_adapter = SimpleNamespace(move_to_device=lambda batch, device: batch)
        self.calls = []

    def forward_train(self, batch):
        if isinstance(self.pipeline, torch.nn.Module):
            assert not self.pipeline.training
            assert not torch.is_grad_enabled()
        batch_size = len(batch)
        config = TrainingConfig()
        video = build_video_flow_match_train_artifacts(
            torch.zeros(batch_size, 1, 3, 2, 2),
            training_config=config,
            noisy_condition_prob=0.5,
            clean_prefix_frames=1,
        )
        mode = sample_conditioning_mode(
            {choice: 1.0 for choice in DynamicsObjective},
            enum_cls=DynamicsObjective,
            device=torch.device("cpu"),
            error_label="Gloo regression",
        )
        self.calls.append({
            "ids": batch.tolist(),
            "mode": mode.value,
            "noisy_condition": bool(video.condition_timesteps.any()),
        })
        action = build_action_flow_match_train_artifacts(
            torch.zeros(batch_size, 2, 2), None, training_config=config,
        )
        delta = float(batch[0] + 1)
        artifacts = DualExpertTrainArtifacts(
            action=DualExpertActionTrainArtifacts(
                flow_pred=action.targets + delta, targets=action.targets,
                timesteps=action.timesteps, scheduler=action.scheduler,
                denoised_actions=action.noisy_actions, action_mask=action.action_mask,
            ),
            video=DualExpertVideoTrainArtifacts(
                flow_pred=video.targets + delta, targets=video.targets,
                timesteps=video.timesteps, scheduler=video.scheduler,
                predicted_latents=video.noisy_latents,
                target_latents=torch.zeros_like(video.targets),
                future_loss_mask=build_video_frame_loss_mask(
                    torch.zeros_like(video.targets), default_target_start=1,
                ),
            ),
            condition_mode=mode.value, program="joint", history_frames=1,
        )
        envelope = DecoderArtifactEnvelope(DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT, artifacts)
        return SimpleNamespace(output=SimpleNamespace(
            policy_output=SimpleNamespace(decoder_artifacts=envelope),
        ))


def _make_runtime(rank: int, world_size: int):
    runtime = object.__new__(RankAwareTrainingRuntime)
    runtime._runtime = SimpleNamespace()
    runtime.strategy = SimpleNamespace(
        is_main_process=rank == 0, device=torch.device("cpu"),
        world_size=world_size, unwrap_model=lambda model: model,
        autocast_context=nullcontext,
    )
    runtime.model = torch.nn.Identity()
    runtime.model.train()
    original_pipeline = object()
    runtime.step_executor = _TinyExecutor(original_pipeline)
    runtime.val_loader = torch.utils.data.DataLoader(
        torch.arange(5), batch_size=2, shuffle=False, num_workers=0,
    )
    runtime.eval_seed = 7341
    runtime.identity = {"fixture": "five_full_unsharded_samples"}
    runtime.train_state = SimpleNamespace(optimizer_step=1000)
    runtime.heldout_episode_ids = [0]
    return runtime, original_pipeline


def _heldout_worker(rank: int, directory: str) -> None:
    output = Path(directory)
    record = {"rank": rank, "ok": False}
    initialized = False
    try:
        torch.set_num_threads(1)
        dist.init_process_group(
            "gloo", init_method=(output / "rendezvous").as_uri(),
            rank=rank, world_size=2, timeout=timedelta(seconds=10),
        )
        initialized = True
        runtime, original_pipeline = _make_runtime(rank, 2)
        # Distinct training RNGs still synchronize through the native broadcasts.
        torch.manual_seed(100 + rank)
        runtime.step_executor.forward_train(torch.tensor([0]))
        record["training_decision"] = runtime.step_executor.calls.pop()
        record["metrics"] = runtime._evaluate_heldout()
        record["eval_calls"] = runtime.step_executor.calls
        assert runtime.model.training
        assert runtime.step_executor.pipeline is original_pipeline
        assert runtime.train_state.optimizer_step == 1000
        assert not torch.cuda.is_initialized()
        record["restored"] = True
        # run() calls this after only the main rank writes metrics.
        dist.barrier()
        record["ok"] = True
    except Exception:
        record["traceback"] = traceback.format_exc()
    finally:
        if initialized:
            dist.destroy_process_group()
        (output / f"rank{rank}.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8",
        )


@pytest.mark.unit
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo backend is unavailable")
def test_two_rank_heldout_pairs_native_collectives_and_keeps_main_metrics(tmp_path: Path) -> None:
    assert not torch.cuda.is_initialized()
    mp.spawn(_heldout_worker, args=(str(tmp_path.resolve()),), nprocs=2, join=True)
    records = [
        json.loads((tmp_path / f"rank{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    assert all(record["ok"] for record in records), json.dumps(records, indent=2)
    main_rank, other_rank = records
    assert main_rank["training_decision"] == other_rank["training_decision"]
    assert main_rank["eval_calls"] == other_rank["eval_calls"]
    assert [call["ids"] for call in main_rank["eval_calls"]] == [[0, 1], [2, 3], [4]]
    assert main_rank["metrics"]["heldout_sample_count"] == 5
    assert other_rank["metrics"] is None
    assert all(record["restored"] for record in records)
    # Rank-zero sampling and native weighted proxies retain serial semantics.
    serial_runtime, original_pipeline = _make_runtime(0, 1)
    serial_metrics = serial_runtime._evaluate_heldout()
    assert main_rank["metrics"]["metrics"] == serial_metrics["metrics"]
    assert serial_runtime.model.training
    assert serial_runtime.step_executor.pipeline is original_pipeline
    assert serial_runtime.train_state.optimizer_step == 1000
    assert not torch.cuda.is_initialized()
