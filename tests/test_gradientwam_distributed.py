from __future__ import annotations

import json
import os
import random
import socket
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, Dataset


def test_existing_libero_entry_rejects_non_torchrun_before_model_load(tmp_path, monkeypatch):
    from gradientwam.distributed_train import _build_runtime
    from open_wam.training.launch import DistributedLaunchContext, LaunchEnvironment

    split = tmp_path / "episodes.json"
    split.write_text(json.dumps({"schema_version": 1, "train_episode_ids": [0], "heldout_episode_ids": [1]}))
    (tmp_path / "settings.yaml").write_text("{}")
    spec = tmp_path / "run.yaml"
    spec.write_text("settings_config: settings.yaml\nepisode_split_json: episodes.json\n")
    monkeypatch.setattr("gradientwam.settings.load_settings", lambda path: SimpleNamespace())
    monkeypatch.setattr(
        DistributedLaunchContext, "from_env",
        classmethod(lambda cls: SimpleNamespace(environment=LaunchEnvironment.SINGLE_PROCESS)),
    )
    with pytest.raises(ValueError, match="Launch with torchrun"):
        _build_runtime(spec, resume=None)


class _ToyDataset(Dataset):
    def __init__(self, size: int = 16) -> None:
        self.data_config = SimpleNamespace(split_seed=812)
        self.sample_weights = tuple(1.0 for _ in range(size))

    def __len__(self) -> int:
        return len(self.sample_weights)

    def __getitem__(self, index: int):
        # Exercise all three RNG streams that the resume sidecar must restore.
        x = index / 8.0 + random.random() * 0.1 + float(np.random.random()) * 0.05
        target = index * 0.03 + float(torch.rand(())) * 0.2
        return torch.tensor([x], dtype=torch.float32), torch.tensor(
            [target], dtype=torch.float32
        )


class _TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(observations)


class _BatchAdapter:
    @staticmethod
    def move_to_device(batch, device):
        return tuple(value.to(device) for value in batch)


class _ToyStepExecutor:
    def __init__(self, pipeline: nn.Module) -> None:
        self.pipeline = pipeline
        self.batch_adapter = _BatchAdapter()
        self.batch_log: list[tuple[list[float], list[float]]] = []

    def forward_train(self, batch):
        observations, targets = batch
        noisy_targets = targets + torch.rand_like(targets) * 0.01
        self.batch_log.append(
            (
                observations.detach().cpu().flatten().tolist(),
                noisy_targets.detach().cpu().flatten().tolist(),
            )
        )
        loss = (self.pipeline(observations) - noisy_targets).square().mean()
        return SimpleNamespace(loss=loss, metrics={"toy_loss": loss.detach()})


class _LogSink:
    def log_event(self, **_kwargs) -> None:
        pass

    def log_metrics(self, **_kwargs) -> None:
        pass

    def close(self) -> None:
        pass


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _worker(rank: int, world_size: int, port: int, root: str) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE=str(world_size),
        WORLD_SIZE=str(world_size),
    )
    if os.name == "nt":
        # This CPU-only test never enters the POSIX artifact-cache lock.
        import sys
        import types

        fcntl_stub = types.ModuleType("fcntl")
        fcntl_stub.LOCK_EX = 2
        fcntl_stub.LOCK_NB = 4
        fcntl_stub.LOCK_SH = 1
        fcntl_stub.LOCK_UN = 8
        fcntl_stub.flock = lambda *_args: None
        sys.modules.setdefault("fcntl", fcntl_stub)

    from open_wam.configs import ExperimentConfig, TrainerConfig, TrainingConfig
    from open_wam.configs.enums import (
        CheckpointMode,
        LoopPolicyName,
        StrategyName,
        TrainerAccelerator,
        TrainerPrecision,
    )
    from open_wam.data.lerobot_v2_latent_sampler_adapters import (
        LocalLatentWeightedTrainSampler,
    )
    from open_wam.training.checkpoints import CheckpointManager
    from open_wam.training.launch import DistributedLaunchContext
    from open_wam.training.loop_policies import StepLoopPolicy
    from open_wam.training.runtime import TrainingRuntime
    from open_wam.training.state import TrainState
    from open_wam.training.strategies import DistributedStrategy
    from gradientwam.distributed_train import (
        EpisodeSubset,
        RankAwareTrainingRuntime,
        ResumeAwareSampler,
    )

    context = DistributedLaunchContext.from_env()
    strategy = DistributedStrategy(
        accelerator=TrainerAccelerator.CPU,
        precision=TrainerPrecision.FP32,
        kind=StrategyName.DDP,
        launch_context=context,
        distributed_timeout_seconds=60,
    )
    checkpoint_root = Path(root) / "checkpoints"
    output_dir = Path(root)
    identity = {
        "arm": "variational_sharing",
        "world_size": world_size,
        "per_rank_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "global_batch_size": world_size * 2,
    }

    def make_runtime(init_seed: int):
        torch.manual_seed(init_seed)
        model = strategy.prepare_model(_TinyPolicy())
        dataset = EpisodeSubset(_ToyDataset(), list(range(16)), seed=441)
        native_sampler = LocalLatentWeightedTrainSampler(
            dataset, world_size=world_size, rank=rank
        )
        sampler = ResumeAwareSampler(native_sampler, dataset, batch_size=1)
        loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0)
        config = ExperimentConfig(
            training=TrainingConfig(
                gradient_accumulation_steps=2, max_grad_norm=None, num_steps=2
            ),
            trainer=TrainerConfig(
                accelerator=TrainerAccelerator.CPU,
                devices=world_size,
                precision=TrainerPrecision.FP32,
                strategy=StrategyName.DDP,
                loop_policy=LoopPolicyName.STEPS,
                enable_checkpointing=True,
                checkpoint_mode=CheckpointMode.FULL_TRAINING_STATE,
                checkpoint_dir=str(checkpoint_root),
                save_interval=1,
                limit_val_batches=0,
                log_every_n_steps=100,
                distributed_timeout_seconds=60,
            ),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        manager = CheckpointManager(
            root_dir=checkpoint_root,
            config=config,
            checkpoint_mode=CheckpointMode.FULL_TRAINING_STATE,
        )

        class _Runtime(TrainingRuntime):
            def _run_all_validation(self, *, limit_batches):
                return None

        runtime = _Runtime(
            config=config,
            model=model,
            strategy=strategy,
            train_loader=loader,
            val_loader=DataLoader(_ToyDataset(2), batch_size=1),
            step_executor=_ToyStepExecutor(model),
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_manager=manager,
            log_sink=_LogSink(),
            train_state=TrainState(run_name="cpu-ddp-test"),
            trainability_report=None,
        )
        adapter = RankAwareTrainingRuntime(
            runtime,
            output_dir=output_dir,
            eval_seed=7,
            identity=identity,
        )
        return runtime, adapter

    live_runtime, live = make_runtime(900)
    initial_parameters = {
        name: value.detach().cpu().clone()
        for name, value in strategy.unwrap_model(live.model).state_dict().items()
    }
    live._run_step_loop(
        StepLoopPolicy(max_steps=2, limit_train_batches=None, limit_val_batches=0)
    )
    checkpoint_step1 = checkpoint_root / "checkpoint_step_1"
    checkpoint_payload = torch.load(
        checkpoint_step1 / "full_training_state.pt",
        map_location="cpu",
        weights_only=True,
    )

    per_rank_batches: list[list[tuple[list[float], list[float]]] | None] = [
        None
    ] * world_size
    dist.all_gather_object(per_rank_batches, live_runtime.step_executor.batch_log[:2])
    if rank == 0:
        observations = torch.tensor(
            [item[0][0] for rank_batches in per_rank_batches for item in rank_batches],
            dtype=torch.float32,
        ).reshape(-1, 1)
        targets = torch.tensor(
            [item[1][0] for rank_batches in per_rank_batches for item in rank_batches],
            dtype=torch.float32,
        ).reshape(-1, 1)
        reference = _TinyPolicy()
        reference.load_state_dict(initial_parameters)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.02)
        reference_scheduler = torch.optim.lr_scheduler.LambdaLR(
            reference_optimizer, lambda _: 1.0
        )
        reference_optimizer.zero_grad(set_to_none=True)
        (reference(observations) - targets).square().mean().backward()
        reference_optimizer.step()
        reference_scheduler.step()
        expected_weight = next(reference.parameters()).detach()
        actual_weight = checkpoint_payload["model_state_dict"]["linear.weight"]
        if not torch.allclose(actual_weight, expected_weight, atol=1e-7, rtol=1e-6):
            raise AssertionError("DDP accumulation differs from the global-batch update")
        if per_rank_batches[0] == per_rank_batches[1]:
            raise AssertionError("the two ranks did not receive different data/noise")

    resumed_runtime, resumed = make_runtime(1200 + rank)
    resumed.resume(str(checkpoint_step1))
    if resumed.train_state.next_batch_index != 2:
        raise AssertionError("runtime did not restore its native batch cursor")
    if resumed.train_loader.sampler.resume_batch_index != 2:
        raise AssertionError("rank wrapper did not restore the mid-epoch sampler cursor")
    resumed._run_step_loop(
        StepLoopPolicy(max_steps=2, limit_train_batches=None, limit_val_batches=0)
    )

    expected_parameters = [
        value.detach().clone() for value in strategy.unwrap_model(live.model).parameters()
    ]
    actual_parameters = [
        value.detach().clone()
        for value in strategy.unwrap_model(resumed.model).parameters()
    ]
    if any(
        not torch.equal(expected, actual)
        for expected, actual in zip(expected_parameters, actual_parameters, strict=True)
    ):
        raise AssertionError("actual RankAwareTrainingRuntime resume changed the update")

    all_parameters: list[list[torch.Tensor] | None] = [None] * world_size
    dist.all_gather_object(all_parameters, actual_parameters)
    if any(
        not torch.equal(all_parameters[0][index], all_parameters[1][index])
        for index in range(len(actual_parameters))
    ):
        raise AssertionError("DDP parameters diverged after resumed training")

    if rank == 0:
        (output_dir / "result.json").write_text(
            json.dumps(
                {
                    "world_size": world_size,
                    "gradient_accumulation_steps": 2,
                    "global_batch_reference_matches": True,
                    "per_rank_data_and_noise_differ": True,
                    "runtime_cursor_restored": True,
                    "rank_rng_and_sampler_state_restored": True,
                    "resumed_update_matches_uninterrupted": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    strategy.close()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo backend is unavailable")
def test_two_rank_gloo_runtime_training_sampling_and_resume(tmp_path: Path) -> None:
    port = _free_port()
    mp.spawn(_worker, args=(2, port, str(tmp_path)), nprocs=2, join=True)
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["world_size"] == 2
    assert result["gradient_accumulation_steps"] == 2
    assert result["global_batch_reference_matches"] is True
    assert result["per_rank_data_and_noise_differ"] is True
    assert result["runtime_cursor_restored"] is True
    assert result["rank_rng_and_sampler_state_restored"] is True
    assert result["resumed_update_matches_uninterrupted"] is True


class _CAGradToyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.common = nn.Parameter(torch.tensor(0.2))
        self.video_only = nn.Parameter(torch.tensor(-0.1))
        self.action_only = nn.Parameter(torch.tensor(0.15))
        self.posterior = nn.Parameter(torch.tensor(0.3))


class _CAGradToyBatchAdapter:
    @staticmethod
    def move_to_device(batch, device):
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


class _CAGradToyExecutor:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.batch_adapter = _CAGradToyBatchAdapter()

    def forward_train(self, batch):
        model = self.pipeline
        video = (model.common * batch["video_x"] + model.video_only - batch["video_y"]).square()
        video = video * float(batch["video_active"])
        action = (model.common * batch["action_x"] + model.action_only - batch["action_y"]).square()
        kl = 0.3 * model.posterior.square()
        return SimpleNamespace(
            loss=video + action + kl,
            task_losses={"video": video, "action": action},
            task_active={"video": bool(batch["video_active"]), "action": True},
            metrics={"toy_loss": (video + action + kl).detach()},
        )


def _cagrad_worker(rank: int, world_size: int, port: int, root: str) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE=str(world_size),
        WORLD_SIZE=str(world_size),
    )
    from gradientwam.cagrad import cagrad_coefficients
    from gradientwam.distributed_train import RankAwareTrainingRuntime
    from gradientwam.settings import GradientWAMMethod, GradientWAMMethodConfig
    from open_wam.configs import ExperimentConfig, TrainerConfig, TrainingConfig
    from open_wam.configs.enums import StrategyName, TrainerAccelerator, TrainerPrecision
    from open_wam.training.runtime import TrainingRuntime
    from open_wam.training.state import TrainState
    from open_wam.training.strategies import DistributedStrategy

    context = __import__("open_wam.training.launch", fromlist=["DistributedLaunchContext"]).DistributedLaunchContext.from_env()
    strategy = DistributedStrategy(
        accelerator=TrainerAccelerator.CPU,
        precision=TrainerPrecision.FP32,
        kind=StrategyName.DDP,
        launch_context=context,
        distributed_timeout_seconds=60,
    )
    torch.manual_seed(204)
    raw_model = _CAGradToyPolicy()
    initial = {name: value.detach().clone() for name, value in raw_model.state_dict().items()}
    model = strategy.prepare_model(raw_model)
    unwrapped = strategy.unwrap_model(model)
    batches = [
        {
            "video_x": torch.tensor(1.0 + rank + micro),
            "video_y": torch.tensor(0.2 * (micro - rank)),
            "action_x": torch.tensor(0.5 + 0.3 * rank + micro),
            "action_y": torch.tensor(-0.1 * (rank + micro)),
            "video_active": not (rank == 1 and micro == 0),
        }
        for micro in range(2)
    ]
    all_batches: list[list[dict] | None] = [None] * world_size
    dist.all_gather_object(all_batches, batches)

    training = TrainingConfig(
        gradient_accumulation_steps=2,
        learning_rate=0.02,
        beta1=0.9,
        beta2=0.95,
        weight_decay=0.05,
        warmup_steps=0,
        max_grad_norm=None,
        num_steps=1,
    )
    trainer = TrainerConfig(
        accelerator=TrainerAccelerator.CPU,
        devices=world_size,
        precision=TrainerPrecision.FP32,
        strategy=StrategyName.DDP,
        log_every_n_steps=100,
        enable_checkpointing=False,
        distributed_timeout_seconds=60,
    )
    config = ExperimentConfig(training=training, trainer=trainer)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.02, betas=(0.9, 0.95), weight_decay=0.05, foreach=False
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    class _LogSink:
        def log_event(self, **_kwargs):
            pass

        def log_metrics(self, **_kwargs):
            pass

    runtime = TrainingRuntime(
        config=config,
        model=model,
        strategy=strategy,
        train_loader=[],
        val_loader=[],
        step_executor=_CAGradToyExecutor(model),
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_manager=None,
        log_sink=_LogSink(),
        train_state=TrainState(run_name="cpu-cagrad-ddp"),
        trainability_report=None,
    )
    adapted = RankAwareTrainingRuntime(
        runtime,
        output_dir=Path(root),
        eval_seed=7,
        identity={"gradientwam": {"method": "cagrad", "cagrad_c": 0.4}},
        method_config=GradientWAMMethodConfig(method=GradientWAMMethod.CAGRAD),
        cagrad_candidates=(unwrapped.common,),
    )
    for batch in batches:
        adapted._train_micro_step(batch)

    if rank == 0:
        reference = _CAGradToyPolicy()
        reference.load_state_dict(initial)
        video_gradient = torch.zeros(())
        action_gradient = torch.zeros(())
        ordinary_gradients = {name: torch.zeros_like(parameter) for name, parameter in reference.named_parameters() if name != "common"}
        for rank_batches in all_batches:
            assert rank_batches is not None
            for batch in rank_batches:
                video = (reference.common * batch["video_x"] + reference.video_only - batch["video_y"]).square()
                video = video * float(batch["video_active"])
                action = (reference.common * batch["action_x"] + reference.action_only - batch["action_y"]).square()
                total = video + action + 0.3 * reference.posterior.square()
                if batch["video_active"]:
                    video_gradient += torch.autograd.grad(video, reference.common, retain_graph=True)[0] / (2 * world_size)
                action_gradient += torch.autograd.grad(action, reference.common, retain_graph=True)[0] / (2 * world_size)
                grads = torch.autograd.grad(total, tuple(reference.parameters()))
                for (name, _parameter), gradient in zip(reference.named_parameters(), grads, strict=True):
                    if name != "common":
                        ordinary_gradients[name] += gradient / (2 * world_size)
        video64, action64 = video_gradient.double(), action_gradient.double()
        gram = (
            (float(video64 * video64), float(video64 * action64)),
            (float(video64 * action64), float(action64 * action64)),
        )
        coefficients = cagrad_coefficients(gram, 0.4)
        reference.common.grad = coefficients[0] * video_gradient + coefficients[1] * action_gradient
        for name, parameter in reference.named_parameters():
            if name != "common":
                parameter.grad = ordinary_gradients[name]
        reference_optimizer = torch.optim.AdamW(
            reference.parameters(), lr=0.02, betas=(0.9, 0.95), weight_decay=0.05, foreach=False
        )
        reference_optimizer.step()
        actual = strategy.unwrap_model(model)
        for name, expected in reference.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[name], expected, rtol=1e-6, atol=1e-7)
    gathered: list[dict[str, torch.Tensor] | None] = [None] * world_size
    dist.all_gather_object(
        gathered,
        {name: value.detach().cpu() for name, value in unwrapped.state_dict().items()},
    )
    assert gathered[0] is not None and gathered[1] is not None
    for name in gathered[0]:
        torch.testing.assert_close(gathered[0][name], gathered[1][name], rtol=0, atol=0)
    strategy.close()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo backend is unavailable")
def test_two_rank_cagrad_window_matches_global_reference(tmp_path: Path) -> None:
    port = _free_port()
    mp.spawn(_cagrad_worker, args=(2, port, str(tmp_path)), nprocs=2, join=True)


def test_episode_split_json_is_strict_and_disjoint(tmp_path: Path) -> None:
    from gradientwam.distributed_train import load_episode_split

    path = tmp_path / "episode_split.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "train_episode_ids": [0, 1, 2],
                "heldout_episode_ids": [3, 4],
            }
        ),
        encoding="utf-8",
    )
    assert load_episode_split(path) == {
        "train_episode_ids": [0, 1, 2],
        "heldout_episode_ids": [3, 4],
    }

    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "train_episode_ids": [0, 1, 1],
                "heldout_episode_ids": [3],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique"):
        load_episode_split(path)

    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "train_episode_ids": [0, 1],
                "heldout_episode_ids": [1, 3],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="disjoint"):
        load_episode_split(path)


def test_resume_identity_ignores_stop_step_and_output_directory() -> None:
    from open_wam.configs import ExperimentConfig, TrainerConfig, TrainingConfig
    from gradientwam.distributed_train import _experiment_config_identity_sha256

    base = ExperimentConfig(
        training=TrainingConfig(num_steps=2, gradient_accumulation_steps=10),
        trainer=TrainerConfig(
            devices=8,
            checkpoint_dir="/runs/short/checkpoints",
            default_root_dir="/runs/short",
            run_name="short",
        ),
    )
    continued = replace(
        base,
        training=replace(base.training, num_steps=50),
        trainer=replace(
            base.trainer,
            devices=3,
            checkpoint_dir="/runs/continued/checkpoints",
            default_root_dir="/runs/continued",
            run_name="continued",
        ),
    )
    assert _experiment_config_identity_sha256(base) == (
        _experiment_config_identity_sha256(continued)
    )
    changed_accumulation = replace(
        continued,
        training=replace(continued.training, gradient_accumulation_steps=8),
    )
    assert _experiment_config_identity_sha256(base) != (
        _experiment_config_identity_sha256(changed_accumulation)
    )
