"""DDP launcher target and rank-local resume helpers for GradientWAM."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml


def load_episode_split(path: str | Path) -> dict[str, list[int]]:
    """Load the small prepare/training split contract and reject leakage."""
    from .settings import load_episode_split as load

    split = load(Path(path))
    return {
        "train_episode_ids": split["train_episode_ids"],
        "heldout_episode_ids": split["heldout_episode_ids"],
    }


@dataclass(frozen=True)
class _ResumeIndex:
    index: int
    restore_rng: bool = False


class EpisodeSubset(torch.utils.data.Dataset):
    """Select explicit episode-owned virtual sample indices without rebuilding weights."""

    def __init__(self, dataset: Any, indices: list[int], *, seed: int) -> None:
        if not indices:
            raise ValueError("explicit episode split produced an empty sample subset")
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)
        self.data_config = dataset.data_config
        weights = getattr(dataset, "sample_weights", None)
        self.sample_weights = (
            tuple(float(weights[index]) for index in self.indices)
            if weights is not None
            else tuple(1.0 for _ in self.indices)
        )
        self.resume_rng_state: dict[str, Any] | None = None
        self.epoch = 0
        self.rank = 0
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int | _ResumeIndex):
        restore_rng = isinstance(index, _ResumeIndex) and index.restore_rng
        local_index = index.index if isinstance(index, _ResumeIndex) else int(index)
        if restore_rng and self.resume_rng_state is not None:
            _restore_rng_state(self.resume_rng_state)
            self.resume_rng_state = None
        return self.dataset[self.indices[local_index]]

    @property
    def episode_ids(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    _episode_for_index(self.dataset, index)
                    for index in self.indices
                }
            )
        )

    def seed_epoch(self, *, epoch: int, rank: int) -> None:
        self.epoch = int(epoch)
        self.rank = int(rank)
        epoch_seed = (self.seed + 1_000_003 * self.epoch + self.rank) % (2**32)
        random.seed(epoch_seed)
        np.random.seed(epoch_seed)
        torch.manual_seed(epoch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(epoch_seed)

    def build_train_sampler(self, *, world_size: int, rank: int):
        from open_wam.data.lerobot_v2_latent_sampler_adapters import (
            LocalLatentWeightedTrainSampler,
        )

        return LocalLatentWeightedTrainSampler(self, world_size=world_size, rank=rank)


class ResumeAwareSampler(torch.utils.data.Sampler):
    """Wrap the native sampler and mark the first sample after a saved cursor."""

    def __init__(self, sampler: Any, dataset: EpisodeSubset, *, batch_size: int) -> None:
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.resume_batch_index = 0
        self.rank = int(sampler.rank)
        self.world_size = int(sampler.world_size)

    def __len__(self) -> int:
        return len(self.sampler)

    def __iter__(self):
        boundary = self.resume_batch_index * self.batch_size
        for position, index in enumerate(self.sampler):
            if self.resume_batch_index > 0 and position == boundary:
                # The runtime may continue through more optimizer steps in the
                # same epoch. Consume the resume marker once so later iterator
                # passes cannot restore stale RNG state a second time.
                self.resume_batch_index = 0
                yield _ResumeIndex(int(index), restore_rng=True)
            else:
                yield int(index)

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(int(epoch))
        self.dataset.seed_epoch(epoch=int(epoch), rank=self.rank)

    def state_dict(self) -> dict[str, Any]:
        return {
            "sampler": _sampler_state(self.sampler),
            "resume_batch_index": int(self.resume_batch_index),
            "batch_size": self.batch_size,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        _restore_sampler_state(self.sampler, state["sampler"], rank=self.rank)
        if int(state["batch_size"]) != self.batch_size:
            raise ValueError("resume batch size differs from saved training cursor")
        self.resume_batch_index = int(state["resume_batch_index"])


def build_explicit_episode_loaders(
    config: Any,
    split: dict[str, list[int]],
    *,
    world_size: int,
    rank: int,
    seed: int,
):
    from torch.utils.data import DataLoader

    train_dataset, heldout_dataset, collate_fn = build_explicit_episode_datasets(
        config, split, seed=seed
    )
    batch_size = int(config.data.train_batch_size)
    native_sampler = train_dataset.build_train_sampler(
        world_size=world_size, rank=rank
    )
    sampler = ResumeAwareSampler(native_sampler, train_dataset, batch_size=batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_fn,
    )
    heldout_loader = DataLoader(
        heldout_dataset,
        batch_size=int(config.data.val_batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    return train_loader, heldout_loader


def _episode_for_index(dataset: Any, index: int) -> int:
    virtual_indices = getattr(dataset, "_virtual_index", None)
    if virtual_indices is not None:
        window_index = int(virtual_indices[index][0])
    else:
        windows = getattr(dataset, "windows", None)
        if windows is None or len(windows) != len(dataset):
            raise ValueError(
                f"Cannot map sample indices to episodes for {type(dataset).__name__}"
            )
        window_index = int(index)
    return int(dataset.windows[window_index].episode_index)


def build_explicit_episode_datasets(
    config: Any, split: dict[str, list[int]], *, seed: int
):
    """Build native local-latent datasets from exact JSON episode IDs."""
    from open_wam.configs import DataSplit, SampleOrderMode
    from open_wam.data import build_train_val_latent_datasets, collate_latent_wam_samples

    if config.trainer.batch_adapter.value != "latents":
        raise ValueError("explicit episode filtering currently requires batch_adapter=latents")
    if config.data.batching.mode.value != "strict":
        raise ValueError("explicit distributed recipe currently requires strict latent batches")
    if config.data.sample_construction.sample_order_mode != SampleOrderMode.REPLACEMENT:
        raise ValueError("explicit distributed recipe requires native replacement sampling")
    all_config = replace(config.data, train_fraction=1.0, split=DataSplit.TRAIN)
    all_train, _ = build_train_val_latent_datasets(all_config)
    windows = list(getattr(all_train, "windows", ()))
    if not windows:
        raise ValueError("native latent dataset exposes no episode windows")
    available = {int(window.episode_index) for window in windows}
    train_ids = set(split["train_episode_ids"])
    heldout_ids = set(split["heldout_episode_ids"])
    if train_ids | heldout_ids != available:
        raise ValueError(
            "episode split must account for every prepared episode: "
            f"missing={sorted(available - train_ids - heldout_ids)}, "
            f"unknown={sorted((train_ids | heldout_ids) - available)}"
        )
    if train_ids & heldout_ids:
        raise ValueError("training and heldout episode IDs must be disjoint")
    train_windows = [window for window in windows if int(window.episode_index) in train_ids]
    heldout_windows = [window for window in windows if int(window.episode_index) in heldout_ids]
    train_indices = [
        index
        for index in range(len(all_train))
        if _episode_for_index(all_train, index) in train_ids
    ]
    train_dataset = EpisodeSubset(all_train, train_indices, seed=seed)
    heldout_config = replace(config.data, split=DataSplit.VAL)
    heldout_dataset = type(all_train)(
        data_config=heldout_config,
        windows=heldout_windows,
    )
    if set(train_dataset.episode_ids) != train_ids:
        raise ValueError("explicit train episode selection lost an episode")
    actual_heldout = {int(window.episode_index) for window in heldout_dataset.windows}
    if actual_heldout != heldout_ids:
        raise ValueError("explicit heldout episode selection lost an episode")
    return train_dataset, heldout_dataset, collate_latent_wam_samples


def validate_distributed_loader(loader: Any, *, world_size: int, rank: int) -> None:
    """Verify native rank ownership without assuming DistributedSampler semantics."""
    dataset_size = len(loader.dataset)
    if dataset_size <= 0:
        raise ValueError("distributed train/heldout dataset must be nonempty")
    if world_size <= 1:
        return
    sampler = getattr(loader.sampler, "sampler", loader.sampler)
    if getattr(sampler, "world_size", None) != world_size or getattr(sampler, "rank", None) != rank:
        raise ValueError(
            f"Native sampler {type(sampler).__name__} has no matching rank/world_size contract"
        )
    if getattr(sampler, "total_size", dataset_size) != dataset_size and not hasattr(sampler, "base_seed"):
        raise ValueError(
            "native sampler would pad or drop a non-replacement sample stream"
        )


def _qualified_type(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    cuda_states = {}
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        cuda_states[str(device)] = torch.cuda.get_rng_state(device)
    return {
        "python": random.getstate(),
        "numpy": (
            str(numpy_state[0]),
            numpy_state[1].tolist(),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        ),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_states,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state[0],
            np.asarray(numpy_state[1], dtype=np.uint32),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        )
    )
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint has CUDA RNG state but CUDA is unavailable")
        current_device = torch.cuda.current_device()
        if set(cuda_states) != {str(current_device)}:
            raise ValueError("rank-local CUDA RNG state does not match local device")
        torch.cuda.set_rng_state(cuda_states[str(current_device)], current_device)


def _sampler_state(sampler: Any) -> dict[str, Any]:
    from torch.utils.data import RandomSampler, SequentialSampler
    from torch.utils.data.distributed import DistributedSampler

    if isinstance(sampler, ResumeAwareSampler):
        return {"type": _qualified_type(sampler), "kind": "resume_wrapper", "state": sampler.state_dict()}
    if all(hasattr(sampler, name) for name in ("world_size", "rank", "epoch")):
        state = {
            "type": _qualified_type(sampler),
            "kind": "native_rank_sampler",
            "epoch": int(sampler.epoch),
            "rank": int(sampler.rank),
            "world_size": int(sampler.world_size),
        }
        for name in ("base_seed", "seed", "drop_last"):
            if hasattr(sampler, name):
                state[name] = getattr(sampler, name)
        return state
    if isinstance(sampler, DistributedSampler):
        return {
            "type": _qualified_type(sampler),
            "kind": "distributed_sampler",
            "epoch": int(sampler.epoch),
            "seed": int(sampler.seed),
            "rank": int(sampler.rank),
            "num_replicas": int(sampler.num_replicas),
            "drop_last": bool(sampler.drop_last),
        }
    if isinstance(sampler, RandomSampler):
        return {
            "type": _qualified_type(sampler),
            "kind": "random_sampler",
            "generator_state": (
                sampler.generator.get_state() if sampler.generator is not None else None
            ),
        }
    if isinstance(sampler, SequentialSampler):
        return {"type": _qualified_type(sampler), "kind": "sequential_sampler"}
    state_dict = getattr(sampler, "state_dict", None)
    if callable(state_dict):
        state = state_dict()
        if isinstance(state, dict):
            return {"type": _qualified_type(sampler), "kind": "state_dict", "state": state}
    raise TypeError(f"Unsupported sampler state contract: {_qualified_type(sampler)}")


def _restore_sampler_state(sampler: Any, state: dict[str, Any], *, rank: int) -> None:
    from torch.utils.data import RandomSampler, SequentialSampler
    world_size = int(dist.get_world_size()) if dist.is_initialized() else 1

    if state.get("type") != _qualified_type(sampler):
        raise ValueError("checkpoint sampler type does not match runtime sampler")
    if state.get("kind") == "resume_wrapper" and isinstance(sampler, ResumeAwareSampler):
        sampler.load_state_dict(state["state"])
        return
    if state.get("kind") == "native_rank_sampler":
        if int(state["rank"]) != rank or int(state["world_size"]) != world_size:
            raise ValueError("checkpoint native sampler topology does not match runtime")
        for name in ("base_seed", "seed", "drop_last"):
            if name in state and getattr(sampler, name, None) != state[name]:
                raise ValueError(f"checkpoint native sampler {name} differs from runtime")
        sampler.set_epoch(int(state["epoch"]))
        return
    if state.get("kind") == "distributed_sampler":
        if int(state["rank"]) != rank:
            raise ValueError("checkpoint sampler rank does not match this process")
        if int(state["num_replicas"]) != world_size:
            raise ValueError("checkpoint sampler world size does not match runtime")
        if int(state["seed"]) != int(sampler.seed):
            raise ValueError("checkpoint sampler seed does not match runtime")
        if bool(state["drop_last"]) != bool(sampler.drop_last):
            raise ValueError("checkpoint sampler drop_last setting does not match")
        sampler.set_epoch(int(state["epoch"]))
        return
    if state.get("kind") == "random_sampler" and isinstance(sampler, RandomSampler):
        generator_state = state.get("generator_state")
        if generator_state is not None:
            if sampler.generator is None:
                raise ValueError("checkpoint RandomSampler generator is missing")
            sampler.generator.set_state(generator_state)
        return
    if state.get("kind") == "sequential_sampler" and isinstance(
        sampler, SequentialSampler
    ):
        return
    if state.get("kind") == "state_dict":
        loader = getattr(sampler, "load_state_dict", None)
        if not callable(loader):
            raise TypeError("runtime sampler cannot load its saved state_dict")
        loader(state["state"])
        return
    raise ValueError("unsupported sampler state in checkpoint")


def _rank_state_path(checkpoint_dir: str | Path) -> Path:
    return Path(checkpoint_dir) / "rank_runtime_state.pt"


def _controller_state(model: Any) -> dict[str, torch.Tensor] | None:
    controller = getattr(getattr(model, "policy_variant", None), "routing_controller", None)
    if controller is None:
        return None
    return {name: value.detach().cpu() for name, value in controller.state_dict().items()}


def save_rank_runtime_state(
    checkpoint_dir: str | Path,
    *,
    rank: int,
    loader: Any,
    model: Any,
    train_state: Any,
    identity: dict[str, Any],
) -> Path:
    """Gather all rank-local continuation state; rank zero writes one sidecar."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(checkpoint_dir)
    local_state = {
        "rank": int(rank),
        "rng": _capture_rng_state(),
        "sampler": _sampler_state(loader.sampler),
        "train_cursor": {
            name: getattr(train_state, name)
            for name in (
                "global_step",
                "optimizer_step",
                "epoch_index",
                "next_batch_index",
                "seen_batches",
            )
        },
        "routing_controller": _controller_state(model),
        "loader_generator_state": (
            loader.generator.get_state() if loader.generator is not None else None
        ),
    }
    world_size = int(dist.get_world_size()) if dist.is_initialized() else 1
    gathered: list[dict[str, Any] | None] | None = [None] * world_size if rank == 0 else None
    if dist.is_initialized():
        dist.gather_object(local_state, object_gather_list=gathered, dst=0)
    else:
        gathered = [local_state]
    target = _rank_state_path(checkpoint_dir)
    if rank == 0:
        payload = {
            "format": "gradientwam-rank-state-v2",
            "world_size": world_size,
            "identity": identity,
            "ranks": gathered,
        }
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    if dist.is_initialized():
        dist.barrier()
    return target


def restore_rank_runtime_state(
    checkpoint_dir: str | Path,
    *,
    rank: int,
    loader: Any,
    model: Any,
    train_state: Any,
    identity: dict[str, Any],
) -> None:
    """Restore sampler/controller state; defer RNG until cursor replay completes."""
    payloads: list[Any] = [None]
    if rank == 0:
        payloads[0] = torch.load(
            _rank_state_path(checkpoint_dir), map_location="cpu", weights_only=True
        )
    if dist.is_initialized():
        dist.broadcast_object_list(payloads, src=0)
    payload = payloads[0]
    world_size = int(dist.get_world_size()) if dist.is_initialized() else 1
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "gradientwam-rank-state-v2"
        or int(payload.get("world_size", -1)) != world_size
        or payload.get("identity") != identity
    ):
        raise ValueError("checkpoint run identity/world size does not match this launch")
    rank_states = payload.get("ranks")
    if not isinstance(rank_states, list) or len(rank_states) != world_size:
        raise ValueError("checkpoint lacks the complete rank state list")
    state = rank_states[rank]
    if not isinstance(state, dict) or int(state.get("rank", -1)) != rank:
        raise ValueError("checkpoint rank-local continuation state is missing")
    _restore_sampler_state(loader.sampler, state["sampler"], rank=rank)
    if any(
        int(state["train_cursor"][name]) != int(getattr(train_state, name))
        for name in state["train_cursor"]
    ):
        raise ValueError("rank sidecar cursor does not match native full checkpoint")
    generator_state = state.get("loader_generator_state")
    if generator_state is not None:
        if loader.generator is None:
            raise ValueError("checkpoint has a DataLoader generator but runtime does not")
        loader.generator.set_state(generator_state)
    controller = getattr(getattr(model, "policy_variant", None), "routing_controller", None)
    controller_state = state.get("routing_controller")
    if controller is not None:
        if not isinstance(controller_state, dict):
            raise ValueError("rank sidecar lacks routing-controller state")
        controller.load_state_dict(controller_state, strict=True)
    elif controller_state is not None:
        raise ValueError("checkpoint has routing-controller state but runtime does not")
    cursor = int(train_state.next_batch_index)
    if cursor > 0:
        if not isinstance(loader.dataset, EpisodeSubset):
            raise ValueError("exact mid-epoch resume requires EpisodeSubset")
        if not isinstance(loader.sampler, ResumeAwareSampler):
            raise ValueError("exact mid-epoch resume requires ResumeAwareSampler")
        loader.dataset.resume_rng_state = state["rng"]
        loader.sampler.resume_batch_index = cursor
    else:
        _restore_rng_state(state["rng"])


def _weighted_action_proxy(action: Any) -> torch.Tensor:
    from open_wam.models.common.route_evidence import route_objective_weighted

    timestep_weight = action.scheduler.training_weight(
        action.timesteps.flatten()
    ).reshape(action.timesteps.shape)
    if action.private_flow_pred is not None:
        if action.prior_shared is None:
            raise ValueError("private branch artifacts are missing observed prior")
        route = route_objective_weighted(
            action.private_flow_pred,
            action.flow_pred,
            action.targets,
            action.action_mask,
            timestep_weight,
            action.prior_shared,
        )
        prior = action.prior_shared.float()
        branch = route["branch_weighted_mse"]
        return ((1.0 - prior) * branch[:, 0] + prior * branch[:, 1]).mean()

    target = action.targets.detach().float()
    error = torch.nn.functional.mse_loss(
        action.flow_pred.float(), target, reduction="none"
    )
    mask = (
        torch.ones_like(target, dtype=torch.float32)
        if action.action_mask is None
        else action.action_mask.float()
    )
    denominator = mask.sum(-1).clamp_min(1.0)
    per_token = (error * timestep_weight.float()[:, :, None] * mask).sum(-1)
    return (per_token / denominator).mean(-1).mean()


def _heldout_batch_proxies(result: Any) -> tuple[torch.Tensor, torch.Tensor, int]:
    from open_wam.models.common.flow_supervision import masked_video_flow_match_loss
    from open_wam.models.decoder_artifacts import (
        DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT,
        DualExpertTrainArtifacts,
    )

    pipeline_output = result.output
    envelope = pipeline_output.policy_output.decoder_artifacts
    if envelope is None:
        raise ValueError("heldout evaluation is missing native decoder artifacts")
    artifacts = envelope.require(
        contract=DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT,
        payload_type=DualExpertTrainArtifacts,
    )
    action_proxy = _weighted_action_proxy(artifacts.action)
    if artifacts.video is None:
        world_proxy = action_proxy.new_zeros(())
    else:
        video = artifacts.video
        world_proxy = masked_video_flow_match_loss(
            flow_pred=video.flow_pred,
            targets=video.targets,
            timesteps=video.timesteps,
            scheduler=video.scheduler,
            future_loss_mask=video.future_loss_mask,
        )
    sample_count = int(artifacts.action.targets.shape[0])
    return action_proxy.detach().float(), world_proxy.detach().float(), sample_count


class RankAwareTrainingRuntime:
    """Narrow adapter that adds rank RNG state and comparable heldout proxies."""

    def __init__(
        self,
        runtime: Any,
        *,
        output_dir: Path,
        eval_seed: int,
        identity: dict[str, Any],
    ) -> None:
        self._runtime = runtime
        self.__dict__.update(runtime.__dict__)
        self.output_dir = output_dir
        self.eval_seed = int(eval_seed)
        self.identity = identity
        runtime._save_checkpoint = self._save_checkpoint

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime, name)

    def _save_checkpoint(self, *, final: bool) -> None:
        from open_wam.training.runtime import TrainingRuntime

        previous = self.train_state.last_checkpoint_path
        TrainingRuntime._save_checkpoint(self._runtime, final=final)
        self.train_state = self._runtime.train_state
        checkpoint = self._runtime.train_state.last_checkpoint_path
        if checkpoint is None:
            return
        path = Path(checkpoint)
        if checkpoint != previous or not _rank_state_path(path).is_file():
            save_rank_runtime_state(
                path,
                rank=self.strategy.rank,
                loader=self.train_loader,
                model=self.strategy.unwrap_model(self.model),
                train_state=self.train_state,
                identity=self.identity,
            )

    def resume(self, checkpoint_path: str) -> None:
        from open_wam.training.checkpoints import CheckpointManager
        from open_wam.training.runtime import TrainingRuntime

        TrainingRuntime.resume(self._runtime, checkpoint_path)
        self.train_state = self._runtime.train_state
        if self.train_state.optimizer_step > int(self.config.training.num_steps):
            raise ValueError(
                "cannot resume with a step limit below the checkpoint's completed optimizer step"
            )
        checkpoint_file = CheckpointManager.resolve_checkpoint_path(checkpoint_path)
        restore_rank_runtime_state(
            checkpoint_file.parent,
            rank=self.strategy.rank,
            loader=self.train_loader,
            model=self.strategy.unwrap_model(self.model),
            train_state=self.train_state,
            identity=self.identity,
        )

    def run(self):
        from open_wam.training.loop_policies import StepLoopPolicy

        if self.config.training.num_steps is None:
            raise ValueError("GradientWAM distributed training requires training.num_steps")
        if self.config.trainer.loop_policy.value != "steps":
            raise ValueError("GradientWAM distributed training requires loop_policy=steps")
        self.log_sink.log_event(
            name="run_start",
            payload={
                "run_name": self.train_state.run_name,
                "strategy": "ddp",
                "world_size": self.strategy.world_size,
                "heldout_split": "episode_split_json",
            },
        )
        self.strategy.zero_grad(self.optimizer)
        try:
            self._run_step_loop(
                StepLoopPolicy(
                    max_steps=int(self.config.training.num_steps),
                    limit_train_batches=self.config.trainer.limit_train_batches,
                    limit_val_batches=0,
                )
            )
            metrics = self._evaluate_heldout()
            if self.strategy.is_main_process:
                assert metrics is not None
                target = self.output_dir / "heldout_proxy_metrics.json"
                temporary = target.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False)
                    + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, target)
                self.log_sink.log_event(
                    name="heldout_denoising_proxy",
                    payload=metrics["metrics"],
                )
            self.strategy.barrier()
            return self.train_state
        finally:
            self.log_sink.close()
            self.strategy.close()

    def _evaluate_heldout(self) -> dict[str, Any] | None:
        if not self.strategy.is_main_process:
            return None
        model = self.strategy.unwrap_model(self.model)
        model.eval()
        self.step_executor.pipeline = model
        random.seed(self.eval_seed)
        np.random.seed(self.eval_seed % (2**32))
        torch.manual_seed(self.eval_seed)
        if self.strategy.device.type == "cuda":
            torch.cuda.manual_seed(self.eval_seed)
        action_total = 0.0
        world_total = 0.0
        sample_total = 0
        with torch.no_grad():
            for batch in self.val_loader:
                device_batch = self.step_executor.batch_adapter.move_to_device(
                    batch, self.strategy.device
                )
                with self.strategy.autocast_context():
                    result = self.step_executor.forward_train(device_batch)
                action_proxy, world_proxy, count = _heldout_batch_proxies(result)
                action_total += float(action_proxy.cpu()) * count
                world_total += float(world_proxy.cpu()) * count
                sample_total += count
        if sample_total <= 0:
            raise ValueError("heldout evaluation produced no samples")
        return {
            "schema_version": 1,
            "status": "heldout_denoising_proxy_completed",
            "run_identity": self.identity,
            "optimizer_step": int(self.train_state.optimizer_step),
            "world_size": int(self.strategy.world_size),
            "eval_seed": self.eval_seed,
            "heldout_episode_ids": self.heldout_episode_ids,
            "heldout_sample_count": sample_total,
            "metrics": {
                "action_denoising_proxy": action_total / sample_total,
                "world_denoising_proxy": world_total / sample_total,
            },
            "semantics": (
                "Native-timestep/mask flow-denoising proxies on heldout episodes; "
                "variational action proxy is the observed-prior-weighted private/shared "
                "branch MSE. These metrics are not closed-loop task success."
            ),
        }


def _resolve_config_path(value: str, *, spec_path: Path) -> Path:
    expanded = os.path.expandvars(value)
    if "${" in expanded:
        raise ValueError(f"Unresolved environment variable in path: {value}")
    path = Path(expanded).expanduser()
    if path.is_absolute():
        return path
    candidates = (Path.cwd() / path, spec_path.parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(value)


def _experiment_config_identity_sha256(config: Any) -> str:
    """Hash the scientific recipe, excluding run length and storage location."""
    from open_wam.configs import serialize_experiment_config

    identity_config = serialize_experiment_config(config)
    identity_config["training"].pop("num_steps", None)
    for field in (
        "checkpoint_dir",
        "default_root_dir",
        "run_name",
        "devices",
        "enable_checkpointing",
    ):
        identity_config["trainer"].pop(field, None)
    return hashlib.sha256(
        json.dumps(
            identity_config,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _build_runtime(spec_path: Path, *, resume: str | None):
    from open_wam.configs import (
        LoopPolicyName,
        resolve_experiment_config,
        validate_experiment_config_runtime_contract,
    )
    from open_wam.configs.enums import (
        CheckpointMode,
        StrategyName,
        TrainerAccelerator,
        TrainerPrecision,
        WandBMode,
    )
    from open_wam.models.policy_variants.dual_expert.variational_sharing import (
        configure_variational_sharing,
    )
    from open_wam.models.visual_tower.public_pretraining import (
        load_public_video_checkpoint_into_tower,
    )
    from open_wam.models.visual_tower.tower import VisualTower
    from open_wam.pipelines.factory import build_variant_pipeline_from_config
    from open_wam.training.checkpoints import CheckpointManager
    from open_wam.training.controls import apply_training_component_controls
    from open_wam.training.data_loading import (
        preflight_runtime_dataset_artifacts,
    )
    from open_wam.training.launch import (
        DistributedLaunchContext,
        validate_training_launch,
    )
    from open_wam.training.logging import build_log_sink
    from open_wam.training.optim import build_optimizer, build_scheduler
    from open_wam.training.runtime import TrainingRuntime, resolve_runtime_output_dir
    from open_wam.training.state import TrainState
    from open_wam.training.step_executor import (
        PipelineTrainStepExecutor,
        build_batch_adapter,
    )
    from open_wam.training.strategies import build_training_strategy

    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("run config must be a YAML mapping")
    required = ("settings_config", "episode_split_json")
    missing = [key for key in required if not spec.get(key)]
    if missing:
        raise ValueError(f"run config is missing required fields: {', '.join(missing)}")
    settings_path = _resolve_config_path(str(spec["settings_config"]), spec_path=spec_path)
    split_path = _resolve_config_path(str(spec["episode_split_json"]), spec_path=spec_path)
    split = load_episode_split(split_path)
    from .settings import load_settings

    settings = load_settings(settings_path)
    context = DistributedLaunchContext.from_env()
    if context.environment.value != "torch_distributed":
        raise ValueError("Launch with torchrun; implicit single-process fallback is disabled")
    seed = int(settings.seed)
    output_dir = Path(settings.output_root)
    steps = int(spec.get("steps", 2))
    config = settings.native_config()
    accumulation_steps = int(config.training.gradient_accumulation_steps)
    if steps <= 0 or accumulation_steps <= 0:
        raise ValueError("steps and gradient_accumulation_steps must be positive")
    trainer = replace(
        config.trainer,
        accelerator=TrainerAccelerator.GPU,
        devices=context.world_size,
        precision=TrainerPrecision.BF16,
        strategy=StrategyName.DDP,
        loop_policy=LoopPolicyName.STEPS,
        enable_checkpointing=True,
        checkpoint_mode=CheckpointMode.FULL_TRAINING_STATE,
        checkpoint_dir=str(output_dir / "checkpoints"),
        default_root_dir=str(output_dir.parent),
        run_name=output_dir.name,
        resume_from=None,
        initialize_weights_from=None,
        enable_jsonl_logging=True,
        enable_wandb=False,
        wandb_mode=WandBMode.DISABLED,
        export_runtime_backbone=False,
        limit_val_batches=0,
    )
    backbone = replace(config.backbone, load_reference_core_weights=False)
    data = replace(config.data, train_fraction=1.0, num_workers=0)
    training = replace(
        config.training,
        num_steps=steps,
    )
    config = validate_experiment_config_runtime_contract(
        resolve_experiment_config(
            replace(config, data=data, backbone=backbone, training=training, trainer=trainer)
        )
    )
    if any(task.enabled and task.max_batches != 0 for task in config.validation.auxiliary_tasks):
        raise ValueError(
            "Disable auxiliary validation tasks; this runner reports only the explicit heldout proxies."
        )
    output_dir = resolve_runtime_output_dir(config)
    if output_dir.exists() and resume is None:
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    dataset_artifacts = preflight_runtime_dataset_artifacts(config)
    strategy = build_training_strategy(config.trainer, launch_context=context)
    try:
        # One common initialization seed gives all arms/ranks the same initial
        # tower and policy; DDP broadcasts rank zero's initialized parameters.
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        train_loader, val_loader = build_explicit_episode_loaders(
            config,
            split,
            world_size=strategy.world_size,
            rank=strategy.rank,
            seed=seed,
        )
        validate_distributed_loader(
            train_loader, world_size=strategy.world_size, rank=strategy.rank
        )
        if strategy.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
        strategy.barrier()

        # Dataset construction is deterministic but reseed before any parameter
        # initialization so four method arms share an identical common start.
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        tower = VisualTower(
            config.backbone,
            action_dim=config.action_decoder.action_dim,
            state_dim=config.data.action_schema.state_dim,
        )
        load_public_video_checkpoint_into_tower(
            tower,
            settings.checkpoint,
            expected_sha256=settings.checkpoint_sha256,
        )
        model = build_variant_pipeline_from_config(config, visual_tower=tower)
        tower.get_runtime_backbone(action_dim=int(config.action_decoder.action_dim))
        model.policy_variant.initialize_for_training(tower)
        trainability_report = apply_training_component_controls(model, config.training)
        sharing_audit = configure_variational_sharing(
            model,
            arm=settings.arm,
            expected_layers=int(config.backbone.num_layers),
            route_seed=settings.route_seed,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        trainable_count = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        trainability_report = replace(
            trainability_report,
            total_parameters=parameter_count,
            trainable_parameters=trainable_count,
        )
        model = strategy.prepare_model(model)
        rank_seed = seed + int(strategy.rank)
        random.seed(rank_seed)
        np.random.seed(rank_seed % (2**32))
        torch.manual_seed(rank_seed)
        if strategy.device.type == "cuda":
            torch.cuda.manual_seed(rank_seed)
        batch_adapter = build_batch_adapter(config.trainer.batch_adapter)
        step_executor = PipelineTrainStepExecutor(
            pipeline=model,
            batch_adapter=batch_adapter,
            training_config=config.training,
        )
        checkpoint_manager = CheckpointManager(
            root_dir=Path(config.trainer.checkpoint_dir),
            config=config,
            checkpoint_mode=config.trainer.checkpoint_mode,
            max_checkpoints_to_keep=config.trainer.max_checkpoints_to_keep,
        )
        run_name = config.trainer.run_name or config.name
        # A two-step smoke can continue to a longer limit or a fresh output
        # directory, while recipe changes remain part of the identity.
        config_sha256 = _experiment_config_identity_sha256(config)
        split_sha256 = hashlib.sha256(
            json.dumps(split, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        identity = {
            "arm": settings.arm,
            "initialization_seed": seed,
            "rank_training_seed_rule": "seed_plus_rank",
            "route_seed": settings.route_seed,
            "public_video_checkpoint_sha256": settings.checkpoint_sha256,
            "experiment_config_sha256": config_sha256,
            "episode_split_sha256": split_sha256,
            "train_episode_ids": split["train_episode_ids"],
            "heldout_episode_ids": split["heldout_episode_ids"],
            "world_size": strategy.world_size,
            "per_rank_batch_size": config.data.train_batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "global_batch_size": (
                strategy.world_size
                * int(config.data.train_batch_size)
                * accumulation_steps
            ),
        }
        if strategy.is_main_process:
            (output_dir / "run_identity.json").write_text(
                json.dumps(identity, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        strategy.barrier()
        optimizer = build_optimizer(model, config.training)
        scheduler = build_scheduler(optimizer, config.training)
        runtime = TrainingRuntime(
            config=config,
            model=model,
            strategy=strategy,
            train_loader=train_loader,
            val_loader=val_loader,
            step_executor=step_executor,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_manager=checkpoint_manager,
            log_sink=build_log_sink(
                config=config, output_dir=output_dir, run_name=run_name, strategy=strategy
            ),
            train_state=TrainState(run_name=run_name),
            trainability_report=trainability_report,
            dataset_artifacts=dataset_artifacts,
        )
        adapted = RankAwareTrainingRuntime(
            runtime,
            output_dir=output_dir,
            eval_seed=int(spec.get("eval_seed", 20261009)),
            identity=identity,
        )
        adapted.heldout_episode_ids = split["heldout_episode_ids"]
        adapted.sharing_audit = sharing_audit
        adapted.run_identity = identity
        if resume is not None:
            adapted.resume(resume)
        return adapted
    except BaseException:
        strategy.close()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True, type=Path)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args(argv)
    runtime = _build_runtime(args.run_config.resolve(), resume=args.resume)
    runtime.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
