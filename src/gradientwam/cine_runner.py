"""Independent Cine v3 training, preparation, data-check, and inference CLI."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .settings import CineSettings, load_cine_settings


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    check_config = commands.add_parser("check-config", help="validate Cine YAML and native geometry")
    check_config.add_argument("--config", required=True, type=Path)

    prepare = commands.add_parser("prepare", help="plan or execute explicit Cine latent preparation")
    prepare.add_argument("--config", required=True, type=Path)
    prepare.add_argument("--device", choices=("cpu", "cuda"), required=True)
    prepare.add_argument("--execute", action="store_true")
    prepare.add_argument("--all-windows", action="store_true")
    prepare.add_argument("--train-episode-limit", type=_positive_int)
    prepare.add_argument("--train-windows-per-episode", type=_positive_int)
    prepare.add_argument("--val-episode-limit", type=_positive_int)
    prepare.add_argument("--val-windows-per-episode", type=_positive_int)

    check_data = commands.add_parser("check-data", help="validate Cine cache and native samples")
    check_data.add_argument("--config", required=True, type=Path)
    check_data.add_argument("--sample-limit", type=_positive_int, default=1)

    train = commands.add_parser("train", help="train one Cine method under torchrun")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--resume", default=None)

    infer = commands.add_parser("infer", help="strictly load model-only Cine weights")
    infer.add_argument("--config", required=True, type=Path)
    infer.add_argument("--checkpoint", required=True, type=Path)
    infer.add_argument("--device", default="cpu")

    return parser


def _load_settings(config_path: Path) -> CineSettings:
    return load_cine_settings(config_path)


def _check_config(settings: CineSettings) -> dict[str, Any]:
    config = settings.native_config()
    return {
        "status": "valid",
        "method": settings.method_config.label,
        "dataset_type": config.data.dataset_type,
        "train_root": str(settings.train_root),
        "validation_root": str(settings.val_root),
        "latent_root": str(settings.latent_root),
        "steps": settings.steps,
        "action_dim": config.data.action_schema.action_dim,
        "action_horizon": config.action_decoder.action_horizon,
        "state_dim": config.data.action_schema.state_dim,
        "latent_frames": config.data.num_frames,
        "frame_chunk_size": config.inference.frame_chunk_size,
        "sample_stride": config.data.sample_stride,
    }


def _prepare(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    limits = (
        args.train_episode_limit,
        args.train_windows_per_episode,
        args.val_episode_limit,
        args.val_windows_per_episode,
    )
    if args.all_windows and any(value is not None for value in limits):
        parser.error("--all-windows cannot be combined with episode/window limits")
    if not args.all_windows and any(value is None for value in limits):
        parser.error("choose --all-windows or provide all four positive episode/window limits")

    settings = _load_settings(args.config)
    settings.validate_fresh_cache_root(execute=args.execute)
    from .cine_preparation import prepare_cine

    result = prepare_cine(
        settings.native_config().data,
        frontend_root=settings.frontend_root,
        tokenizer_root=settings.tokenizer_root,
        device=args.device,
        execute=args.execute,
        train_episode_limit=args.train_episode_limit,
        train_windows_per_episode=args.train_windows_per_episode,
        val_episode_limit=args.val_episode_limit,
        val_windows_per_episode=args.val_windows_per_episode,
        all_windows=args.all_windows,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


def _check_data(settings: CineSettings, sample_limit: int) -> dict[str, Any]:
    from open_wam.data.cine_v3_latent import build_cine_latent_train_val_datasets
    from open_wam.data.cine_v3 import contained_path

    native = settings.native_config()
    train, validation = build_cine_latent_train_val_datasets(native.data)
    result: dict[str, Any] = {
        "status": "valid",
        "dataset_type": native.data.dataset_type,
        "train_samples": len(train),
        "validation_samples": len(validation),
        "train_source_identity": train.manifest["sources"]["train"],
        "validation_source_identity": train.manifest["sources"]["validation"],
        "selection": train.manifest.get("selection"),
        "manifest_sha256": hashlib.sha256(
            contained_path(settings.latent_root, native.data.adapter_options.get("cache_manifest", "manifest.json")).read_bytes()
        ).hexdigest(),
        "checked": {"train": 0, "validation": 0},
    }
    for split, dataset in (("train", train), ("validation", validation)):
        limit = min(sample_limit, len(dataset))
        for index in range(limit):
            sample = dataset[index]
            if sample.video_latents.shape[1] != native.data.num_frames:
                raise ValueError(f"{split} sample has the wrong latent time dimension")
            if sample.actions.shape != (native.data.action_schema.action_horizon, 7):
                raise ValueError(f"{split} sample has the wrong action shape")
            if sample.state.shape[-1] != 7 or sample.proprio_context_frames.shape[0] != native.data.num_frames:
                raise ValueError(f"{split} sample has the wrong state/proprio shape")
            result["checked"][split] += 1
    return result


def _cine_loaders(config, strategy, *, seed: int):
    from open_wam.data import collate_latent_wam_samples
    from open_wam.data.cine_v3_latent import build_cine_latent_train_val_datasets
    from .distributed_train import EpisodeSubset, ResumeAwareSampler, validate_distributed_loader

    train_dataset, val_dataset = build_cine_latent_train_val_datasets(config.data)
    from open_wam.configs import BatchingMode
    if config.data.batching.mode is not BatchingMode.STRICT:
        raise ValueError("Cine distributed training requires strict latent batches.")
    train_subset = EpisodeSubset(train_dataset, list(range(len(train_dataset))), seed=seed)
    batch_size = int(config.data.train_batch_size)
    native_sampler = train_subset.build_train_sampler(
        world_size=strategy.world_size,
        rank=strategy.rank,
    )
    sampler = ResumeAwareSampler(native_sampler, train_subset, batch_size=batch_size)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_latent_wam_samples,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config.data.val_batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_latent_wam_samples,
    )
    validate_distributed_loader(
        train_loader,
        world_size=strategy.world_size,
        rank=strategy.rank,
    )
    return train_loader, val_loader, train_dataset, val_dataset


def _build_cine_runtime(settings: CineSettings, *, resume: str | None):
    import numpy as np
    import random

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
    from open_wam.training.data_loading import preflight_runtime_dataset_artifacts
    from open_wam.training.launch import DistributedLaunchContext, LaunchEnvironment
    from open_wam.training.strategies import build_training_strategy
    from .distributed_train import _experiment_config_identity_sha256

    context = DistributedLaunchContext.from_env()
    if context.environment is not LaunchEnvironment.TORCH_DISTRIBUTED:
        raise ValueError("Launch Cine training with torchrun; implicit single-process fallback is disabled.")
    output_dir = settings.output_root
    if output_dir.exists() and resume is None:
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    config = settings.native_config()
    if settings.steps <= 0 or config.training.gradient_accumulation_steps <= 0:
        raise ValueError("Cine steps and gradient accumulation must be positive.")
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
    config = validate_experiment_config_runtime_contract(
        resolve_experiment_config(
            replace(
                config,
                data=replace(config.data, train_fraction=1.0, num_workers=0),
                training=replace(config.training, num_steps=settings.steps),
                trainer=trainer,
            )
        )
    )
    if any(task.enabled and task.max_batches != 0 for task in config.validation.auxiliary_tasks):
        raise ValueError("Cine uses only the explicit heldout-prior evaluation path.")

    strategy = build_training_strategy(config.trainer, launch_context=context)
    try:
        random.seed(settings.seed)
        np.random.seed(settings.seed % (2**32))
        torch.manual_seed(settings.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(settings.seed)
        train_loader, val_loader, train_dataset, val_dataset = _cine_loaders(
            config,
            strategy,
            seed=settings.seed,
        )
        dataset_artifacts = preflight_runtime_dataset_artifacts(config)
        manifest_path = settings.latent_root / config.data.adapter_options.get("cache_manifest", "manifest.json")
        manifest = train_dataset.manifest
        selection = manifest.get("selection", {
            "train": train_dataset.manifest["samples"]["train"],
            "validation": train_dataset.manifest["samples"]["validation"],
            "window_stride": config.data.sample_stride,
        })
        identity = {
            **settings.identity(),
            "initialization_seed": settings.seed,
            "rank_training_seed_rule": "seed_plus_rank",
            "public_video_checkpoint_sha256": settings.checkpoint_sha256,
            "experiment_config_sha256": _experiment_config_identity_sha256(config),
            "cine_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "cine_source_identities": manifest["sources"],
            "cine_selection": selection,
            "world_size": strategy.world_size,
            "per_rank_batch_size": config.data.train_batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "global_batch_size": (
                strategy.world_size
                * int(config.data.train_batch_size)
                * int(config.training.gradient_accumulation_steps)
            ),
        }
        from .runtime_factory import build_rank_aware_runtime

        return build_rank_aware_runtime(
            settings=settings,
            config=config,
            strategy=strategy,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=output_dir,
            dataset_artifacts=dataset_artifacts,
            identity=identity,
            heldout_episode_ids=sorted(
                {int(entry["episode_index"]) for entry in val_dataset.entries}
            ),
            eval_seed=settings.eval_seed,
            resume=resume,
            heldout_split_label="cine_validation_root",
        )
    except BaseException:
        strategy.close()
        raise


def _run_inference(settings: CineSettings, checkpoint: Path, device: str) -> dict[str, Any]:
    from .cine_inference import load_cine_policy_for_inference

    model = load_cine_policy_for_inference(settings, checkpoint, device=device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "status": "model_only_checkpoint_loaded",
        "method": settings.method_config.label,
        "checkpoint": str(checkpoint.resolve()),
        "device": device,
        "parameter_count": parameter_count,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return _prepare(args, parser)

    settings = _load_settings(args.config)
    if args.command == "check-config":
        print(json.dumps(_check_config(settings), indent=2, sort_keys=True))
        return 0
    if args.command == "check-data":
        print(json.dumps(_check_data(settings, args.sample_limit), indent=2, sort_keys=True))
        return 0
    if args.command == "train":
        runtime = _build_cine_runtime(settings, resume=args.resume)
        runtime.run()
        return 0
    if args.command == "infer":
        report = _run_inference(settings, args.checkpoint, args.device)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
