"""Shared GradientWAM model/optimizer/runtime construction for distributed entries."""
from __future__ import annotations

import json
from dataclasses import replace
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _configure_cagrad_compilation(method_config: Any) -> None:
    if not method_config.uses_cagrad:
        return
    from torch._functorch import config as functorch_config

    # Task gradients reuse the forward graph, so its saved buffers cannot be donated.
    functorch_config.donated_buffer = False
    print("[cagrad_compile] donated_buffer=False", flush=True)


def build_rank_aware_runtime(
    *,
    settings: Any,
    config: Any,
    strategy: Any,
    train_loader: Any,
    val_loader: Any,
    output_dir: Path,
    dataset_artifacts: tuple[Any, ...],
    identity: dict[str, Any],
    heldout_episode_ids: list[int] | tuple[int, ...],
    eval_seed: int,
    resume: str | None = None,
    heldout_split_label: str = "episode_split_json",
    load_public_video_checkpoint: bool = True,
):
    """Build the common model, method controls, optimizers, and training runtime."""
    from open_wam.models.visual_tower.public_pretraining import (
        load_public_video_checkpoint_into_tower,
    )
    from open_wam.models.visual_tower.tower import VisualTower
    from open_wam.pipelines.factory import build_variant_pipeline_from_config
    from open_wam.training.checkpoints import CheckpointManager
    from open_wam.training.controls import apply_training_component_controls
    from open_wam.training.logging import build_log_sink
    from open_wam.training.optim import build_optimizer, build_scheduler
    from open_wam.training.runtime import TrainingRuntime
    from open_wam.training.state import TrainState
    from open_wam.training.step_executor import PipelineTrainStepExecutor, build_batch_adapter
    from .distributed_train import RankAwareTrainingRuntime

    try:
        _configure_cagrad_compilation(settings.method_config)
        seed = int(settings.seed)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        if strategy.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
        strategy.barrier()

        tower = VisualTower(
            config.backbone,
            action_dim=config.action_decoder.action_dim,
            state_dim=config.data.action_schema.state_dim,
        )
        if load_public_video_checkpoint:
            load_public_video_checkpoint_into_tower(
                tower,
                settings.checkpoint,
                expected_sha256=settings.checkpoint_sha256,
            )
        model = build_variant_pipeline_from_config(config, visual_tower=tower)
        tower.get_runtime_backbone(action_dim=int(config.action_decoder.action_dim))
        model.policy_variant.initialize_for_training(tower)
        trainability_report = apply_training_component_controls(model, config.training)
        cagrad_candidates: tuple[torch.nn.Parameter, ...] = ()
        if settings.method_config.legacy_v02:
            from open_wam.models.policy_variants.dual_expert.variational_sharing import (
                configure_variational_sharing,
            )

            trainability_audit = configure_variational_sharing(
                model,
                arm=settings.arm,
                expected_layers=int(config.backbone.num_layers),
                route_seed=settings.route_seed,
            )
        else:
            from .cagrad_training import (
                TRAINABILITY_SCOPE_ID,
                cagrad_candidate_parameters,
                configure_native_trainability,
            )

            trainability_audit = configure_native_trainability(
                model, expected_layers=int(config.backbone.num_layers)
            )
            if trainability_audit["scope_id"] != TRAINABILITY_SCOPE_ID:
                raise ValueError("Unexpected GradientWAM trainability scope")
            if settings.method_config.uses_vrfm:
                from open_wam.models.policy_variants.dual_expert.vrfm import configure_vrfm

                trainability_audit["vrfm"] = configure_vrfm(
                    model,
                    latent_dim=settings.method_config.latent_dim,
                    kl_weight=settings.method_config.kl_weight,
                )
            if settings.method_config.uses_cagrad:
                cagrad_candidates = cagrad_candidate_parameters(model)

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
                config=config,
                output_dir=output_dir,
                run_name=run_name,
                strategy=strategy,
            ),
            train_state=TrainState(run_name=run_name),
            trainability_report=trainability_report,
            dataset_artifacts=dataset_artifacts,
        )
        adapted = RankAwareTrainingRuntime(
            runtime,
            output_dir=output_dir,
            eval_seed=int(eval_seed),
            identity=identity,
            method_config=None if settings.method_config.legacy_v02 else settings.method_config,
            cagrad_candidates=cagrad_candidates,
            trainability_audit=trainability_audit,
        )
        adapted.heldout_episode_ids = list(heldout_episode_ids)
        adapted.heldout_split_label = heldout_split_label
        adapted.sharing_audit = trainability_audit
        adapted.run_identity = identity
        if resume is not None:
            adapted.resume(resume)
        return adapted
    except BaseException:
        strategy.close()
        raise
