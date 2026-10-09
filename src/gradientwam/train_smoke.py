"""Portable two-update smoke, adapted from the reviewed round02 runner.

The OpenWAM factory, sharing setup, optimizer and checkpoint implementations
are reused; local orchestration/lease documents are not deployment inputs.
"""
from __future__ import annotations
import gc
import json
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any


def _update(stack, batch, state, *, torch, deadline):
    """Reviewed native 10-microbatch accumulation; only logging is portable."""
    model, strategy, optimizer = (stack[k] for k in ('model', 'strategy', 'optimizer'))
    model.train()
    losses = []
    cagrad = stack.get('cagrad_accumulator')
    applied_lr = optimizer.param_groups[0]['lr']
    for _ in range(10):
        if time.monotonic() >= deadline:
            raise TimeoutError('Smoke time budget exceeded.')
        with strategy.autocast_context():
            result = stack['executor'].forward_train(batch)
            loss = result.loss / 10
        if cagrad is not None:
            if result.task_losses is None or result.task_active is None:
                raise ValueError('CAGrad requires graph-bearing task losses and native activity masks.')
            scaler = getattr(strategy, 'grad_scaler', None)
            scale = float(scaler.get_scale()) if scaler is not None and scaler.is_enabled() else 1.0
            cagrad.accumulate(result.task_losses, result.task_active, scale=scale * 0.1)
        if not torch.isfinite(loss).item():
            raise ValueError('Nonfinite loss; optimizer not advanced.')
        strategy.backward(loss)
        losses.append(float(result.loss.detach().cpu()))
        state.global_step += 1
        state.seen_batches += 1
        state.next_batch_index += 1
    cagrad_report = cagrad.finalize(model.parameters()) if cagrad is not None else None
    strategy.unscale_(optimizer)
    norm = strategy.clip_grad_norm_(model.parameters(), 2.)
    if not torch.isfinite(norm).item():
        raise ValueError('Nonfinite gradient; optimizer not advanced.')
    from open_wam.training.optim import _normalize_optimizer_state_dtypes
    _normalize_optimizer_state_dtypes(optimizer)
    strategy.optimizer_step(optimizer)
    stack['scheduler'].step()
    strategy.zero_grad(optimizer)
    if cagrad is not None:
        from .cagrad_training import CAGradGradientAccumulator
        stack['cagrad_accumulator'] = CAGradGradientAccumulator(
            stack['cagrad_candidates'], c=stack['method_config'].cagrad_c
        )
    state.optimizer_step += 1
    state.last_checkpoint_path = None
    report = {'optimizer_step': state.optimizer_step, 'microbatch_cursor': state.next_batch_index,
            'mean_loss': sum(losses)/10, 'grad_norm': float(norm.detach().cpu()),
            'applied_learning_rate': applied_lr, 'next_learning_rate': stack['scheduler'].get_last_lr()[0],
            'optimizer_state_dtypes': _optimizer_state_dtypes(optimizer, torch)}
    if cagrad_report is not None:
        report['cagrad'] = cagrad_report
    return report


def run(settings, *, resume: Path | None = None):
    """Two successful updates total; optional strict step1 resume to step2 only."""
    import signal
    import numpy as np
    import torch
    from open_wam.training.state import TrainState
    from . import checkpoint
    from .data_check import load_sample

    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Explicitly select one free GPU with CUDA_VISIBLE_DEVICES.')
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('One visible BF16-capable CUDA device is required.')
    if settings.output_root.exists():
        raise FileExistsError('Select a fresh output directory; existing outputs are never overwritten.')
    ancestor = settings.output_root.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < 65 * 1024**3:
        raise RuntimeError('At least 65 GiB free space is required for the one full-state checkpoint.')
    config = settings.native_config()
    random.seed(settings.seed); np.random.seed(settings.seed)
    torch.manual_seed(settings.seed); torch.cuda.manual_seed(settings.seed)
    batch, sample_report = load_sample(settings, config)
    metadata = {**settings.identity(), 'sample_sha256': sample_report['sample_sha256']}
    sampler = {'episode_id': settings.episode_id, 'dataset_index': sample_report['sample_index'],
               'sample_sha256': sample_report['sample_sha256'], 'next_microbatch_cursor': 10,
               'microbatch_dataset_indices': [sample_report['sample_index']]*20}
    if resume is not None and not resume.is_file():
        raise FileNotFoundError(resume)
    settings.output_root.mkdir(parents=True, exist_ok=False)
    journal = settings.output_root/'progress.jsonl'
    deadline = time.monotonic() + settings.max_minutes*60
    def expired(*_):
        raise TimeoutError('Smoke time budget exceeded.')
    previous = None
    if hasattr(signal, 'SIGALRM'):
        if signal.getitimer(signal.ITIMER_REAL)[0]:
            raise RuntimeError('Existing process timer; use a separate process.')
        previous = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, settings.max_minutes*60)
    def record(event, **fields):
        with journal.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'event':event, **fields}, allow_nan=False)+'\n')
    def build(load):
        return _build_stack(config, base_checkpoint=settings.checkpoint, load_public_base=load,
            checkpoint_sha256=settings.checkpoint_sha256, arm=settings.arm, route_seed=settings.route_seed,
            torch=torch, method_config=settings.method_config)
    try:
        record('started', **metadata)
        stack = build(resume is None)
        state = TrainState()
        reports = []
        if resume is None:
            device_batch = stack['adapter'].move_to_device(batch, stack['strategy'].device)
            reports.append(_update(stack, device_batch, state, torch=torch, deadline=deadline))
            record('optimizer_step', **reports[-1])
            del device_batch
            resume = settings.output_root/'step1_full_state.pt'
            checkpoint.save_step1(resume, model=stack['model'], optimizer=stack['optimizer'],
                scheduler=stack['scheduler'], strategy=stack['strategy'], train_state=state,
                sampler_state=sampler, extra_generator_states={}, metadata=metadata,
                cuda_devices=(0,), max_bytes=64*1024**3)
            del stack
            gc.collect(); torch.cuda.empty_cache()
            stack = build(False)
        device_batch = stack['adapter'].move_to_device(batch, stack['strategy'].device)
        restored = checkpoint.load_step1(resume, model=stack['model'], optimizer=stack['optimizer'],
            scheduler=stack['scheduler'], strategy=stack['strategy'], expected_metadata=metadata, cuda_devices=(0,))
        state = restored['train_state']
        if (state.optimizer_step != 1 or state.global_step != 10 or state.next_batch_index != 10
                or restored['sampler_state'] != sampler or restored['extra_generator_states'] != {}):
            raise ValueError('Checkpoint is not the matching step1/sample/cursor boundary.')
        checkpoint.restore_rng(restored['rng_state'], cuda_devices=(0,))
        reports.append(_update(stack, device_batch, state, torch=torch, deadline=deadline))
        record('optimizer_step', **reports[-1])
        assert state.optimizer_step == 2 and state.next_batch_index == 20
        result = {'status':'two_update_smoke_completed', 'arm':settings.arm, 'step_reports':reports,
                  'restored_checkpoint':str(resume), 'sample':sample_report,
                  'optimizer_step':state.optimizer_step, 'microbatch_cursor':state.next_batch_index,
                  'control_evaluation':False, 'uninterrupted_numerical_parity_verified':False}
        (settings.output_root/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        return result
    except BaseException as exc:
        record('failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

def _parameter_manifest(model, *, expected_numel: int | None = None) -> tuple[list[dict[str, Any]], set[int]]:
    all_named = list(model.named_parameters(remove_duplicate=False))
    owner_ids = [id(parameter) for _, parameter in all_named]
    if len(owner_ids) != len(set(owner_ids)):
        raise ValueError("training_duplicate_parameter_owners")
    trainable = [(name, parameter) for name, parameter in all_named if parameter.requires_grad]
    manifest = [
        {"name": name, "numel": parameter.numel(), "dtype": str(parameter.dtype)}
        for name, parameter in trainable
    ]
    count = sum(item["numel"] for item in manifest)
    if expected_numel is not None and count != expected_numel:
        raise ValueError("training_trainable_parameter_count_mismatch")
    if any(parameter.dtype != __import__("torch").float32 for _, parameter in trainable):
        raise ValueError("training_parameters_must_remain_fp32")
    return manifest, {id(parameter) for _, parameter in trainable}

def _optimizer_state_dtypes(optimizer, torch) -> dict[str, Any]:
    result: dict[str, set[str]] = {}
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                result.setdefault(str(name), set()).add(str(value.dtype))
    for name in ("exp_avg", "exp_avg_sq"):
        if result.get(name) and result[name] != {"torch.float32"}:
            raise ValueError("training_adamw_moments_must_remain_fp32")
    if not optimizer.state:
        raise ValueError("training_adamw_state_missing_after_update")
    return {
        "tensor_dtypes": {name: sorted(dtypes) for name, dtypes in result.items()},
        "state_parameter_count": len(optimizer.state),
    }

def _build_stack(config, *, base_checkpoint: Path, load_public_base: bool,
                 checkpoint_sha256: str, arm: str, route_seed: int, torch,
                 method_config=None) -> dict[str, Any]:
    from open_wam.configs.enums import TrainerAccelerator, TrainerPrecision
    from open_wam.models.visual_tower.public_pretraining import load_public_video_checkpoint_into_tower
    from open_wam.models.visual_tower.tower import VisualTower
    from open_wam.pipelines.factory import build_variant_pipeline_from_config
    from open_wam.training.optim import build_optimizer, build_scheduler
    from open_wam.training.step_executor import LatentBatchAdapter, PipelineTrainStepExecutor
    from open_wam.training.strategies import SingleDeviceStrategy

    tower = VisualTower(
        config.backbone,
        action_dim=int(config.data.action_schema.action_dim),
        state_dim=int(config.data.action_schema.state_dim),
    )
    if load_public_base:
        load_public_video_checkpoint_into_tower(
            tower, base_checkpoint,
            expected_sha256=checkpoint_sha256,
        )
    pipeline = build_variant_pipeline_from_config(config, visual_tower=tower)
    cagrad_candidates = ()
    if method_config is None:
        from open_wam.models.policy_variants.dual_expert.variational_sharing import configure_variational_sharing
        sharing_audit = configure_variational_sharing(
            pipeline, arm=arm, expected_layers=30, route_seed=route_seed
        )
    elif method_config.legacy_v02:
        from open_wam.models.policy_variants.dual_expert.variational_sharing import configure_variational_sharing
        sharing_audit = configure_variational_sharing(
            pipeline, arm=method_config.legacy_arm, expected_layers=30, route_seed=route_seed
        )
    else:
        from .cagrad_training import cagrad_candidate_parameters, configure_native_trainability
        sharing_audit = configure_native_trainability(
            pipeline, expected_layers=int(config.backbone.num_layers)
        )
        if method_config.uses_vrfm:
            from open_wam.models.policy_variants.dual_expert.vrfm import configure_vrfm
            sharing_audit['vrfm'] = configure_vrfm(
                pipeline, latent_dim=method_config.latent_dim, kl_weight=method_config.kl_weight
            )
        if method_config.uses_cagrad:
            cagrad_candidates = cagrad_candidate_parameters(pipeline)
    raw_model = pipeline
    expected_numel = 2_600_806_456 if arm == 'variational_sharing' and method_config is None else None
    before_manifest, _ = _parameter_manifest(raw_model, expected_numel=expected_numel)
    strategy = SingleDeviceStrategy(
        accelerator=TrainerAccelerator.GPU, precision=TrainerPrecision.BF16
    )
    model = strategy.prepare_model(raw_model)
    after_manifest, trainable_ids = _parameter_manifest(
        strategy.unwrap_model(model), expected_numel=expected_numel
    )
    if before_manifest != after_manifest:
        raise ValueError("training_trainable_manifest_changed_after_device_move")
    optimizer = build_optimizer(model, config.training)
    optimizer_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    if len(optimizer_ids) != len(set(optimizer_ids)) or set(optimizer_ids) != trainable_ids:
        raise ValueError("training_optimizer_owner_set_mismatch")
    owners = {id(parameter): name for name, parameter in model.named_parameters()}
    optimizer_names = [[owners[id(parameter)] for parameter in group["params"]] for group in optimizer.param_groups]
    if optimizer_names != [[entry["name"] for entry in after_manifest]]:
        raise ValueError("training_optimizer_parameter_order_mismatch")
    scheduler = build_scheduler(optimizer, config.training)
    cagrad_accumulator = None
    if method_config is not None and method_config.uses_cagrad:
        from .cagrad_training import CAGradGradientAccumulator
        cagrad_accumulator = CAGradGradientAccumulator(
            cagrad_candidates, c=method_config.cagrad_c
        )
    adapter = LatentBatchAdapter()
    executor = PipelineTrainStepExecutor(
        pipeline=pipeline, batch_adapter=adapter, training_config=config.training
    )
    return {
        "model": model,
        "pipeline": pipeline,
        "strategy": strategy,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "adapter": adapter,
        "executor": executor,
        "parameter_manifest": after_manifest,
        "sharing_audit": sharing_audit,
        "cagrad_accumulator": cagrad_accumulator,
        "method_config": method_config,
        "cagrad_candidates": cagrad_candidates,
        "optimizer_names": optimizer_names,
    }
