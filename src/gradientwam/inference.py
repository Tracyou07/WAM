"""Strict model-only inference loading for the four GradientWAM methods."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from .settings import Settings, load_settings

if TYPE_CHECKING:
    import torch
    from open_wam.pipelines.variant_pipeline import VariantPipeline


def load_policy_for_inference(
    settings: Settings | str | Path,
    checkpoint: str | Path,
    device: str | torch.device = 'cpu',
) -> VariantPipeline:
    """Assemble the declared method, strictly restore model tensors, return eval.

    Requires a raw tensor state dict or a one-key model_state_dict/state_dict
    wrapper. No optimizer, training state or resume identity is restored. Base
    weights/frontends are not loaded: provide precomputed latents and text context
    to the returned native pipeline, or configure frontends separately.
    """
    from open_wam.models.policy_variants.dual_expert.vrfm import configure_vrfm
    from open_wam.pipelines import build_variant_pipeline_from_config
    from open_wam.runtime.checkpoints import (
        CheckpointCompatibilityPolicy, load_pipeline_checkpoint, resolve_checkpoint_file,
    )

    values = load_settings(Path(settings)) if isinstance(settings, (str, Path)) else settings
    if not isinstance(values, Settings):
        raise TypeError('settings must be Settings or a settings YAML path')
    if values.method_config.legacy_v02:
        raise ValueError('Inference loader supports only the four new methods; legacy routes are rejected')
    checkpoint_file = resolve_checkpoint_file(checkpoint)
    if checkpoint_file.name == 'full_training_state.pt':
        raise ValueError('Inference requires a model-only checkpoint, not full training state')
    config = values.native_config()
    config = replace(config, backbone=replace(config.backbone,
        load_reference_core_weights=False, load_wan_vae_frontend=False, load_text_conditioning=False))
    pipeline = build_variant_pipeline_from_config(config)
    if values.method_config.uses_vrfm:
        configure_vrfm(pipeline, latent_dim=values.method_config.latent_dim,
                       kl_weight=values.method_config.kl_weight)
    load_pipeline_checkpoint(pipeline, checkpoint_file, map_location='cpu',
        compatibility=CheckpointCompatibilityPolicy.STRICT, require_model_only=True)
    return pipeline.to(device=device).eval().requires_grad_(False)


__all__ = ['load_policy_for_inference']
