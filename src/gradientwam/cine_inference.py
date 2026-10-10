"""Strict model-only checkpoint loader for the independent Cine entry."""
from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from open_wam.pipelines.variant_pipeline import VariantPipeline


def load_cine_policy_for_inference(
    settings,
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
) -> VariantPipeline:
    """Build the configured Cine method and strictly load model-only weights."""
    from open_wam.models.policy_variants.dual_expert.vrfm import configure_vrfm
    from open_wam.pipelines import build_variant_pipeline_from_config
    from open_wam.runtime.checkpoints import (
        CheckpointCompatibilityPolicy,
        load_pipeline_checkpoint,
        resolve_checkpoint_file,
    )

    checkpoint_file = resolve_checkpoint_file(checkpoint)
    if checkpoint_file.name == "full_training_state.pt":
        raise ValueError("Cine inference requires a model-only checkpoint.")
    config = settings.native_config()
    config = replace(
        config,
        backbone=replace(
            config.backbone,
            load_reference_core_weights=False,
            load_wan_vae_frontend=False,
            load_text_conditioning=False,
        ),
    )
    pipeline = build_variant_pipeline_from_config(config)
    if settings.method_config.uses_vrfm:
        configure_vrfm(
            pipeline,
            latent_dim=settings.method_config.latent_dim,
            kl_weight=settings.method_config.kl_weight,
        )
    load_pipeline_checkpoint(
        pipeline,
        checkpoint_file,
        map_location="cpu",
        compatibility=CheckpointCompatibilityPolicy.STRICT,
        require_model_only=True,
    )
    return pipeline.to(device=device).eval().requires_grad_(False)



def denormalize_cine_actions(actions, *, settings, run_identity: dict):
    """Restore train-normalized action values after validating the run/cache identity.

    Pass the parsed `run_identity.json` written beside the training checkpoints.
    For Gaussian targets this returns the original seven action units; for
    `none` it returns the input unchanged.
    """
    import hashlib
    import json
    import torch

    from open_wam.configs import ActionNormalizationConfig
    from open_wam.configs.enums import ActionNormalizationMode
    from open_wam.data.cine_v3 import contained_path
    from open_wam.data.cine_v3_latent import build_cine_latent_train_val_datasets
    from open_wam.data.action_normalization import denormalize_action_targets

    if not isinstance(actions, torch.Tensor) or actions.ndim < 1 or actions.shape[-1] != 7:
        raise ValueError("Cine actions must be a tensor whose final dimension is seven.")
    if not actions.is_floating_point() or not torch.isfinite(actions).all():
        raise ValueError("Cine action values must be finite floating point.")
    native = settings.native_config()
    manifest_path = contained_path(
        settings.latent_root,
        native.data.adapter_options.get("cache_manifest", "manifest.json"),
    )
    raw = manifest_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if not isinstance(run_identity, dict) or run_identity.get("cine_manifest_sha256") != digest:
        raise ValueError("Model run identity does not match the current Cine manifest.")
    manifest = json.loads(raw.decode("utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError("Cine action conversion requires a complete manifest.")
    train, validation = build_cine_latent_train_val_datasets(native.data)
    manifest = train.manifest
    expected_identity = {
        "cine_source_identities": manifest["sources"],
        "cine_selection": manifest["selection"],
        "action_semantics": manifest["action_semantics"],
        "action_normalization": manifest["action_normalization"],
    }
    if any(run_identity.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("Model run identity and Cine action/cache identity differ.")
    if (
        manifest["action_semantics"] != settings.action_semantics
        or manifest["action_normalization"] != settings.action_normalization
    ):
        raise ValueError("Cine manifest semantics/normalization differ from the selected config.")
    if actions.shape[-1] != int(native.data.action_schema.action_dim):
        raise ValueError("Cine action dimension differs from the native config.")
    normalization_mode = ActionNormalizationMode(settings.action_normalization)
    if normalization_mode is ActionNormalizationMode.NONE:
        return actions
    if normalization_mode is not ActionNormalizationMode.GAUSSIAN:
        raise ValueError("Only train-stat Gaussian Cine action normalization can be inverted.")
    statistics = manifest["action_statistics"]
    train_ids = sorted({int(item["episode_index"]) for item in manifest["samples"]["train"]})
    if (
        statistics.get("source") != "train"
        or statistics.get("root") != str(settings.train_root)
        or statistics.get("episode_indices") != train_ids
    ):
        raise ValueError("Cine Gaussian statistics are not bound to the selected train source.")
    normalization = ActionNormalizationConfig(
        mode=ActionNormalizationMode.GAUSSIAN,
        mean=tuple(float(value) for value in statistics["mean"]),
        std=tuple(float(value) for value in statistics["std"]),
    )
    return denormalize_action_targets(actions, normalization=normalization)


__all__ = ["denormalize_cine_actions", "load_cine_policy_for_inference"]
