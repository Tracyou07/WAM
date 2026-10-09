"""Load public video-only OpenWAM pretraining before native policy attachment."""
from __future__ import annotations

import hashlib
import mmap
from pathlib import Path

import torch

from .tower import VisualTower


# The public video checkpoint serialized dummy action/state adapters. They are
# not action-policy pretraining, even when their shapes happen to match.
_PUBLIC_SOURCE_AUXILIARY_KEYS = frozenset({
    "core.action_embedder.bias",
    "core.action_embedder.weight",
    "core.action_proj_out.bias",
    "core.action_proj_out.weight",
    "core.action_text_proj.linear_1.bias",
    "core.action_text_proj.linear_1.weight",
    "core.action_text_proj.linear_2.bias",
    "core.action_text_proj.linear_2.weight",
    "core.action_time_conditioner.time_embedder.linear_1.bias",
    "core.action_time_conditioner.time_embedder.linear_1.weight",
    "core.action_time_conditioner.time_embedder.linear_2.bias",
    "core.action_time_conditioner.time_embedder.linear_2.weight",
    "core.action_time_conditioner.time_proj.bias",
    "core.action_time_conditioner.time_proj.weight",
    "core.runtime_stream_adapters.action_register_adapter.0.bias",
    "core.runtime_stream_adapters.action_register_adapter.0.weight",
    "core.runtime_stream_adapters.action_register_adapter.2.bias",
    "core.runtime_stream_adapters.action_register_adapter.2.weight",
    "core.runtime_stream_adapters.role_embedding.weight",
    "core.runtime_stream_adapters.state_register_adapter.0.bias",
    "core.runtime_stream_adapters.state_register_adapter.0.weight",
    "core.runtime_stream_adapters.state_register_adapter.2.bias",
    "core.runtime_stream_adapters.state_register_adapter.2.weight",
})

# Target initialization is a separate schema. These conditioning modules are
# allowed in the native target, but never in the fixed public source format.
_NATIVE_TARGET_INITIALIZED_KEYS = _PUBLIC_SOURCE_AUXILIARY_KEYS | frozenset({
    "core.proprio_context_encoder.proj.bias",
    "core.proprio_context_encoder.proj.weight",
    "core.proprio_hidden_context_encoder.proj.bias",
    "core.proprio_hidden_context_encoder.proj.weight",
    "core.generalist_mode_context_encoder.embedding.weight",
})


def load_public_video_checkpoint_into_tower(
    visual_tower: VisualTower,
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict:
    """Assign verified video tensors into a CPU tower before attach/optimizer.

    Only the pure ``model_state_dict`` public pretraining format is accepted.
    Required video/frontend tensors have exact key, shape and dtype coverage;
    native action/state auxiliary initialization is retained and reported.
    """
    blocks = getattr(visual_tower.core, "blocks", ())
    if not blocks or len(blocks) != visual_tower.config.num_layers:
        raise ValueError("Load public video pretraining before native policy attach")
    target = visual_tower.state_dict()
    if any(tensor.device.type != "cpu" for tensor in target.values()):
        raise ValueError("Public video pretraining must load into a real CPU tower")
    path = Path(checkpoint_path)
    actual_sha256 = None
    if expected_sha256 is not None:
        with path.open("rb") as handle:
            actual_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError("Public video checkpoint SHA256 mismatch")
    # Optimizer writes must never propagate into the input checkpoint file.
    with torch.serialization.set_default_mmap_options(mmap.MAP_PRIVATE):
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict) or set(payload) != {"model_state_dict"}:
        raise ValueError("Expected pure public video model_state_dict checkpoint")
    source = payload["model_state_dict"]
    if not isinstance(source, dict) or not source:
        raise ValueError("Public video model_state_dict must be nonempty")
    if any(not isinstance(key, str) or not key.startswith("visual_tower.") or not isinstance(value, torch.Tensor)
           for key, value in source.items()):
        raise ValueError("Unexpected non-video key or non-tensor in public checkpoint")
    source = {key.removeprefix("visual_tower."): value for key, value in source.items()}
    required = target.keys() - _NATIVE_TARGET_INITIALIZED_KEYS
    missing_source_auxiliary = sorted(_PUBLIC_SOURCE_AUXILIARY_KEYS - source.keys())
    if missing_source_auxiliary:
        raise ValueError(f"Fixed public auxiliary keys missing: {missing_source_auxiliary}")
    missing = sorted(required - source.keys())
    unexpected = sorted(source.keys() - required - _PUBLIC_SOURCE_AUXILIARY_KEYS)
    mismatched = sorted(key for key in required & source.keys()
        if source[key].shape != target[key].shape or source[key].dtype != target[key].dtype)
    if missing or unexpected or mismatched:
        raise ValueError(f"Public video coverage mismatch: missing={missing}, "
                         f"unexpected={unexpected}, shape_or_dtype={mismatched}")
    video_state = {key: source[key] for key in required}
    incompatible = visual_tower.load_state_dict(video_state, strict=False, assign=True)
    if (incompatible.unexpected_keys
        or set(incompatible.missing_keys) != target.keys() & _NATIVE_TARGET_INITIALIZED_KEYS):
        raise ValueError("Native public video load violated declared auxiliary boundary")
    loaded = visual_tower.state_dict()
    if any(not torch.equal(loaded[key], source[key]) for key in required):
        raise ValueError("Native video tensors differ after assignment")
    return {
        "checkpoint_path": str(path), "checkpoint_bytes": path.stat().st_size,
        "actual_sha256": actual_sha256, "mmap_mode": "MAP_PRIVATE",
        "source_tensor_keys": len(source), "loaded_video_keys": sorted(required),
        "loaded_video_numel": sum(source[key].numel() for key in required),
        "missing_required_video_keys": missing, "unexpected_video_keys": unexpected,
        "shape_or_dtype_mismatches": mismatched, "video_values_exactly_equal": True,
        "source_auxiliary_keys_ignored": sorted(source.keys() & _PUBLIC_SOURCE_AUXILIARY_KEYS),
        "missing_public_auxiliary_keys": missing_source_auxiliary,
        "native_auxiliary_keys_initialized": sorted(incompatible.missing_keys),
        "action_policy_pretrained_keys": 0,
        "scope": "video-only warm start; not action-policy or optimizer resume",
    }
