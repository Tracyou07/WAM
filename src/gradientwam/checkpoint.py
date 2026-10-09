"""Single-device step1 full checkpoint; caller restores sampler/extra RNG first.

Portable copy of the reviewed round02 full-state checkpoint helper.
Only load checkpoints from a trusted run with matching metadata.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from open_wam.artifacts import load_tensor_artifact
from open_wam.training.checkpoint_storage import _atomic_torch_save
from open_wam.training.optim import _normalize_optimizer_state_dtypes
from open_wam.training.state import TrainState

__all__ = ["save_step1", "load_step1", "restore_rng"]
_FORMAT = "round02-single-device-raw-adamw-step1-v1"


def _param_names(model, optimizer) -> list[list[str]]:
    owners = {id(p): name for name, p in model.named_parameters()}
    groups = []
    ids = []
    for group in optimizer.param_groups:
        group_ids = [id(p) for p in group["params"]]
        if any(i not in owners for i in group_ids):
            raise ValueError("optimizer_parameter_not_owned_by_model")
        groups.append([owners[i] for i in group_ids])
        ids.extend(group_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate_optimizer_parameter")
    if set(ids) != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError("optimizer_trainable_owner_set_mismatch")
    return groups


def _safe(value: Any) -> None:
    if isinstance(value, torch.Tensor) or type(value) in (type(None), bool, int, float, str, bytes):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) not in (str, int):
                raise TypeError("checkpoint_nonprimitive_key")
            _safe(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _safe(item)
        return
    raise TypeError("checkpoint_nonprimitive_value: " + type(value).__name__)


def _storage_bytes(value: Any, seen: set) -> int:
    if isinstance(value, torch.Tensor):
        storage = value.untyped_storage()
        key = (str(value.device), storage.data_ptr())
        if key in seen:
            return 0
        seen.add(key)
        return storage.nbytes()
    if isinstance(value, Mapping):
        return sum(_storage_bytes(v, seen) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_storage_bytes(v, seen) for v in value)
    return 0


def _cuda_scope(model, cuda_devices) -> tuple[int, ...]:
    devices = tuple(cuda_devices)
    if len(devices) > 1 or any(type(i) is not int or i < 0 for i in devices):
        raise ValueError("single_leased_cuda_device_required")
    actual = {p.device.index for p in (*model.parameters(), *model.buffers()) if p.device.type == "cuda"}
    if actual != set(devices):
        raise ValueError("cuda_rng_scope_mismatch")
    return devices


def _capture_rng(cuda_devices) -> dict:
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])),
        "cpu": torch.get_rng_state(),
        "cuda_devices": list(cuda_devices),
        "cuda": {i: torch.cuda.get_rng_state(i) for i in cuda_devices},
    }


def save_step1(
    path: str | Path, *, model, optimizer, scheduler, strategy, train_state: TrainState,
    sampler_state: Any, extra_generator_states: Mapping[str, Any], metadata: Mapping,
    cuda_devices=(), max_bytes: int = 64 << 30,
) -> dict:
    """Save once at successful step1/zero-grad boundary; no automatic copies."""
    path = Path(path).expanduser()
    if path.exists() or path.is_symlink():
        raise FileExistsError("checkpoint_path_exists: " + str(path))
    if any(path.parent.glob("." + path.name + ".tmp.*")):
        raise FileExistsError("checkpoint_temp_exists: " + str(path))
    if not path.parent.is_dir():
        raise FileNotFoundError("checkpoint_parent_missing: " + str(path.parent))
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("checkpoint_cap_must_be_positive")
    if train_state.optimizer_step != 1 or train_state.global_step < 1:
        raise ValueError("checkpoint_requires_successful_step1")
    if any(p.grad is not None for g in optimizer.param_groups for p in g["params"]):
        raise ValueError("checkpoint_requires_zero_grad_set_to_none")
    devices = _cuda_scope(model, cuda_devices)
    names = _param_names(model, optimizer)
    if not isinstance(extra_generator_states, Mapping) or any(type(k) is not str for k in extra_generator_states):
        raise TypeError("extra_generator_states_requires_named_mapping")
    # JSON normalizes metadata to ordinary containers accepted by weights-only.
    metadata = json.loads(json.dumps(dict(metadata), sort_keys=True, allow_nan=False))
    payload = {
        "format": _FORMAT,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "strategy_state_dict": strategy.state_dict(),
        "train_state": train_state.state_dict(),
        "optimizer_param_names": names,
        "sampler_state": sampler_state,
        "sampler_state_saved": sampler_state is not None,
        "extra_generator_states": dict(extra_generator_states),
        "metadata": metadata,
        "rng_state": _capture_rng(devices),
    }
    _safe(payload)
    if _storage_bytes(payload, set()) > max_bytes:
        raise ValueError("checkpoint_tensor_payload_exceeds_cap")
    # One sole launcher: exclusive empty target reserves this name for this
    # attempt; native atomic writer only replaces our own reservation.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    reservation = os.fstat(descriptor)
    os.close(descriptor)
    try:
        # Recheck after reservation; never let native writer unlink stale evidence.
        if any(path.parent.glob("." + path.name + ".tmp.*")):
            raise FileExistsError("checkpoint_temp_exists: " + str(path))
        _atomic_torch_save(payload, path)
    except BaseException:
        current = path.stat()
        if current.st_ino == reservation.st_ino and current.st_size == 0:
            path.unlink()  # only this attempt's empty reservation
        raise
    size = path.stat().st_size
    if size > max_bytes:
        path.unlink()  # only this attempt's newly written oversize output
        raise ValueError("checkpoint_serialized_file_exceeds_cap")
    return {"path": str(path.resolve()), "bytes": size,
            "sampler_state_saved": sampler_state is not None,
            "extra_generator_names": list(extra_generator_states), "optimizer_param_names": names}


def load_step1(
    path: str | Path, *, model, optimizer, scheduler, strategy,
    expected_metadata: Mapping, cuda_devices=(),
) -> dict:
    """Strict full restore; return caller states. Global RNG is NOT restored."""
    devices = _cuda_scope(model, cuda_devices)
    payload = load_tensor_artifact(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format") != _FORMAT:
        raise ValueError("checkpoint_format_mismatch")
    _safe(payload)
    metadata = json.loads(json.dumps(dict(expected_metadata), sort_keys=True, allow_nan=False))
    if payload["metadata"] != metadata:
        raise ValueError("checkpoint_metadata_mismatch")
    if payload["optimizer_param_names"] != _param_names(model, optimizer):
        raise ValueError("optimizer_param_names_mismatch")
    if tuple(payload["rng_state"]["cuda_devices"]) != devices:
        raise ValueError("cuda_rng_scope_mismatch")
    required_cursor = {"global_step", "optimizer_step", "epoch_index", "next_batch_index", "seen_batches"}
    if not required_cursor <= payload["train_state"].keys():
        raise ValueError("checkpoint_trainstate_cursor_missing")
    state = TrainState.from_state_dict(payload["train_state"])
    if state.optimizer_step != 1 or state.global_step < 1:
        raise ValueError("checkpoint_requires_successful_step1")
    # Preflight key sets before PyTorch can partially mutate a model on failure.
    saved_model = payload["model_state_dict"]
    if set(saved_model) != set(model.state_dict()):
        raise ValueError("model_state_keys_mismatch")
    for name, parameter in model.named_parameters():
        value = saved_model[name]
        if value.shape != parameter.shape or value.dtype != parameter.dtype:
            raise ValueError("model_parameter_shape_dtype_mismatch: " + name)
    model.load_state_dict(saved_model, strict=True)  # preserves route controller hooks
    optimizer.load_state_dict(payload["optimizer_state_dict"])  # raw IDs, same owner order
    _normalize_optimizer_state_dtypes(optimizer)
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    strategy.load_state_dict(payload["strategy_state_dict"])
    return {"train_state": state, "sampler_state": payload["sampler_state"],
            "sampler_state_saved": payload["sampler_state_saved"],
            "extra_generator_states": payload["extra_generator_states"],
            "rng_state": payload["rng_state"], "metadata": payload["metadata"]}


def restore_rng(rng_state: Mapping, *, cuda_devices=()) -> None:
    """Call LAST, after caller sampler/generator/cursor and all object setup."""
    devices = tuple(cuda_devices)
    if devices != tuple(rng_state["cuda_devices"]) or set(rng_state["cuda"]) != set(devices):
        raise ValueError("cuda_rng_scope_mismatch")
    if len(devices) > 1 or any(type(i) is not int or i < 0 for i in devices):
        raise ValueError("single_leased_cuda_device_required")
    state = rng_state["numpy"]
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32),
                         int(state[2]), int(state[3]), float(state[4])))
    random.setstate(rng_state["python"])
    torch.set_rng_state(rng_state["cpu"])
    for device in devices:
        torch.cuda.set_rng_state(rng_state["cuda"][device], device)
