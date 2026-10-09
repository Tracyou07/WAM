"""Native single-sample checks; no frontend encoding or model construction."""
from __future__ import annotations
from typing import Any


def load_sample(settings, config):
    import hashlib
    import json
    import torch
    from open_wam.data import build_train_val_latent_datasets, collate_latent_wam_samples
    from open_wam.models.visual_tower.prompt_cache import OfflinePromptCache
    if not settings.prompt_fingerprint:
        raise ValueError('Pin the encoder fingerprint from your verified cache generation before data checks/training.')
    train, _ = build_train_val_latent_datasets(config.data)
    virtual = getattr(train, '_virtual_index', [(i,0) for i in range(len(train.windows))])
    index = next((i for i,(w,_) in enumerate(virtual)
                  if int(train.windows[w].episode_index)==settings.episode_id), None)
    if index is None:
        raise ValueError('Selected episode has no complete paired latent window.')
    sample = train[index]
    if sample.condition_latents is None or sample.negative_text_context is None:
        raise ValueError('Independent condition and real empty-text cache are required.')
    if sample.video_latents.shape[1] != sample.condition_latents.shape[1]:
        raise ValueError('Video/condition time axes disagree.')
    for name, tensor in vars(sample).items():
        if isinstance(tensor,torch.Tensor) and not torch.isfinite(tensor).all():
            raise ValueError('Nonfinite sample tensor: '+name)
    for mask in (sample.action_mask,sample.state_mask):
        if mask is None or not ((mask==0)|(mask==1)).all() or mask.sum()<=0:
            raise ValueError('Nonempty binary supervision masks required.')
    if not sample.task_text:
        raise ValueError('Real task text is missing.')
    cache = OfflinePromptCache(settings.prompt_root,max_text_tokens=config.backbone.max_text_tokens,
        text_dim=config.backbone.text_dim,expected_encoder_fingerprint=settings.prompt_fingerprint)
    embeddings = cache.encode_prompts((sample.task_text,''),device=torch.device('cpu'),dtype=torch.float32)
    if any(float(row.abs().sum())==0 for row in embeddings):
        raise ValueError('Task/empty embeddings must come from real text encoding.')
    raw_audit = _audit_native_sample_raw_alignment(train,sample,config,torch)
    digest=hashlib.sha256(json.dumps(sample.metadata,sort_keys=True,default=str).encode())
    for name,tensor in sorted(vars(sample).items()):
        if isinstance(tensor,torch.Tensor):
            digest.update(name.encode()); digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    report={'episode_id':settings.episode_id,'sample_index':index,'sample_sha256':digest.hexdigest(),
            'video_shape':list(sample.video_latents.shape),'action_shape':list(sample.actions.shape),
            'raw_alignment':raw_audit,'heldout_evaluation':False}
    return collate_latent_wam_samples([sample]),report

def _audit_native_sample_raw_alignment(train_dataset, sample, config, torch) -> dict[str, Any]:
    from open_wam.data.row_action_targets import resolve_row_key
    from open_wam.data.action_normalization import normalize_action_targets

    metadata = sample.metadata
    window_index = int(metadata["trajectory_window_index"])
    window = train_dataset.windows[window_index]
    source = train_dataset._load_sample_source(window, include_condition_latents=True)
    rows = source.rows
    def row_value(frame_index: int, key: str):
        row = rows[frame_index]
        return row[resolve_row_key(row, key)]

    observed_ids = [int(value) for value in metadata["observed_frame_ids"]]
    if not observed_ids or any(frame < 0 or frame >= len(rows) for frame in observed_ids):
        raise ValueError("native_observed_frames_outside_raw_episode_rows")

    alignment = metadata.get("lingbot_window_action_alignment", {})
    actions_per_latent = int(
        config.data.action_schema.action_horizon // config.data.num_frames
    )
    leading_steps = int(alignment.get("leading_zero_action_steps", -1))
    if (
        int(alignment.get("prefix_actions", -1)) != actions_per_latent
        or int(alignment.get("required_action_num", -1)) != int(sample.actions.shape[0])
        or int(alignment.get("latent_num_frames", -1)) != int(sample.video_latents.shape[1])
        or leading_steps < 0
        or leading_steps > int(sample.actions.shape[0])
    ):
        raise ValueError("native_action_prefix_contract_does_not_match_sample")

    sample_start = int(metadata["sample_start_frame"])
    sample_end = int(metadata["sample_end_frame"])
    action_start = sample_start + int(alignment["action_start_offset"])
    if action_start != observed_ids[0]:
        raise ValueError("native_action_alignment_frame_origin_mismatch")
    if leading_steps:
        leading_target = sample.actions[:leading_steps]
        expected_leading_mask = float(alignment["leading_zero_action_mask"])
        if (
            float(leading_target.abs().max()) > 1e-7
            or not torch.all(sample.action_mask[:leading_steps] == expected_leading_mask).item()
        ):
            raise ValueError("native_action_prefix_values_or_mask_mismatch")

    source_capacity = int(sample.actions.shape[0]) - leading_steps
    raw_action_count = max(0, min(source_capacity, sample_end - action_start, len(rows) - action_start))
    action_key = config.data.action_target.source_key
    raw_action_rows = torch.stack(
        [
            torch.as_tensor(
                row_value(frame_index, action_key), dtype=torch.float32
            )
            for frame_index in range(action_start, action_start + raw_action_count)
        ],
        dim=0,
    ) if raw_action_count else torch.zeros((0, config.data.action_schema.action_dim))
    expected_actions = normalize_action_targets(
        raw_action_rows, normalization=config.data.action_target.normalization
    )
    actual_actions = sample.actions[leading_steps : leading_steps + raw_action_count].float()
    action_error = (
        float((actual_actions - expected_actions).abs().max())
        if raw_action_count
        else 0.0
    )
    if action_error > 1e-5:
        raise ValueError("native_action_targets_do_not_match_aligned_raw_rows")
    if raw_action_count and not torch.all(
        sample.action_mask[leading_steps : leading_steps + raw_action_count] == 1
    ).item():
        raise ValueError("native_raw_action_rows_not_marked_valid")

    state_key = config.data.action_target.pose_source_key
    state_anchor = int(metadata["state_anchor_frame"])
    raw_state = torch.as_tensor(
        row_value(state_anchor, state_key), dtype=torch.float32
    )
    actual_state = sample.state.reshape(-1, sample.state.shape[-1])[-1].float()
    state_error = float((actual_state - raw_state).abs().max())
    if state_error > 1e-5:
        raise ValueError("native_state_target_does_not_match_raw_anchor_row")

    context_frame = int(metadata["proprio_context_frame_index"])
    context_state = sample.proprio_context_state.reshape(
        -1, sample.proprio_context_state.shape[-1]
    )[0].float()
    expected_context_state = torch.as_tensor(
        row_value(context_frame, state_key), dtype=torch.float32
    )
    context_state_error = float((context_state - expected_context_state).abs().max())
    if context_state_error > 1e-5:
        raise ValueError("native_proprio_context_state_does_not_match_raw_row")

    context_frames = sample.proprio_context_frames.float()
    if context_frames.shape[0] != len(observed_ids):
        raise ValueError("native_proprio_context_frame_count_mismatch")
    expected_context_frames = torch.stack(
        [
            torch.as_tensor(row_value(index, state_key), dtype=torch.float32)
            for index in observed_ids
        ],
        dim=0,
    )
    context_frames_error = float((context_frames - expected_context_frames).abs().max())
    if context_frames_error > 1e-5:
        raise ValueError("native_proprio_context_frames_do_not_match_raw_rows")

    boundary_offsets = sorted(set((0, max(0, raw_action_count - 1))) )
    action_boundary_rows = [
        {
            "sample_action_row": leading_steps + offset,
            "raw_episode_row": action_start + offset,
            "target_values": [float(value) for value in actual_actions[offset].tolist()],
        }
        for offset in boundary_offsets
        if raw_action_count
    ]
    return {
        "status": "passed_value_by_value_against_raw_episode_rows",
        "raw_episode_row_count": len(rows),
        "action_source_row_start_end_inclusive": [
            action_start,
            action_start + raw_action_count - 1 if raw_action_count else None,
        ],
        "raw_action_rows_compared": raw_action_count,
        "action_target_max_abs_error_after_native_normalization": action_error,
        "native_leading_zero_prefix_rows": leading_steps,
        "native_prefix_mask_value": float(alignment["leading_zero_action_mask"]),
        "action_boundary_rows": action_boundary_rows,
        "state_anchor_raw_row": state_anchor,
        "state_anchor_max_abs_error": state_error,
        "proprio_context_state_raw_row": context_frame,
        "proprio_context_state_max_abs_error": context_state_error,
        "proprio_context_frame_ids_first_last": [observed_ids[0], observed_ids[-1]],
        "proprio_context_frames_compared": len(observed_ids),
        "proprio_context_frames_max_abs_error": context_frames_error,
    }
