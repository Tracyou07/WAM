"""Explicit Cine v3 selection and native Wan/UMT5 cache preparation."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from open_wam.artifacts.files import atomic_json, sha256_file
from open_wam.configs import DataConfig
from open_wam.data.cine_v3 import COLOR_CAMERA, CineV3Repository, contained_path, fit_cine_action_statistics
from open_wam.data.latent_temporal import raw_window_frames_for_latents
from .settings import _direct_path


def _hash_mapping(values: dict) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def _load_frontends(frontend_root: Path, tokenizer_root: Path, device: torch.device):
    """Load only local encoders, never the video/action transformer."""
    from open_wam.cli.encode_prompt_cache import _encoder_files_sha256
    from open_wam.configs.backbone import SharedVideoTransformerConfig
    from open_wam.models.visual_tower.reference_assets import LingbotReferenceAssets
    from open_wam.models.visual_tower.vae_encoding import load_vae

    for directory in (frontend_root/'vae', frontend_root/'text_encoder', tokenizer_root):
        if not directory.is_dir():
            raise FileNotFoundError(f'Missing local frontend component: {directory}')
    backbone = SharedVideoTransformerConfig(
        pretrained_model_name_or_path=str(frontend_root),
        tokenizer_subdir=str(tokenizer_root), load_text_conditioning=True,
        load_wan_vae_frontend=False, load_reference_core_weights=False,
    )
    text_hashes = _encoder_files_sha256(backbone)
    vae_hashes = {p.name: sha256_file(p) for p in sorted((frontend_root/'vae').iterdir()) if p.is_file()}
    if not text_hashes or not vae_hashes:
        raise ValueError('Encoder components must contain local config and weight files')
    text_assets = LingbotReferenceAssets.maybe_load(backbone)
    if not text_assets.has_text_encoder:
        raise RuntimeError('Native UMT5 assets did not load')
    dtype = torch.float32 if device.type == 'cpu' else torch.bfloat16
    vae = load_vae(str(frontend_root/'vae'), device, dtype)
    encoding = {
        'vae_identity': _hash_mapping(vae_hashes),
        'text_encoder_identity': _hash_mapping(text_hashes),
        'vae_files_sha256': vae_hashes,
        'text_encoder_files_sha256': text_hashes,
        'contract': 'cine-wan-normalized-umt5-center-crop-v1',
        'vae_dtype': str(dtype), 'text_storage_dtype': 'torch.bfloat16',
        'max_text_tokens': backbone.max_text_tokens,
    }
    return vae, text_assets, encoding


def _selection(repo: CineV3Repository, episode_limit, window_limit, raw_frames: int, stride: int):
    records = repo.episode_records
    if episode_limit is not None:
        if episode_limit > len(records):
            raise ValueError('Requested episode limit exceeds the selected source root')
        records = records[:episode_limit]
    if not records:
        raise ValueError('Cine source has no selected episodes')
    samples, starts = [], {}
    for record in records:
        episode = record['episode_index']
        window_starts = range(1, record['length'] - raw_frames + 1, stride)
        values = list(window_starts if window_limit is None else window_starts[:window_limit])
        if not values:
            raise ValueError(f'Episode {episode} has no complete window with an observed preceding frame')
        starts[str(episode)] = values
        samples.extend({'episode_index': episode, 'raw_start': start} for start in values)
    return samples, {'episode_indices': [r['episode_index'] for r in records],
                     'window_count': len(samples), 'raw_starts': starts,
                     'available_episode_count': len(repo.episode_records)}


def prepare_cine(
    data_config: DataConfig, *, frontend_root: Path, tokenizer_root: Path,
    device: str = 'cpu', execute: bool = False,
    train_episode_limit: int | None = None, train_windows_per_episode: int | None = None,
    val_episode_limit: int | None = None, val_windows_per_episode: int | None = None,
    all_windows: bool = False,
) -> dict:
    """Plan by default; execute an explicit subset or the complete declared grid."""
    from open_wam.configs.enums import CineActionSemantics

    limits = (train_episode_limit, train_windows_per_episode, val_episode_limit, val_windows_per_episode)
    if type(all_windows) is not bool or (all_windows and any(x is not None for x in limits)):
        raise ValueError('Choose all-windows or four explicit positive limits, not both')
    if not all_windows and any(type(x) is not int or x <= 0 for x in limits):
        raise ValueError('Preparation requires all four positive episode/window limits or all-windows')
    if not data_config.local_root or not data_config.val_local_root or not data_config.latent_root:
        raise ValueError('Explicit train, validation and prepared roots are required')
    train_root, val_root, cache_root, frontend_root, tokenizer_root = [
        _direct_path(str(p)) for p in (data_config.local_root, data_config.val_local_root,
                                     data_config.latent_root, frontend_root, tokenizer_root)]
    if any(a == b or a in b.parents or b in a.parents for a, b in (
        (train_root, val_root), (cache_root, train_root), (cache_root, val_root),
        (cache_root, frontend_root), (cache_root, tokenizer_root))):
        raise ValueError('Derived Cine cache must be separate from raw inputs and model assets')
    if execute and cache_root.exists():
        raise FileExistsError('Cine preparation requires a fresh directory; existing caches are never overwritten')
    target = contained_path(cache_root, data_config.adapter_options.get('cache_manifest', 'manifest.json'))
    if target in (cache_root/'manifest.pending.json', cache_root/'preparation_plan.json'):
        raise ValueError('The selected cache manifest name is reserved for preparation state')
    if data_config.camera_names != (COLOR_CAMERA,) or data_config.latent_camera_names != (COLOR_CAMERA,):
        raise ValueError('Cine preparation requires the single real color camera')
    if (data_config.frame_stride != 1 or data_config.num_frames != 9
            or data_config.action_schema.action_horizon != 36
            or data_config.action_schema.action_dim != 7 or data_config.action_schema.state_dim != 7):
        raise ValueError('Cine preparation requires 33 consecutive raw frames, nine latents and raw7 supervision')
    stride = data_config.sample_stride
    if type(stride) is not int or stride <= 0:
        raise ValueError('data.sample_stride must be a positive raw-window start stride')
    if min(data_config.canonical_height, data_config.canonical_width) <= 0:
        raise ValueError('Cine image dimensions must be positive')
    semantics = CineActionSemantics(data_config.adapter_options.get('action_semantics', 'raw_joint_command'))
    train, val = CineV3Repository(train_root), CineV3Repository(val_root)
    raw_frames = raw_window_frames_for_latents(data_config.num_frames)
    train_samples, train_selection = _selection(train, train_episode_limit, train_windows_per_episode, raw_frames, stride)
    val_samples, val_selection = _selection(val, val_episode_limit, val_windows_per_episode, raw_frames, stride)
    selection = {'scope': 'all_windows' if all_windows else 'subset', 'window_stride': stride,
                 'train': train_selection, 'validation': val_selection}
    result = {'status': 'plan_only', 'cache_root': str(cache_root), 'selection': selection,
              'raw_window_frames': raw_frames, 'latent_frames': data_config.num_frames,
              'source_fps': 30, 'policy_model_constructed': False,
              'action_semantics': semantics.value, 'action_normalization': data_config.adapter_options.get('action_normalization','none')}
    if not execute:
        return result

    from open_wam.data.cine_v3 import validate_cine_split_sources
    from open_wam.data.cine_v3_latent import SCHEMA_VERSION, build_cine_latent_train_val_datasets
    from open_wam.data.preparation.frames import fit_frames
    from open_wam.models.visual_tower.vae_encoding import encode_clip

    validate_cine_split_sources(train, val, train_selection['episode_indices'], val_selection['episode_indices'])
    compute_device = torch.device(device)
    if compute_device.type not in ('cpu', 'cuda'):
        raise ValueError('Preparation supports an explicitly chosen CPU or CUDA device')
    # Fail before model loading if the selected frame/task records are invalid.
    tasks = {}
    for split, repo, entries in (('train', train, train_samples), ('validation', val, val_samples)):
        for entry in entries:
            rows = repo.read_rows(entry['episode_index'], entry['raw_start']-1, entry['raw_start']+raw_frames)
            ids = {row['task_index'] for row in rows}
            if len(ids) != 1:
                raise ValueError('A Cine preparation window cannot span multiple task prompts')
            tasks[(split, entry['episode_index'], entry['raw_start'])] = repo.tasks[ids.pop()]
    statistics = fit_cine_action_statistics(train, train_selection['episode_indices'])
    cache_root.mkdir(parents=True, exist_ok=False)
    atomic_json(cache_root/'preparation_plan.json', result)
    vae, text_assets, encoding = _load_frontends(frontend_root, tokenizer_root, compute_device)
    prompt_dir = cache_root/'prompt_cache/embeddings'
    prompt_dir.mkdir(parents=True)
    prompts = sorted(set(tasks.values()) | {''})
    prompt_paths = {}
    for prompt in prompts:
        encoded = text_assets.encode_prompts([prompt], device=compute_device, dtype=torch.bfloat16)
        if encoded is None or encoded.ndim != 3 or encoded.shape[0] != 1 or not torch.isfinite(encoded).all():
            raise ValueError('Native text encoder returned invalid embeddings')
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        relative = f'prompt_cache/embeddings/{digest}.pt'
        torch.save(encoded[0].detach().cpu().contiguous(), cache_root/relative)
        prompt_paths[prompt] = relative
    # Text is encoded once; do not retain its large encoder on the GPU during VAE encoding.
    if getattr(text_assets, 'text_encoder', None) is not None:
        text_assets.text_encoder.to('cpu')
    text_assets = None

    manifest = {'schema_version': SCHEMA_VERSION, 'complete': True, 'fps': 30,
        'camera': COLOR_CAMERA, 'action_semantics': semantics.value,
        'action_normalization': result['action_normalization'],
        'raw_window_frames': raw_frames, 'latent_frames': data_config.num_frames,
        'latent_stride_frames': 4,
        'canonical_size': [data_config.canonical_height, data_config.canonical_width],
        'encoding': encoding, 'sources': {'train': train.identity, 'validation': val.identity},
        'action_statistics': statistics, 'selection': selection, 'samples': {}}
    for split, repo, entries in (('train', train, train_samples), ('validation', val, val_samples)):
        (cache_root/split).mkdir()
        manifest['samples'][split] = []
        for entry in entries:
            episode, start = entry['episode_index'], entry['raw_start']
            raw = repo.read_rgb_window(episode, start-1, raw_frames+1)
            rgb = fit_frames(raw.permute(0,2,3,1).numpy(), data_config.canonical_height,
                             data_config.canonical_width, 'center_crop')
            video = encode_clip(vae, rgb[1:], normalize=True).permute(3,0,1,2).contiguous()
            condition = encode_clip(vae, rgb[:1], normalize=True).permute(3,0,1,2).contiguous()
            if video.shape[1] != data_config.num_frames or condition.shape != (video.shape[0],1,*video.shape[2:]):
                raise ValueError('Actual VAE output violates the 33-to-nine plus independent-prefix contract')
            if not torch.isfinite(video).all() or not torch.isfinite(condition).all():
                raise ValueError('VAE produced nonfinite Cine latents')
            prompt = tasks[(split, episode, start)]
            relative = f'{split}/episode{episode:06d}-start{start:06d}.pt'
            payload = {'video_latents': video.to(device='cpu', dtype=torch.bfloat16),
                'condition_latents': condition.to(device='cpu', dtype=torch.bfloat16),
                'text_context_path': prompt_paths[prompt], 'negative_text_context_path': prompt_paths[''],
                'task_text': prompt, 'task_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                'frame_ids': list(range(start, start+raw_frames)), 'condition_frame_id': start-1,
                'episode_index': episode, 'source_identity': repo.identity, 'encoding': encoding}
            torch.save(payload, cache_root/relative)
            manifest['samples'][split].append({**entry, 'path': relative})

    pending = 'manifest.pending.json'
    atomic_json(cache_root/pending, manifest)
    check_config = replace(data_config, adapter_options={**data_config.adapter_options, 'cache_manifest': pending})
    train_set, val_set = build_cine_latent_train_val_datasets(check_config)
    train_set[0], val_set[0]
    atomic_json(target, manifest)
    (cache_root/pending).unlink()
    return {**result, 'status': 'prepared', 'manifest': str(target),
            'frontend_models_loaded': True,
            'manifest_sha256': sha256_file(target), 'encoding': encoding,
            'train_windows': len(train_set), 'validation_windows': len(val_set)}
