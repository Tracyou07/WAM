"""Strict Cine cache -> native LatentWAMSample without a v2 compatibility tree."""
from __future__ import annotations

import json
import hashlib
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from open_wam.artifacts import load_tensor_artifact
from open_wam.configs import DataConfig, ActionNormalizationConfig
from open_wam.configs.enums import (ActionMappingMode, ActionNormalizationMode,
    ActionTargetRepresentation, CineActionSemantics)
from .action_normalization import normalize_action_targets
from .cine_v3 import COLOR_CAMERA, CineV3Repository, contained_path, validate_cine_split_sources
from .latent_contracts import LatentWAMSample
from .latent_temporal import latent_anchor_positions, raw_window_frames_for_latents

SCHEMA_VERSION = 'cine_v3_latent_v1'


def _validate_encoding(value):
    if not isinstance(value, dict) or any(not isinstance(value.get(key), str) or not value[key]
        for key in ('vae_identity', 'text_encoder_identity')):
        raise ValueError('Cache requires explicit VAE/text encoder identities')


def validate_cine_selection_manifest(manifest: dict, repositories: dict | None = None) -> None:
    """Single native selection validator; callers may reuse already-open repos."""
    if repositories is None:
        repositories = {split: CineV3Repository(manifest['sources'][split]['root'])
                        for split in ('train','validation')}
    selection = manifest.get('selection')
    if not isinstance(selection, dict) or selection.get('scope') not in ('subset','all_windows'):
        raise ValueError('Cine manifest must declare an explicit subset/all_windows selection')
    stride = selection.get('window_stride')
    if type(stride) is not int or stride <= 0:
        raise ValueError('Cine selection window_stride must be positive')
    for split, repo in repositories.items():
        declaration = selection.get(split)
        entries = manifest['samples'][split]
        if not isinstance(declaration,dict) or not isinstance(entries,list):
            raise ValueError('Cine selection requires explicit split declarations and sample lists')
        actual = {}
        for entry in entries:
            if not isinstance(entry,dict) or type(entry.get('episode_index')) is not int or type(entry.get('raw_start')) is not int:
                raise ValueError('Cine selected samples need integer episode/start IDs')
            actual.setdefault(entry['episode_index'], []).append(entry['raw_start'])
        ids = sorted(actual)
        for index in ids:
            if index not in repo._episodes or any(
                start < 1 or start + manifest['raw_window_frames'] > repo._episodes[index]['length']
                or (start-1) % stride != 0 for start in actual[index]
            ):
                raise ValueError('Cine selected window is outside its episode or the declared stride grid')
        starts = {str(index): sorted(actual[index]) for index in ids}
        if (declaration.get('episode_indices') != ids or declaration.get('raw_starts') != starts
                or declaration.get('window_count') != len(entries)
                or declaration.get('available_episode_count') != len(repo.episode_records)):
            raise ValueError('Cine declared selection disagrees with cached windows/source episode count')
        if selection['scope'] == 'all_windows':
            expected = {str(record['episode_index']): list(range(1,
                record['length'] - manifest['raw_window_frames'] + 1, stride)) for record in repo.episode_records}
            if starts != expected:
                raise ValueError('Cine all_windows cache omits source episodes or stride-aligned windows')


class CineV3LatentDataset(Dataset[LatentWAMSample]):
    def __init__(self, config: DataConfig, repo: CineV3Repository, root: Path,
                 manifest: dict, entries: list[dict], normalization: ActionNormalizationConfig):
        self.data_config, self.repository, self.cache_root = config, repo, root
        self.manifest, self.entries, self.normalization = manifest, entries, normalization
        self.action_statistics = manifest['action_statistics']
        self._prompt_cache = OrderedDict()

    def __len__(self):
        return len(self.entries)

    def _text_tensor(self, payload: dict, key: str, prompt: str):
        relative = payload.get(key+'_path')
        value = payload.get(key)
        if relative is None:
            return value
        if value is not None:
            raise ValueError('Cine text path and inline tensor are mutually exclusive')
        path = contained_path(self.cache_root, relative)
        digest = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        if path.name != digest+'.pt':
            raise ValueError('Cine prompt cache filename does not match the raw task SHA256')
        if path not in self._prompt_cache:
            self._prompt_cache[path] = load_tensor_artifact(path, map_location='cpu')
            if len(self._prompt_cache) > 8:
                self._prompt_cache.popitem(last=False)
        self._prompt_cache.move_to_end(path)
        return self._prompt_cache[path]

    def __getitem__(self, index: int) -> LatentWAMSample:
        entry = self.entries[index]
        payload = load_tensor_artifact(contained_path(self.cache_root, entry['path']), map_location='cpu')
        if not isinstance(payload, dict) or payload.get('encoding') != self.manifest['encoding']:
            raise ValueError('Cine cache encoder identity mismatch')
        if payload.get('source_identity') != self.repository.identity:
            raise ValueError('Cine payload source identity belongs to a different physical split')
        episode, start = entry['episode_index'], entry['raw_start']
        latent_count, raw_count = self.manifest['latent_frames'], self.manifest['raw_window_frames']
        frame_ids = list(range(start, start + raw_count))
        if payload.get('episode_index') != episode or payload.get('frame_ids') != frame_ids or payload.get('condition_frame_id') != start-1:
            raise ValueError('Cine cache episode/prefix/frame IDs do not match the time contract')
        rows = self.repository.read_rows(episode, start-1, start+raw_count)
        if len({row['task_index'] for row in rows}) != 1:
            raise ValueError('Cine window crosses a task change')
        task_text = self.repository.tasks[rows[0]['task_index']]
        task_sha = hashlib.sha256(task_text.encode('utf-8')).hexdigest()
        if ('task_text' in payload and payload['task_text'] != task_text) or ('task_sha256' in payload and payload['task_sha256'] != task_sha):
            raise ValueError('Cine payload task text identity differs from the raw prompt')
        if any(key+'_path' in payload for key in ('text_context','negative_text_context')) and not (
            payload.get('task_text') == task_text or payload.get('task_sha256') == task_sha
        ):
            raise ValueError('Cine shared text paths require raw task text/SHA binding')
        video, condition = (payload.get(key) for key in ('video_latents','condition_latents'))
        text = self._text_tensor(payload, 'text_context', task_text)
        if not isinstance(video, torch.Tensor) or video.ndim != 4 or video.shape[1] != latent_count or min(video.shape) <= 0:
            raise ValueError('Cine cached video shape disagrees with actual VAE temporal contract')
        if not isinstance(condition, torch.Tensor) or condition.shape != (video.shape[0], 1, *video.shape[2:]):
            raise ValueError('Cine requires an independently encoded one-frame observed prefix')
        if not isinstance(text, torch.Tensor) or text.ndim != 2 or min(text.shape) <= 0:
            raise ValueError('Cine text embedding must have shape [L,D]')
        negative = self._text_tensor(payload, 'negative_text_context', '')
        if negative is not None and (not isinstance(negative, torch.Tensor) or negative.ndim != 2 or min(negative.shape) <= 0 or negative.shape[-1] != text.shape[-1]):
            raise ValueError('Cine negative text embedding shape mismatch')
        for value in (video,condition,text,negative):
            if value is not None and (not value.is_floating_point() or not torch.isfinite(value).all()):
                raise ValueError('Cine cached tensors must be finite floating point')
        # First latent is causal singleton; the next latent owns commands 0..3.
        # Drop only the final frame's outgoing command, never early commands.
        real_actions = torch.tensor([row['action'] for row in rows[1:-1]], dtype=torch.float32)
        real_actions = normalize_action_targets(real_actions, normalization=self.normalization)
        actions = torch.cat((torch.zeros(4,7), real_actions), dim=0)
        action_mask = torch.cat((torch.zeros(4,7), torch.ones_like(real_actions)), dim=0)
        anchors = latent_anchor_positions(raw_frame_count=raw_count, latent_num_frames=latent_count,
                                        layout=self.data_config.latent_temporal_layout)
        state = torch.tensor(rows[0]['observation.state'], dtype=torch.float32)[None]
        # Frame states remain at true source anchors. Native prefix/pre-chunk
        # projection selects only the preceding causal chunk boundary.
        proprio = torch.tensor([rows[1+i]['observation.state'] for i in anchors], dtype=torch.float32)
        metadata = {'dataset_type': 'cine_v3_latent', 'source_root': str(self.repository.root),
            'episode_index': episode, 'raw_start': start, 'fps': 30,
            'action_semantics': self.manifest['action_semantics'], 'action_tokens_per_frame': 4,
            'action_frame_indices': [-1]*4 + frame_ids[:-1],
            'observed_frame_indices': [frame_ids[i] for i in anchors],
            'condition_frame_id': start-1, 'proprio_context_frame_indices': [frame_ids[i] for i in anchors],
            'history_frames': 1, 'sampled_chunk_size': self.data_config.sample_construction.chunk_size,
            'sampled_window_size': self.data_config.sample_construction.window_size,
            'loss_frame_start': 1, 'loss_frame_end': latent_count,
            'latent_loss_frame_start': 0, 'latent_loss_frame_end': latent_count,
            'action_loss_frame_start': 1, 'action_loss_frame_end': latent_count,
            'chunk_origin_frame': 1, 'singleton_chunk_frame': 0, 'frame_shift': 1,
            'raw_frame_count': raw_count, 'latent_frames': latent_count,
            'source_identity': self.repository.identity, 'cache_encoding': self.manifest['encoding']}
        return LatentWAMSample(video_latents=video.float(), condition_latents=condition.float(),
            actions=actions, action_mask=action_mask, state=state, state_mask=torch.ones_like(state),
            proprio_context_frames=proprio, proprio_context_frames_mask=torch.ones_like(proprio),
            task_text=task_text, text_context=text.float(),
            negative_text_context=None if negative is None else negative.float(), metadata=metadata)


def build_cine_latent_train_val_datasets(config: DataConfig):
    """Explicit physical roots; statistics belong to selected training episodes."""
    if not config.local_root or not config.val_local_root or not config.latent_root:
        raise ValueError('Cine needs explicit train, validation and cache roots')
    train_root, val_root, cache_root = (Path(value).expanduser().resolve()
        for value in (config.local_root,config.val_local_root,config.latent_root))
    if train_root == val_root or train_root in val_root.parents or val_root in train_root.parents:
        raise ValueError('Cine train and validation must be separate physical roots')
    if any(cache_root == source or cache_root in source.parents or source in cache_root.parents
           for source in (train_root,val_root)):
        raise ValueError('Cine cache must be separate from raw source roots')
    if config.camera_names != (COLOR_CAMERA,) or config.latent_camera_names != (COLOR_CAMERA,):
        raise ValueError('Cine requires the actual single color camera')
    if config.action_schema.action_dim != 7 or config.action_schema.state_dim != 7:
        raise ValueError('Cine model input requires raw action7/state7 without padding')
    if config.action_target.representation is not ActionTargetRepresentation.RAW or config.action_mapping.mode is not ActionMappingMode.NONE:
        raise ValueError('Cine joint commands require raw, unmapped targets')
    if config.action_target.normalization.mode is not ActionNormalizationMode.NONE:
        raise ValueError('Use Cine train-stat normalization to avoid applying a second normalization')
    if config.frame_stride != 1 or config.train_fraction != 1.:
        raise ValueError('Cine uses consecutive 30-FPS frames and explicitly separate roots')
    options = config.adapter_options
    if set(options) - {'cache_manifest','action_semantics','action_normalization'}:
        raise ValueError('Unknown Cine adapter options')
    semantics = CineActionSemantics(options.get('action_semantics','raw_joint_command'))
    mode = ActionNormalizationMode(options.get('action_normalization','none'))
    if mode not in (ActionNormalizationMode.NONE, ActionNormalizationMode.GAUSSIAN):
        raise ValueError('Cine supports none or train-stat gaussian normalization')
    manifest = json.loads(contained_path(cache_root, options.get('cache_manifest','manifest.json')).read_text(encoding='utf-8'))
    if manifest.get('schema_version') != SCHEMA_VERSION or manifest.get('complete') is not True or manifest.get('fps') != 30 or manifest.get('camera') != COLOR_CAMERA:
        raise ValueError('Incompatible Cine cache schema/FPS/camera')
    if manifest.get('action_semantics') != semantics.value:
        raise ValueError('Cine action-semantics declaration differs from cache')
    if manifest.get('action_normalization') != mode.value:
        raise ValueError('Cine action normalization differs from the prepared cache')
    count = manifest['latent_frames']
    if type(count) is not int or count < 2 or count != config.num_frames:
        raise ValueError('Cine cache and native latent frame count differ')
    if manifest.get('raw_window_frames') != raw_window_frames_for_latents(count) or manifest.get('latent_stride_frames') != 4:
        raise ValueError('Cine raw/latent temporal compression is not native Wan stride4')
    if config.action_schema.action_horizon != 4*count or manifest.get('canonical_size') != [config.canonical_height,config.canonical_width]:
        raise ValueError('Cine model action geometry/canonical resolution differs from cache')
    _validate_encoding(manifest.get('encoding'))
    train, val = CineV3Repository(train_root), CineV3Repository(val_root)
    for split, repo in (('train',train),('validation',val)):
        if manifest['sources'].get(split) != repo.identity:
            raise ValueError('Cine cache source identity is stale or split-mismatched')
    validate_cine_selection_manifest(manifest, {'train':train,'validation':val})
    if manifest['selection']['window_stride'] != config.sample_stride:
        raise ValueError('Cine prepared window stride differs from data.sample_stride')
    selected = {}
    for split, repo, limit in (('train',train,config.max_train_episodes),('validation',val,config.max_val_episodes)):
        entries = manifest['samples'][split]
        seen_windows, seen_paths, seen_files = set(), set(), set()
        for entry in entries:
            path = contained_path(cache_root, entry['path'])
            if not path.is_file():
                raise ValueError('Cine manifest payload is missing or not a file')
            window = (entry['episode_index'], entry['raw_start'])
            stat = path.stat()
            physical = (stat.st_dev,stat.st_ino)
            if window in seen_windows or path in seen_paths or physical in seen_files:
                raise ValueError('Cine split cache contains a duplicate window/payload')
            seen_windows.add(window)
            seen_paths.add(path)
            seen_files.add(physical)
        ids = sorted({entry['episode_index'] for entry in entries})
        if limit is not None: ids = ids[:limit]
        selected[split] = [entry for entry in entries if entry['episode_index'] in ids]
        if not selected[split]: raise ValueError('Cine split cache contains no selected samples')
        for entry in selected[split]:
            if type(entry['episode_index']) is not int or entry['episode_index'] not in repo._episodes or type(entry['raw_start']) is not int or entry['raw_start'] < 1:
                raise ValueError('Cine cache needs a valid episode and independent preceding frame')
            if entry['raw_start'] + manifest['raw_window_frames'] > repo._episodes[entry['episode_index']]['length']:
                raise ValueError('Cine cached window crosses its episode boundary')
            contained_path(cache_root, entry['path'])
    validate_cine_split_sources(train,val,
        sorted({entry['episode_index'] for entry in selected['train']}),
        sorted({entry['episode_index'] for entry in selected['validation']}))
    stats = manifest['action_statistics']
    train_ids = sorted({entry['episode_index'] for entry in selected['train']})
    expected_count = sum(train._episodes[index]['length'] for index in train_ids)
    if stats.get('source') != 'train' or stats.get('root') != str(train_root) or stats.get('episode_indices') != train_ids or stats.get('count') != expected_count:
        raise ValueError('Cine statistics must describe exactly the selected train episodes, never validation')
    mean, std = torch.tensor(stats['mean']), torch.tensor(stats['std'])
    if mean.shape != (7,) or std.shape != (7,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Cine train statistics must be finite seven-channel mean/std')
    normalization = ActionNormalizationConfig(mode=mode,
        mean=tuple(stats['mean']) if mode is ActionNormalizationMode.GAUSSIAN else (),
        std=tuple(stats['std']) if mode is ActionNormalizationMode.GAUSSIAN else ())
    return (CineV3LatentDataset(config,train,cache_root,manifest,selected['train'],normalization),
            CineV3LatentDataset(config,val,cache_root,manifest,selected['validation'],normalization))
