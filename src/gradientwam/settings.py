"""Direct-path delivery configuration; no private machine or project lock files."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import yaml

ARMS = ('native_joint', 'same_capacity_deterministic_kv_blend',
        'variational_sharing', 'forced_private_world_gradient_off')
CAMERAS = ('observation.images.image', 'observation.images.wrist_image')


def load_episode_split(path: Path) -> dict:
    """Explicit episode partition shared by preparation and distributed readers."""
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict) or set(value) != {'schema_version', 'train_episode_ids', 'heldout_episode_ids'} or value.get('schema_version') != 1 or type(value.get('schema_version')) is not int:
        raise ValueError('Split JSON requires schema_version=1, train_episode_ids and heldout_episode_ids.')
    for name in ('train_episode_ids', 'heldout_episode_ids'):
        ids = value[name]
        if (not isinstance(ids, list) or not ids or
                any(type(i) is not int or i < 0 for i in ids) or len(ids) != len(set(ids))):
            raise ValueError(f'{name} must be a nonempty list of unique nonnegative integer episode IDs.')
    if set(value['train_episode_ids']) & set(value['heldout_episode_ids']):
        raise ValueError('Train and heldout episodes must be disjoint.')
    return {'schema_version':1, 'train_episode_ids':sorted(value['train_episode_ids']),
            'heldout_episode_ids':sorted(value['heldout_episode_ids'])}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _expand(value):
    if isinstance(value, str):
        value = re.sub(r'\$\{([A-Za-z_][A-Za-z_0-9]*):-([^}]*)\}',
                       lambda m: os.environ.get(m[1]) or m[2], value)
        expanded = os.path.expandvars(value)
        if re.search(r'\$\{[^}]+\}', expanded):
            raise ValueError('Set the environment variables referenced by the config.')
        return expanded
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _direct_path(value) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError('Asset and output paths must be absolute direct paths.')
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Use direct asset paths, not symbolic links.')
    return path.resolve()


@dataclass(frozen=True)
class Settings:
    arm: str
    seed: int
    route_seed: int
    episode_id: int
    checkpoint: Path
    checkpoint_sha256: str
    dataset_root: Path
    frontend_root: Path
    tokenizer_root: Path
    preparation_root: Path
    output_root: Path
    prompt_fingerprint: str
    max_minutes: int
    native: dict

    @property
    def latent_root(self) -> Path:
        return self.preparation_root / 'latents' / self.dataset_root.name / 'latents'

    @property
    def prompt_root(self) -> Path:
        return self.preparation_root / 'prompt_cache'

    def identity(self) -> dict:
        return {'arm': self.arm, 'seed': self.seed, 'route_seed': self.route_seed,
                'episode_id': self.episode_id, 'base_sha256': self.checkpoint_sha256,
                'native_config_sha256': hashlib.sha256(json.dumps(self.native, sort_keys=True).encode()).hexdigest()}

    def native_config(self):
        from open_wam.configs.loader import load_experiment_config
        from open_wam.configs import enums
        with tempfile.TemporaryDirectory(prefix='gradientwam-config-') as td:
            path = Path(td) / 'experiment.yaml'
            path.write_text(yaml.safe_dump(self.native), encoding='utf-8')
            config = load_experiment_config(path)
        return replace(config, trainer=replace(config.trainer,
            accelerator=enums.TrainerAccelerator.GPU, devices=1,
            precision=enums.TrainerPrecision.BF16, strategy=enums.StrategyName.SINGLE_DEVICE,
            enable_checkpointing=False, enable_jsonl_logging=False, enable_wandb=False,
            wandb_mode=enums.WandBMode.DISABLED, export_runtime_backbone=False,
            limit_val_batches=0, checkpoint_dir=None, resume_from=None))


def load_settings(path: Path) -> Settings:
    path = path.resolve()
    raw = _expand(yaml.safe_load(path.read_text(encoding='utf-8')))
    arm = raw['arm']
    if arm not in ARMS:
        raise ValueError('Unknown arm; choose one of the four frozen core arms.')
    assets, run = raw['assets'], raw['run']
    names = ('dataset_root', 'frontend_root', 'tokenizer_root', 'preparation_root')
    paths = {name: _direct_path(assets[name]) for name in names}
    checkpoint = _direct_path(raw['checkpoint']['path'])
    preparation = paths['preparation_root']
    for protected in (paths['dataset_root'], paths['frontend_root'], paths['tokenizer_root'], checkpoint):
        if preparation == protected or preparation in protected.parents or protected in preparation.parents:
            raise ValueError('Preparation output must be separate from source assets.')
    output = _direct_path(run['output_root'])
    for protected in (*paths.values(), checkpoint):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('Training output must be separate from all input/preparation assets.')
    if not re.fullmatch(r'[a-f0-9]{64}', raw['checkpoint']['sha256']):
        raise ValueError('checkpoint.sha256 must be a lowercase SHA256 digest.')
    if assets['prompt_encoder_fingerprint'] and not re.fullmatch(r'[a-f0-9]{64}', assets['prompt_encoder_fingerprint']):
        raise ValueError('prompt_encoder_fingerprint must be independently pinned.')
    if type(run['episode_id']) is not int or run['episode_id'] < 0:
        raise ValueError('episode_id must be a nonnegative integer.')
    if not 1 <= int(run['max_minutes']) <= 60:
        raise ValueError('This two-update smoke is limited to 1–60 minutes.')
    native = yaml.safe_load((path.parent / raw['experiment']).read_text(encoding='utf-8'))
    latent_root = paths['preparation_root'] / 'latents' / paths['dataset_root'].name / 'latents'
    prompt_root = paths['preparation_root'] / 'prompt_cache'
    data = native['data']
    data.update(local_root=str(paths['dataset_root']), latent_root=str(latent_root),
                camera_names=list(CAMERAS), latent_camera_names=list(CAMERAS),
                replay_status_path=None, empty_text_embedding_path=str(prompt_root/'embeddings'/f'{hashlib.sha256(b"").hexdigest()}.pt'),
                num_workers=0, train_fraction=1.0, split='train')
    for layout, camera in zip(data['view_layout'], CAMERAS, strict=True):
        layout['source_name'] = camera
    native['backbone'].update(pretrained_model_name_or_path=str(paths['frontend_root']),
        vae_subdir=str(paths['frontend_root']/'vae'), text_encoder_subdir=str(paths['frontend_root']/'text_encoder'),
        tokenizer_subdir=str(paths['tokenizer_root']), load_wan_vae_frontend=False,
        load_text_conditioning=False, load_reference_core_weights=False,
        prompt_cache=({'root': str(prompt_root), 'encoder_fingerprint': assets['prompt_encoder_fingerprint']}
                      if assets['prompt_encoder_fingerprint'] else None))
    t = native['training']
    required = {'gradient_accumulation_steps':10,'optimizer_name':'adamw', 'scheduler_name':'constant_with_warmup',
                'learning_rate':1e-5,'beta1':.9,'beta2':.95,'weight_decay':.1,'warmup_steps':10,
                'max_grad_norm':2.0,'action_loss_weight':1.0,'latent_loss_weight':1.0}
    if any(t.get(k) != v for k,v in required.items()):
        raise ValueError('Experiment differs from the frozen optimizer/loss recipe.')
    if native['backbone']['num_layers'] != 30 or native['policy_variant']['video_prefix_frames'] != 1:
        raise ValueError('The delivery requires the native 30-layer, one-prefix-frame profile.')
    return Settings(arm=arm, seed=int(run['seed']), route_seed=int(run['route_seed']),
        episode_id=run['episode_id'], checkpoint=checkpoint, checkpoint_sha256=raw['checkpoint']['sha256'],
        output_root=output, prompt_fingerprint=assets['prompt_encoder_fingerprint'],
        max_minutes=int(run['max_minutes']), native=native, **paths)
