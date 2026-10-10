"""Direct-path delivery configuration; no private machine or project lock files."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
from enum import Enum
import hashlib
import json
import os
import math
from numbers import Real
from pathlib import Path
import re
import tempfile
from collections.abc import Mapping

import yaml

class GradientWAMMethod(str, Enum):
    BASELINE = 'baseline'
    VRFM = 'vrfm'
    CAGRAD = 'cagrad'
    VRFM_CAGRAD = 'vrfm_cagrad'


ARMS = tuple(method.value for method in GradientWAMMethod)
TRAINABILITY_SCOPE_ID = 'native_action_plus_final_video_shared_kv_v1'
LEGACY_ARMS = (
    'native_joint',
    'same_capacity_deterministic_kv_blend',
    'variational_sharing',
    'forced_private_world_gradient_off',
)
CAMERAS = ('observation.images.image', 'observation.images.wrist_image')


@dataclass(frozen=True)
class GradientWAMMethodConfig:
    method: GradientWAMMethod | None = GradientWAMMethod.BASELINE
    latent_dim: int = 32
    kl_weight: float = 0.001
    cagrad_c: float = 0.4
    legacy_v02: bool = False
    legacy_arm: str | None = None

    def __post_init__(self) -> None:
        if self.legacy_v02:
            if self.method is not None or self.legacy_arm not in LEGACY_ARMS:
                raise ValueError('legacy_v02 requires exactly one supported legacy arm.')
        elif self.method is None or self.legacy_arm is not None:
            raise ValueError('New GradientWAM configs require a new method value.')
        elif not isinstance(self.method, GradientWAMMethod):
            object.__setattr__(self, 'method', GradientWAMMethod(self.method))
        if type(self.latent_dim) is not int or self.latent_dim <= 0:
            raise ValueError('gradientwam.latent_dim must be a positive integer.')
        for name, value, upper_exclusive in (
            ('kl_weight', self.kl_weight, None),
            ('cagrad_c', self.cagrad_c, 1.0),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f'gradientwam.{name} must be numeric.')
            value = float(value)
            if not math.isfinite(value) or value < 0 or (upper_exclusive is not None and value >= upper_exclusive):
                interval = '[0, 1)' if upper_exclusive is not None else 'finite and nonnegative'
                raise ValueError(f'gradientwam.{name} must be {interval}.')
            object.__setattr__(self, name, value)

    @property
    def uses_vrfm(self) -> bool:
        return self.method in (GradientWAMMethod.VRFM, GradientWAMMethod.VRFM_CAGRAD)

    @property
    def uses_cagrad(self) -> bool:
        return self.method in (GradientWAMMethod.CAGRAD, GradientWAMMethod.VRFM_CAGRAD)

    @property
    def label(self) -> str:
        return self.legacy_arm if self.legacy_v02 else self.method.value

    def identity(self) -> dict:
        if self.legacy_v02:
            return {'method': 'legacy_v02', 'arm': self.legacy_arm, 'legacy_v02': True}
        return {
            'method': self.method.value,
            'latent_dim': self.latent_dim,
            'kl_weight': self.kl_weight,
            'cagrad_c': self.cagrad_c,
            'legacy_v02': False,
        }


def parse_method_config(raw: Mapping) -> GradientWAMMethodConfig:
    if not isinstance(raw, Mapping):
        raise TypeError('settings config must be a mapping.')
    legacy = raw.get('legacy_v02', False)
    if type(legacy) is not bool:
        raise TypeError('legacy_v02 must be a boolean.')
    if legacy:
        if 'gradientwam' in raw:
            raise ValueError('legacy_v02 cannot be combined with a new gradientwam method.')
        arm = raw.get('arm')
        if arm not in LEGACY_ARMS:
            raise ValueError('legacy_v02 requires one supported historical arm.')
        return GradientWAMMethodConfig(
            method=None,
            legacy_v02=True,
            legacy_arm=arm,
        )
    if 'arm' in raw:
        raise ValueError('Historical arm requires explicit legacy_v02: true.')
    values = raw.get('gradientwam', {})
    if not isinstance(values, Mapping):
        raise TypeError('gradientwam must be a mapping.')
    unknown = set(values) - {'method', 'latent_dim', 'kl_weight', 'cagrad_c'}
    if unknown:
        raise ValueError(f'Unknown gradientwam settings: {sorted(unknown)}.')
    try:
        method = GradientWAMMethod(values.get('method', GradientWAMMethod.BASELINE))
    except ValueError as exc:
        supported = ', '.join(method.value for method in GradientWAMMethod)
        raise ValueError(f'Unknown GradientWAM method; choose one of: {supported}.') from exc
    return GradientWAMMethodConfig(
        method=method,
        latent_dim=values.get('latent_dim', 32),
        kl_weight=values.get('kl_weight', 0.001),
        cagrad_c=values.get('cagrad_c', 0.4),
    )


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
    method_config: GradientWAMMethodConfig
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

    @property
    def arm(self) -> str:
        """Compatibility label for reports; new runs expose their method name."""
        return self.method_config.label

    def identity(self) -> dict:
        scope = 'legacy_v02' if self.method_config.legacy_v02 else TRAINABILITY_SCOPE_ID
        return {'gradientwam': self.method_config.identity(), 'trainability_scope': scope,
                'seed': self.seed, 'route_seed': self.route_seed,
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
    method_config = parse_method_config(raw)
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
    return Settings(method_config=method_config, seed=int(run['seed']), route_seed=int(run['route_seed']),
        episode_id=run['episode_id'], checkpoint=checkpoint, checkpoint_sha256=raw['checkpoint']['sha256'],
        output_root=output, prompt_fingerprint=assets['prompt_encoder_fingerprint'],
        max_minutes=int(run['max_minutes']), native=native, **paths)

@dataclass(frozen=True)
class CineSettings:
    """Independent, direct-path settings for the Cine v3 training entry."""

    config_path: Path
    method_config: GradientWAMMethodConfig
    seed: int
    route_seed: int
    eval_seed: int
    steps: int
    checkpoint: Path
    checkpoint_sha256: str
    frontend_root: Path
    tokenizer_root: Path
    train_root: Path
    val_root: Path
    latent_root: Path
    prompt_cache_root: Path
    output_root: Path
    action_semantics: str
    action_normalization: str
    native: Any
    config_sha256: str

    @property
    def arm(self) -> str:
        return self.method_config.label

    def native_config(self):
        return self.native

    def identity(self) -> dict:
        return {
            "gradientwam": self.method_config.identity(),
            "trainability_scope": TRAINABILITY_SCOPE_ID,
            "seed": self.seed,
            "route_seed": self.route_seed,
            "base_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "train_root": str(self.train_root),
            "validation_root": str(self.val_root),
            "latent_root": str(self.latent_root),
            "action_semantics": self.action_semantics,
            "action_normalization": self.action_normalization,
            "native_config_sha256": hashlib.sha256(
                json.dumps(self.native, default=str, sort_keys=True).encode()
            ).hexdigest(),
        }

    def validate_fresh_cache_root(self, *, execute: bool = False) -> None:
        """Refuse to mix a new preparation with an existing cache."""
        if self.latent_root.exists() and (
            execute or any(self.latent_root.iterdir())
        ):
            raise FileExistsError(
                f"Cine preparation requires a fresh cache root: {self.latent_root}"
            )


def _merge_config(base: dict, overlay: Mapping) -> dict:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _assert_write_path_separate(write_path: Path, inputs: tuple[Path, ...], *, label: str) -> None:
    for input_path in inputs:
        if (
            write_path == input_path
            or write_path in input_path.parents
            or input_path in write_path.parents
        ):
            raise ValueError(f"{label} must not overlap input paths.")


def load_cine_settings(path: Path) -> CineSettings:
    """Load a Cine method config without passing through legacy LIBERO Settings."""
    from open_wam.configs.enums import ActionNormalizationMode, CineActionSemantics
    from open_wam.configs.loader import load_experiment_config

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = _expand(yaml.safe_load(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1 or type(raw.get("schema_version")) is not int:
        raise ValueError("Cine config requires schema_version: 1.")
    allowed = {"schema_version", "experiment", "gradientwam", "assets", "data", "preparation", "run"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown Cine config fields: {sorted(unknown)}.")
    method_config = parse_method_config(raw)
    if method_config.legacy_v02:
        raise ValueError("Cine supports only baseline, vrfm, cagrad, and vrfm_cagrad.")

    assets = raw.get("assets")
    data_overlay = raw.get("data")
    preparation = raw.get("preparation")
    run = raw.get("run")
    if not all(isinstance(value, Mapping) for value in (assets, data_overlay, preparation, run)):
        raise TypeError("Cine assets, data, preparation, and run sections must be mappings.")
    required_assets = {"checkpoint", "checkpoint_sha256", "frontend_root", "tokenizer_root"}
    if set(assets) != required_assets:
        raise ValueError(f"Cine assets must contain exactly: {sorted(required_assets)}.")
    required_data = {"dataset_name", "dataset_type", "local_root", "val_local_root", "latent_root"}
    if not required_data <= set(data_overlay):
        raise ValueError(f"Cine data is missing fields: {sorted(required_data - set(data_overlay))}.")
    if data_overlay["dataset_name"] != "cine_v3" or data_overlay["dataset_type"] != "cine_v3_latent":
        raise ValueError("Cine requires dataset_name=cine_v3 and dataset_type=cine_v3_latent.")
    if set(preparation) - {"prompt_cache_root"} or "prompt_cache_root" not in preparation:
        raise ValueError("Cine preparation requires prompt_cache_root and no unknown fields.")
    if set(run) - {"output_root", "steps", "seed", "route_seed", "eval_seed"}:
        raise ValueError(f"Unknown Cine run fields: {sorted(set(run) - {'output_root', 'steps', 'seed', 'route_seed', 'eval_seed'})}.")
    if not {"output_root", "steps", "seed"} <= set(run):
        raise ValueError("Cine run requires output_root, steps, and seed.")
    if type(run["steps"]) is not int or run["steps"] <= 0:
        raise ValueError("run.steps must be a positive integer.")
    for name in ("seed", "route_seed", "eval_seed"):
        value = run.get(name, run.get("seed"))
        if type(value) is not int or value < 0:
            raise ValueError(f"run.{name} must be a nonnegative integer.")

    checkpoint = _direct_path(assets["checkpoint"])
    frontend_root = _direct_path(assets["frontend_root"])
    tokenizer_root = _direct_path(assets["tokenizer_root"])
    train_root = _direct_path(data_overlay["local_root"])
    val_root = _direct_path(data_overlay["val_local_root"])
    latent_root = _direct_path(data_overlay["latent_root"])
    prompt_cache_root = _direct_path(preparation["prompt_cache_root"])
    output_root = _direct_path(run["output_root"])
    if not re.fullmatch(r"[a-f0-9]{64}", str(assets["checkpoint_sha256"])):
        raise ValueError("assets.checkpoint_sha256 must be a lowercase SHA256 digest.")
    if prompt_cache_root != latent_root / "prompt_cache":
        raise ValueError("preparation.prompt_cache_root must be GW_CINE_LATENT_ROOT/prompt_cache.")
    _assert_write_path_separate(latent_root, (train_root, val_root, checkpoint, frontend_root, tokenizer_root), label="Cine cache root")
    _assert_write_path_separate(output_root, (latent_root, train_root, val_root, checkpoint, frontend_root, tokenizer_root), label="Cine output root")

    options = data_overlay.get("adapter_options", {})
    if not isinstance(options, Mapping):
        raise TypeError("Cine adapter_options must be a mapping.")
    semantics_enum = CineActionSemantics(options.get("action_semantics", "raw_joint_command"))
    normalization_enum = ActionNormalizationMode(options.get("action_normalization", "none"))
    if normalization_enum not in (ActionNormalizationMode.NONE, ActionNormalizationMode.GAUSSIAN):
        raise ValueError("Cine action normalization supports only none or gaussian.")
    semantics = semantics_enum.value
    normalization = normalization_enum.value
    target = data_overlay.get("action_target", {})
    if target.get("include_gripper", False) is not False:
        raise ValueError("Cine requires raw7 joint actions, identity state encoding, and no gripper.")
    schema = data_overlay.get("action_schema", {})
    expected_schema = {"action_dim": 7, "action_horizon": 36, "state_dim": 7, "state_horizon": 1}
    if schema != expected_schema:
        raise ValueError("Cine action_schema must be raw action7/state7 with action_horizon=36.")
    if data_overlay.get("num_frames") != 9 or data_overlay.get("frame_stride") != 1:
        raise ValueError("Cine requires nine latent frames and consecutive source frames.")
    camera = ("observation.images.color",)
    if tuple(data_overlay.get("camera_names", ())) != camera or tuple(data_overlay.get("latent_camera_names", ())) != camera:
        raise ValueError("Cine requires the single observation.images.color camera.")

    experiment_name = raw.get("experiment")
    if not isinstance(experiment_name, str) or not experiment_name:
        raise ValueError("Cine experiment must name the native OpenWAM YAML.")
    experiment_path = (path.parent / experiment_name).resolve()
    if not experiment_path.is_relative_to(path.parent.resolve()) or not experiment_path.is_file():
        raise ValueError("Cine experiment must be a YAML file beside the method config.")
    native_raw = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    if not isinstance(native_raw, dict):
        raise ValueError("Cine native experiment must be a YAML mapping.")
    native_raw["data"] = _merge_config(native_raw.get("data", {}), data_overlay)
    native_raw["data"].update(
        local_root=str(train_root),
        val_local_root=str(val_root),
        latent_root=str(latent_root),
        split="train",
        train_fraction=1.0,
        num_workers=0,
    )
    # Cine's declared sampling default is one window per 32-action group.
    if "sample_stride" not in data_overlay:
        native_raw["data"]["sample_stride"] = 32
    # Cine's encoders are separate from the verified OpenWAM runtime checkpoint.
    native_raw["backbone"]["pretrained_model_name_or_path"] = None
    native_raw["backbone"]["vae_subdir"] = str(frontend_root / "vae")
    native_raw["backbone"]["text_encoder_subdir"] = str(frontend_root / "text_encoder")
    native_raw["backbone"]["tokenizer_subdir"] = str(tokenizer_root)
    native_raw["backbone"]["load_wan_vae_frontend"] = False
    native_raw["backbone"]["load_text_conditioning"] = False
    native_raw["backbone"]["load_reference_core_weights"] = False
    native_raw["trainer"]["devices"] = 1
    with tempfile.TemporaryDirectory(prefix="gradientwam-cine-config-") as td:
        native_path = Path(td) / "experiment.yaml"
        native_path.write_text(yaml.safe_dump(native_raw, sort_keys=False), encoding="utf-8")
        native = load_experiment_config(native_path)
    if (
        native.data.action_schema.action_dim != 7
        or native.data.action_schema.action_horizon != 36
        or native.data.action_schema.state_dim != 7
        or native.data.num_frames != 9
        or native.action_decoder.action_dim != 7
        or native.action_decoder.action_horizon != 36
        or native.inference.frame_chunk_size != 9
    ):
        raise ValueError("Cine method config and native decoder geometry disagree.")
    if native.data.sample_stride <= 0:
        raise ValueError("Cine data.sample_stride must be a positive raw-window start stride.")
    from open_wam.configs.enums import ActionTargetRepresentation, ActionTargetStateEncoding
    if (
        native.data.action_target.representation is not ActionTargetRepresentation.RAW
        or native.data.action_target.state_encoding is not ActionTargetStateEncoding.IDENTITY
        or native.data.action_target.include_gripper
    ):
        raise ValueError("Cine native config requires raw joint actions, identity state, and no gripper.")
    identity_raw = dict(raw)
    identity_raw["run"] = dict(run)
    identity_raw["run"].pop("steps", None)
    identity_raw["run"].pop("output_root", None)
    config_digest = hashlib.sha256(
        json.dumps(identity_raw, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    return CineSettings(
        config_path=path,
        method_config=method_config,
        seed=run["seed"],
        route_seed=run.get("route_seed", run["seed"]),
        eval_seed=run.get("eval_seed", run["seed"]),
        steps=run["steps"],
        checkpoint=checkpoint,
        checkpoint_sha256=str(assets["checkpoint_sha256"]),
        frontend_root=frontend_root,
        tokenizer_root=tokenizer_root,
        train_root=train_root,
        val_root=val_root,
        latent_root=latent_root,
        prompt_cache_root=prompt_cache_root,
        output_root=output_root,
        action_semantics=semantics,
        action_normalization=normalization,
        native=native,
        config_sha256=config_digest,
    )
