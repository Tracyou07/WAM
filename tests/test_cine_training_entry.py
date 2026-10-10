from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gradientwam.settings import load_cine_settings
from gradientwam import cine_runner


@pytest.fixture
def cine_env(tmp_path, monkeypatch):
    roots = {
        "GW_CINE_TRAIN_ROOT": tmp_path / "train_2.4k",
        "GW_CINE_VAL_ROOT": tmp_path / "validation_200",
        "GW_CINE_LATENT_ROOT": tmp_path / "latent-cache",
        "GW_CHECKPOINT": tmp_path / "weights.safetensors",
        "GW_FRONTEND_ROOT": tmp_path / "frontend",
        "GW_TOKENIZER_ROOT": tmp_path / "tokenizer",
        "GW_OUTPUT_ROOT": tmp_path / "runs" / "vrfm",
    }
    for key, value in roots.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.setenv("GW_CHECKPOINT_SHA256", "a" * 64)
    return roots


@pytest.mark.parametrize(
    ("filename", "method"),
    [
        ("baseline.yaml", "baseline"),
        ("vrfm.yaml", "vrfm"),
        ("cagrad.yaml", "cagrad"),
        ("vrfm_cagrad.yaml", "vrfm_cagrad"),
    ],
)
def test_cine_settings_loads_all_methods_and_native_geometry(cine_env, filename, method):
    config_path = Path(__file__).parents[1] / "configs" / "cine_v3" / filename

    settings = load_cine_settings(config_path)
    native = settings.native_config()

    assert settings.method_config.method.value == method
    assert settings.checkpoint == cine_env["GW_CHECKPOINT"]
    assert settings.frontend_root == cine_env["GW_FRONTEND_ROOT"]
    assert settings.tokenizer_root == cine_env["GW_TOKENIZER_ROOT"]
    assert settings.output_root == cine_env["GW_OUTPUT_ROOT"]
    assert native.data.local_root == str(cine_env["GW_CINE_TRAIN_ROOT"])
    assert native.data.val_local_root == str(cine_env["GW_CINE_VAL_ROOT"])
    assert native.data.latent_root == str(cine_env["GW_CINE_LATENT_ROOT"])
    assert native.data.dataset_type == "cine_v3_latent"
    assert native.data.num_frames == 9
    assert native.data.action_schema.action_dim == 7
    assert native.data.action_schema.action_horizon == 36
    assert native.data.action_schema.state_dim == 7
    assert native.action_decoder.action_horizon == 36
    assert native.inference.frame_chunk_size == 9
    assert native.data.sample_stride == 32


def test_cine_settings_rejects_overlapping_cache_root(cine_env, monkeypatch):
    monkeypatch.setenv(
        "GW_CINE_LATENT_ROOT",
        str(cine_env["GW_CINE_TRAIN_ROOT"] / "cache"),
    )
    config_path = Path(__file__).parents[1] / "configs" / "cine_v3" / "baseline.yaml"

    with pytest.raises(ValueError, match="overlap"):
        load_cine_settings(config_path)


@pytest.mark.parametrize(
    "args",
    [
        ["--train-episode-limit", "1"],
        ["--train-episode-limit", "1", "--train-windows-per-episode", "1",
         "--val-episode-limit", "1"],
        ["--train-episode-limit", "0", "--train-windows-per-episode", "1",
         "--val-episode-limit", "1", "--val-windows-per-episode", "1"],
    ],
)
def test_prepare_requires_all_windows_or_four_positive_limits(cine_env, args):
    config_path = Path(__file__).parents[1] / "configs" / "cine_v3" / "baseline.yaml"

    with pytest.raises(SystemExit):
        cine_runner.main(["prepare", "--config", str(config_path), "--device", "cpu", *args])


def test_prepare_routes_explicit_selection_to_preparer(cine_env, monkeypatch, capsys):
    config_path = Path(__file__).parents[1] / "configs" / "cine_v3" / "vrfm.yaml"
    calls = []
    module = SimpleNamespace(prepare_cine=lambda *a, **kw: calls.append((a, kw)) or {"status": "planned"})
    monkeypatch.setitem(__import__("sys").modules, "gradientwam.cine_preparation", module)

    result = cine_runner.main([
        "prepare", "--config", str(config_path), "--device", "cpu",
        "--train-episode-limit", "2", "--train-windows-per-episode", "3",
        "--val-episode-limit", "1", "--val-windows-per-episode", "1",
    ])

    assert result == 0
    assert len(calls) == 1
    data_config, = calls[0][0]
    kwargs = calls[0][1]
    assert data_config.local_root == str(cine_env["GW_CINE_TRAIN_ROOT"])
    assert kwargs == {
        "frontend_root": cine_env["GW_FRONTEND_ROOT"],
        "tokenizer_root": cine_env["GW_TOKENIZER_ROOT"],
        "device": "cpu",
        "execute": False,
        "train_episode_limit": 2,
        "train_windows_per_episode": 3,
        "val_episode_limit": 1,
        "val_windows_per_episode": 1,
        "all_windows": False,
    }
    assert "planned" in capsys.readouterr().out


def test_check_config_does_not_build_model(cine_env, capsys):
    config_path = Path(__file__).parents[1] / "configs" / "cine_v3" / "baseline.yaml"

    assert cine_runner.main(["check-config", "--config", str(config_path)]) == 0
    report = yaml.safe_load(capsys.readouterr().out)
    assert report["status"] == "valid"
    assert report["method"] == "baseline"
    assert report["action_horizon"] == 36
    assert report["frame_chunk_size"] == 9

def _copy_cine_config(tmp_path: Path, source: Path, *, steps: int | None = None,
                     output_root: Path | None = None, normalization: str | None = None,
                     prompt_cache_root: Path | None = None) -> Path:
    import shutil

    config_dir = tmp_path / "cine-config"
    config_dir.mkdir(parents=True, exist_ok=True)
    target = config_dir / source.name
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    shutil.copy2(source.parent / raw["experiment"], config_dir / raw["experiment"])
    if steps is not None:
        raw["run"]["steps"] = steps
    if output_root is not None:
        raw["run"]["output_root"] = str(output_root)
    if normalization is not None:
        raw["data"]["adapter_options"]["action_normalization"] = normalization
    if prompt_cache_root is not None:
        raw["preparation"]["prompt_cache_root"] = str(prompt_cache_root)
    target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return target


def test_cine_settings_allows_nested_read_only_assets(cine_env, monkeypatch):
    frontend = cine_env["GW_FRONTEND_ROOT"]
    monkeypatch.setenv("GW_TOKENIZER_ROOT", str(frontend / "tokenizer"))
    monkeypatch.setenv("GW_CHECKPOINT", str(frontend / "weights.safetensors"))
    config_path = Path(__file__).parents[1] / "configs" / "cine_v3" / "baseline.yaml"

    settings = load_cine_settings(config_path)

    assert settings.tokenizer_root == frontend / "tokenizer"
    assert settings.checkpoint == frontend / "weights.safetensors"


def test_cine_settings_requires_preparer_prompt_cache_location(cine_env, tmp_path):
    source = Path(__file__).parents[1] / "configs" / "cine_v3" / "baseline.yaml"
    config_path = _copy_cine_config(
        tmp_path,
        source,
        prompt_cache_root=cine_env["GW_CINE_LATENT_ROOT"] / "other-prompts",
    )

    with pytest.raises(ValueError, match="prompt_cache_root"):
        load_cine_settings(config_path)


def test_cine_settings_resume_identity_ignores_step_budget_and_output_root(cine_env, tmp_path):
    source = Path(__file__).parents[1] / "configs" / "cine_v3" / "baseline.yaml"
    base_path = _copy_cine_config(
        tmp_path / "base",
        source,
        steps=1,
        output_root=tmp_path / "run-1",
    )
    continued_path = _copy_cine_config(
        tmp_path / "continued",
        source,
        steps=3,
        output_root=tmp_path / "run-2",
    )

    base = load_cine_settings(base_path)
    continued = load_cine_settings(continued_path)

    assert base.identity() == continued.identity()


def test_cine_resume_entry_routes_identity_for_larger_step_budget_and_new_output(
    cine_env, tmp_path, monkeypatch
):
    import torch
    from open_wam.training.launch import DistributedLaunchContext, LaunchEnvironment
    from open_wam.training.runtime import TrainingRuntime
    from gradientwam import cine_runner
    from gradientwam.distributed_train import RankAwareTrainingRuntime
    from gradientwam import runtime_factory

    source = Path(__file__).parents[1] / "configs" / "cine_v3" / "baseline.yaml"
    base_path = _copy_cine_config(tmp_path / "base", source, steps=1, output_root=tmp_path / "run-1")
    continued_path = _copy_cine_config(tmp_path / "continued", source, steps=2, output_root=tmp_path / "run-2")
    base_settings = load_cine_settings(base_path)
    continued_settings = load_cine_settings(continued_path)

    context = DistributedLaunchContext(
        rank=0, local_rank=0, world_size=1, local_world_size=1,
        environment=LaunchEnvironment.TORCH_DISTRIBUTED,
    )
    monkeypatch.setattr(
        DistributedLaunchContext,
        "from_env",
        classmethod(lambda cls: context),
    )

    class FakeStrategy:
        rank = 0
        world_size = 1
        is_main_process = True
        device = torch.device("cpu")

        def barrier(self):
            pass

        def close(self):
            pass

        def unwrap_model(self, model):
            return model

    monkeypatch.setattr(
        "open_wam.training.strategies.build_training_strategy",
        lambda *args, **kwargs: FakeStrategy(),
    )

    manifest = {
        "sources": {"train": {"root": str(cine_env["GW_CINE_TRAIN_ROOT"])},
                    "validation": {"root": str(cine_env["GW_CINE_VAL_ROOT"])}},
        "selection": {"scope": "subset", "window_stride": 32},
        "samples": {"train": [{"episode_index": 0, "raw_start": 1}],
                    "validation": [{"episode_index": 2, "raw_start": 1}]},
    }
    import json
    latent_root = cine_env["GW_CINE_LATENT_ROOT"]
    latent_root.mkdir(parents=True, exist_ok=True)
    (latent_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    train_dataset = SimpleNamespace(manifest=manifest, entries=manifest["samples"]["train"])
    val_dataset = SimpleNamespace(manifest=manifest, entries=manifest["samples"]["validation"])
    monkeypatch.setattr(
        cine_runner,
        "_cine_loaders",
        lambda *args, **kwargs: (object(), object(), train_dataset, val_dataset),
    )
    monkeypatch.setattr(
        "open_wam.training.data_loading.preflight_runtime_dataset_artifacts",
        lambda config: (),
    )

    identity_by_call = []
    training_resume_calls = []
    rank_resume_calls = []

    class FakeTrainingRuntime:
        def __init__(self, config, strategy, train_loader, val_loader):
            self.config = config
            self.strategy = strategy
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.model = object()
            self.train_state = SimpleNamespace(
                global_step=1,
                optimizer_step=1,
                epoch_index=0,
                next_batch_index=0,
                seen_batches=1,
            )
            self._save_checkpoint = lambda **kwargs: None

    def fake_build(**kwargs):
        identity_by_call.append(kwargs["identity"])
        runtime = FakeTrainingRuntime(
            kwargs["config"], kwargs["strategy"], kwargs["train_loader"], kwargs["val_loader"]
        )
        adapted = RankAwareTrainingRuntime(
            runtime,
            output_dir=kwargs["output_dir"],
            eval_seed=kwargs["eval_seed"],
            identity=kwargs["identity"],
            method_config=kwargs["settings"].method_config,
        )
        adapted.heldout_episode_ids = list(kwargs["heldout_episode_ids"])
        if kwargs["resume"] is not None:
            adapted.resume(kwargs["resume"])
        return adapted

    monkeypatch.setattr(runtime_factory, "build_rank_aware_runtime", fake_build)
    monkeypatch.setattr(
        TrainingRuntime,
        "resume",
        lambda self, checkpoint: training_resume_calls.append(checkpoint),
    )
    monkeypatch.setattr(
        "gradientwam.distributed_train.restore_rank_runtime_state",
        lambda *args, **kwargs: rank_resume_calls.append(kwargs["identity"]),
    )
    checkpoint = tmp_path / "checkpoint" / "full_training_state.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()

    cine_runner._build_cine_runtime(base_settings, resume=None)
    resumed = cine_runner._build_cine_runtime(continued_settings, resume=str(checkpoint))

    assert resumed.config.training.num_steps == 2
    assert identity_by_call[0] == identity_by_call[1]
    assert training_resume_calls == [str(checkpoint)]
    assert rank_resume_calls == [identity_by_call[0]]


def test_gaussian_cine_inference_denormalizes_only_matching_run_manifest(
    cine_env, tmp_path, monkeypatch
):
    import hashlib
    import json
    import torch
    from gradientwam import cine_inference

    source = Path(__file__).parents[1] / "configs" / "cine_v3" / "baseline.yaml"
    config_path = _copy_cine_config(tmp_path, source, normalization="gaussian")
    settings = load_cine_settings(config_path)
    settings.latent_root.mkdir(parents=True, exist_ok=True)
    selection = {
        "scope": "subset",
        "window_stride": 32,
        "train": {"episode_indices": [0], "raw_starts": {"0": [1]},
                  "window_count": 1, "available_episode_count": 1},
        "validation": {"episode_indices": [2], "raw_starts": {"2": [1]},
                       "window_count": 1, "available_episode_count": 1},
    }
    sources = {
        "train": {"root": str(settings.train_root), "digest": "train-id"},
        "validation": {"root": str(settings.val_root), "digest": "val-id"},
    }
    manifest = {
        "complete": True,
        "raw_window_frames": 33,
        "sources": sources,
        "selection": selection,
        "action_semantics": "raw_joint_command",
        "action_normalization": "gaussian",
        "action_statistics": {
            "source": "train",
            "root": str(settings.train_root),
            "episode_indices": [0],
            "mean": [1.0] * 7,
            "std": [2.0] * 7,
        },
        "samples": {
            "train": [{"episode_index": 0, "raw_start": 1}],
            "validation": [{"episode_index": 2, "raw_start": 1}],
        },
    }
    manifest_path = settings.latent_root / "manifest.json"
    raw = json.dumps(manifest).encode()
    manifest_path.write_bytes(raw)
    train_repository = SimpleNamespace(episode_records=({"episode_index": 0, "length": 40},))
    val_repository = SimpleNamespace(episode_records=({"episode_index": 2, "length": 40},))
    data_config = SimpleNamespace(sample_stride=32)
    train_dataset = SimpleNamespace(
        manifest=manifest, repository=train_repository, data_config=data_config,
        entries=manifest["samples"]["train"],
    )
    val_dataset = SimpleNamespace(
        manifest=manifest, repository=val_repository, data_config=data_config,
        entries=manifest["samples"]["validation"],
    )
    monkeypatch.setattr(
        "open_wam.data.cine_v3_latent.build_cine_latent_train_val_datasets",
        lambda config: (train_dataset, val_dataset),
    )
    run_identity = {
        "cine_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "cine_source_identities": sources,
        "cine_selection": selection,
        "action_semantics": "raw_joint_command",
        "action_normalization": "gaussian",
    }

    restored = cine_inference.denormalize_cine_actions(
        torch.zeros((2, 7)), settings=settings, run_identity=run_identity
    )
    assert torch.equal(restored, torch.ones((2, 7)))

    with pytest.raises(ValueError, match="manifest"):
        cine_inference.denormalize_cine_actions(
            torch.zeros((2, 7)),
            settings=settings,
            run_identity={**run_identity, "cine_manifest_sha256": "stale"},
        )





def test_real_cine_dataset_trains_and_resumes_through_shared_runtime(tmp_path):
    import hashlib
    from dataclasses import replace

    import torch
    from tests.test_cine_v3 import cache_manifest, config as cine_data_config, raw_repo
    from tests.test_variational_native_pipeline import fixture
    from gradientwam.distributed_train import (
        _experiment_config_identity_sha256,
        _rank_state_path,
    )
    from gradientwam.runtime_factory import build_rank_aware_runtime
    from gradientwam.settings import (
        GradientWAMMethod,
        GradientWAMMethodConfig,
        TRAINABILITY_SCOPE_ID,
    )
    from gradientwam.cine_runner import _cine_loaders
    from open_wam.configs import resolve_experiment_config, validate_experiment_config_runtime_contract
    from open_wam.configs.enums import (
        BatchAdapterName,
        CheckpointMode,
        LoopPolicyName,
        StrategyName,
        TrainerAccelerator,
        TrainerPrecision,
        WandBMode,
    )
    from open_wam.training.launch import DistributedLaunchContext
    from open_wam.training.strategies import build_training_strategy

    torch.set_num_threads(1)
    train_root = raw_repo(tmp_path / "train")
    validation_root = raw_repo(tmp_path / "validation", action_offset=1000.0)
    cache_root = tmp_path / "latent-cache"
    manifest = cache_manifest(train_root, validation_root, cache_root)
    import json
    manifest["action_normalization"] = "none"
    (cache_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    data = cine_data_config(train_root, validation_root, cache_root)
    data = replace(
        data,
        sample_construction=replace(data.sample_construction, chunk_size=2, window_size=8),
    )

    *_, tiny_native = fixture(return_config=True)
    method_settings = SimpleNamespace(
        method_config=GradientWAMMethodConfig(method=GradientWAMMethod.BASELINE),
        seed=57,
        route_seed=57,
        checkpoint=tmp_path / "unused-public-checkpoint.pt",
        checkpoint_sha256="a" * 64,
        arm="baseline",
    )
    output_dir = tmp_path / "runtime-output"

    def config_for_steps(step_count):
        training = replace(
            tiny_native.training,
            chunk_size=2,
            window_size=8,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            warmup_steps=0,
            num_steps=step_count,
        )
        trainer = replace(
            tiny_native.trainer,
            accelerator=TrainerAccelerator.CPU,
            devices=1,
            precision=TrainerPrecision.FP32,
            strategy=StrategyName.SINGLE_DEVICE,
            batch_adapter=BatchAdapterName.LATENTS,
            loop_policy=LoopPolicyName.STEPS,
            enable_checkpointing=True,
            checkpoint_mode=CheckpointMode.FULL_TRAINING_STATE,
            checkpoint_dir=str(output_dir / "checkpoints"),
            default_root_dir=str(output_dir.parent),
            run_name=output_dir.name,
            save_interval=None,
            enable_jsonl_logging=False,
            enable_wandb=False,
            wandb_mode=WandBMode.DISABLED,
            export_runtime_backbone=False,
            limit_val_batches=0,
        )
        native = replace(
            tiny_native,
            data=data,
            action_decoder=replace(
                tiny_native.action_decoder,
                action_dim=7,
                action_horizon=36,
            ),
            training=training,
            inference=replace(tiny_native.inference, frame_chunk_size=9),
            trainer=trainer,
        )
        return validate_experiment_config_runtime_contract(resolve_experiment_config(native))

    def start_runtime(config, resume=None):
        context = DistributedLaunchContext.from_env()
        strategy = build_training_strategy(config.trainer, launch_context=context)
        train_loader, val_loader, train_dataset, val_dataset = _cine_loaders(
            config, strategy, seed=method_settings.seed
        )
        identity = {
            "arm": method_settings.arm,
            "gradientwam": method_settings.method_config.identity(),
            "trainability_scope": TRAINABILITY_SCOPE_ID,
            "initialization_seed": method_settings.seed,
            "rank_training_seed_rule": "seed_plus_rank",
            "route_seed": method_settings.route_seed,
            "public_video_checkpoint_sha256": method_settings.checkpoint_sha256,
            "experiment_config_sha256": _experiment_config_identity_sha256(config),
            "cine_manifest_sha256": hashlib.sha256(
                (cache_root / "manifest.json").read_bytes()
            ).hexdigest(),
            "cine_source_identities": manifest["sources"],
            "cine_selection": manifest["selection"],
            "world_size": strategy.world_size,
            "per_rank_batch_size": config.data.train_batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "global_batch_size": (
                strategy.world_size
                * config.data.train_batch_size
                * config.training.gradient_accumulation_steps
            ),
        }
        runtime = build_rank_aware_runtime(
            settings=method_settings,
            config=config,
            strategy=strategy,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=output_dir,
            dataset_artifacts=(),
            identity=identity,
            heldout_episode_ids=sorted(
                {int(entry["episode_index"]) for entry in val_dataset.entries}
            ),
            eval_seed=88,
            resume=resume,
            heldout_split_label="cine_validation_root",
            load_public_video_checkpoint=False,
        )
        return runtime, identity

    short, short_identity = start_runtime(config_for_steps(1))
    initial = {
        name: value.detach().clone()
        for name, value in short.strategy.unwrap_model(short.model).named_parameters()
        if value.requires_grad
    }
    state1 = short.run()
    checkpoint = state1.last_checkpoint_path
    assert checkpoint is not None and Path(checkpoint).is_dir()
    assert (Path(checkpoint) / "full_training_state.pt").is_file()
    assert _rank_state_path(Path(checkpoint)).is_file()
    updated = {
        name: value.detach().clone()
        for name, value in short.strategy.unwrap_model(short.model).named_parameters()
        if value.requires_grad
    }
    assert any(not torch.equal(initial[name], updated[name]) for name in initial)

    continued, continued_identity = start_runtime(
        config_for_steps(2),
        resume=str(checkpoint),
    )
    assert continued_identity == short_identity
    state2 = continued.run()

    assert state2.optimizer_step == 2
    assert state2.last_checkpoint_path is not None
