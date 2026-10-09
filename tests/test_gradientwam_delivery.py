"""Delivery checks: tiny CPU only, never download assets or construct full model."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gradientwam.settings import ARMS, load_settings, load_episode_split
from gradientwam import train_smoke as smoke
from gradientwam import checkpoint
from open_wam.configs import TrainingConfig, enums
from open_wam.training.optim import build_optimizer, build_scheduler
from open_wam.training.state import TrainState
from open_wam.training.strategies import SingleDeviceStrategy

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def environment(tmp_path, monkeypatch):
    for name, value in {
        'CHECKPOINT': str(tmp_path/'assets/model.pt'), 'CHECKPOINT_SHA256': 'a'*64,
        'DATASET_ROOT': str(tmp_path/'dataset'), 'FRONTEND_ROOT': str(tmp_path/'frontend'),
        'TOKENIZER_ROOT': str(tmp_path/'tokenizer'), 'PREPARATION_ROOT': str(tmp_path/'prepared'),
        'OUTPUT_ROOT': str(tmp_path/'output'), 'PROMPT_FINGERPRINT': 'b'*64,
    }.items():
        monkeypatch.setenv('GW_'+name, value)
    return tmp_path


@pytest.mark.parametrize('arm', ARMS)
def test_four_configs_parse_without_assets(environment, arm):
    settings = load_settings(ROOT/f'configs/gradientwam/{arm}.yaml')
    config = settings.native_config()
    assert settings.arm == arm
    assert config.backbone.num_layers == config.policy_variant.num_action_layers == 30
    assert config.training.gradient_accumulation_steps == 10
    assert not config.backbone.load_reference_core_weights
    assert not config.backbone.load_wan_vae_frontend
    assert not config.backbone.load_text_conditioning
    assert str(config.policy_variant.program) == 'decoupled_same_step'
    assert not settings.checkpoint.exists()
    assert replace(settings, output_root=environment/'resume').identity() == settings.identity()
    assert replace(settings, arm='different').identity() != settings.identity()


def test_preparation_print_only_and_source_overlap_rejected(environment, monkeypatch):
    from gradientwam.preprocess_smoke import prepare
    path = ROOT/'configs/gradientwam/variational_sharing.yaml'
    monkeypatch.setenv('GW_PROMPT_FINGERPRINT', '')
    settings = load_settings(path)
    result = prepare(settings)
    assert result['status'] == 'commands_only' and len(result['commands']) == 3
    assert all(Path(cmd[1]).is_file() for cmd in result['commands'])
    assert not settings.preparation_root.exists()
    monkeypatch.setenv('GW_PREPARATION_ROOT', str(settings.dataset_root/'inside'))
    with pytest.raises(ValueError, match='separate from source'):
        load_settings(path)


def test_split_preparation_exact_ranges_and_no_condition_truncation(environment):
    from gradientwam.preprocess_smoke import prepare
    path=environment/'episodes.json'
    path.write_text(json.dumps({'schema_version':1,'train_episode_ids':[5,2,1],'heldout_episode_ids':[9]}))
    settings=load_settings(ROOT/'configs/gradientwam/variational_sharing.yaml')
    result=prepare(settings,episodes_file=path)
    assert result['episode_ids']==[1,2,5,9]
    encoders=result['commands'][:-2]
    ranges=[(int(c[c.index('--start-episode')+1]),int(c[c.index('--episodes')+1])) for c in encoders]
    assert ranges==[(1,2),(5,1),(9,1)]
    assert '--max-files' not in result['commands'][-2]
    assert not settings.preparation_root.exists()


@pytest.mark.parametrize('value',[{'schema_version':1,'train_episode_ids':[1],'heldout_episode_ids':[1]}, {'schema_version':1,'train_episode_ids':[],'heldout_episode_ids':[2]}, {'schema_version':1,'train_episode_ids':[True],'heldout_episode_ids':[2]}, {'schema_version':1,'train_episode_ids':[1,1],'heldout_episode_ids':[2]}])
def test_invalid_episode_split_rejected(tmp_path,value):
    path=tmp_path/'episodes.json';path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_episode_split(path)


@pytest.mark.parametrize('arm', ARMS)
def test_portable_build_attaches_arm_before_optimizer_on_native_tiny_cpu(tmp_path, monkeypatch, arm):
    from tests.test_variational_native_pipeline import fixture
    from open_wam.models.visual_tower import tower as tower_module
    from open_wam.models.visual_tower import public_pretraining
    from open_wam.models.policy_variants.dual_expert import variational_sharing
    from open_wam.pipelines import factory
    from open_wam.training import optim, strategies
    *_, config = fixture(return_config=True)
    config = replace(config,
        data=replace(config.data, action_schema=replace(config.data.action_schema, action_dim=7, state_dim=8)),
        action_decoder=replace(config.action_decoder, action_dim=7))
    source = tower_module.VisualTower(config.backbone, action_dim=7, state_dim=8)
    path = tmp_path/'base.pt'
    torch.save({'model_state_dict': {'visual_tower.'+k:v for k,v in source.state_dict().items()}}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    events = []
    real_tower, real_load, real_factory = tower_module.VisualTower, public_pretraining.load_public_video_checkpoint_into_tower, factory.build_variant_pipeline_from_config
    real_configure, real_manifest, real_optim = variational_sharing.configure_variational_sharing, smoke._parameter_manifest, optim.build_optimizer

    def tower(*a, **kw):
        events.append('tower'); return real_tower(*a, **kw)
    def load(*a, **kw):
        events.append('load'); assert kw['expected_sha256'] == digest
        return real_load(*a, **kw)
    def build(*a, **kw):
        events.append('factory'); return real_factory(*a, **kw)
    def configure(pipeline, **kw):
        events.append('configure'); assert kw['arm'] == arm and kw['expected_layers'] == 30
        # Only test geometry changes. The actual method implementation remains native.
        kw['expected_layers'] = 2
        return real_configure(pipeline, **kw)
    def optimizer(*a, **kw):
        events.append('optimizer'); return real_optim(*a, **kw)
    def strategy(**kw):
        assert kw == dict(accelerator=enums.TrainerAccelerator.GPU, precision=enums.TrainerPrecision.BF16)
        return SingleDeviceStrategy(accelerator=enums.TrainerAccelerator.CPU, precision=enums.TrainerPrecision.FP32)
    def manifest(model, *, expected_numel=None):
        assert expected_numel == (2_600_806_456 if arm == 'variational_sharing' else None)
        return real_manifest(model)
    monkeypatch.setattr(tower_module, 'VisualTower', tower)
    monkeypatch.setattr(public_pretraining, 'load_public_video_checkpoint_into_tower', load)
    monkeypatch.setattr(factory, 'build_variant_pipeline_from_config', build)
    monkeypatch.setattr(variational_sharing, 'configure_variational_sharing', configure)
    monkeypatch.setattr(optim, 'build_optimizer', optimizer)
    monkeypatch.setattr(strategies, 'SingleDeviceStrategy', strategy)
    monkeypatch.setattr(smoke, '_parameter_manifest', manifest)
    stack = smoke._build_stack(config, base_checkpoint=path, load_public_base=True,
                              checkpoint_sha256=digest, arm=arm, route_seed=17, torch=torch)
    assert events == ['tower','load','factory','configure','optimizer']
    assert stack['sharing_audit']['duplicate_parameter_ids'] == []
    events.clear()
    smoke._build_stack(config, base_checkpoint=path, load_public_base=False,
                       checkpoint_sha256=digest, arm=arm, route_seed=17, torch=torch)
    assert events == ['tower','factory','configure','optimizer']


def test_actual_update_and_checkpoint_match_uninterrupted_tiny_cpu(tmp_path):
    torch.manual_seed(901)
    config = TrainingConfig(gradient_accumulation_steps=10, learning_rate=1e-5,
        beta1=.9, beta2=.95, weight_decay=.1, warmup_steps=10, scheduler_name='constant_with_warmup')
    def build():
        model = torch.nn.Linear(1,1)
        strategy = SingleDeviceStrategy(accelerator=enums.TrainerAccelerator.CPU, precision=enums.TrainerPrecision.FP32)
        optimizer = build_optimizer(model, config)
        executor = SimpleNamespace(forward_train=lambda _: SimpleNamespace(loss=model(torch.randn(1,1)).square().mean()))
        return dict(model=model,strategy=strategy,optimizer=optimizer,scheduler=build_scheduler(optimizer,config),executor=executor)
    def update(stack,state):
        return smoke._update(stack,None,state,torch=torch,deadline=float('inf'))
    stack=build();state=TrainState()
    first=update(stack,state)
    path=tmp_path/'step1.pt'
    kwargs=dict(sampler_state={'cursor':10},extra_generator_states={},metadata={'arm':'tiny'},cuda_devices=())
    checkpoint.save_step1(path,model=stack['model'],optimizer=stack['optimizer'],scheduler=stack['scheduler'],
        strategy=stack['strategy'],train_state=state,max_bytes=1<<20,**kwargs)
    direct=update(stack,state)
    resumed=build()
    with pytest.raises(ValueError):
        checkpoint.load_step1(path,model=resumed['model'],optimizer=resumed['optimizer'],scheduler=resumed['scheduler'],
            strategy=resumed['strategy'],expected_metadata={'arm':'wrong'},cuda_devices=())
    restored=checkpoint.load_step1(path,model=resumed['model'],optimizer=resumed['optimizer'],scheduler=resumed['scheduler'],
        strategy=resumed['strategy'],expected_metadata=kwargs['metadata'],cuda_devices=())
    assert restored['sampler_state']=={'cursor':10}
    checkpoint.restore_rng(restored['rng_state'],cuda_devices=())
    second=update(resumed,restored['train_state'])
    assert second == direct
    assert first['applied_learning_rate'] == pytest.approx(1e-6)
    assert second['applied_learning_rate'] == pytest.approx(2e-6)
    assert restored['train_state'].optimizer_step == 2
    assert restored['train_state'].next_batch_index == 20
    for name,value in stack['model'].state_dict().items():
        torch.testing.assert_close(value,resumed['model'].state_dict()[name],rtol=0,atol=0)
