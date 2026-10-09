"""Delivery checks: tiny CPU only, never download assets or construct full model."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gradientwam.settings import ARMS, LEGACY_ARMS, GradientWAMMethod, load_settings, load_episode_split
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
    changed_method = (
        GradientWAMMethod.BASELINE
        if settings.method_config.method is GradientWAMMethod.CAGRAD
        else GradientWAMMethod.CAGRAD
    )
    changed = replace(settings, method_config=replace(
        settings.method_config, method=changed_method
    ))
    assert changed.identity() != settings.identity()


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


@pytest.mark.parametrize('arm', LEGACY_ARMS)
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


@pytest.mark.parametrize('method', list(GradientWAMMethod))
def test_new_method_native_tiny_build_and_optimizer_update(tmp_path, monkeypatch, method):
    from open_wam.data import LatentWAMBatch
    from tests.test_variational_native_pipeline import fixture
    from open_wam.training import strategies
    from gradientwam.settings import GradientWAMMethodConfig, TRAINABILITY_SCOPE_ID

    *_, policy_batch, latent, text, config = fixture(return_config=True)
    def cpu_strategy(**_kwargs):
        return SingleDeviceStrategy(
            accelerator=enums.TrainerAccelerator.CPU,
            precision=enums.TrainerPrecision.FP32,
        )
    monkeypatch.setattr(strategies, 'SingleDeviceStrategy', cpu_strategy)
    method_config = GradientWAMMethodConfig(method=method)
    stack = smoke._build_stack(
        config,
        base_checkpoint=tmp_path/'unused.pt',
        load_public_base=False,
        checkpoint_sha256='a'*64,
        arm=method.value,
        route_seed=31,
        torch=torch,
        method_config=method_config,
    )
    assert stack['sharing_audit']['scope_id'] == TRAINABILITY_SCOPE_ID
    assert stack['parameter_manifest']
    assert all(entry['dtype'] == 'torch.float32' for entry in stack['parameter_manifest'])
    policy = stack['pipeline'].policy_variant
    assert not hasattr(policy, 'routing_controller')
    assert not any('private_video_to_k' in name for name, _ in stack['pipeline'].named_parameters())
    assert hasattr(policy, 'vrfm') is method_config.uses_vrfm
    if method_config.uses_cagrad:
        assert stack['cagrad_candidates']
        if method_config.uses_vrfm:
            assert not {id(parameter) for parameter in policy.vrfm.posterior.parameters()} & {
                id(parameter) for parameter in stack['cagrad_candidates']
            }

    state = TrainState()
    latent_batch = LatentWAMBatch(
        video_latents=latent,
        actions=policy_batch.actions,
        action_mask=policy_batch.action_mask,
        state=policy_batch.state,
        task_text=(None,),
        text_context=text,
        condition_latents=policy_batch.extra['condition_latents'],
        proprio_context_frames=policy_batch.extra['proprio_context_frames'],
        proprio_context_frames_mask=policy_batch.extra['proprio_context_frames_mask'],
        metadata=({'valid_video_frames': 4},),
    )
    device_batch = stack['adapter'].move_to_device(latent_batch, stack['strategy'].device)
    if method is GradientWAMMethod.VRFM_CAGRAD:
        diagnostic = stack['executor'].forward_train(device_batch)
        trainable = [
            (name, parameter)
            for name, parameter in stack['pipeline'].named_parameters()
            if parameter.requires_grad
        ]
        video_grads = torch.autograd.grad(
            diagnostic.task_losses['video'],
            tuple(parameter for _, parameter in trainable),
            retain_graph=True,
            allow_unused=True,
        )
        action_grads = torch.autograd.grad(
            diagnostic.task_losses['action'],
            tuple(parameter for _, parameter in trainable),
            retain_graph=True,
            allow_unused=True,
        )
        nonzero_overlap = {
            id(parameter)
            for (_, parameter), video_grad, action_grad in zip(
                trainable, video_grads, action_grads, strict=True
            )
            if video_grad is not None and action_grad is not None
            and bool(torch.count_nonzero(video_grad).item())
            and bool(torch.count_nonzero(action_grad).item())
        }
        posterior_ids = {
            id(parameter) for parameter in policy.vrfm.posterior.parameters()
        }
        candidate_ids = {id(parameter) for parameter in stack['cagrad_candidates']}
        assert nonzero_overlap <= candidate_ids | posterior_ids, (
            'CAGrad candidates miss a parameter with nonzero gradients from both tasks'
        )
        action_only = [
            (name, parameter, video_grad, action_grad)
            for (name, parameter), video_grad, action_grad in zip(
                trainable, video_grads, action_grads, strict=True
            )
            if 'action_expert.' in name or '.action_block.' in name
            or 'vrfm.action_projection.' in name
        ]
        assert action_only
        assert all(
            gradient is None or not bool(torch.count_nonzero(gradient).item())
            for _, _, gradient, _ in action_only
        ), 'video-only attention mask must sever action-stream influence on video loss'
        assert any(
            gradient is not None and bool(torch.count_nonzero(gradient).item())
            for _, _, _, gradient in action_only
        ), 'action-only generator parameters must retain their native action gradients'
        action_expert_blocks = [
            item for item in action_only
            if 'action_expert.' in item[0] or '.action_block.' in item[0]
        ]
        action_projection = [
            item for item in action_only if 'vrfm.action_projection.' in item[0]
        ]
        max_abs = lambda gradient: 0.0 if gradient is None else float(gradient.detach().abs().max().item())
        evidence = {
            'action_expert_blocks_video_max_abs': max(
                (max_abs(video_grad) for _, _, video_grad, _ in action_expert_blocks),
                default=0.0,
            ),
            'action_expert_blocks_action_max_abs': max(
                (max_abs(action_grad) for _, _, _, action_grad in action_expert_blocks),
                default=0.0,
            ),
            'action_projection_video_max_abs': max(
                (max_abs(video_grad) for _, _, video_grad, _ in action_projection),
                default=0.0,
            ),
            'action_projection_action_max_abs': max(
                (max_abs(action_grad) for _, _, _, action_grad in action_projection),
                default=0.0,
            ),
        }
        print(f'VIDEO_ONLY_MASK_GRAD_EVIDENCE {evidence}')
    report = smoke._update(stack, device_batch, state, torch=torch, deadline=float('inf'))
    assert report['optimizer_step'] == 1
    if method_config.uses_cagrad:
        assert report['cagrad']['applied'] is True
        assert report['cagrad']['common_parameter_numel'] > 0
    if method is GradientWAMMethod.VRFM_CAGRAD:
        final_video = policy.packed_block_stack.packed_blocks[-1].video_block
        graph_common_numel = sum(
            parameter.numel()
            for (_, parameter), video_grad, action_grad in zip(
                trainable, video_grads, action_grads, strict=True
            )
            if id(parameter) in candidate_ids
            and video_grad is not None and action_grad is not None
        )
        assert report['cagrad']['common_parameter_numel'] == graph_common_numel

        # Strict step-1 -> step-2 resume also restores the RNG used by the VRFM
        # posterior sample and starts with an empty CAGrad accumulation window.
        from gradientwam import checkpoint
        accumulator = stack['cagrad_accumulator']
        assert all(not gradient.any() for gradient in accumulator.video_grads)
        assert all(not gradient.any() for gradient in accumulator.action_grads)
        metadata = {
            'gradientwam': method_config.identity(),
            'trainability_scope': TRAINABILITY_SCOPE_ID,
        }
        checkpoint_path = tmp_path/'vrfm_cagrad_step1.pt'
        checkpoint.save_step1(
            checkpoint_path,
            model=stack['model'],
            optimizer=stack['optimizer'],
            scheduler=stack['scheduler'],
            strategy=stack['strategy'],
            train_state=state,
            sampler_state={'next_microbatch_cursor': 10},
            extra_generator_states={},
            metadata=metadata,
            cuda_devices=(),
            max_bytes=2<<30,
        )
        direct_report = smoke._update(
            stack, device_batch, state, torch=torch, deadline=float('inf')
        )

        torch.manual_seed(717)
        resumed_stack = smoke._build_stack(
            config,
            base_checkpoint=tmp_path/'unused.pt',
            load_public_base=False,
            checkpoint_sha256='a'*64,
            arm=method.value,
            route_seed=31,
            torch=torch,
            method_config=method_config,
        )
        restored = checkpoint.load_step1(
            checkpoint_path,
            model=resumed_stack['model'],
            optimizer=resumed_stack['optimizer'],
            scheduler=resumed_stack['scheduler'],
            strategy=resumed_stack['strategy'],
            expected_metadata=metadata,
            cuda_devices=(),
        )
        resumed_state = restored['train_state']
        assert resumed_state.optimizer_step == 1 and resumed_state.global_step == 10
        assert all(not gradient.any() for gradient in resumed_stack['cagrad_accumulator'].video_grads)
        assert all(not gradient.any() for gradient in resumed_stack['cagrad_accumulator'].action_grads)
        checkpoint.restore_rng(restored['rng_state'], cuda_devices=())
        resumed_report = smoke._update(
            resumed_stack,
            device_batch,
            resumed_state,
            torch=torch,
            deadline=float('inf'),
        )
        assert resumed_report == direct_report
        assert resumed_state.optimizer_step == 2
        for name, value in stack['model'].state_dict().items():
            torch.testing.assert_close(
                value,
                resumed_stack['model'].state_dict()[name],
                rtol=0,
                atol=0,
            )


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
