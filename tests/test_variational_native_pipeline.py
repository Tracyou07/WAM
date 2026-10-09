"""Native preparation, decoder and inference; synthetic CPU correctness only."""
import copy
import io
import random

import pytest
import torch

from open_wam.configs import (ActionSchemaConfig, DualExpertActionDecoderConfig,
    DualExpertPolicyConfig, ExperimentConfig, InferenceConfig, RobotWinDataConfig, TrainingConfig)
from open_wam.models.policy_variants.contracts import PolicyTrainBatch, PolicyInferContext
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
from open_wam.pipelines import build_variant_pipeline_from_config
from open_wam.models.policy_variants.dual_expert import variational_sharing as extension


def fixture(return_config=False):
    torch.set_num_threads(1)
    torch.manual_seed(109)
    config=ExperimentConfig(
        data=RobotWinDataConfig(num_frames=4,action_schema=ActionSchemaConfig(
            action_dim=4,action_horizon=4,state_dim=4,state_horizon=1)),
        backbone=SharedVideoTransformerConfig(implementation='shared_transformer',
            hidden_size=32,num_layers=2,num_heads=4,attention_head_dim=8,ffn_dim=64,
            text_dim=16,freq_dim=8,load_reference_core_weights=False,
            load_text_conditioning=False,load_wan_vae_frontend=False),
        policy_variant=DualExpertPolicyConfig(hidden_size=32,program='decoupled_same_step',
            video_prefix_frames=1,num_action_layers=2,
            sequence_contract='legacy_prefix_single_frame_perchunk_proprio',
            proprio_context_mode='per_chunk_additive',context_condition_latent_source='single_frame_condition_latent',
            history_stream_visibility='video_only',use_condition_latents=True,require_condition_latents=True,
            noisy_video_condition_prob=0.,joint_timestep_coupling='independent'),
        action_decoder=DualExpertActionDecoderConfig(hidden_size=32,action_dim=4,action_horizon=4),
        training=TrainingConfig(chunk_size=2,window_size=8,enabled_objectives=('action','latent'),
            action_loss_weight=1.,latent_loss_weight=1.),
        inference=InferenceConfig(frame_chunk_size=2,video_num_inference_steps=3,
            action_num_inference_steps=3,use_cache=True,guidance_scale=1.,action_guidance_scale=1.))
    pipeline=build_variant_pipeline_from_config(config)
    latents=torch.randn(1,48,4,4,4)
    prefix=torch.randn(1,48,1,4,4)
    batch=PolicyTrainBatch(actions=torch.randn(1,4,4),state=torch.randn(1,4),extra={
        'condition_latents':prefix,'proprio_context_frames':torch.randn(1,4,4),
        'proprio_context_frames_mask':torch.ones(1,4,4)})
    result=(pipeline,batch,latents,torch.randn(1,3,16))
    return (*result,config) if return_config else result


def configure(pipeline,arm):
    return extension.configure_variational_sharing(pipeline,arm=arm,expected_layers=2,route_seed=617)


def train(pipeline,batch,latent,text):
    random.seed(213)
    torch.manual_seed(213)
    return pipeline.forward_train_from_latents(latent,batch,text_context=text)


@pytest.mark.parametrize('arm',['native_joint','same_capacity_deterministic_kv_blend',
    'variational_sharing','forced_private_world_gradient_off'])
def test_native_caller_four_arms_and_ownership(arm):
    pipeline,batch,latent,text=fixture()
    audit=configure(pipeline,arm)
    output=train(pipeline,batch,latent,text)
    assert torch.isfinite(output.decoder_output.loss)
    assert output.policy_output.aux['current_block_coupling']=='decoupled_same_step'
    assert output.policy_output.aux['video_condition_source']=='condition_latents_prefix'
    assert audit['last_layer_index']==1 and audit['duplicate_parameter_ids']==[]
    names=[n for n,p in pipeline.named_parameters() if p.requires_grad and '.video_block.' in n]
    assert names and all('packed_blocks.1.video_block.attn1.to_k.' in n or
        'packed_blocks.1.video_block.attn1.to_v.' in n for n in names)
    assert len(pipeline.visual_tower.core.blocks)==0
    assert len(pipeline.policy_variant.action_expert.blocks)==0
    output.decoder_output.loss.backward()


def test_initial_pipeline_value_common_action_gradient_and_prior_target_invariance():
    native,batch,latent,text=fixture()
    sharing=copy.deepcopy(native)
    configure(native,'native_joint')
    configure(sharing,'variational_sharing')
    a=train(native,batch,latent,text)
    b=train(sharing,batch,latent,text)
    torch.testing.assert_close(a.decoder_output.loss,b.decoder_output.loss,rtol=2e-5,atol=2e-6)
    a.decoder_output.loss.backward()
    b.decoder_output.loss.backward()
    other=dict(sharing.named_parameters())
    for name,param in native.named_parameters():
        if 'action_expert.' in name or '.action_block.' in name:
            if param.grad is not None:
                torch.testing.assert_close(param.grad,other[name].grad,rtol=3e-5,atol=3e-6)
    posterior=b.decoder_output.metrics['route_posterior_shared_mean']
    torch.testing.assert_close(posterior,torch.tensor(.5))
    controller=sharing.policy_variant.routing_controller
    with torch.no_grad():
        controller.prior.linear.weight.fill_(.15)
    first=train(sharing,batch,latent,text).policy_output.decoder_artifacts.payload.action.prior_shared
    changed=copy.copy(batch)
    changed.actions=batch.actions+100
    second=train(sharing,changed,latent+1000,text).policy_output.decoder_artifacts.payload.action.prior_shared
    torch.testing.assert_close(first,second,rtol=0,atol=0)


def test_actual_native_predictions_preserve_private_phi_and_world_eta_isolation():
    pipeline,batch,latent,text=fixture()
    configure(pipeline,'variational_sharing')
    output=train(pipeline,batch,latent,text)
    artifacts=output.policy_output.decoder_artifacts.payload
    last=pipeline.policy_variant.packed_block_stack.packed_blocks[-1]
    phi=tuple(last.video_block.attn1.to_k.parameters())+tuple(last.video_block.attn1.to_v.parameters())
    eta=tuple(last.private_video_to_k.parameters())+tuple(last.private_video_to_v.parameters())
    private=artifacts.action.private_flow_pred.float().square().mean()
    assert all(g is None or not g.any() for g in torch.autograd.grad(private,phi,allow_unused=True,retain_graph=True))
    assert any(g is not None and g.any() for g in torch.autograd.grad(private,eta,allow_unused=True,retain_graph=True))
    shared=artifacts.action.flow_pred.float().square().mean()
    assert any(g is not None and g.any() for g in torch.autograd.grad(shared,phi,allow_unused=True,retain_graph=True))
    world=artifacts.video.flow_pred.float().square().mean()
    assert all(g is None or not g.any() for g in torch.autograd.grad(world,eta,allow_unused=True))


def test_bfloat16_cpu_caller_has_float32_evidence_and_finite_backward():
    pipeline,batch,latent,text=fixture()
    configure(pipeline,'variational_sharing')
    with torch.autocast('cpu',dtype=torch.bfloat16):
        output=train(pipeline,batch,latent,text)
    assert output.decoder_output.loss.dtype==torch.float32
    output.decoder_output.loss.backward()
    gradients=[p.grad for p in pipeline.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


def test_native_state_dict_stays_tensor_only_for_official_export():
    pipeline,_,_,_=fixture()
    configure(pipeline,'variational_sharing')
    pipeline.policy_variant.routing_controller.sampler.begin_chunk(torch.tensor([.5]))
    assert all(isinstance(value,torch.Tensor) for value in pipeline.state_dict().values())


def test_public_video_checkpoint_loads_before_native_attach(tmp_path):
    from open_wam.models.visual_tower import VisualTower
    pipeline,batch,latent,text,config=fixture(return_config=True)
    source=VisualTower(config.backbone,action_dim=1,state_dim=1)
    path=tmp_path/'video_only.pt'
    torch.save({'model_state_dict':{'visual_tower.'+key:value for key,value in source.state_dict().items()}},path)
    target=VisualTower(config.backbone,action_dim=4,state_dim=4)
    report=extension.load_public_video_checkpoint_into_tower(target,path)
    assert report['missing_required_video_keys']==[] and report['action_policy_pretrained_keys']==0
    assert report['source_auxiliary_keys_ignored']  # source uses one-dimensional dummy action/state
    expected=source.core.blocks[-1].attn1.to_v.weight.detach().clone()
    original_id=id(target.core.blocks[-1].attn1.to_v.weight)
    assembled=build_variant_pipeline_from_config(config,visual_tower=target)
    configure(assembled,'variational_sharing')
    last=assembled.policy_variant.packed_block_stack.packed_blocks[-1]
    assert id(last.video_block.attn1.to_v.weight)==original_id
    torch.testing.assert_close(last.video_block.attn1.to_v.weight,expected,rtol=0,atol=0)
    torch.testing.assert_close(last.private_video_to_v.weight,expected,rtol=0,atol=0)
    assert torch.isfinite(train(assembled,batch,latent,text).decoder_output.loss)
    with pytest.raises(ValueError,match='before.*attach'):
        extension.load_public_video_checkpoint_into_tower(target,path)


def test_public_video_loader_rejects_missing_video_projection(tmp_path):
    from open_wam.models.visual_tower import VisualTower
    *_,config=fixture(return_config=True)
    target=VisualTower(config.backbone,action_dim=4,state_dim=4)
    state={'visual_tower.'+key:value for key,value in target.state_dict().items()}
    state.pop('visual_tower.core.blocks.1.attn1.to_k.weight')
    path=tmp_path/'incomplete.pt'; torch.save({'model_state_dict':state},path)
    with pytest.raises(ValueError,match='video'):
        extension.load_public_video_checkpoint_into_tower(target,path)


def test_preloaded_tower_factory_rejects_config_or_attachment_mismatch():
    from open_wam.models.visual_tower import VisualTower
    pipeline,*_,config=fixture(return_config=True)
    wrong=VisualTower(config.backbone,action_dim=1,state_dim=4)
    with pytest.raises(ValueError,match='match resolved'):
        build_variant_pipeline_from_config(config,visual_tower=wrong)
    with pytest.raises(ValueError,match='before policy attach'):
        build_variant_pipeline_from_config(config,visual_tower=pipeline.visual_tower)


def test_inference_native_ode_samples_once_and_cache_receives_fixed_route(monkeypatch):
    pipeline,batch,latent,text=fixture()
    configure(pipeline,'variational_sharing')
    controller=pipeline.policy_variant.routing_controller
    calls=[]
    original=controller.sampler.begin_chunk
    def begin(*args,**kwargs):
        value=original(*args,**kwargs)
        calls.append(value.clone())
        return value
    monkeypatch.setattr(controller.sampler,'begin_chunk',begin)
    routing_calls=[]
    last=pipeline.policy_variant.packed_block_stack.packed_blocks[-1]
    def hook(module,args,kwargs):
        if kwargs.get('routing_mode') is not None:
            routing_calls.append(kwargs['route_choices'].clone())
    handle=last.register_forward_pre_hook(hook,with_kwargs=True)
    output=pipeline.forward_infer_step_from_latents(latent[:,:,:1],
        PolicyInferContext(state=batch.state,initial_route_uniform=torch.tensor([.1])),text_context=text)
    handle.remove()
    assert len(calls)==1 and len(routing_calls)>=3
    assert all(torch.equal(calls[0],route) for route in routing_calls)
    assert torch.isfinite(output.decoder_output.action_pred).all()


def test_full_cpu_adamw_scheduler_data_and_route_rng_resume():
    pipeline,batch,latent,text=fixture()
    configure(pipeline,'variational_sharing')
    parameters=[p for p in pipeline.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=1e-4)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.9)
    output=train(pipeline,batch,latent,text)
    output.decoder_output.loss.backward()
    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    controller=pipeline.policy_variant.routing_controller
    controller.sampler.begin_chunk(torch.tensor([.5]))
    buffer=io.BytesIO()
    torch.save({'model':pipeline.state_dict(),'optimizer':optimizer.state_dict(),
        'scheduler':scheduler.state_dict(),'data_position':1,'torch_rng':torch.get_rng_state(),
        'python_rng':random.getstate()},buffer)
    expected_route=controller.sampler.begin_chunk(torch.tensor([.5]))
    next_output=train(pipeline,batch,latent,text)
    next_output.decoder_output.loss.backward(); optimizer.step(); scheduler.step()
    expected={n:p.detach().clone() for n,p in pipeline.named_parameters()}
    restored,_,_,_=fixture(); configure(restored,'variational_sharing')
    restored_optimizer=torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad],lr=1e-4)
    restored_scheduler=torch.optim.lr_scheduler.StepLR(restored_optimizer,step_size=1,gamma=.9)
    buffer.seek(0); state=torch.load(buffer,weights_only=True)
    restored.load_state_dict(state['model'],strict=True)
    restored_optimizer.load_state_dict(state['optimizer']); restored_scheduler.load_state_dict(state['scheduler'])
    torch.set_rng_state(state['torch_rng']); random.setstate(state['python_rng'])
    assert state['data_position']==1
    actual_route=restored.policy_variant.routing_controller.sampler.begin_chunk(torch.tensor([.5]))
    assert torch.equal(actual_route,expected_route)
    resumed=train(restored,batch,latent,text)
    resumed.decoder_output.loss.backward(); restored_optimizer.step(); restored_scheduler.step()
    for name,param in restored.named_parameters():
        torch.testing.assert_close(param,expected[name],rtol=0,atol=0)
    assert scheduler.state_dict()==restored_scheduler.state_dict()
