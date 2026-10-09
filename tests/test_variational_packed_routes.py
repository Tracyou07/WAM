"""Actual native paired blocks: routing, ownership, gradients and cache parity."""
import pytest
import torch

import open_wam.configs
from open_wam.models.common.denoising_cache import DenoisingCache
from open_wam.models.policy_variants.dual_expert.attention_packed import build_dual_expert_packed_coupling_attention_profile
from open_wam.models.policy_variants.dual_expert.packed_block import DualExpertPackedBlockStack
from tests.test_dual_expert_packed_block import _make_video_core, _make_action_expert, _build_inputs


def case(batch=1):
    torch.set_num_threads(1)
    torch.manual_seed(71)
    stack = DualExpertPackedBlockStack(_make_video_core().blocks, _make_action_expert().blocks).eval()
    inputs = _build_inputs(batch=batch, seed=83)
    profile = build_dual_expert_packed_coupling_attention_profile(
        num_video_frames=3, video_tokens_per_frame=1, num_action_frames=2,
        action_tokens_per_frame=1, chunk_size_frames=1, prefix_condition_frames=1,
        current_block_coupling='decoupled_same_step', history_stream_visibility='video_only',
        device=torch.device('cpu'), build_dense_masks=True, build_flex_masks=False)
    mask = profile.self_attention_mask
    assert not mask[:6, 6:].any()
    assert mask[6:, :6].any()
    inputs['video_attention_mask'] = mask[:6][None,None]
    inputs['action_attention_mask'] = mask[6:][None,None]
    for block in stack.packed_blocks:
        block.video_block.requires_grad_(False)
    for projection in (stack.packed_blocks[-1].video_block.attn1.to_k, stack.packed_blocks[-1].video_block.attn1.to_v):
        projection.requires_grad_(True)
    return stack, inputs, profile


def enable(stack):
    stack.enable_private_video_kv(expected_layers=2)
    return stack.packed_blocks[-1]


def test_last_layer_only_independent_values_and_storage():
    stack, _, _ = case()
    last = enable(stack)
    assert not hasattr(stack.packed_blocks[0], 'private_video_to_k')
    for name in ('k','v'):
        original = getattr(last.video_block.attn1, 'to_'+name)
        private = getattr(last, 'private_video_to_'+name)
        for a,b in zip(original.parameters(), private.parameters(), strict=True):
            assert a is not b and a.data_ptr() != b.data_ptr()
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert not any(p.requires_grad for p in last.private_video_norm_k.parameters())
    with pytest.raises(ValueError, match='layer'):
        stack.enable_private_video_kv(expected_layers=30)


def test_initial_native_parity_and_world_executed_once():
    stack, inputs, _ = case()
    baseline = stack(**inputs)
    last = enable(stack)
    calls = {'video':0, 'action':0}
    handles=[]
    for name, block in [('video',last.video_block),('action',last.action_block)]:
        def hook(module, args, result, stream=name):
            calls[stream] += 1
        handles.append(block.ffn.register_forward_hook(hook))
    world, (private,shared) = stack(**inputs, routing_mode='two_routes')
    assert calls == {'video':1, 'action':2}
    for actual,expected in [(world,baseline[0]),(private,baseline[1]),(shared,baseline[1])]:
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    for handle in handles:
        handle.remove()
    for mode in ('private','blend','selected'):
        _, action = stack(**inputs, routing_mode=mode, prior_shared=torch.tensor([.5]), route_choices=torch.tensor([0]))
        torch.testing.assert_close(action,baseline[1],rtol=1e-5,atol=1e-6)


def test_true_two_layer_graph_private_phi_zero_shared_phi_nonzero():
    stack, inputs, _ = case()
    last = enable(stack)
    phi = tuple(last.video_block.attn1.to_k.parameters()) + tuple(last.video_block.attn1.to_v.parameters())
    eta = tuple(last.private_video_to_k.parameters()) + tuple(last.private_video_to_v.parameters())
    upstream = stack.packed_blocks[0].action_block.attn1.to_q.weight
    world,(private,shared) = stack(**inputs,routing_mode='two_routes')
    grads = torch.autograd.grad(private.square().mean(), (*phi,*eta,upstream), allow_unused=True,retain_graph=True)
    assert all(g is None or not g.any() for g in grads[:len(phi)])
    assert any(g is not None and g.any() for g in grads[len(phi):-1])
    assert grads[-1] is not None and grads[-1].any()  # no detach of shared upstream graph
    assert any(g is not None and g.any() for g in torch.autograd.grad(shared.square().mean(),phi,allow_unused=True,retain_graph=True))
    assert all(g is None or not g.any() for g in torch.autograd.grad(world.square().mean(),eta,allow_unused=True))


def test_selected_route_is_per_sample_and_preserves_world():
    stack, inputs, _ = case(batch=2)
    last = enable(stack)
    with torch.no_grad():
        last.private_video_to_v.weight.add_(.2)
    world,(private,shared)=stack(**inputs,routing_mode='two_routes')
    selected_world,selected=stack(**inputs,routing_mode='selected',route_choices=torch.tensor([0,1]))
    torch.testing.assert_close(selected[0],private[0])
    torch.testing.assert_close(selected[1],shared[1])
    torch.testing.assert_close(selected_world,world)
    assert not torch.allclose(private,shared)


def test_video_to_action_leak_is_rejected():
    stack, inputs, _ = case()
    enable(stack)
    inputs['video_attention_mask']=torch.ones_like(inputs['video_attention_mask'])
    with pytest.raises(ValueError,match='video.*action'):
        stack(**inputs,routing_mode='private')


@pytest.mark.parametrize('mode',['private','blend','selected'])
@torch.no_grad()
def test_real_private_cache_matches_recomputation_across_ode_steps(mode):
    stack, inputs, profile=case()
    last=enable(stack)
    last.private_video_to_v.weight.add_(.2)
    cache=DenoisingCache()
    cache.bind(profile=profile,invariant_tokens=profile.token_layout.noise_id==1,
        stream_lengths=(6,4),num_layers=2)
    for step in range(3):
        changed={**inputs}
        for name,n in [('video_hidden_states',3),('action_hidden_states',2)]:
            changed[name]=inputs[name].clone()
            changed[name][:,:n]+=step/3
        args=dict(routing_mode=mode,prior_shared=torch.tensor([.4]),route_choices=torch.tensor([0]))
        reference=stack(**changed,**args)
        cached=stack(**changed,denoising_cache=cache,**args)
        for expected,actual in zip(reference,cached,strict=True):
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=2e-6)
    assert all(entry.ready for layer in cache.layers for entry in layer)
    with pytest.raises(ValueError,match='fixed|route|scope'):
        stack(**inputs,denoising_cache=cache,routing_mode='selected' if mode=='private' else mode,
            prior_shared=torch.tensor([.7]),route_choices=torch.tensor([1]))


def test_activation_checkpointing_keeps_both_route_gradients():
    stack,inputs,_=case()
    last=enable(stack)
    _,(private,shared)=stack(**inputs,routing_mode='two_routes',use_activation_checkpointing=True)
    (private.square().mean()+shared.square().mean()).backward()
    assert last.private_video_to_v.weight.grad is not None
    assert last.video_block.attn1.to_v.weight.grad is not None
