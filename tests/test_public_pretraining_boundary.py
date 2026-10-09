"""Native tiny checkpoint input boundary; no full model reload."""
import pytest
import torch

from open_wam.configs import SharedVideoTransformerConfig
from open_wam.models.visual_tower import VisualTower
from open_wam.models.visual_tower.public_pretraining import load_public_video_checkpoint_into_tower


@pytest.fixture
def public_pair(tmp_path):
    config = SharedVideoTransformerConfig(hidden_size=32,num_layers=1,
        num_heads=4,attention_head_dim=8,ffn_dim=64,text_dim=16,freq_dim=8)
    source = VisualTower(config,action_dim=1,state_dim=1)
    target = VisualTower(config,action_dim=4,state_dim=4)
    state = {'visual_tower.'+name:value for name,value in source.state_dict().items()}
    return target,tmp_path/'video.pt',state


@pytest.mark.parametrize('key',[
    'visual_tower.core.action_unknown_future_module.weight',
    'visual_tower.core.runtime_stream_adapters.future_adapter.weight',
    'visual_tower.core.proprio_hidden_context_encoder.proj.weight',
])
def test_unknown_source_auxiliary_key_is_rejected_before_assignment(public_pair,key):
    target,path,state = public_pair
    state[key] = torch.zeros(1)
    before = target.core.blocks[0].attn1.to_k.weight
    torch.save({'model_state_dict':state},path)
    with pytest.raises(ValueError,match='(auxiliary|unexpected)'):
        load_public_video_checkpoint_into_tower(target,path)
    assert target.core.blocks[0].attn1.to_k.weight is before


def test_fixed_public_auxiliary_key_cannot_be_missing(public_pair):
    target,path,state = public_pair
    del state['visual_tower.core.action_proj_out.bias']
    torch.save({'model_state_dict':state},path)
    with pytest.raises(ValueError,match='auxiliary'):
        load_public_video_checkpoint_into_tower(target,path)


def test_target_known_new_conditioning_keys_keep_native_initialization(public_pair):
    target,path,state = public_pair
    target.core.configure_proprio_context_encoder(enabled=True,state_dim=4)
    target.core.configure_proprio_hidden_context_encoder(enabled=True,state_dim=4)
    target.core.configure_generalist_mode_context_encoder(enabled=True)
    names = ('core.proprio_context_encoder.proj.weight','core.proprio_context_encoder.proj.bias',
        'core.proprio_hidden_context_encoder.proj.weight','core.proprio_hidden_context_encoder.proj.bias',
        'core.generalist_mode_context_encoder.embedding.weight')
    before = {name:target.state_dict()[name].clone() for name in names}
    torch.save({'model_state_dict':state},path)
    report = load_public_video_checkpoint_into_tower(target,path)
    assert len(report['source_auxiliary_keys_ignored'])==23
    assert len(report['native_auxiliary_keys_initialized'])==28
    for name,value in before.items():
        torch.testing.assert_close(target.state_dict()[name],value,rtol=0,atol=0)


def test_target_unknown_auxiliary_module_cannot_be_silently_initialized(public_pair):
    target,path,state = public_pair
    target.core.action_unknown_future_module = torch.nn.Linear(4,32)
    torch.save({'model_state_dict':state},path)
    with pytest.raises(ValueError,match='video'):
        load_public_video_checkpoint_into_tower(target,path)
