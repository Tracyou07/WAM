"""Shipped data entries activate the same base scope and two-step run budget."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from gradientwam.cagrad_training import configure_native_trainability
from gradientwam.settings import parse_method_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('entry', ['gradientwam', 'cine_v3'])
def test_shipped_arm_recipes_match_except_method_and_train_state_projection(entry):
    reference = None
    for arm in ['baseline', 'vrfm', 'cagrad', 'vrfm_cagrad']:
        raw = yaml.safe_load((ROOT / 'configs' / entry / f'{arm}.yaml').read_text())
        method = parse_method_config(raw)
        assert method.label == arm
        assert method.train_proprio_projection, 'the declared base scope must train state projection'
        common = deepcopy(raw)
        common['gradientwam'].pop('method')
        if reference is None:
            reference = common
        else:
            assert common == reference
        if entry == 'cine_v3':
            assert raw['run']['steps'] == 2
    if entry == 'gradientwam':
        run = yaml.safe_load((ROOT / 'configs/gradientwam_distributed.example.yaml').read_text())
        assert run['steps'] == 2


@pytest.mark.parametrize('method', ['baseline', 'vrfm_cagrad'])
def test_real_native_state_projection_receives_training_gradient(method):
    from tests.test_variational_native_pipeline import fixture, train
    from open_wam.models.policy_variants.dual_expert.vrfm import configure_vrfm

    pipeline, batch, latent, text = fixture()
    scope = configure_native_trainability(pipeline, expected_layers=2,
                                         train_proprio_projection=True)
    if method == 'vrfm_cagrad':
        configure_vrfm(pipeline, latent_dim=4)
    projection = pipeline.visual_tower.core.proprio_hidden_context_encoder.proj
    before = projection.weight.detach().clone()
    optimizer = torch.optim.AdamW([p for p in pipeline.parameters() if p.requires_grad], lr=1e-5)
    train(pipeline, batch, latent, text).decoder_output.loss.backward()
    assert scope['hidden_proprio_parameter_count'] > 0
    assert projection.weight.grad is not None and projection.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(projection.weight, before)
