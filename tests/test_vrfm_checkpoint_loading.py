"""Model-only strict loading and real prior generation, using tiny native WAM."""
from dataclasses import replace
import importlib
import importlib.util
from pathlib import Path

import pytest
import torch
import yaml

from gradientwam.settings import Settings, GradientWAMMethodConfig, GradientWAMMethod, load_settings
from open_wam.configs import serialize_experiment_config
from open_wam.models.policy_variants.contracts import PolicyInferContext
from open_wam.pipelines import build_variant_pipeline_from_config
from tests.test_variational_native_pipeline import fixture
from tests.test_vrfm_native import extension


def loader():
    assert importlib.util.find_spec('gradientwam.inference') is not None, 'VRFM inference loader is missing'
    return importlib.import_module('gradientwam.inference')


def settings(tmp_path, method='vrfm'):
    *_, config = fixture(return_config=True)
    return Settings(method_config=GradientWAMMethodConfig(method=GradientWAMMethod(method), latent_dim=4),
        seed=3, route_seed=3, episode_id=0, checkpoint=tmp_path/'absent-base.pt', checkpoint_sha256='a'*64,
        dataset_root=tmp_path/'dataset', frontend_root=tmp_path/'frontend', tokenizer_root=tmp_path/'tokenizer',
        preparation_root=tmp_path/'prepared', output_root=tmp_path/'output', prompt_fingerprint='',
        max_minutes=5, native=serialize_experiment_config(config))


def source_policy(values):
    pipeline = build_variant_pipeline_from_config(values.native_config())
    if values.method_config.uses_vrfm:
        extension().configure_vrfm(pipeline, latent_dim=values.method_config.latent_dim,
                                   kl_weight=values.method_config.kl_weight)
    return pipeline.eval()


@pytest.mark.parametrize('method', ['baseline', 'cagrad', 'vrfm', 'vrfm_cagrad'])
def test_model_reload_and_actual_prior_generation(tmp_path, method, monkeypatch):
    values = settings(tmp_path, method)
    source = source_policy(values)
    checkpoint = tmp_path/'model_state.pt'
    torch.save({'model_state_dict': source.state_dict()}, checkpoint)
    loaded = loader().load_policy_for_inference(values, checkpoint, device='cpu')
    assert not loaded.training
    assert not any(p.requires_grad for p in loaded.parameters())
    assert hasattr(loaded.policy_variant, 'vrfm') == values.method_config.uses_vrfm
    assert not any('private_video' in n or 'routing_controller' in n for n in loaded.state_dict())
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], tensor, rtol=0, atol=0)
    if values.method_config.uses_vrfm:
        monkeypatch.setattr(loaded.policy_variant.vrfm, 'posterior_sample',
            lambda **kwargs: pytest.fail('reloaded inference used posterior'))
    _, batch, latent, text = fixture()
    def generate(model):
        torch.manual_seed(149)
        return model.forward_infer_step_from_latents(latent[:, :, :1],
            PolicyInferContext(state=batch.state), text_context=text).decoder_output
    expected, actual = generate(source), generate(loaded)
    torch.testing.assert_close(expected.action_pred, actual.action_pred, rtol=0, atol=0)
    torch.testing.assert_close(expected.aux['predicted_latents'], actual.aux['predicted_latents'], rtol=0, atol=0)


@pytest.mark.parametrize('kind', ['missing', 'unexpected', 'shape', 'wrong_method', 'wrong_latent_dim'])
def test_checkpoint_incompatibilities_fail_strictly(tmp_path, kind):
    values = settings(tmp_path)
    state = source_policy(values).state_dict()
    key = 'policy_variant.vrfm.video_projection.weight'
    if kind == 'missing':
        state.pop(key)
    elif kind == 'unexpected':
        state['policy_variant.packed_block_stack.packed_blocks.1.private_video_to_k.weight'] = torch.zeros(32, 32)
    elif kind == 'shape':
        state[key] = state[key][:1]
    elif kind == 'wrong_method':
        values = replace(values, method_config=replace(values.method_config, method=GradientWAMMethod.BASELINE))
    else:
        values = replace(values, method_config=replace(values.method_config, latent_dim=8))
    checkpoint = tmp_path/'model_state.pt'
    torch.save({'model_state_dict': state}, checkpoint)
    from open_wam.runtime.checkpoints import CheckpointCompatibilityError
    with pytest.raises(CheckpointCompatibilityError):
        loader().load_policy_for_inference(values, checkpoint)


@pytest.mark.parametrize('payload_type', ['full_state', 'non_tensor_model_entry'])
def test_requires_model_only_and_does_not_silently_drop_model_entries(tmp_path, payload_type):
    values = settings(tmp_path)
    state = source_policy(values).state_dict()
    payload = {'model_state_dict': state}
    if payload_type == 'full_state':
        payload['optimizer_state_dict'] = {'state': {}, 'param_groups': []}
    else:
        state['unrecognized_non_tensor_entry'] = 5
    path = tmp_path/'model_state.pt'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='model-only'):
        loader().load_policy_for_inference(values, path)


def test_legacy_settings_rejected_before_model_construction(tmp_path):
    values = replace(settings(tmp_path), method_config=GradientWAMMethodConfig(
        method=None, legacy_v02=True, legacy_arm='native_joint'))
    with pytest.raises(ValueError, match='legacy'):
        loader().load_policy_for_inference(values, tmp_path/'absent.pt')


def test_real_settings_path_assembly_without_pretrained_assets(tmp_path):
    values = settings(tmp_path)
    native = values.native
    native['backbone']['num_layers'] = 30
    native['policy_variant']['num_action_layers'] = 30
    native['data']['view_layout'] = native['data']['view_layout'][:2]
    native['training'].update(gradient_accumulation_steps=10, optimizer_name='adamw',
        scheduler_name='constant_with_warmup', learning_rate=1e-5, beta1=.9, beta2=.95,
        weight_decay=.1, warmup_steps=10, max_grad_norm=2., action_loss_weight=1., latent_loss_weight=1.)
    experiment = tmp_path/'tiny_native.yaml'
    experiment.write_text(yaml.safe_dump(native), encoding='utf-8')
    raw = {'experiment': experiment.name, 'gradientwam': {'method': 'vrfm', 'latent_dim': 4},
           'checkpoint': {'path': str(values.checkpoint), 'sha256': values.checkpoint_sha256},
           'assets': {key: str(getattr(values, key)) for key in
                      ('dataset_root', 'frontend_root', 'tokenizer_root', 'preparation_root')},
           'run': {'seed': 3, 'route_seed': 3, 'episode_id': 0, 'max_minutes': 5,
                   'output_root': str(values.output_root)}}
    raw['assets']['prompt_encoder_fingerprint'] = ''
    settings_path = tmp_path/'settings.yaml'
    settings_path.write_text(yaml.safe_dump(raw), encoding='utf-8')
    parsed = load_settings(settings_path)
    model = source_policy(parsed)
    path = tmp_path/'model_state.pt'
    torch.save({'model_state_dict': model.state_dict()}, path)
    loaded = loader().load_policy_for_inference(settings_path, path)
    assert len(loaded.policy_variant.packed_block_stack.packed_blocks) == 30
    assert hasattr(loaded.policy_variant, 'vrfm')
    assert not values.checkpoint.exists() and not values.frontend_root.exists()
