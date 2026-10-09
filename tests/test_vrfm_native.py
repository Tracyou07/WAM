"""Small real OpenWAM forwards; no pretrained assets or GPU needed."""
import copy
import importlib
import importlib.util
import random
from dataclasses import replace

import pytest
import torch

from tests.test_variational_native_pipeline import fixture, train
from open_wam.models.policy_variants.contracts import PolicyInferContext


def extension():
    name = 'open_wam.models.policy_variants.dual_expert.vrfm'
    assert importlib.util.find_spec(name) is not None, 'native VRFM attachment is missing'
    return importlib.import_module(name)


def nonzero(loss, parameters, retain_graph=True):
    return any(g is not None and g.abs().sum() > 0 for g in
               torch.autograd.grad(loss, tuple(parameters), allow_unused=True,
                                   retain_graph=retain_graph))


def test_baseline_task_losses_keep_graph_without_new_parameters():
    pipeline, batch, latent, text = fixture()
    before = {n: id(p) for n, p in pipeline.named_parameters()}
    result = train(pipeline, batch, latent, text).decoder_output
    assert 'task_losses' in result.aux, 'decoder must expose differentiable task losses'
    losses = result.aux['task_losses']
    assert set(losses) == {'video', 'action'}
    assert all(value.requires_grad for value in losses.values())
    torch.testing.assert_close(result.loss, losses['video'] + losses['action'])
    assert result.aux['vrfm_kl_loss'].item() == 0
    assert before == {n: id(p) for n, p in pipeline.named_parameters()}


def test_native_vrfm_both_streams_reach_posterior_and_kl():
    pipeline, batch, latent, text = fixture()
    next(pipeline.visual_tower.core.parameters()).requires_grad_(False)
    before = {id(p): p.requires_grad for p in pipeline.parameters()}
    module = extension()
    audit = module.configure_vrfm(pipeline, latent_dim=4, kl_weight=.03)
    assert all(p.requires_grad == before[id(p)] for p in pipeline.parameters() if id(p) in before)
    assert audit['existing_trainability_preserved']
    assert not any('private_video' in name or 'routing_controller' in name for name, _ in pipeline.named_parameters())
    output = train(pipeline, batch, latent, text)
    losses = output.decoder_output.aux['task_losses']
    posterior = module.vrfm_posterior_parameters(pipeline)
    assert posterior
    assert nonzero(losses['video'], posterior)
    assert nonzero(losses['action'], posterior)
    assert nonzero(output.decoder_output.aux['vrfm_kl_loss'], posterior)
    assert nonzero(losses['video'], pipeline.policy_variant.vrfm.video_projection.parameters())
    assert nonzero(losses['action'], pipeline.policy_variant.vrfm.video_projection.parameters())
    assert nonzero(losses['action'], pipeline.policy_variant.vrfm.action_projection.parameters())
    torch.testing.assert_close(output.decoder_output.loss,
        losses['video'] + losses['action'] + output.decoder_output.aux['vrfm_kl_loss'])
    output.decoder_output.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in posterior if p.grad is not None)


def test_eval_prior_never_calls_posterior_and_targets_cannot_select_z(monkeypatch):
    pipeline, batch, latent, text = fixture()
    extension().configure_vrfm(pipeline, latent_dim=4)
    pipeline.eval()
    def forbidden(*args, **kwargs):
        pytest.fail('evaluation accessed the posterior')
    monkeypatch.setattr(pipeline.policy_variant.vrfm, 'posterior_sample', forbidden)
    first = train(pipeline, batch, latent, text)
    changed = copy.copy(batch)
    changed.actions = batch.actions + 100
    second = train(pipeline, changed, latent + 100, text)
    torch.testing.assert_close(first.policy_output.aux['vrfm_z'], second.policy_output.aux['vrfm_z'], rtol=0, atol=0)
    assert first.decoder_output.aux['vrfm_kl_loss'].item() == 0


def test_actual_generation_prior_once_shared_across_ode_cfg_and_cache(monkeypatch):
    pipeline, batch, latent, text = fixture()
    extension().configure_vrfm(pipeline, latent_dim=4)
    pipeline.policy_variant.inference_config = replace(pipeline.policy_variant.inference_config,
        guidance_scale=3., action_guidance_scale=2.)
    vrfm = pipeline.policy_variant.vrfm
    draws = []
    original = vrfm.prior_sample
    def draw(*args, **kwargs):
        value = original(*args, **kwargs)
        draws.append(value.detach().clone())
        return value
    monkeypatch.setattr(vrfm, 'prior_sample', draw)
    monkeypatch.setattr(vrfm, 'posterior_sample', lambda *args, **kwargs: pytest.fail('generate accessed posterior'))
    injected = []
    original_condition = vrfm.condition
    def condition(z):
        injected.append(z.detach().clone())
        return original_condition(z)
    monkeypatch.setattr(vrfm, 'condition', condition)
    # Generate must use the prior even if a caller forgot pipeline.eval().
    output = pipeline.forward_infer_step_from_latents(latent[:, :, :1],
        PolicyInferContext(state=batch.state), text_context=text,
        negative_text_context=-text)
    assert len(draws) == 1 and len(injected) >= 6
    assert all(torch.equal(z, draws[0]) for z in injected)
    assert torch.isfinite(output.decoder_output.action_pred).all()


def test_checkpoint_recompute_and_rng_restore_do_not_redraw():
    pipeline, batch, latent, text = fixture()
    extension().configure_vrfm(pipeline, latent_dim=4)
    checkpointed = copy.deepcopy(pipeline)
    checkpointed.policy_variant.config = replace(checkpointed.policy_variant.config, use_activation_checkpointing=True)
    first = train(pipeline, batch, latent, text)
    first.decoder_output.loss.backward()
    after_first = torch.get_rng_state()
    second = train(checkpointed, batch, latent, text)
    second.decoder_output.loss.backward()
    assert torch.equal(torch.get_rng_state(), after_first)
    torch.testing.assert_close(first.decoder_output.loss, second.decoder_output.loss, rtol=0, atol=0)
    other = dict(checkpointed.named_parameters())
    for name, p in pipeline.named_parameters():
        if p.grad is not None:
            torch.testing.assert_close(p.grad, other[name].grad, rtol=2e-5, atol=2e-6)
    saved = copy.deepcopy(pipeline.state_dict())
    rng = torch.get_rng_state()
    py_rng = random.getstate()
    expected = pipeline.forward_train_from_latents(latent, batch, text_context=text)
    restored = fixture()[0]
    extension().configure_vrfm(restored, latent_dim=4)
    restored.load_state_dict(saved, strict=True)
    torch.set_rng_state(rng)
    random.setstate(py_rng)
    actual = restored.forward_train_from_latents(latent, batch, text_context=text)
    torch.testing.assert_close(expected.decoder_output.loss, actual.decoder_output.loss, rtol=0, atol=0)


@pytest.mark.parametrize('kwargs', [{'latent_dim': 0}, {'latent_dim': True}, {'kl_weight': -1}, {'kl_weight': float('nan')}])
def test_invalid_attachment_rejected_without_mutating_pipeline(kwargs):
    pipeline, *_ = fixture()
    before = set(pipeline.state_dict())
    with pytest.raises(ValueError):
        extension().configure_vrfm(pipeline, **kwargs)
    assert set(pipeline.state_dict()) == before


@pytest.mark.parametrize('enable_vrfm', [False, True])
@pytest.mark.parametrize('mode', ['bucket', 'padded', 'packed'])
def test_variable_batch_task_losses_preserve_sample_mean_and_graph(mode, enable_vrfm):
    from open_wam.configs import BatchingConfig, BatchingMode
    from open_wam.data.latent_batching import LatentBatchCollator
    from open_wam.data.latent_contracts import LatentWAMSample
    from open_wam.models.policy_variants.contracts import PolicyTrainBatch
    torch.set_num_threads(1)
    pipeline, *_ = fixture()
    if enable_vrfm:
        extension().configure_vrfm(pipeline, latent_dim=4)
    samples = [LatentWAMSample(video_latents=torch.randn(48, frames, 4, 4),
        actions=torch.randn(frames, 4), action_mask=torch.ones(frames, 4),
        state=torch.randn(1, 4), state_mask=torch.ones(1, 4),
        condition_latents=torch.randn(48, 1, 4, 4),
        proprio_context_frames=torch.randn(frames, 4), proprio_context_frames_mask=torch.ones(frames, 4),
        text_context=torch.randn(3, 16), metadata={'sampled_chunk_size': 2,
            'sampled_window_size': 8, 'history_frames': 2, 'action_tokens_per_frame': 1})
        for frames in (4, 6)]
    collated = LatentBatchCollator(BatchingConfig(mode=BatchingMode(mode)))(samples)
    batch = PolicyTrainBatch(actions=collated.actions, action_mask=collated.action_mask, state=collated.state,
        extra={'batching_mode': BatchingMode(mode), 'sequence_lengths': collated.sequence_lengths,
               'tensor_lengths': collated.tensor_lengths, 'metadata': collated.metadata,
               'condition_latents': collated.condition_latents,
               'proprio_context_frames': collated.proprio_context_frames,
               'proprio_context_frames_mask': collated.proprio_context_frames_mask})
    result = pipeline.forward_train_from_latents(collated.video_latents, batch, text_context=collated.text_context)
    decoder = result.decoder_output
    assert 'task_losses' in decoder.aux, 'variable-batch bridge lost task gradients'
    for task in ('video', 'action'):
        expected = torch.stack([sample.decoder_output.aux['task_losses'][task]
                                for sample in result.sample_outputs]).mean()
        torch.testing.assert_close(decoder.aux['task_losses'][task], expected)
        assert decoder.aux['task_losses'][task].requires_grad
    torch.testing.assert_close(decoder.loss, sum(decoder.aux['task_losses'].values()) + decoder.aux['vrfm_kl_loss'])
    decoder.loss.backward()


def test_bfloat16_native_vrfm_has_finite_posterior_backward():
    pipeline, batch, latent, text = fixture()
    extension().configure_vrfm(pipeline, latent_dim=4)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = train(pipeline, batch, latent, text)
    assert output.decoder_output.aux['vrfm_kl_loss'].dtype == torch.float32
    output.decoder_output.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in pipeline.parameters() if p.grad is not None)


def test_generation_cache_parity_under_same_prior_and_noise_rng():
    pipeline, batch, latent, text = fixture()
    extension().configure_vrfm(pipeline, latent_dim=4)
    pipeline.eval()
    uncached = copy.deepcopy(pipeline)
    uncached.policy_variant.inference_config = replace(uncached.policy_variant.inference_config, use_cache=False)
    def generate(model):
        torch.manual_seed(709)
        return model.forward_infer_step_from_latents(latent[:, :, :1],
            PolicyInferContext(state=batch.state), text_context=text).decoder_output
    a, b = generate(pipeline), generate(uncached)
    torch.testing.assert_close(a.action_pred, b.action_pred, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(a.aux['predicted_latents'], b.aux['predicted_latents'], rtol=2e-5, atol=2e-6)
