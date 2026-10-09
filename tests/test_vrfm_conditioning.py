"""Posterior evidence, latent KL, and attachment lifecycle regressions."""
import copy

import pytest
import torch

from tests.test_vrfm_native import extension
from tests.test_variational_native_pipeline import fixture, configure


def inputs():
    return dict(clean_video=torch.randn(2, 48, 3, 2, 2),
                noisy_video=torch.randn(2, 48, 3, 2, 2),
                clean_action=torch.randn(2, 4, 4), noisy_action=torch.randn(2, 4, 4),
                video_timesteps=torch.tensor([[0., 20., 40.], [0., 600., 900.]]),
                action_timesteps=torch.tensor([[10., 10., 30., 30.], [500., 500., 800., 800.]]),
                text_context=torch.randn(2, 3, 16),
                proprio_state=torch.tensor([[1., 2., 3., 4.], [11., 12., 13., 14.]]),
                action_mask=torch.ones(2, 4, 4))


def test_gaussian_kl_independent_closed_form_and_gradient():
    mean = torch.tensor([[1., 2.]], requires_grad=True)
    logvar = torch.zeros_like(mean, requires_grad=True)
    kl = extension().diagonal_gaussian_kl(mean, logvar)
    assert kl.item() == 2.5  # unit covariance, KL = half squared mean norm
    kl.backward()
    torch.testing.assert_close(mean.grad, torch.tensor([[1., 2.]]))
    torch.testing.assert_close(logvar.grad, torch.zeros_like(logvar))


def test_posterior_keeps_batch_rows_independent_and_reads_all_evidence():
    pipeline, *_ = fixture()
    extension().configure_vrfm(pipeline, latent_dim=4)
    vrfm = pipeline.policy_variant.vrfm
    values = inputs()
    captured = []
    handle = vrfm.posterior.register_forward_hook(lambda module, args, result: captured.append(result.detach().clone()))
    vrfm.posterior_sample(**values)
    batched = captured[-1]
    for index in range(2):
        vrfm.posterior_sample(**{key: value[index:index+1] for key, value in values.items()})
        torch.testing.assert_close(captured[-1][0], batched[index], atol=2e-6, rtol=2e-5)
    for key in ('clean_video', 'noisy_video', 'clean_action', 'noisy_action',
                'video_timesteps', 'action_timesteps', 'text_context', 'proprio_state'):
        vrfm.posterior_sample(**{**values, key: values[key] + 50})
        assert not torch.allclose(captured[-1], batched), f'posterior ignored {key}'
    handle.remove()


def test_masked_actions_cannot_change_posterior_parameters():
    pipeline, *_ = fixture()
    extension().configure_vrfm(pipeline, latent_dim=4)
    vrfm = pipeline.policy_variant.vrfm
    values = inputs()
    values['action_mask'][:, 2:] = 0
    seen = []
    handle = vrfm.posterior.register_forward_hook(lambda module, args, result: seen.append(result.detach().clone()))
    vrfm.posterior_sample(**values)
    changed = copy.deepcopy(values)
    changed['clean_action'][:, 2:] = 1e6
    changed['noisy_action'][:, 2:] = -1e6
    vrfm.posterior_sample(**changed)
    torch.testing.assert_close(seen[0], seen[1], atol=0, rtol=0)
    handle.remove()


def test_attach_rejects_duplicate_and_legacy_in_both_orders():
    pipeline, *_ = fixture()
    extension().configure_vrfm(pipeline)
    with pytest.raises(ValueError, match='already'):
        extension().configure_vrfm(pipeline)
    with pytest.raises(ValueError, match='VRFM'):
        configure(pipeline, 'variational_sharing')
    legacy, *_ = fixture()
    configure(legacy, 'variational_sharing')
    with pytest.raises(ValueError, match='legacy'):
        extension().configure_vrfm(legacy)
