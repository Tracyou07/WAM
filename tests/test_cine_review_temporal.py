"""Independent tiny CPU checks for the Cine/native temporal interface."""
from dataclasses import replace

import pytest
import torch


def test_actual_small_wan_vae_distinguishes_raw_and_latent_frames():
    from diffusers import AutoencoderKLWan
    from open_wam.models.visual_tower.vae_encoding import encode_clip

    torch.set_num_threads(1)
    torch.manual_seed(803)
    vae = AutoencoderKLWan(
        base_dim=4, decoder_base_dim=4, z_dim=4, num_res_blocks=1,
        dim_mult=[1, 2, 4, 4], temperal_downsample=[False, True, True],
        latents_mean=[0.] * 4, latents_std=[1.] * 4,
    ).eval().requires_grad_(False)
    raw = torch.rand(33, 3, 16, 16)
    contiguous = encode_clip(vae, raw, normalize=True)
    incorrectly_subsampled = encode_clip(vae, raw[::4], normalize=True)
    independent_frame = encode_clip(vae, raw[:1], normalize=True)
    assert contiguous.shape[0] == 9
    assert incorrectly_subsampled.shape[0] == 3
    assert independent_frame.shape[0] == 1
    torch.testing.assert_close(contiguous[:1], independent_frame, rtol=2e-6, atol=2e-7)
    assert torch.isfinite(contiguous).all()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('chunk_size', [2, 4])
def test_native_cine_prefix_masks_and_previous_chunk_state(monkeypatch, tmp_path, chunk_size):
    from tests.test_variational_native_pipeline import fixture
    from open_wam.configs import ViewLayoutConfig
    from open_wam.data.latent_temporal import latent_anchor_positions
    from open_wam.models.policy_variants.contracts import PolicyInferContext, PolicyTrainBatch
    from open_wam.models.policy_variants.dual_expert import packed_training
    from open_wam.models.policy_variants.dual_expert.vrfm import (
        configure_vrfm, vrfm_posterior_parameters,
    )
    from open_wam.pipelines import build_variant_pipeline_from_config
    from gradientwam.cagrad_training import (
        CAGradGradientAccumulator, cagrad_candidate_parameters,
        configure_native_trainability,
    )

    _, _, _, _, config = fixture(return_config=True)
    data = replace(
        config.data, num_frames=9, frame_stride=1,
        camera_names=('observation.images.color',),
        latent_camera_names=('observation.images.color',),
        canonical_height=224, canonical_width=448,
        view_layout=(ViewLayoutConfig('observation.images.color', 'color', 0, 0, 224, 448),),
        action_schema=replace(config.data.action_schema, action_dim=7,
                              action_horizon=36, state_dim=7),
        action_target=replace(config.data.action_target, representation='raw',
                              state_encoding='identity'),
    )
    config = replace(config, data=data,
                     action_decoder=replace(config.action_decoder, action_dim=7,
                                            action_horizon=36),
                     training=replace(config.training, chunk_size=chunk_size),
                     inference=replace(config.inference, frame_chunk_size=9))
    pipeline = build_variant_pipeline_from_config(config)
    configure_native_trainability(pipeline, expected_layers=2)
    configure_vrfm(pipeline, latent_dim=4, kl_weight=.03)
    assert pipeline.policy_variant.action_tokens_per_frame == 4

    raw_start = 10
    anchors = latent_anchor_positions(raw_frame_count=33, latent_num_frames=9,
                                     layout='wan_causal_stride4')
    assert anchors == [0, 4, 8, 12, 16, 20, 24, 28, 32]
    state_frames = torch.zeros(1, 9, 7)
    state_frames[0, :, 0] = torch.tensor(anchors) + raw_start
    state_frames[0, :, -1] = 1
    prefix_state = torch.tensor([[raw_start - 1., 0, 0, 0, 0, 0, 1]])
    actions = torch.zeros(1, 36, 7)
    raw_commands = torch.arange(32 * 7, dtype=torch.float32).reshape(32, 7) / 100
    actions[0, 4:] = raw_commands
    action_mask = torch.ones_like(actions)
    action_mask[:, :4] = 0
    batch = PolicyTrainBatch(
        actions=actions, action_mask=action_mask, state=prefix_state,
        extra={
            'condition_latents': torch.randn(1, 48, 1, 4, 4),
            'proprio_context_frames': state_frames,
            'proprio_context_frames_mask': torch.ones_like(state_frames),
            'metadata': ({
                'sampled_chunk_size': chunk_size, 'sampled_window_size': 8,
                'history_frames': 1, 'loss_frame_start': 1, 'loss_frame_end': 9,
                'latent_loss_frame_start': 0, 'latent_loss_frame_end': 9,
                'action_loss_frame_start': 1, 'action_loss_frame_end': 9,
                'chunk_origin_frame': 1, 'singleton_chunk_frame': 0,
                'frame_shift': 1, 'action_tokens_per_frame': 4,
            },),
        },
    )
    projected = []
    original = packed_training.project_hidden_proprio_context_to_frames

    def capture(*args, **kwargs):
        value = original(*args, **kwargs)
        projected.append(value.detach().clone())
        return value

    monkeypatch.setattr(packed_training, 'project_hidden_proprio_context_to_frames', capture)
    video = torch.randn(1, 48, 9, 4, 4)
    text = torch.randn(1, 3, 16)
    output = pipeline.forward_train_from_latents(video, batch, text_context=text)
    assert len(projected) == 1
    assert projected[0].shape == (1, 10, 7)
    torch.testing.assert_close(
        projected[0][0, :, 0],
        torch.tensor([9., 9., 10., 10., 18., 18., 26., 26., 34., 34.]
                     if chunk_size == 2 else
                     [9., 9., 10., 10., 10., 10., 26., 26., 26., 26.]),
        rtol=0, atol=0,
    )
    artifacts = output.policy_output.decoder_artifacts.payload
    assert artifacts.video.target_latents.shape[2] == 10
    assert artifacts.video.future_loss_mask[:, :, :1].count_nonzero() == 0
    assert artifacts.video.future_loss_mask[:, :, 1:].all()
    assert artifacts.action.action_mask[:, :4].count_nonzero() == 0
    assert artifacts.action.action_mask[:, 4:].all()
    torch.testing.assert_close(batch.actions[:, 4:], raw_commands[None], rtol=0, atol=0)
    assert output.policy_output.aux['video_condition_source'] == 'condition_latents_prefix'

    candidates = cagrad_candidate_parameters(pipeline)
    ordinary = tuple(vrfm_posterior_parameters(pipeline)) + tuple(
        pipeline.policy_variant.vrfm.action_projection.parameters()
    )
    expected = torch.autograd.grad(output.decoder_output.loss, ordinary, retain_graph=True)
    accumulator = CAGradGradientAccumulator(candidates, c=.4)
    accumulator.accumulate(output.decoder_output.aux['task_losses'],
                           {'video': True, 'action': True}, scale=1.)
    output.decoder_output.loss.backward()
    report = accumulator.finalize(pipeline.parameters())
    assert report['applied']
    for parameter, gradient in zip(ordinary, expected, strict=True):
        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
    trainable = [p for p in pipeline.parameters() if p.requires_grad]
    assert all(torch.isfinite(p.grad).all() for p in trainable if p.grad is not None)
    torch.nn.utils.clip_grad_norm_(trainable, 2.)
    optimizer = torch.optim.AdamW(trainable, lr=.001)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if chunk_size == 4:
        import copy
        from gradientwam import checkpoint
        from open_wam.configs.enums import TrainerAccelerator, TrainerPrecision
        from open_wam.training.state import TrainState
        from open_wam.training.strategies import SingleDeviceStrategy

        strategy = SingleDeviceStrategy(accelerator=TrainerAccelerator.CPU,
                                        precision=TrainerPrecision.FP32)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        state = TrainState(global_step=1, optimizer_step=1, next_batch_index=1,
                           seen_batches=1)
        metadata = {'method': 'vrfm_cagrad', 'cine_geometry': [33, 9, 36, 7, 7]}
        checkpoint_path = tmp_path / 'cine_native_step1.pt'
        checkpoint.save_step1(
            checkpoint_path, model=pipeline, optimizer=optimizer, scheduler=scheduler,
            strategy=strategy, train_state=state, sampler_state={'next_window': 1},
            extra_generator_states={}, metadata=metadata, cuda_devices=(), max_bytes=1 << 26,
        )

        def next_update(model, opt, sched):
            model.train()
            result = model.forward_train_from_latents(video, batch, text_context=text)
            task_accumulator = CAGradGradientAccumulator(cagrad_candidate_parameters(model), c=.4)
            task_accumulator.accumulate(result.decoder_output.aux['task_losses'],
                                        {'video': True, 'action': True}, scale=1.)
            result.decoder_output.loss.backward()
            task_accumulator.finalize(model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            return result.decoder_output.loss.detach().clone()

        expected_loss = next_update(pipeline, optimizer, scheduler)
        expected_rng = torch.get_rng_state().clone()
        expected_parameters = copy.deepcopy(pipeline.state_dict())
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        restored = build_variant_pipeline_from_config(config)
        configure_native_trainability(restored, expected_layers=2)
        configure_vrfm(restored, latent_dim=4, kl_weight=.03)
        restored_optimizer = torch.optim.AdamW(
            (p for p in restored.parameters() if p.requires_grad), lr=.001,
        )
        restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1.)
        loaded = checkpoint.load_step1(
            checkpoint_path, model=restored, optimizer=restored_optimizer,
            scheduler=restored_scheduler, strategy=strategy,
            expected_metadata=metadata, cuda_devices=(),
        )
        assert loaded['train_state'].optimizer_step == 1
        assert loaded['sampler_state'] == {'next_window': 1}
        checkpoint.restore_rng(loaded['rng_state'], cuda_devices=())
        actual_loss = next_update(restored, restored_optimizer, restored_scheduler)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        for name, value in expected_parameters.items():
            torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
        actual_optimizer = restored_optimizer.state_dict()
        assert actual_optimizer['param_groups'] == expected_optimizer['param_groups']
        for index, values in expected_optimizer['state'].items():
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(actual_optimizer['state'][index][key], value, rtol=0, atol=0)
                else:
                    assert actual_optimizer['state'][index][key] == value
        assert restored_scheduler.state_dict() == scheduler.state_dict()

    pipeline.eval()
    vrfm = pipeline.policy_variant.vrfm
    monkeypatch.setattr(vrfm, 'posterior_sample',
                        lambda **kwargs: pytest.fail('heldout/inference called posterior'))
    heldout = pipeline.forward_train_from_latents(video, batch, text_context=text)
    assert heldout.decoder_output.aux['vrfm_kl_loss'].item() == 0
    draws = []
    prior_sample = vrfm.prior_sample

    def sample_prior(*args, **kwargs):
        z = prior_sample(*args, **kwargs)
        draws.append(z.detach().clone())
        return z

    monkeypatch.setattr(vrfm, 'prior_sample', sample_prior)
    generated = pipeline.forward_infer_step_from_latents(
        batch.extra['condition_latents'], PolicyInferContext(state=prefix_state),
        text_context=text,
    )
    assert len(draws) == 1
    assert generated.decoder_output.action_pred.shape == (1, 36, 7)
    assert generated.decoder_output.aux['predicted_video_latents'].shape[2] == 9
    assert torch.isfinite(generated.decoder_output.action_pred).all()
    assert not torch.cuda.is_initialized()
