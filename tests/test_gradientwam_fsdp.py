"""FSDP controls preserve the bounded method and legacy DDP continuation."""
from pathlib import Path
from types import SimpleNamespace
import os
import shlex
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_strategy_parser_defaults_to_legacy_ddp_and_validates_choices():
    from gradientwam.settings import parse_distributed_strategy
    from open_wam.configs.enums import StrategyName

    assert parse_distributed_strategy(None) is StrategyName.DDP
    assert parse_distributed_strategy('fsdp') is StrategyName.FSDP
    for value in ['single_device', 'zero', True, 1]:
        with pytest.raises(ValueError, match='ddp.*fsdp'):
            parse_distributed_strategy(value)


def test_shipped_distributed_entries_choose_fsdp_with_two_step_budget():
    raw = yaml.safe_load((ROOT / 'configs/gradientwam_distributed.example.yaml').read_text())
    assert raw['strategy'] == 'fsdp'
    assert raw['steps'] == 2
    for arm in ['baseline', 'vrfm', 'cagrad', 'vrfm_cagrad']:
        cine = yaml.safe_load((ROOT / 'configs/cine_v3' / f'{arm}.yaml').read_text())
        assert cine['run']['strategy'] == 'fsdp'
        assert cine['run']['steps'] == 2


def test_fsdp_checkpoint_identity_binds_contract_and_offload_only_for_fsdp(monkeypatch):
    from gradientwam.runtime_factory import _distributed_checkpoint_identity
    from open_wam.configs.enums import StrategyName

    previous = {'method': 'baseline', 'world_size': 3}
    monkeypatch.delenv('OPEN_WAM_FSDP_CPU_OFFLOAD', raising=False)
    assert _distributed_checkpoint_identity(previous, StrategyName.DDP) == previous
    gpu = _distributed_checkpoint_identity(previous, StrategyName.FSDP)
    assert gpu['fsdp_contract'] == {'version': 'replicated_common_v1', 'cpu_offload': False}
    monkeypatch.setenv('OPEN_WAM_FSDP_CPU_OFFLOAD', '1')
    cpu = _distributed_checkpoint_identity(previous, StrategyName.FSDP)
    assert cpu != gpu
    assert cpu['fsdp_contract']['cpu_offload'] is True
    assert _distributed_checkpoint_identity(previous, StrategyName.DDP) == previous
    assert 'fsdp_contract' not in previous


@pytest.mark.skipif(os.name == 'nt', reason='the public launcher requires POSIX bash')
@pytest.mark.parametrize('override', [None, 'ddp'])
def test_quickstart_passes_fsdp_default_or_explicit_legacy_ddp_to_torchrun(tmp_path, override):
    # Intercept model/data checks and torchrun while executing the real YAML generation.
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    python = bin_dir / 'python'
    python.write_text('#!/bin/sh\nif [ "$1" = "-m" ]; then exit 0; fi\n'
                      f'exec {shlex.quote(sys.executable)} "$@"\n')
    python.chmod(0o755)
    torchrun = bin_dir / 'torchrun'
    torchrun.write_text(f'#!{sys.executable}\n'
        'import json, os, sys, yaml\nfrom pathlib import Path\n'
        "spec = yaml.safe_load(Path(sys.argv[sys.argv.index('--run-config') + 1]).read_text())\n"
        "Path(os.environ['CAPTURE_RUN_SPEC']).write_text(json.dumps(spec))\n")
    torchrun.chmod(0o755)
    capture = tmp_path / 'run.json'
    env = {**os.environ, 'PATH': f'{bin_dir}{os.pathsep}{os.environ["PATH"]}',
           'GW_PROMPT_FINGERPRINT': 'test-only', 'TRAIN_STEPS': '2', 'NPROC_PER_NODE': '3',
           'CAPTURE_RUN_SPEC': str(capture), 'CHECK_ONLY': '0', 'PREPARE': '0'}
    env.pop('TRAIN_STRATEGY', None)
    if override is not None:
        env['TRAIN_STRATEGY'] = override
    result = subprocess.run(['bash', str(ROOT / 'scripts/quickstart.sh'),
                             str(tmp_path / 'settings.yaml'), str(tmp_path / 'split.json')],
                            env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    import json
    spec = json.loads(capture.read_text())
    assert spec['strategy'] == (override or 'fsdp')
    assert spec['steps'] == 2


def test_fsdp_shards_layers_before_root_and_keeps_selected_parameters_replicated(monkeypatch):
    from tests.test_training_strategies import _Pipeline
    from open_wam.training.strategies import _apply_composable_fsdp_sharding

    model = _Pipeline()
    common = tuple(model.root_projection.parameters())
    calls = []
    monkeypatch.setattr('torch.distributed.fsdp.fully_shard',
                        lambda module, **kwargs: calls.append((module, kwargs)))
    _apply_composable_fsdp_sharding(model, mesh=SimpleNamespace(), mp_policy=SimpleNamespace(),
                                   ignored_parameters=common)
    assert calls[-1][0] is model
    assert len(calls) > 1
    assert all(kwargs['ignored_params'] == set(common) for _, kwargs in calls)


def test_fsdp_prepare_does_not_move_complete_model_before_sharding(monkeypatch):
    from tests.test_training_strategies import _Pipeline
    from open_wam.training import strategies
    from open_wam.configs.enums import StrategyName, TrainerPrecision

    strategy = object.__new__(strategies.DistributedStrategy)
    strategy.kind = StrategyName.FSDP
    strategy.device = torch.device('cpu')
    strategy.distributed = True
    strategy.precision = TrainerPrecision.FP32
    strategy._device_mesh = SimpleNamespace()
    model = _Pipeline()
    def reject_full_transfer(*args, **kwargs):
        raise AssertionError('complete model was transferred before FSDP sharding')
    monkeypatch.setattr(model, 'to', reject_full_transfer)
    monkeypatch.setattr(strategies, '_apply_block_activation_checkpointing', lambda _: None)
    monkeypatch.setattr(strategies, '_apply_composable_fsdp_sharding', lambda model, **kwargs: model)
    assert strategy.prepare_model(model) is model


@pytest.mark.parametrize('accelerator', ['cpu', 'gpu'])
@pytest.mark.parametrize('method', ['baseline', 'vrfm_cagrad'])
@pytest.mark.parametrize('mixed_precision', [False, True], ids=['fp32', 'bf16'])
def test_three_rank_native_fsdp_full_state_resume(tmp_path, method, mixed_precision, accelerator):
    import json
    from tests.test_gradientwam_distributed import _worker, _free_port

    if accelerator == 'gpu' and (torch.cuda.device_count() < 3 or not dist.is_nccl_available()):
        pytest.skip('three free CUDA GPUs and NCCL required')
    mp.spawn(_worker, args=(3, _free_port(), str(tmp_path), True, 'fsdp', accelerator, method, mixed_precision),
             nprocs=3, join=True)
    result = json.loads((tmp_path / 'result.json').read_text())
    assert result['resumed_update_matches_uninterrupted']
    assert result['optimizer_and_scheduler_match']
    assert result['rank_rng_and_sampler_state_restored']
    assert result['heldout_prior_evaluation_passed']
    assert result['strategy'] == 'fsdp'


class _ShardedOracleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.video = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.SiLU(), torch.nn.Linear(8, 1))
        self.action = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.SiLU(), torch.nn.Linear(8, 1))
        self.shared = torch.nn.Linear(8, 8)
        self.one_task = torch.nn.Parameter(torch.zeros(1))

    def module_topology(self):
        from open_wam.models.policy_variants import PolicyModuleTopology
        return PolicyModuleTopology(visual_runtime_modules=(),
                                    fsdp_atomic_modules=(self.video[0], self.video[2],
                                                        self.action[0], self.action[2]))

    def forward(self, x):
        y = self.shared(x)
        return self.video(y)[:, 0], self.action(y)[:, 0] + self.one_task


def _fsdp_global_update_oracle(rank, world, port, root, accelerator):
    import copy
    import json
    import os
    from gradientwam.cagrad import cagrad_coefficients
    from gradientwam.cagrad_training import CAGradGradientAccumulator
    from open_wam.training.strategies import DistributedStrategy
    from open_wam.training.launch import DistributedLaunchContext
    from open_wam.configs.enums import StrategyName, TrainerAccelerator, TrainerPrecision
    from torch.distributed.tensor import DTensor

    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), RANK=str(rank),
                      LOCAL_RANK=str(rank), WORLD_SIZE=str(world))
    torch.manual_seed(371)
    strategy = DistributedStrategy(accelerator=TrainerAccelerator(accelerator),
        precision=TrainerPrecision.FP32, kind=StrategyName.FSDP,
        launch_context=DistributedLaunchContext.from_env(), distributed_timeout_seconds=90)
    plain = _ShardedOracleModel()
    reference = copy.deepcopy(plain).to(strategy.device)
    common = (*plain.shared.parameters(), plain.one_task)
    model = strategy.prepare_model(plain, ignored_parameters=common)
    assert any(isinstance(p, DTensor) for p in model.parameters())
    assert all(not isinstance(p, DTensor) for p in common)
    accumulator = CAGradGradientAccumulator(common, c=.4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, foreach=False)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=.001, foreach=False)

    def inputs(r, micro):
        return torch.arange(8, device=strategy.device, dtype=torch.float32)[None] / 5 + r / 4 + micro / 7

    for micro in range(2):
        video, action = model(inputs(rank, micro))
        losses = {'video': (video - .3).square().mean(),
                  'action': (action - 1.2).square().mean() * (rank != world - 1)}
        strategy.set_partial_backward(model, enabled=True)
        accumulator.accumulate(losses, {'video': True, 'action': rank != world - 1}, scale=.5,
                               include_inactive_tasks=True)
        strategy.set_partial_backward(model, enabled=False)
        (sum(losses.values()) / 2).backward()
    report = accumulator.finalize(model.parameters())
    assert report['common_parameter_count'] == 2
    strategy.unscale_(optimizer, synchronize_replicated=False)

    all_inputs = torch.cat([inputs(r, micro) for r in range(world) for micro in range(2)])
    video, action = reference(all_inputs)
    active = torch.tensor([r != world - 1 for r in range(world) for _ in range(2)],
                          device=strategy.device)
    reference_losses = [(video - .3).square().mean(), ((action - 1.2).square() * active).mean()]
    reference_common = tuple(reference.shared.parameters())
    task_grads = [torch.autograd.grad(loss, reference_common, retain_graph=True)
                  for loss in reference_losses]
    gram = [[sum(torch.dot(left.double().flatten(), right.double().flatten()).item()
                 for left, right in zip(task_grads[i], task_grads[j]))
             for j in range(2)] for i in range(2)]
    coefficients = cagrad_coefficients(gram, .4)
    torch.testing.assert_close(torch.tensor(report['coefficients']), torch.tensor(coefficients),
                               atol=1e-6, rtol=1e-5)
    sum(reference_losses).backward()
    for p, gv, ga in zip(reference_common, *task_grads):
        p.grad = coefficients[0] * gv + coefficients[1] * ga
    for name, p in model.named_parameters():
        actual = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
        torch.testing.assert_close(actual, dict(reference.named_parameters())[name].grad,
                                   atol=1e-6, rtol=1e-5)
    norm = strategy.clip_grad_norm_(model.parameters(), .1)
    reference_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), .1)
    torch.testing.assert_close(norm, reference_norm, atol=1e-6, rtol=1e-5)
    optimizer.step()
    reference_optimizer.step()
    for name, p in model.named_parameters():
        actual = p.full_tensor() if isinstance(p, DTensor) else p
        torch.testing.assert_close(actual, reference.state_dict()[name], atol=1e-6, rtol=1e-5)
    if rank == 0:
        (Path(root) / 'oracle.json').write_text(json.dumps({'global_update_matches': True}))
    strategy.close()


@pytest.mark.parametrize('accelerator', ['cpu', 'gpu'])
def test_fsdp_cagrad_update_matches_dense_global_batch_with_inactive_rank(tmp_path, accelerator):
    import json
    from tests.test_gradientwam_distributed import _free_port
    if accelerator == 'gpu' and (torch.cuda.device_count() < 3 or not dist.is_nccl_available()):
        pytest.skip('three free CUDA GPUs and NCCL required')
    mp.spawn(_fsdp_global_update_oracle, args=(3, _free_port(), str(tmp_path), accelerator), nprocs=3, join=True)
    assert json.loads((tmp_path / 'oracle.json').read_text())['global_update_matches']


def test_inactive_fsdp_task_runs_zero_backward_without_marking_connection():
    from gradientwam.cagrad_training import CAGradGradientAccumulator
    common = torch.nn.Parameter(torch.tensor(1.))
    observed = []
    common.register_hook(lambda grad: observed.append(float(grad)))
    accumulator = CAGradGradientAccumulator((common,), c=.4)
    losses = {'video': common * 2, 'action': common * 3}
    accumulator.accumulate(losses, {'video': True, 'action': False}, scale=1,
                           include_inactive_tasks=True)
    assert observed == [2., 0.]
    assert accumulator.active == [True, False]
    assert accumulator.connected == [[True, False]]
    assert accumulator.action_grads[0] == 0
    accumulator.close()
