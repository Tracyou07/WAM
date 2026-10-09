"""CPU reference for the weighted regression surrogate, not a training integration."""
import ast
import enum
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'src' / 'open_wam'


def route_objective_weighted(pred_private, pred_shared, target, action_mask,
                             timestep_weight, prior_shared):
    """[B,T,D], [B,T,D]|None, [B,T], [B]; mirror official token reduction."""
    if target.ndim != 3 or pred_private.shape != target.shape or pred_shared.shape != target.shape:
        raise ValueError('expected matching [B,T,D] predictions and target')
    b, t, d = target.shape
    if not b or not t or not d or timestep_weight.shape != (b, t) or prior_shared.shape != (b,):
        raise ValueError('invalid geometry')
    if timestep_weight.requires_grad or (action_mask is not None and action_mask.requires_grad):
        raise ValueError('weights and mask must be parameter/route independent')
    if action_mask is not None and action_mask.shape != target.shape:
        raise ValueError('mask must use official [B,T,D] layout')
    m = torch.ones_like(target, dtype=torch.float32) if action_mask is None else action_mask.float()
    w = timestep_weight.float()
    if not torch.isfinite(w).all() or not torch.isfinite(m).all() or (w < 0).any() or (m < 0).any():
        raise ValueError('invalid fixed weights/mask')
    if not torch.isfinite(prior_shared).all() or ((prior_shared < .1) | (prior_shared > .9)).any():
        raise ValueError('prior outside fixed [0.1,0.9] contract')
    denom = m.sum(-1).clamp_min(1.0)
    branch = []
    for pred in (pred_private, pred_shared):
        e = torch.nn.functional.mse_loss(pred.float(), target.float().detach(), reduction='none')
        token_loss = (e * w[:, :, None] * m).sum(-1) / denom
        branch.append(token_loss.mean(-1))
    ell = torch.stack(branch, -1)
    if not torch.isfinite(ell).all():
        raise ValueError('nonfinite branch loss')
    p = torch.stack((1 - prior_shared, prior_shared), -1).float()
    joint = p.log() - ell
    log_q = joint.log_softmax(-1)
    q = log_q.exp()
    nll = -joint.logsumexp(-1)
    return {'loss': nll.mean(), 'per_sample_nll': nll, 'branch_weighted_mse': ell,
            'posterior': q, 'kl': (q * (log_q - p.log())).sum(-1),
            'coefficient': w[:, :, None] * m / (t * denom[:, :, None])}


def load_original_nodes(relative, names, env):
    """Execute unchanged, reviewed AST definitions; omit package initialization."""
    path = SOURCE / relative
    tree = ast.parse(path.read_text(encoding='utf-8'))
    selected = [node for node in tree.body if getattr(node, 'name', None) in names]
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id in names for x in node.targets):
            selected.append(node)
    header = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[header] + selected, type_ignores=[]))
    exec(compile(module, str(path), 'exec'), env)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20261008)
    checks = {}
    env = {'torch': torch, 'Any': Any}
    load_original_nodes('models/action_decoders/dual_expert_decoder.py', {'_masked_action_flow_match_loss'}, env)
    official = env['_masked_action_flow_match_loss']
    p = torch.tensor([.1, .5, .9], requires_grad=True)
    a = torch.randn(3, 4, 5, requires_grad=True)
    target = torch.randn_like(a)
    mask = torch.rand_like(a) > .3
    mask[0, 1] = False
    mask[2] = False  # official reduction retains empty sample in B denominator
    weights = torch.tensor([[.2, 3., 0., 1.], [5., 1., .5, 2.], [1., 1., 1., 1.]])
    class Scheduler:
        def training_weight(self, timesteps):
            return weights.flatten()
    baseline = official(flow_pred=a, targets=target, timesteps=torch.zeros(3, 4),
                        scheduler=Scheduler(), action_mask=mask, action_dim=5)
    same = route_objective_weighted(a, a, target, mask, weights, p)
    torch.testing.assert_close(same['loss'], baseline, atol=2e-6, rtol=2e-6)
    ga = torch.autograd.grad(same['loss'], a, retain_graph=True)[0]
    gb = torch.autograd.grad(baseline, a, retain_graph=True)[0]
    torch.testing.assert_close(ga, gb, atol=2e-6, rtol=2e-6)
    checks['unchanged_official_loss_identical_branch_value_gradient_parity'] = True
    torch.testing.assert_close(same['posterior'][:, 1], p, atol=2e-6, rtol=2e-6)
    assert (ga[2] == 0).all() and (ga[0, 2] == 0).all()
    checks['zero_weight_empty_tokens_empty_sample_match_official'] = True
    other = torch.randn_like(a, requires_grad=True)
    out = route_objective_weighted(a, other, target, mask, weights, p)
    elbo = (out['posterior'] * out['branch_weighted_mse']).sum(-1) + out['kl']
    torch.testing.assert_close(elbo, out['per_sample_nll'], atol=2e-6, rtol=2e-6)
    detached = out['posterior'].detach()
    prior2 = torch.stack((1-p, p), -1)
    surrogate = (detached * (out['branch_weighted_mse'] + detached.log() - prior2.log())).sum(-1).mean()
    direct = torch.autograd.grad(out['loss'], (a, other, p), retain_graph=True)
    indirect = torch.autograd.grad(surrogate, (a, other, p))
    for x, y in zip(direct, indirect):
        torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-6)
    checks['weighted_evidence_elbo_and_envelope_gradient_identity'] = True
    # Counterexample: token means [1,4] versus global coordinate mean [1,4,4]/3.
    simple = torch.tensor([[[1., 99.], [2., 2.]]])
    m = torch.tensor([[[1., 0.], [1., 1.]]])
    token = route_objective_weighted(simple, simple, torch.zeros_like(simple), m, torch.ones(1,2), torch.tensor([.5]))
    global_mean = (simple.square()*m).sum()/m.sum()
    assert abs(token['loss'].item()-2.5) < 1e-6 and global_mean.item() == 3.
    checks['v01_reduction_counterexample'] = True
    # Execute the actual visibility kernel over every pair of 24 synthetic tokens.
    class StrEnum(str, enum.Enum):
        pass
    env.update({'StrEnum': StrEnum, 'IntEnum': enum.IntEnum})
    load_original_nodes('configs/enums.py', {'HistoryStreamVisibility'}, env)
    load_original_nodes('models/common/packed_token_layout.py', {'PackedTokenStream'}, env)
    load_original_nodes('contracts/sample_metadata.py', {'DYNAMICS_CONDITIONAL_HISTORY_PREVIOUS_BOUNDARY_VIDEO_ONLY'}, env)
    names = {'ACTION_NOISY_TO_VIDEO_COUPLING', 'ACTION_THEN_VIDEO_COUPLING',
             'CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY', 'DECOUPLED_SAME_STEP_COUPLING',
             'JOINT_COUPLING', 'VIDEO_NOISY_TO_ACTION_COUPLING'}
    load_original_nodes('models/common/attention_contracts.py', names, env)
    load_original_nodes('models/common/chunked_attention_visibility.py',
                        {'_previous_boundary_frame_ids', 'build_history_stream_visibility_mask',
                         '_build_chunked_self_attention_visibility'}, env)
    tokens = torch.tensor([[chunk, stream, clean] for chunk in range(6) for stream in range(2) for clean in range(2)])
    chunk, stream, clean = tokens.T
    pairs = {}
    for name, values in [('seq', torch.zeros(24, dtype=torch.long)), ('block_id', 2*chunk+stream),
                         ('chunk', chunk), ('noise', clean), ('stream', stream),
                         ('effective_frame', chunk), ('valid', torch.ones(24, dtype=torch.bool))]:
        pairs['q_'+name], pairs['kv_'+name] = values[:,None], values[None,:]
    opts = dict(window_size=100, chunk_size=1, chunk_origin_frame=0,
                prefix_condition_frames=0, singleton_chunk_frame=None,
                current_block_coupling='decoupled_same_step',
                history_stream_visibility=env['HistoryStreamVisibility'].VIDEO_ONLY,
                conditional_history_policy='none')
    visibility = env['_build_chunked_self_attention_visibility'](**pairs, **opts)
    v_to_a = (stream[:,None] == 0) & (stream[None,:] == 1)
    assert not visibility[v_to_a].any()
    assert visibility[(stream[:,None] == 1) & (stream[None,:] == 0)].any()
    for prefix in (0, 1, 2):
        for singleton in (None, 0, 2):
            opts.update(prefix_condition_frames=prefix, singleton_chunk_frame=singleton)
            assert not env['_build_chunked_self_attention_visibility'](**pairs, **opts)[v_to_a].any()
    checks['official_visibility_kernel_no_video_reads_action_9_settings'] = True
    opts.update(prefix_condition_frames=0, singleton_chunk_frame=None,
                history_stream_visibility=env['HistoryStreamVisibility'].FULL)
    assert env['_build_chunked_self_attention_visibility'](**pairs, **opts)[v_to_a].any()
    opts.update(history_stream_visibility=env['HistoryStreamVisibility'].VIDEO_ONLY,
                current_block_coupling='joint')
    assert env['_build_chunked_self_attention_visibility'](**pairs, **opts)[v_to_a].any()
    checks['visibility_negative_controls_full_history_and_joint_detect_leak'] = True
    # Pre-fixed sensitivity grid in units of the OFFICIAL weighted per-sample loss.
    sensitivity = []
    for prior in (.1, .5, .9):
        for delta in (0., .001, .01, .04, .1, .4054651081081644, 1., 2.1972245773362196, 5.):
            logit = torch.logit(torch.tensor(prior, dtype=torch.float64)) + delta
            q = logit.sigmoid().item()
            sensitivity.append({'prior_shared': prior, 'ell_private_minus_shared': delta,
                                'posterior_shared': q, 'absolute_change': abs(q-prior)})
    result = {'torch': torch.__version__, 'device': 'cpu', 'seed': 20261008,
              'passed': len(checks), 'checks': checks, 'sensitivity_grid': sensitivity,
              'counterexample': {'official_token_mean': 2.5, 'v01_global_mean': 3.0},
              'executed_original_source': ['OpenWAM _masked_action_flow_match_loss (unchanged AST)',
                   'OpenWAM _build_chunked_self_attention_visibility and helpers (unchanged AST)'],
              'full_model_or_benchmark_reproduction': False, 'gpu_work': False}
    (ROOT/'operator_v02_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'passed': len(checks), 'checks': checks}, indent=2))


if __name__ == '__main__':
    main()
