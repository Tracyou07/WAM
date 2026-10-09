# VRFM + CAGrad: formula, integration contract and independent audit

2026-10-09. This document audits the approved new method and the standalone
solver. It does not certify another owner's model/trainer implementation or
report a training or control improvement. The implementation plan is
[2026-10-09-vrfm-cagrad.md](superpowers/plans/2026-10-09-vrfm-cagrad.md).

## Sources and permission boundary

| Source | What was checked | Reuse status |
| --- | --- | --- |
| Liu et al., [Conflict-Averse Gradient Descent](https://arxiv.org/html/2110.14048), NeurIPS 2021 | Eq. (3), the dual and Algorithm 1 | Mathematical reference for an independent implementation. |
| [Official CAGrad repository](https://github.com/Cranial-XIX/CAGrad/tree/dc3d48152b6196945cfd56144879b9d42353b095) | Fixed commit `dc3d48152b6196945cfd56144879b9d42353b095`; [toy.py](https://github.com/Cranial-XIX/CAGrad/blob/dc3d48152b6196945cfd56144879b9d42353b095/toy.py#L278), [nyuv2/utils.py](https://github.com/Cranial-XIX/CAGrad/blob/dc3d48152b6196945cfd56144879b9d42353b095/nyuv2/utils.py#L377) and [LICENSE](https://github.com/Cranial-XIX/CAGrad/blob/dc3d48152b6196945cfd56144879b9d42353b095/LICENSE) | MIT, copyright 2022 Bo Liu. No author implementation was copied into this solver or its tests. These new files use the repository's AGPL-3.0-only license. |
| Guo and Schwing, [Variational Rectified Flow Matching](https://proceedings.mlr.press/v267/guo25i.html), ICML 2025; [full text](https://arxiv.org/html/2502.09616) | Eq. (5), training Algorithm 1, inference Algorithm 2 and code links | The paper specifies a Gaussian posterior, fixed Gaussian prior and trajectory-fixed latent. This bounded check did not confirm an official VRFM method repository or its software license. No VRFM source code was copied; do not label this WAM adaptation an official-code reproduction. |

The VRFM paper links `YangLing0818/consistency_flow_matching` as a baseline
implementation. That link does not establish a VRFM implementation or grant
a license for an unidentified VRFM codebase. A missing confirmed link is not
proof that no code exists.

Official CAGrad `toy.py` divides its direction by `1+c` and uses additive
epsilons. The NYUv2 function is hard-coded for three tasks, adds epsilons and
offers `rescale=0`, `1`, or `2`; its default is not the raw paper direction.
Our two-task interface implements the unrescaled mathematical problem with
explicit degeneracy handling, not bitwise parity with those regularized
reference programs. No SciPy, NumPy or Torch dependency is needed by the solver.

## Two-task objective and coefficients

Let `gv` and `ga` be gradients of the already weighted video and action losses
on the **same structurally common generator parameter subset**. Define

\[
g_0=(g_v+g_a)/2,\quad G_{ij}=g_i^Tg_j,\quad \rho=c\|g_0\|,
\qquad 0\le c<1.
\]

The paper's primal problem maximizes the worse directional improvement:

\[
\max_d\min(g_v^Td,g_a^Td)\quad\text{subject to }\|d-g_0\|\le\rho.
\]

For `w=(x,1-x)`, its dual minimizes
`F(x)=gw.T@g0 + rho*norm(gw)` on `[0,1]`, where
`gw=x*gv+(1-x)*ga`. When `gw` is nonzero,

\[
d=g_0+\rho g_w/\|g_w\|,
\quad k_v=\tfrac12+\rho x/\|g_w\|,
\quad k_a=\tfrac12+\rho(1-x)/\|g_w\|.
\]

The returned coefficients `(kv,ka)` are **not probabilities**, and must not
be renormalized to sum to one. There is no division by `1+c` or `1+c*c`.

For reproducible two-task solving, write `a=G00`, `b=G11`, `h=G01` and
`s=a+b-2h`. The dual derivative has linear slope `(a-b)/2`, plus the
derivative of a norm. Its norm center is `(b-h)/s` and squared minimum is
`(a*b-h*h)/s`. Comparing the linear slope with `rho*sqrt(s)` identifies a
monotone boundary optimum; otherwise the unique smooth zero of the derivative
is solved algebraically and clipped to `[0,1]`. This derivation is independent
of the author's numerical optimizer.

## Direct trainer interface

```python
from gradientwam.cagrad import cagrad_coefficients

# gram64 comes from the complete optimizer-batch, globally averaged gradients.
kv, ka = cagrad_coefficients(gram64.detach().cpu().tolist(), c=0.4)
for parameter, gv, ga in common_parameter_gradients:
    parameter.grad = kv * gv + ka * ga
```

The trainer owns these preconditions; the scalar solver cannot verify them:

1. Apply configured video/action loss weights **before differentiation**.
   Multiplying the action contribution after CAGrad is a different algorithm.
   Preserve native timestep weights, masks and token reductions; there is no
   hidden per-task norm normalization.
2. Both losses use the same forward sample, including the same sampled `z`.
   Accumulate each task gradient over all microbatches, then average across
   distributed ranks before building the Gram and solving once. Averaging
   Gram matrices or per-rank coefficients is not equivalent to solving on
   globally averaged gradients. Do not reduce the composed direction twice.
3. Form the Gram on common generator parameters only, with float64 dot
   products. Structural ownership is not determined by whether a gradient
   happens to be numerically zero in this batch. Parameters used by only one
   task retain that task's weighted gradient, without CAGrad coefficients.
4. Exclude posterior parameters from CAGrad. They receive the ordinary sum
   of weighted video loss, weighted action loss and weighted KL gradients.
   KL is not a third coordinated task. The posterior inputs must not create
   an undeclared KL gradient path into generator parameters.
5. Unscale any mixed-precision gradients consistently before composition;
   clip once after composing all parameter gradients, then step AdamW.
   `c=0` is the mean on the common subset, not the unhalved sum of two losses.
   This normalization must be explicit in baseline comparisons.

CAGrad is symmetric in the supplied tasks. For example, `gv=2` and `ga=-1`
give `d=0.3` at `c=0.4`: `ga*d<0`, so an SGD step can increase action loss.
It is not an action-protection constraint. The paper's gradient-descent
analysis does not automatically extend to AdamW moments, clipping, selective
parameter composition, stochastic training or closed-loop success.

## Invalid and degenerate inputs

Inputs must be a finite symmetric PSD 2x2 real matrix and a finite scalar
`c` in `[0,1)`. Invalid shape, nonfinite values, material asymmetry, negative
diagonal or a Cauchy-Schwarz violation raise `ValueError`, including at `c=0`.
There is no silent fallback for corrupt gradients.

A common positive scaling of the Gram is removed internally to avoid overflow;
it does not change task norms relative to each other. Relative symmetry/PSD
roundoff up to `1e-12` is symmetrized/clamped. Compute the Gram in float64;
this tolerance is not permission to repair a materially invalid FP32 matrix.
Nearly cancelling gradients below the precision of the input Gram cannot be
reconstructed by a scalar solver.

| Situation | Deterministic result |
| --- | --- |
| `c=0` | `(0.5,0.5)`, the mean gradient. |
| Both gradients zero, or their mean is zero | `(0.5,0.5)`; the trust ball has radius zero and `d=0`. |
| Exactly one gradient zero | `(0.5,0.5)`; the mean is a feasible primal optimum among nonunique choices. |
| Identical nonzero gradients | `((1+c)/2,(1+c)/2)`, so `d=(1+c)*g0`. |
| Collinear positive `gv=2, ga=1, c=.4` | `d=2.1`; no hidden rescaling. |
| Opposed unequal `gv=2, ga=-1, c=.4` | `d=0.3`, despite an adverse action directional derivative. |

## Independent VRFM method-boundary audit

The approved WAM adaptation has a continuous diagonal-Gaussian training
posterior `q_phi(z | clean/noisy video, clean/noisy action, times, context)`.
For `logvar=log(sigma**2)`, use

\[
z=\mu+\exp(\tfrac12\mathrm{logvar})\odot\epsilon,
\quad\epsilon\sim N(0,I),
\qquad
K_i=\tfrac12\sum_j(\mu_{ij}^2+e^{\mathrm{logvar}_{ij}}-1-\mathrm{logvar}_{ij}).
\]

The stated objective is `weighted_Lv + weighted_La + beta * mean_i(K_i)`.
Summing the latent dimension and averaging the batch are explicit conventions;
changing to a latent-dimension mean changes the effective KL weight. The
default `latent_dim=32`, `beta=.001` and `c=.4` are configurable engineering
choices, not validated WAM hyperparameters.

| Boundary | Independent audit conclusion / integration acceptance condition |
| --- | --- |
| Continuous shared `z` | One sample conditions both native streams; there is no binary shared/private switch, additional expert or private K/V. Verify both streams actually respond to `z`. |
| Posterior only during training | Paired targets are legitimate posterior training inputs. Posterior parameters must receive reparameterized reconstruction gradients and analytic KL gradients. |
| Fixed prior at inference and heldout proxies | Sample `N(0,I)` without targets or posterior invocation. Hold one `z` fixed over an entire trajectory, including CFG calls and checkpoint recomputation where relevant. Explicit caller-supplied posterior latents must not become a default evaluation path. |
| Original WAM contracts | Preserve masks, schedulers, available context and data ordering. Adding `z` must not expose clean future or action targets through another generator input. |
| Separation from legacy routes | Four new arms are baseline, vrfm, cagrad, vrfm_cagrad. The legacy discrete mixture/private-path parameters, gates and resume identities must not be silently enabled or reused. |
| Scope of the variational statement | The cited VRFM paper models conditional velocities. Native weighted WAM losses and a configurable beta form a WAM adaptation; they do not automatically prove a clean joint video/action trajectory ELBO or inherit every marginal-distribution theorem. |

Mathematical ingredients and the separation of responsibilities are consistent.
Actual target-leakage, shared conditioning, fixed-latent, posterior-gradient,
global accumulation and checkpoint tests remain model/trainer integration
acceptance gates; this standalone solver does not certify them. Prior-only
heldout FM losses are proxies, not closed-loop control outcomes. A continuous
latent can also collapse or be ignored; merely adding KL does not establish
useful multi-modality.

An independent read of the in-progress native source additionally confirmed:

- `vrfm.py` computes KL as latent sum / batch mean. Posterior input summaries
  are detached, while reparameterized `z` remains differentiable; the generator
  projections are separate from the helper's posterior parameter list.
- `prior_sample` accepts a batch size, device and optional RNG, not targets.
  `configure_vrfm` rejects legacy routing/private K/V and preserves existing
  trainability flags.
- `packed_training.py` selects the posterior only in VRFM train mode, uses
  prior/zero KL in eval mode and projects the same `z` to both streams.
- `inference.py` draws the prior latent outside the ODE prediction closure and
  reuses it within that closure, including the shared CFG path.

This is a source-level observation, not a completed runtime leakage test.
The inspected snapshots had SHA-256:

| Source | SHA-256 |
| --- | --- |
| `src/open_wam/models/policy_variants/dual_expert/vrfm.py` | `d07858f5caf45fd471cc8ed60bcde4ccfed1805ead7cb9af02ea4f9e29af4698` |
| `src/open_wam/models/policy_variants/dual_expert/packed_training.py` | `726f2aa02b62f20f67df5a59f918245bf60a8a6b51d906c0822b4653ffa214d3` |
| `src/open_wam/models/policy_variants/dual_expert/inference.py` | `c59702eb5170d1b05765839b290d901f36614b11a1e1214f3dcd2a1f6b311224` |

## Executed verification

Test-first: the initial `c=0` check failed because the solver was absent.
After implementation, the focused command was:

```bash
python -m pytest tests/test_cagrad.py -q -p no:cacheprovider
```

41 CPU checks passed. These include 640 deterministic random gradient pairs
across five `c` values, hand-derived collinear/opposed/zero examples, scale
invariance, near cancellation, finite overflow prevention and invalid inputs.
The random reference solves the two-dimensional **primal** using extrema and
intersections on the trust-ball circle; it does not call the production dual
solver. It checks feasibility and the optimal worst-task inner product rather
than demanding a unique coefficient representation in degenerate cases.

This is numerical/operator validation only. No full-width model, GPU,
distributed training, asset download or benchmark was run by this task.
The solver was also loaded and exercised under Python `-I -S` with no site
packages; Torch, NumPy and SciPy were not imported. Syntax and whitespace
checks passed on the three owned delivery files. No other contributor's
source was edited, and no branch change, commit or push was performed.

The same focused tests were subsequently run in the PM-provisioned pinned
Linux CPU environment: Python 3.12.3, Torch 2.11.0+cpu, CUDA hidden/uninitialized,
offline model access and one BLAS/OpenMP thread. Result: **41 passed in 3.25s**.
The persistent UTF-8 log is `outputs/vrfm-cagrad-checks/algorithm-cagrad-linux-cpu.log`
(an ignored local validation artifact). No dependencies were installed or
older environments changed by this task.
