# VRFM model integration

Public interface is
`open_wam.models.policy_variants.dual_expert.vrfm.configure_vrfm(pipeline, latent_dim=32, kl_weight=0.001)`.
Call after native assembly/trainability selection and before optimizer/DDP/FSDP construction.
It preserves every existing parameter's `requires_grad`; do not call legacy
`configure_variational_sharing` for any of the four new arms.

`vrfm_posterior_parameters(pipeline)` identifies only q encoder parameters.
The new `policy_variant.vrfm.video_projection` and `action_projection` belong
to the generator, not the posterior. New parameters are trainable. The Gaussian
posterior, fixed unit Gaussian prior, and injection projections have no private
K/V or additional expert. Disabled baseline/CAGrad add no modules/parameters.

Important for CAGrad: in the decoupled native profile, `video_projection` receives
BOTH world and action reconstruction gradients, because action queries read
video K/V. It is eligible for the common generator set along with final shared
K/V. Do not classify all projections as single-task solely by their names.
`action_projection` receives only action gradients in that profile; other native
coupling programs may change reachability. Identify common parameters using the
actual task graph. Posterior parameters remain excluded irrespective of reachability.

Decoder `aux['task_losses']` has differentiable, native-weighted `video` and
`action` scalars in every arm. `aux['vrfm_kl_loss']` is weighted KL (zero without
VRFM). Decoder `loss` equals their sum. Do not detach these before autograd.

Training posterior is selected by the VRFM module's train mode, propagated by
`pipeline.train()`. Heldout loss calls must use `pipeline.eval()`; they sample
the fixed prior and emit zero KL even when using `forward_train_from_latents`.
Real `forward_infer_step*` always samples the prior regardless of train mode.
One z is sampled outside the ODE/checkpointed blocks and shared by both streams.
Restore pipeline mode after heldout evaluation. Standard torch RNG determines
sampling and must be checkpointed by the trainer.

## Posterior and injection

q is a diagonal continuous Gaussian, using a 128-unit SiLU MLP on detached
per-channel clean/noisy video means and standard deviations, per-dimension
clean/noisy action moments, native video/action timestep moments, resolved
text-context mean, and proprioceptive moments. Action statistics use the original
valid-channel action mask (not a loss-only history mask); masked coordinates
cannot change q. In variable-length execution the existing bridge trims padding
before these statistics. Training q may see clean future targets; this is
intentional posterior conditioning. Its outputs never enter generation.

`z = mu + exp(0.5*logvar)*epsilon` uses one torch RNG draw per sample forward;
log variance is clamped to [-20,10] for finite mixed-precision exponentiation.
KL is `beta/2 * mean_B(sum_z(mu^2 + exp(logvar) - 1 - logvar))`, computed in
float32. Reconstruction and KL both differentiate to the encoder. Raw paired
inputs are detached at q, preventing q from adding a target-conditioned gradient
path through upstream context modules.

Two bias-free trainable linear projections map the SAME z to native video/action
hidden widths. Add the broadcast vector to all native packed stream tokens
after embedding and before the paired block stack. Original clean/noisy values,
attention visibility, loss masks, schedulers, and timesteps remain authoritative.
No z draws occur within blocks. Inference closes over one prior z outside the
trajectory loop, including positive/negative CFG and cache calls. There is no
learned prior and no target-dependent route selection.

The pooling posterior is an initial engineering implementation, not evidence of
research gains or a validated optimal representation. The runtime configuration
stays outside the native ExperimentConfig and belongs to training's method
identity. Attach identical native training ranges for all four arms BEFORE this
function. Existing generator flags are unchanged; projection parameters belong
to the generator and posterior helper returns only the four q MLP tensors.

## Verification ledger

- Initial focused checks: nine expected feature failures (missing attachment and
  missing decoder task-loss contract), followed by nine passing native checks.
- Added posterior batch-independence and two-way legacy rejection checks failed,
  then passed after scalar moments kept batch rows separate and legacy attachment
  rejected an already configured VRFM pipeline. Thirteen focused checks passed.
- Six variable-batch checks first failed on missing differentiable task aux,
  then passed with graph-preserving per-sample mean aggregation. Final acceptance
  includes variable batch, CPU BF16, CFG/cache parity, and checkpoint recomputation.
- Ruling: no commits, branch operations, subagents or remote/GPU work; PM owns
  integration and algorithm owns independent review. Model owner reviews its diff.

## CPU validation

The final independent Linux CPU environment has Python 3.12.3, torch 2.11.0+cpu,
diffusers 0.37.1, transformers 5.10.4 and pytest 9.1.1. It uses
requirements/reproduce-linux.txt. A generic equivalent acceptance command is:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
.venv-cpu/bin/python \
  -m pytest tests/test_vrfm_native.py tests/test_vrfm_conditioning.py -q
```

Logs are in `outputs/vrfm-cagrad-checks/infra-*.log`. No package installation was
performed by this model owner; PM owns CPU environment setup.

The native acceptance command additionally covered existing variational pipeline,
packed-block, variable-batch and decoder-artifact tests: 158 passed in 48.32s.
The CPU BF16 test reports the existing native RMSNorm input/weight dtype fallback
warning; all gradients were finite. The subsequent final focused run, including
the cross-task `video_projection` assertion, passed 21 tests in 32.42s (same warning).
These test sets overlap; do not add their counts as independent checks.

The complete `pytest -q` attempt stopped at collection with two unrelated errors:
`tests/test_gradientwam_training_methods.py` imported `gradientwam.cagrad_training`
while the training owner had not yet created that module;
`tests/test_pypi_distributions.py` required absent `.github/workflows/publish-pypi.yml`.
The result was 1 skipped, 2 collection errors in 78.12s. These are recorded in
`infra-full-suite.log`; the complete repository suite is not claimed green.

Final native regression command:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
.venv-cpu/bin/python -m pytest \
  tests/test_vrfm_native.py tests/test_vrfm_conditioning.py \
  tests/test_variational_native_pipeline.py tests/test_dual_expert_packed_block.py \
  tests/test_variable_batch_pipeline.py tests/test_decoder_artifact_contracts.py -q
```

Owned model changes are `dual_expert/vrfm.py`, native `variant.py`,
`packed_training.py`, `inference.py`, `batch_execution.py`, `dual_stream_execution.py`,
the two-way legacy guard in `variational_sharing.py`, `dual_expert_decoder.py`,
and `pipelines/variable_batching.py`. No native training files were changed by
the model owner. The focused test files are `test_vrfm_native.py` and
`test_vrfm_conditioning.py`. No full-width GPU/control outcome is established.

With native dimensions C=48, action_dim=7, text_dim=4096, video width=3072,
action width=2048 and z_dim=32, source-derived extra counts are 561,600 posterior
parameters plus 163,840 generator projection parameters, 725,440 total. These are
architecture counts, not measured GPU memory or full training evidence.

Final model-owner self-review covers input detach, masks, mode transition,
checkpoint recomputation and both configuration orders. Independent review is
owned by the algorithm engineer; no subagent was dispatched.

## Model-only checkpoint inference

`gradientwam.inference.load_policy_for_inference(settings, checkpoint, device='cpu')`
accepts a `gradientwam.settings.Settings` instance or its run-settings YAML path.
It assembles the declared native architecture without base/frontend downloads,
attaches VRFM only for `vrfm`/`vrfm_cagrad`, invokes the native strict checkpoint
loader, moves the complete pipeline to the requested device and returns it in
eval mode with all parameters frozen. It constructs no optimizer. Legacy settings
are rejected. Baseline and CAGrad share their native inference architecture;
VRFM and VRFM+CAGrad share their VRFM architecture. This is architecture/tensor
compatibility checking, not full training-resume or hyperparameter identity.

Require a complete **model-only** `.pt` checkpoint: a raw tensor state dict, or
exactly `{'model_state_dict': pipeline.state_dict()}` (the native `model_state.pt`
format), or exactly a `state_dict` wrapper. All model entries must be named
tensors. Legacy private-path parameters, missing/unexpected model keys or shape mismatches are not
ignored. The small optional native `require_model_only=True` guard rejects extra
training fields and non-tensor model entries before normalization. Existing
native callers retain their prior default behavior. Point a run/step directory
at its exported model_state.pt; full-only training-state payloads are rejected.
Export model tensors from the unwrapped pipeline on the training host when
only a resumable checkpoint exists; this API does not extract optimizer state.

For the public settings paths, supply the same `GW_*` path/config environment
variables used to parse run settings. Asset contents and the public base weights
are not read by this loader. The returned pipeline expects already encoded
observations and text; RGB VAE/T5 frontend setup remains a separate operation.
Set a torch RNG seed AFTER loading for repeatable prior/noise draws.

```python
import torch
from gradientwam.inference import load_policy_for_inference
from open_wam.models.policy_variants.contracts import PolicyInferContext

policy = load_policy_for_inference(
    'configs/gradientwam/vrfm_cagrad.yaml',
    '/absolute/path/to/run/checkpoint_step_100/model_state.pt',
    device='cpu',
)
torch.manual_seed(42)
with torch.inference_mode():
    result = policy.forward_infer_step_from_latents(
        observed_video_latents,  # [B,48,T_observed,H_latent,W_latent], observations only
        PolicyInferContext(state=observed_robot_state),
        text_context=prompt_embedding,  # [B,L,text_dim], same encoder as training
    )
actions = result.decoder_output.action_pred
```

Initial checkpoint-entry checks failed on the missing API before implementation.
Final `test_vrfm_checkpoint_loading.py` plus existing `test_checkpoint_runtime.py`:
**28 passed in 22.15s**, log `outputs/vrfm-cagrad-checks/infra-inference-green.log`.
This includes all four methods' exact native save/reload/prior-generation parity,
strict missing/unexpected/shape/wrong-method/wrong-z-width rejection, model-only
payload enforcement, legacy rejection, and real settings-YAML assembly using
30 layers of tiny hidden width without pretrained assets. Added delivery files
are `src/gradientwam/inference.py` and `tests/test_vrfm_checkpoint_loading.py`;
the native loader adds only its optional model-only guard. No full-suite rerun
was performed after this entrypoint change, as requested by PM.
