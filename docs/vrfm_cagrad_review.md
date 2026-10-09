# VRFM + CAGrad independent review

Review date: 2026-10-09. Branch: `feat/vrfm-cagrad`. Base/HEAD:
`f85b35c722379b85d37d6a72ba21bdc2dc82049b`. Reviewed the shared working
tree, including new untracked files; this is not a frozen commit review.
The reviewer changed no model/training source, ran no GPU/remote job and
created no branch/commit. The only review artifact is this file; CPU
evidence is under `outputs/vrfm-cagrad-checks/algorithm-review-*.log`.

## Current verdict

The model's posterior/KL, prior-only evaluation, actual generation and fixed-z
cache/checkpoint paths follow the approved plan. The method-aware inference
loader is present and performs strict model-only loading before real native
generation. The trainer implements accumulation and global rank averaging
before the CAGrad solve, excludes the posterior from coordination, and clips
after composition. These observations are engineering evidence, not research
performance evidence.

The confirmed variable-batch P2 is now fixed and independently retested on the
final source. No open blocking finding remains within this bounded algorithm
and native CPU integration review. Final candidates are final video K/V plus
the VRFM video projection; action parameters and the action projection retain
ordinary gradients. Structural reachability is retained even when a shared
task gradient is numerically zero.

Signed off by the independent algorithm reviewer on 2026-10-09 for the source
hashes recorded below. This completes this review; PM integration checks and
branch/merge decisions are separate. CUDA/full-width/control performance is
outside this sign-off.

## Resolved finding

### [P2, resolved] Variable-batch task activity was lost before the CAGrad trainer

- Original location, before the fix: `src/open_wam/training/step_executor.py:385` (the top-level
  artifact lookup in `_native_dual_expert_task_activity`), together with
  `src/open_wam/pipelines/variable_batching.py:273` (the aggregate policy
  output) and `src/gradientwam/distributed_train.py:650` (required-activity
  guard). The equivalent smoke guard is in `src/gradientwam/train_smoke.py:31`.
- Trigger: run `cagrad` or `vrfm_cagrad` with a native `bucket`, `padded` or
  `packed` batch that takes the per-sample output aggregation path. A real
  two-sample native batch with frame lengths 4 and 6 reproduces the issue.
- Cause: variable batching correctly averages differentiable task losses and
  KL, but its aggregate `PolicyTrainOutput` has `decoder_artifacts=None`.
  Each `sample_outputs` entry still has the native artifact/masks. The new
  executor activity helper only examines the aggregate artifact and returns
  `None`, rather than aggregating activity from the sample artifacts.
- Effect: `TrainStepResult.task_losses` contains `action` and `video`, while
  `task_active` is `None`; the CAGrad micro-step raises
  `ValueError("CAGrad requires graph-bearing task losses and native activity masks")`
  before backward/update. Model-only variable-batch tests do not exercise
  this executor/trainer boundary.
- Evidence: independent CPU probe in
  `outputs/vrfm-cagrad-checks/algorithm-review-native-probe.log` reports this
  exact result for all three modes, with two sample outputs and no top-level
  artifact. The probe uses the actual native pipeline, bounded trainability,
  VRFM attachment, `LatentBatchCollator`, `LatentBatchAdapter` and
  `PipelineTrainStepExecutor`, without substituting fake decoder outputs.
- Implemented fix: `step_executor.py:385` recursively derives activity from
  native sample outputs and ORs each task across the aggregate batch. Fully
  masked tasks remain inactive. The duplicate gradient-module activity helper
  was removed; both trainers consume the executor's result.
- Independent closure evidence: `algorithm-review-variable-final.log` checks
  real two-sample, frame-length 4/6 batches in all three modes. Each mode runs
  a two-microbatch optimizer window with one inactive/one active action sample,
  then a separate window with all action samples inactive. All six updates
  pass: active windows apply CAGrad, inactive windows preserve the ordinary
  total gradients exactly. The native posterior and action-projection
  gradients match their ordinary loss gradients exactly in every case.
  The latest six training-method tests also pass, including recursive activity
  when both task masks are empty. This closes the reproduced P2.

## Model and decoder audit

- `dual_expert/vrfm.py`: diagonal Gaussian reparameterization; analytic weighted
  KL in FP32 to fixed N(0,I); one posterior sample outside checkpointed blocks.
  Detached input summaries preserve ordinary posterior reconstruction/KL
  gradients. Per-sample scalar moments preserve batch rows. This is the
  explicitly approved pooling adaptation, not an exact original encoder.
- Original valid action-channel masks enter posterior statistics; loss-only
  history masks remain native. Variable input padding is trimmed before
  posterior statistics. Differentiable per-sample task/KL means survive
  variable-batch aggregation; executor activity now survives that aggregation.
- `dual_expert_decoder.py`: total loss equals the native weighted video/action
  losses plus weighted KL; task losses retain their graph. Debug latent output
  is detached, while the generator conditioning path remains differentiable.
- `packed_training.py`: eval selects the prior and zero KL; actual generation
  in `dual_expert/inference.py` always selects the prior even in train mode.
  One latent is sampled before the ODE closure and reused for both streams and
  positive/negative CFG calls.
- `models/common/video_action_inference.py:139`: new positive and negative
  cache objects are created for each denoising stage. No observed cache path
  reuses a previous trajectory's latent-conditioned tensors. Native cache
  parity and checkpoint recomputation checks cover this behavior.
- No private K/V or added policy experts are introduced. VRFM attachment
  preserves existing trainability; legacy/new attachments reject both orders.
- `gradientwam/inference.py`: constructs the declared architecture, attaches
  VRFM before strict loading, rejects full-state/non-tensor model wrappers,
  and returns an eval/frozen pipeline. Precomputed latents/text are its stated
  frontend contract; full pretrained assets are not downloaded by this helper.

## Training and continuation audit

- `configure_native_trainability` restricts native training to the action
  expert/packed action blocks and final video self-attention K/V. Four new
  arms call this before VRFM attachment and optimizer construction.
- Final candidates include final K/V and only the VRFM video projection;
  task-graph reachability chooses the common subset for each optimizer window.
  The video projection receives both losses in the approved native profile.
  The action projection, action expert and packed action blocks retain ordinary
  gradients. No posterior parameters enter the candidate set.
- The native mask matches this boundary. In
  `chunked_attention_visibility.py:131`, historical keys are video-only;
  same-chunk clean/noisy visibility preserves stream separation for the
  decoupled/prefix profile (`:195`, `:264`). Video queries cannot read action
  keys. The independent all-trainable native probe found no omitted nonzero
  common parameter outside candidates after excluding the ordinary posterior.
  Native action parameters had autograd-connected but exactly zero video
  gradients; perturbing them left video loss bit-identical. Expanding buffers
  to all action parameters would therefore be unjustified for this profile.
- Common selection uses connectivity, not `count_nonzero` of accumulated
  gradients. Shared parameters stay coordinated at stationary points or after
  gradient cancellation. The zero-gradient regression passes on final source.
- The adapter forwards the underlying DDP module. Task differentiation and
  ordinary backward therefore do not enter DDP's reducer. At the optimizer
  boundary, accumulated task buffers and connectivity flags are reduced
  across ranks, and the Gram/solver use those global mean gradients.
  Only remaining ordinary gradients are manually averaged; composed common
  gradients are not reduced again. CAGrad requires DDP; FSDP is rejected.
- The full native posterior gradient is the ordinary video + action + KL
  gradient. An independent real native VRFM+CAGrad probe compared every
  posterior gradient before/after composition and passed. The final variable
  batch probe repeats exact preservation across two microbatches and checks the
  ordinary action projection as well. The tiny native common subset is four
  K/V tensors (2,112 elements), plus one video-projection tensor (128 elements)
  with VRFM; these counts are fixture evidence, not full-model memory estimates.
- Clipping occurs once after composition/unscale, then AdamW. CAGrad c=0
  gives the official unrescaled common-task mean; it does not equal the
  baseline sum on common parameters. No AdamW displacement or action
  non-degradation guarantee follows from the solver.
- Resume checkpoints are optimizer-boundary snapshots. Native full-state
  loading restores model/optimizer/scheduler/strategy; rank sidecars check
  method/settings/scope, topology, data split, batch/accumulation and cursors.
  Rank RNG is restored after cursor replay; boundary-only saves make a
  persistent partially accumulated CAGrad buffer unnecessary.
- The same forward graph supplies both task gradients. VRFM uses extra global
  Torch RNG draws, so equal seeds across methods do not establish identical
  noise tensors. This is a later strict paired-experiment limitation, not a
  present algorithm correctness defect. Public method documentation explicitly
  states this boundary.

## Independent verification

Environment: the PM-provisioned pinned Linux CPU environment, Python 3.12.3,
Torch 2.11.0+cpu, offline assets, single-thread BLAS, `CUDA_VISIBLE_DEVICES=''`.
No packages were installed. The native probe reports CUDA uninitialized.

The independent native probe passed posterior-gradient preservation and
reproduced the activity omission above. Its first harness attempt used an
incorrect sample-module import and stopped before running any checks; it is
not a source/test failure. The corrected probe uses the existing native sample
and collator modules.

Focused independent suite:

```bash
python -m pytest tests/test_vrfm_native.py tests/test_vrfm_conditioning.py \
  tests/test_vrfm_checkpoint_loading.py tests/test_gradientwam_training_methods.py \
  tests/test_gradientwam_vrfm_cagrad_settings.py \
  tests/test_gradientwam_distributed.py -q -p no:cacheprovider
```

Result: **60 passed, 1 warning in 77.87s**; exit code 0, recorded in
`outputs/vrfm-cagrad-checks/algorithm-review-focused.log`. The warning is the
native CPU BF16 RMSNorm input/weight mismatch fallback; it is not a failed
gradient/finite-value assertion. This suite includes the actual two-rank Gloo
CAGrad global-reference update and separate two-rank TrainingRuntime continuation
tests with a toy policy, plus model-only reload followed by real prior generation.
This suite ran before the final activity/accumulator changes; the targeted final
verification below covers those changes without treating earlier tests as a
fresh full-suite result.

Additional independent continuation probe:
`outputs/vrfm-cagrad-checks/algorithm-review-native-resume.log` passed with
the actual tiny native VRFM+CAGrad pipeline and public
`gradientwam.checkpoint.save_step1/load_step1/restore_rng` functions. After
restoring the first update, second-update loss, every model tensor, AdamW
state, scheduler state and next Torch RNG state were bit-identical to
uninterrupted execution. A changed method identity was rejected before loading.
This is a single-device CPU check with one microbatch per update; it does not
establish combined native distributed-runtime cursor replay or CUDA parity.

Final incremental verification, on the final activity/candidate/accumulator
source hashes below:

- `algorithm-review-variable-final.log`: six actual native VRFM+CAGrad
  accumulated optimizer updates across `bucket`, `padded` and `packed`;
  mixed versus fully inactive action masks, exact ordinary posterior/action
  projection gradients, inactive-task fallback, finite gradients, clipping,
  AdamW advancement and accumulator-hook cleanup all pass. CUDA stays
  uninitialized. Each active window selects five tensors / 2,240 elements.
- `python -m pytest tests/test_gradientwam_training_methods.py -q
  -p no:cacheprovider`: **6 passed in 29.70s**, exit code 0, in
  `algorithm-review-training-final.log`. Includes shared zero-task-gradient
  retention and recursive all-masked activity.
- `algorithm-review-common-dependency.log`: independent native audit with and
  without VRFM, all trainable parameters, zero action-to-video gradients,
  no omitted nonzero common parameter outside the bounded candidates, and
  video-loss invariance under action-parameter perturbation. This probe ran
  before removal of the numerical-zero filter; the final source removes that
  filter and the targeted zero-gradient regression above verifies its behavior.

## Declined to judge / remaining evidence limits

- Full-width CUDA/BF16 execution, NCCL/eight-rank behavior, memory fit and
  closed-loop control gains were neither run nor approved by this review.
- Combined native VRFM+CAGrad optimizer continuation is verified on one CPU
  device. Combined native DDP/rank-cursor continuation remains unverified;
  toy DDP continuation/global update parity has a different scope.
- This reviewer did not rerun or certify the full repository suite. PM's final
  integration run is separate evidence and is not duplicated by this review.
- No claim of validated posterior capacity, hyperparameters, four-arm gains,
  strict cross-method stochastic pairing or original-paper reproduction.

## Working-tree snapshot

Representative SHA256 values at review/probe dispatch (case-insensitive):

| File | SHA256 |
|---|---|
| `src/open_wam/models/policy_variants/dual_expert/vrfm.py` | `d07858f5caf45fd471cc8ed60bcde4ccfed1805ead7cb9af02ea4f9e29af4698` |
| `src/open_wam/models/policy_variants/dual_expert/inference.py` | `c59702eb5170d1b05765839b290d901f36614b11a1e1214f3dcd2a1f6b311224` |
| `src/open_wam/models/policy_variants/dual_expert/packed_training.py` | `726f2aa02b62f20f67df5a59f918245bf60a8a6b51d906c0822b4653ffa214d3` |
| `src/gradientwam/cagrad_training.py` | `122b43823c736486b71d901511020b4566b17c18848ccfae26352317e4478ca5` |
| `src/gradientwam/distributed_train.py` | `d8a10dcf0ce7b077bcdab279c7af2e7f6cc5e49a31d4486985ea5573194ef630` |
| `src/gradientwam/train_smoke.py` | `d20b33fd8105991e289b4b470dc7dcb7ce8b72ec7a96b296c8b9dae1c9ba9171` |
| `src/gradientwam/inference.py` | `eda8ab2844b70e32f554f6bbb8911b1bd9cde685fdbe086df13ae5c96f242cd0` |
| `src/open_wam/training/step_executor.py` | `abbff43f8c426ca9f12bd313c4db4ebccf6a6b8e1c1baa614f0c30d5cb6c9fe4` |
| `src/open_wam/pipelines/variable_batching.py` | `2f5892fed3abb2d4f466b31e888a913f0f666fe6859c7e3728eafd58624bec7e` |

The table above is the initial review snapshot, preserving provenance for the
earlier focused suite and checkpoint probe. Final incrementally reviewed files:

| File | Final SHA256 |
|---|---|
| `src/gradientwam/cagrad_training.py` | `bae8599672ece87598ce994a79997d66a0c4c3c26025240bf238e1544788cb4c` |
| `src/open_wam/training/step_executor.py` | `8312de87d9442dc79941843dcf074abeb313227359447e18b1e1af20d3801114` |
| `tests/test_gradientwam_training_methods.py` | `1dbad36caa909cd1aeacae0713d645f7fd2637f128c9cc26b89b721aa578c7fd` |
| `tests/test_gradientwam_delivery.py` | `068a1c6da9cb75d7b276662af6e127d4b0b9441a96f10d9d7d9ec03228da2649` |

The distributed trainer and smoke caller hashes remain those in the initial
table. Final source hashes were checked again after the incremental run. Later
source mutations require corresponding verification; this sign-off does not
extend to unknown future changes.
