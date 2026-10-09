# VRFM + CAGrad training handoff

## Method settings schema (2026-10-09)

Each PM-owned settings YAML (`configs/gradientwam/*.yaml`, referenced by the run spec's `settings_config`) carries application-level method fields under `gradientwam`. Keep these outside OpenWAM's native `ExperimentConfig` and native trainability selectors:

```yaml
gradientwam:
  method: vrfm_cagrad  # baseline | vrfm | cagrad | vrfm_cagrad
  latent_dim: 32
  kl_weight: 0.001
  cagrad_c: 0.4
```

The typed enum lives in `gradientwam.settings.GradientWAMMethod` with values `baseline`, `vrfm`, `cagrad`, and `vrfm_cagrad`. These names map directly to four public arms. When omitted for a new config, method defaults to baseline, with latent_dim=32, kl_weight=0.001, and cagrad_c=0.4; the last three values remain in checkpoint/run identity even when a method does not consume one, so edits cannot silently resume a different recipe. Validate latent_dim as a positive integer, KL weight as finite and nonnegative, and CAGrad c as finite and in `[0, 1)`.

Call `configure_vrfm(pipeline, latent_dim=32, kl_weight=0.001)` before optimizer construction only for `vrfm` and `vrfm_cagrad`. Enable the training combiner only for `cagrad` and `vrfm_cagrad`. The pipeline decoder contract supplies graph-carrying `aux['task_losses'] = {'video': weighted_video_loss, 'action': weighted_action_loss}` and graph-carrying scalar `aux['vrfm_kl_loss']` (zero for non-VRFM arms). Heldout evaluation and inference use the fixed prior; they never encode targets or sample the training posterior.

Use one explicit, budget-bounded native trainability profile in all four new arms: freeze the assembled pipeline, re-enable all native action-expert blocks, then re-enable only the final paired video block's shared `attn1.to_k` and `attn1.to_v`. Do not attach private K/V or a routing controller. This intentionally narrows the broad `visual_tower.runtime_backbone` selector in the current experiment YAML, matching the old native-joint controlled range. A future full-video profile needs its own named budget/config; it is not the new-method default. Masks, noise schedules, and the video/action experts stay native.

VRFM adds its posterior network and two stream-conditioning projections; these method parameters are trainable only in the VRFM arms. The four methods share the same native trainable parameter mask. The structural CAGrad candidates are the final shared video K/V, plus `vrfm.video_projection` in the VRFM arms; the posterior and `vrfm.action_projection` are excluded. Runtime narrows these candidates by native task-graph connectivity, never by whether a computed gradient happens to be nonzero. A zero gradient caused by cancellation remains part of the common geometry. The native `decoupled_same_step` + `video_only` mask blocks video queries from action-stream keys; the tiny-model audit measured exactly zero video-gradient max-absolute value for action-expert/action-block parameters and `vrfm.action_projection`, while their action-gradient max-absolute values were 2.0490100 and 0.8477128 respectively. The video projection remains a shared candidate; the action projection and action expert keep ordinary total-loss gradients. The VRFM posterior receives ordinary video + action + KL gradients. Apply accumulation and cross-rank task-gradient averaging before solving; combine once, clip once, then AdamW.

Task activity comes from native action and future-video masks. For variable-batch aggregate outputs, each task is active when at least one sample has valid tokens; a fully masked batch leaves both objectives inactive. In that case CAGrad is skipped and the ordinary total-gradient path remains in effect. This handles aggregate outputs whose top-level decoder artifact is absent while retaining per-sample native artifacts.

The GradientWAM distributed launcher currently fixes the strategy to DDP. For CAGrad, suppress DDP's automatic synchronization throughout the optimizer accumulation window, average the two task-gradient buffers across ranks, solve once from those global means, install the composed common gradients, and manually average only the remaining ordinary gradients. This avoids both per-rank solves and a second synchronization of already-global common gradients. Do not claim FSDP support for this path; its sharded-gradient integration is not implemented.

Old v0.2 private-K/V routes are not defaults and do not map to any new method. Compatibility is opt-in only: require `legacy_v02: true` together with an allowed old `arm`, and preserve both in checkpoint identity. Reject a legacy `arm` without the marker; a missing new `method` may default only to baseline when no legacy arm is present; unknown values fail closed.

## CAGrad buffer memory estimate

CAGrad uses two additional persistent FP32 task-gradient buffers for the common generator subset, plus the model's ordinary `.grad`. The controlled profile has 18,880,512 shared final-K/V parameters, so the two buffers are 151,044,096 bytes (144.05 MiB) per DDP rank for baseline/CAGrad. VRFM+CAGrad adds 98,304 `video_projection` parameters, for 18,978,816 candidate parameters and 151,830,528 bytes (144.75 MiB) per rank at the full structural candidate size. Exclude short-lived autograd and communication-bucket buffers. Runtime graph connectivity may reduce the actual common subset; inspect it after model attachment before a GPU run.

## Static parameter and memory estimate

The previously audited v0.2 variational smoke manifest had 2,600,806,456 trainable parameters, including 18,880,512 private K/V parameters and a 17-parameter route prior. Removing those legacy-only parameters gives 2,581,925,927 trainable native generator parameters for baseline/CAGrad. Current native dimensions are 48 video latent channels, action dim 7, text dim 4096, video width 3072, action width 2048, and VRFM latent dim 32. The posterior input has 4×48 + 4×7 + 4 + 4096 + 2 = 4,322 features and its network has 561,600 parameters; the two stream projections have 163,840 parameters, adding 725,440. Thus VRFM/VRFM+CAGrad are estimated at 2,582,651,367 trainable parameters. These counts are source/config-derived; verify on the target full model before making a hardware-fit claim.

| Method(s) | Trainable parameters | FP32 trainable weights + gradients + AdamW moments | Extra CAGrad buffers |
|---|---:|---:|---:|
| baseline, cagrad | 2,581,925,927 | 41.31 GB / 38.47 GiB minimum | cagrad: 151.0 MB / 144.0 MiB |
| vrfm, vrfm_cagrad | 2,582,651,367 | 41.32 GB / 38.48 GiB minimum | vrfm_cagrad: 151.8 MB / 144.75 MiB |

If about 20.43 GB of public video weights are resident on each DDP rank, a rough combined estimate is about 61.7 GB per rank before activations, CUDA/NCCL workspace, temporary gradients, or allocator reserve. This is not a strict lower bound: the public-weight byte count overlaps with trainable state, including the final K/V parameters, so the sum may double-count those weights. DDP replicates state per rank; adding GPUs does not divide a rank's memory. The 64 GiB full-checkpoint cap remains unverified until serialized full-state size is measured. No GPU fit or run is claimed by these arithmetic estimates.

## Implementation status and validation

- Native trainability, VRFM/CAGrad integration, variable-batch activity aggregation, optimizer accumulation, and checkpoint identity/resume are implemented in the training path.
- A focused CPU run passed 6 tests in 24.42s (19 deselected): all four tiny native method updates, recursive variable-batch activity including the all-masked fallback, zero-valued shared-task-gradient retention, and the VRFM+CAGrad step-1 to step-2 checkpoint resume.
- The same tiny-model audit measured action-expert/action-block video-gradient max-absolute value 0.0 and action-gradient max-absolute value 2.0490100; `vrfm.action_projection` measured 0.0 and 0.8477128 respectively. The mask source is `src/open_wam/models/common/chunked_attention_visibility.py`: `build_history_stream_visibility_mask` returns `kv_stream == VIDEO` for `VIDEO_ONLY` (lines 112–132), while decoupled same-chunk clean attention requires `kv_stream == q_stream` (lines 199–204).
- A subsequent four-file integration run was interrupted before pytest produced a summary. Treat that run as incomplete; the parent owns the final targeted CPU gate. No CUDA or multi-GPU run was performed.
