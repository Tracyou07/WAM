# Matched baseline versus VRFM+CAGrad

This compares the project's bounded OpenWAM baseline with VRFM+CAGrad. It does
not reproduce the upstream full-backbone FSDP recipe. Both LIBERO (`gradientwam`)
and RM75/Cine (`gradientwam-cine`) remain available.

| Condition | Project OpenWAM baseline | VRFM+CAGrad |
| --- | --- | --- |
| Base trainable parameters | Action expert/packed action blocks, final video K/V, native state projection | Same base scope, plus VRFM modules |
| Initialization | Same public video checkpoint and native action initialization | Same; added VRFM initialization does not consume native RNG |
| Data, split, preparation and interaction | Fixed shared inputs and `decoupled_same_step` program | Same |
| Distributed strategy | DDP | DDP; CAGrad does not currently support FSDP |
| Actual global batch | `WORLD_SIZE * train_batch_size * gradient_accumulation_steps` | Must match |
| Optimizer and base LR | AdamW, 1e-5 | Same |
| Actual entry update budget | Two-update smoke | Two-update smoke; 10000 is an explicit future setting |
| VRFM / CAGrad | Off | On |

Use the same native experiment YAML, public checkpoint SHA256, train and
evaluation seeds, prepared cache and episode split. Use a different output
directory for each arm. The example launchers default to eight ranks; this is
a default, not a GPU reservation. Record actual topology and global batch.
VRFM adds trainable parameters, so this is not a parameter-count-matched study.

The primary two-arm comparison tests the combined change. The `vrfm` and
`cagrad` configs isolate individual contributions when needed. Offline denoising
losses do not establish online control success.

## Explicit budget and full-state continuation

- LIBERO reads `steps` from the distributed run YAML. The native experiment's
  `training.num_steps: 10000` is overridden by that actual run budget.
- Cine reads `run.steps` from its method settings YAML.
- To continue a two-step checkpoint to step 10000, explicitly raise the total
  budget to 10000, use the same method and base scope, then name the full-state
  checkpoint directory. This runs through total step 10000, not 10000 extra steps.

```bash
# LIBERO: RUN_CONFIG contains the same settings/split and the new total budget.
RUN_CONFIG=/absolute/run.yaml NPROC_PER_NODE=3 \
  RESUME_CHECKPOINT=/absolute/run/checkpoints/checkpoint_step_2 \
  bash scripts/train_8gpu.sh

# Cine: the same original config with only run.steps increased.
NPROC_PER_NODE=3 RESUME_CHECKPOINT=/absolute/run/checkpoints/checkpoint_step_2 \
  bash scripts/train_cine.sh /absolute/cine.yaml
```

Keep the process count and effective batch equal to the saved run. Full-state
continuation restores model, optimizer, scheduler, strategy/scaler, step state,
rank-local RNG and sampling cursor. A model-only `model_state.pt` is for weights
or inference, not full-state continuation. Changing the method, training scope,
checkpoint identity, prepared data or topology is rejected.

The state-projection scope is explicit. Settings saved before this scope was
added (without `train_proprio_projection`) retain the original identity and
frozen state projection. Resume those checkpoints with their original settings;
do not silently substitute the new shipped configs. The historical v0.2 scope
remains separate and still requires `legacy_v02: true`.

## Executed engineering checks (2026-10-12)

The updated scope/initialization checks passed 64 CPU tests; the matched-recipe
and existing delivery checks passed 44. The distributed regression passed six
checks, including an actual three-rank Gloo run of a small native OpenWAM model
with VRFM, CAGrad and trainable state projection. Restarting at step one through
step three matched uninterrupted model tensors, Adam state, scheduler, next
batches/noise and rank agreement. A changed-scope identity was rejected.

These tests use synthetic inputs and a two-layer native fixture. They validate
the update/recovery path, not full-width H20 memory fit, CUDA bitwise behavior,
LIBERO success or performance on private robot data.
