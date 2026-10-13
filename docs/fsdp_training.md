# FSDP2 training and full-state continuation

The LIBERO and RM75/Cine distributed entries support FSDP2 for all four methods.
The base trainable scope, public initialization and AdamW recipe stay the same.
New examples and the LIBERO quickstart select FSDP; their actual budget remains
two optimizer updates. Eight ranks, batch one and accumulation ten yield global
batch 80. Use the same strategy, process count, batch and budget in both arms.

## Configuration

LIBERO reads `strategy: fsdp` from the distributed run YAML referenced by
`RUN_CONFIG`. Cine reads `run.strategy: fsdp` from its method YAML. Both also
accept `ddp`. Missing fields retain DDP for compatibility with existing settings
and checkpoint identities. The quickstart's `TRAIN_STRATEGY` defaults to `fsdp`;
set it to `ddp` when continuing a matching older DDP checkpoint.

```bash
# LIBERO: copy the example and supply real settings/split paths.
RUN_CONFIG=/absolute/run-fsdp.yaml NPROC_PER_NODE=8 \
  bash scripts/train_8gpu.sh

# Cine: the shipped method config already selects FSDP.
NPROC_PER_NODE=8 bash scripts/train_cine.sh configs/cine_v3/vrfm_cagrad.yaml
```

Select free GPUs through `CUDA_VISIBLE_DEVICES`. These commands require prepared
data and public weights; installation/config checks do not establish GPU fit.
An 8xA100 40 GB run still needs a full-width two-update smoke with its actual data.

## Gradient and memory policy

FSDP2 shards bulk video/action weights, ordinary gradients and optimizer state.
Sharding begins from the CPU model, layer by layer, avoiding a whole-model CUDA
transfer before sharding. The structural common candidates remain replicated:
final video self-attention K/V, the enabled native state projection, and the VRFM
video projection when present. The same policy applies to the baseline even
though it does not use CAGrad. The VRFM posterior and action projection keep their
ordinary task-plus-KL gradients.

CAGrad accumulates the two task gradients on those replicated leaves. Two partial
backwards retain the graph and temporarily suppress FSDP reduction/resharding;
the ordinary total-loss backward restores FSDP and reduces each microbatch into
parameter shards. At the update boundary the shared task buffers are globally
averaged before the solve. Other replicated gradients are averaged once; distinct
DTensor shards are never all-reduced again. A locally inactive objective takes
a zero backward through its graph so collective ordering stays consistent.

FSDP reduces replicated model/optimizer memory, but CAGrad retains its two shared
FP32 buffers and incurs extra backward work. Transient parameter all-gathers,
activations and full-state checkpoint saves still require measured headroom.
Optional `OPEN_WAM_FSDP_CPU_OFFLOAD=1` uses the native FSDP offload path; keep that
setting fixed across paired arms and full-state continuation.

## Continue a matching run

Raise only the total budget (`steps` for LIBERO, `run.steps` for Cine) to 10000
after the two-update smoke, then supply the full checkpoint directory:

```bash
RUN_CONFIG=/absolute/run-fsdp.yaml NPROC_PER_NODE=8 \
  RESUME_CHECKPOINT=/absolute/run/checkpoints/checkpoint_step_2 \
  bash scripts/train_8gpu.sh

NPROC_PER_NODE=8 \
  RESUME_CHECKPOINT=/absolute/cine-run/checkpoints/checkpoint_step_2 \
  bash scripts/train_cine.sh /absolute/cine-fsdp.yaml
```

This continues through total step 10000. Full state restores model, AdamW,
scheduler, strategy/scaler, update state and every rank's RNG/sampling cursor.
Identity checks bind the method, base scope, data, batch, topology, strategy,
FSDP gradient contract and offload setting. A DDP checkpoint is continued with
DDP; switching strategy starts a new experiment. Historical `legacy_v02` routes
remain DDP-only. A `model_state.pt` file is an inference/weights artifact, not
a substitute for the full-state checkpoint plus rank sidecar.

## Executed checks

On the pinned PyTorch 2.11.0+cu128 environment, three-rank NCCL tests have passed
small native baseline and VRFM+CAGrad training/recovery in both FP32 and BF16.
Restarting at step one through step three matched uninterrupted model tensors,
Adam state, scheduler, rank-local RNG, next batches/noise and rank agreement.
These are synthetic two-layer fixtures, not full-width A100 memory measurements.
Three-rank Gloo tests also execute real CPU FSDP2 with activation checkpointing
enabled and a changing learning-rate scheduler. Both methods pass recovery and
native prior-only heldout evaluation with FP32 and BF16 strategy configuration.
The CPU strategy computes in FP32 for both configuration variants; actual BF16
execution was checked in the earlier CUDA run. An independent dense global-batch oracle matches CAGrad
coefficients, replicated and sharded gradients, global clipping norm and AdamW
updates when the action objective is inactive on one rank. Repeating the added
activation-checkpointing and masked-rank checks under CUDA is pending GPU availability.
The final scoped CPU regression passed **229 tests**, with **55 CUDA-dependent
checks skipped**, in 146.91 seconds. It also covers the existing DDP continuation,
both entry settings, generated quickstart YAML and native checkpoint/runtime
contracts. Counts from overlapping runs are not additive. The earlier CUDA
run's four native recovery checks passed; its separate masked-rank fixture
failed before executing because a required topology argument was missing.
That fixture is fixed and passed the actual CPU FSDP oracle above.
CUDA continuation combining CPU offload with the new replicated candidates has
not been executed; default no-offload recovery is the validated path.
