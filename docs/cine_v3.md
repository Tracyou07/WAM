# Cine v3: a parallel GradientWAM entry

`gradientwam-cine` targets the RM75/Cine LeRobot v3 dataset. The existing `gradientwam` LIBERO commands and configurations remain available. Both entries share OpenWAM, the four method choices, CAGrad gradient coordination and the native training runtime.

## Data meaning

Use one explicitly selected training root and a separate validation root. The example scale directories are `train_0.6k`, `train_1.2k`, `train_2.4k`, `train_4.8k`, `train_9.6k`, and `train_all`; these may overlap and must not be concatenated as disjoint examples. The validation directory is `validation_200`.

The adapter consumes one `observation.images.color` camera at 30 FPS, seven raw state values `[x,y,z,qx,qy,qz,qw]`, and seven raw action values named `joint_1` through `joint_7`. It reads v3 Parquet/task/episode metadata and video-file time offsets. It does not invent a wrist view, a gripper channel or an eighth state value, and it does not resample to 20 FPS.

The default action label is `raw_joint_command`: the original numeric commands are preserved. Their field names do not prove absolute angles versus increments. An explicit `joint_delta` declaration records a protocol choice and does not apply a difference operation. Establish the producer's physical command convention before using predictions on a robot.

## Setup and paths

Use the pinned Linux environment and local model/frontend assets described in the [installation instructions](../README.md#install) and [asset guide](reproduction_setup.md). Installation and configuration checks do not start training or download model weights. After updating an existing checkout, reinstall its editable package so the new console entrypoint is registered.

```bash
export GW_CINE_TRAIN_ROOT=/absolute/datasets/cine/train_0.6k
export GW_CINE_VAL_ROOT=/absolute/datasets/cine/validation_200
export GW_CINE_LATENT_ROOT=/absolute/prepared/cine-small
export GW_OUTPUT_ROOT=/absolute/runs/cine-vrfm-cagrad-001
export GW_CHECKPOINT=/absolute/assets/openwam/model_state.pt
export GW_CHECKPOINT_SHA256=1d22c4159fd77beba6e5e41b484c35280cce8f15ba21c5d2adfd4138ca82c73f
export GW_FRONTEND_ROOT=/absolute/assets/frontend
export GW_TOKENIZER_ROOT=/absolute/assets/tokenizer
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CFG=configs/cine_v3/vrfm_cagrad.yaml

gradientwam-cine check-config --config "$CFG"
```

All paths must be real, direct paths. The prepared cache and per-run output must not overlap each other or any raw source/model asset. Read-only model components may share a parent directory. Keep one prepared cache shared between method comparisons, and use a fresh output directory per run.

## Prepare and verify

Preparation is explicit and records the selected episode/window scope in the cache manifest. Start with a small selection to measure the actual encoder cost:

```bash
# Print the preparation plan without loading the encoders.
gradientwam-cine prepare --config "$CFG" --device cpu \
  --train-episode-limit 2 --train-windows-per-episode 1 \
  --val-episode-limit 1 --val-windows-per-episode 1

# Execute the same selection into a fresh prepared-cache directory.
gradientwam-cine prepare --config "$CFG" --device cpu \
  --train-episode-limit 2 --train-windows-per-episode 1 \
  --val-episode-limit 1 --val-windows-per-episode 1 --execute

gradientwam-cine check-data --config "$CFG" --sample-limit 1
```

That cache contains the declared subset, not the full 600-episode scale. A full preparation uses `--all-windows --execute` with a different fresh cache directory. This means all complete windows on the declared `data.sample_stride` grid: the default start stride is 32 raw frames, while each clip contains 33 consecutive frames. Change `data.sample_stride` explicitly to request denser overlap; this does not alter the 30-FPS video clock. Overlapping windows can make the cache large, so measure time and bytes on the small selection first. `--device cuda` is for an explicitly selected free preparation GPU on your machine.

The source clip has 33 consecutive frames. The real Wan VAE must produce nine target latents; sampling nine RGB frames before encoding would be a different and incorrect timeline. The preceding observed frame is encoded independently. Native prefix conditioning adds it to give ten model-visible video frames.

Action samples have 36 slots: four masked history slots followed by the original 32 commands aligned to the source window. The final raw frame's outgoing command belongs to the next window. State at the preceding frame supplies the initial condition; per-latent state anchors go through native previous-chunk-boundary selection. The implementation checks these indices instead of inferring them from equal tensor dimensions.

Action normalization defaults to `none`. With `data.adapter_options.action_normalization: gaussian`, statistics are fitted on the selected training episodes and reused for validation. Cache and checkpoint identities bind the selection, declared semantics, normalization and encoder identities. A prepared latent tensor alone is not an accepted complete cache.

## Train, validate and continue

The four settings files are `baseline.yaml`, `vrfm.yaml`, `cagrad.yaml` and `vrfm_cagrad.yaml` under `configs/cine_v3/`. Start with their two-update budget, then explicitly set `run.steps` for the intended experiment.

```bash
# Checks the cache, launches synchronized training and reports heldout proxies.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
  bash scripts/train_cine.sh "$CFG"

# Configuration/data checks only; no model training.
CHECK_ONLY=1 bash scripts/train_cine.sh "$CFG"

# Raise run.steps in your config before continuing a matching run.
RESUME_CHECKPOINT=/absolute/runs/cine-vrfm-cagrad-001/checkpoints/checkpoint_step_2 \
  NPROC_PER_NODE=8 bash scripts/train_cine.sh "$CFG"
```

Any positive process count is supported by the launcher; select only free physical GPUs. DDP replicates the model on every rank, so additional GPUs do not pool their memory. Actual full-model fit and throughput must be measured. Keep process count, global batch, selected cache and update budget equal across method comparisons.

The declared training action horizon is 36. The native model factory requires a matching decoder horizon; the inference profile therefore uses nine latent frames per chunk and four action slots per latent. Training uses the separately declared chunked attention geometry. These settings are an explicit Cine adaptation, not a reproduction of the archived FastWAM recipe.

Heldout evaluation uses the separate validation dataset and the VRFM prior. Its action/video denoising losses are offline metrics, not robot success rates. Public OpenWAM initialization supplies video pretraining rather than a trained RM75 policy. See [validation](cine_v3_validation.md) for what was actually executed.

## Load a trained policy

The following command checks strict model-only checkpoint loading. It does not run a robot or generate a rollout:

```bash
gradientwam-cine infer --config "$CFG" \
  --checkpoint /absolute/runs/cine-vrfm-cagrad-001/checkpoints/checkpoint_step_2/model_state.pt \
  --device cpu
```

For a native inference client, `gradientwam.cine_inference.load_cine_policy_for_inference(settings, checkpoint, device)` returns the configured `VariantPipeline`. Supply the native observation/latent batch and use its prior-only generation path. For Gaussian-normalized targets, convert predicted actions back to the source units with `denormalize_cine_actions(actions, settings=settings, run_identity=identity)`, where `identity` is the parsed `run_identity.json` from the same training run. The helper checks the cache and source identities before applying the selected training mean/std. No absolute-angle/increment conversion is performed.
