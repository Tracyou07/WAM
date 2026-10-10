# Cine v3 GradientWAM training entry

This file defines the training-side schema and CLI for `gradientwam-cine`, alongside the existing LIBERO entry. The current Cine interface is based on `configs/cine_v3/cine_v3_experiment.yaml`; PM owns the public method configs, package entrypoint, launcher, and quickstart. Infra owns the LeRobot v3 reader, latent cache schema, and temporal alignment.

## Shared native runtime

The Cine entry reuses `LatentBatchAdapter`, `PipelineTrainStepExecutor`, `TrainingRuntime`, and `RankAwareTrainingRuntime`. The latter preserves the four existing method behaviors: baseline, VRFM, CAGrad, and VRFM+CAGrad, including heldout-prior evaluation, synchronized CAGrad, strict resume, and checkpoint identity.

The common model, optimizer and runtime construction is factored into `gradientwam.runtime_factory.build_rank_aware_runtime`. It accepts typed settings/config, prebuilt train/validation loaders, cache/split identity and resume state, then builds the native model, optimizer, `TrainingRuntime` and `RankAwareTrainingRuntime`. The LIBERO route retains its loader construction; Cine supplies datasets from Infra's `build_cine_latent_train_val_datasets(data_config)`.

`CineSettings` parses the Cine YAML independently, reuses `parse_method_config` and `GradientWAMMethodConfig`, rejects legacy v0.2 routes, and exposes `method_config`, `native_config()`, `seed`, `route_seed`, `checkpoint`, `checkpoint_sha256`, `frontend_root`, `tokenizer_root`, `output_root`, and `identity()`. Its native config merges the selected Cine data roots and method-independent settings into the PM-owned experiment YAML.

## CLI

The console entrypoint is `gradientwam-cine = gradientwam.cine_runner:main`.

```text
gradientwam-cine check-config --config CONFIG.yaml
gradientwam-cine prepare --config CONFIG.yaml --device {cpu,cuda} (--all-windows | --train-episode-limit N --train-windows-per-episode N --val-episode-limit N --val-windows-per-episode N) [--execute]
gradientwam-cine check-data --config CONFIG.yaml [--sample-limit N]
torchrun --standalone --nnodes=1 --nproc-per-node=N -m gradientwam.cine_runner train --config CONFIG.yaml [--resume CHECKPOINT]
gradientwam-cine infer --config CONFIG.yaml --checkpoint MODEL_ONLY.pt
```

`check-config` validates YAML, paths, method, and native geometry without building a model. `prepare` calls `gradientwam.cine_preparation.prepare_cine`; it reports the chosen selection by default and writes cache artifacts only with `--execute`. Selection must be explicit: either `--all-windows` or all four positive train/validation episode/window limits. `check-data` checks source identities, prepared manifest, and selected native samples. `train` requires a torchrun environment, accepts any positive process count, and reads its positive step budget from `run.steps`; `--resume` accepts only a strict full-state training checkpoint. `infer` strictly loads model-only weights.

## Config and environment

The four method YAMLs share the same Cine experiment, data roots, and preparation contract. They vary only in `gradientwam.method` and its method parameters.

```yaml
schema_version: 1
experiment: cine_v3_experiment.yaml
gradientwam:
  method: vrfm_cagrad # baseline | vrfm | cagrad | vrfm_cagrad
  latent_dim: 32
  kl_weight: 0.001
  cagrad_c: 0.4
assets:
  checkpoint: ${GW_CHECKPOINT}
  checkpoint_sha256: ${GW_CHECKPOINT_SHA256}
  frontend_root: ${GW_FRONTEND_ROOT}
  tokenizer_root: ${GW_TOKENIZER_ROOT}
data:
  dataset_name: cine_v3
  dataset_type: cine_v3_latent
  local_root: ${GW_CINE_TRAIN_ROOT}
  val_local_root: ${GW_CINE_VAL_ROOT}
  latent_root: ${GW_CINE_LATENT_ROOT}
  camera_names: [observation.images.color]
  latent_camera_names: [observation.images.color]
  canonical_height: 224
  canonical_width: 448
  num_frames: 9
  frame_stride: 1
  action_schema:
    action_dim: 7
    action_horizon: 36
    state_dim: 7
    state_horizon: 1
  action_target:
    representation: raw
    source_key: action
    pose_source_key: observation.state
    state_encoding: identity
    include_gripper: false
  adapter_options:
    cache_manifest: manifest.json
    action_semantics: raw_joint_command
    action_normalization: none # or gaussian using selected train episodes only
preparation:
  prompt_cache_root: ${GW_CINE_LATENT_ROOT}/prompt_cache
run:
  output_root: ${GW_OUTPUT_ROOT}
  steps: 10000
  seed: 20261008
  eval_seed: 20261009
```

Required environment variables are `GW_CINE_TRAIN_ROOT` (one explicitly selected train scale), `GW_CINE_VAL_ROOT` (the separate validation root), `GW_CINE_LATENT_ROOT`, `GW_CHECKPOINT`, `GW_CHECKPOINT_SHA256`, `GW_FRONTEND_ROOT`, `GW_TOKENIZER_ROOT`, and `GW_OUTPUT_ROOT`. The latent cache root must be fresh for preparation and physically separate from the train/validation roots, model assets, and each run's output. A cache can be shared across method arms only when source, selection, normalization, VAE, T5, and prompt identities all match.

Cine actions are the seven raw joint-command values, retained as-is and labeled `raw_joint_command`; there is no differencing, EEF conversion, or gripper. State encoding is identity for the seven supplied state values. Normalization defaults to `none`; optional Gaussian statistics are fit only on the explicitly selected train episodes and reused for validation. Include the semantic label, normalization mode/statistics digest, root-qualified train and validation source identities, exact selected samples, and encoder identities in the manifest and training identity. The preparation window-start stride defaults to 32 through data.sample_stride to avoid redundant overlapping encodings; an explicit stride of 1 remains available for overlapping windows.

The native experiment uses `data.action_schema.action_horizon=36`, `action_decoder.action_horizon=36`, and `inference.frame_chunk_size=9`: nine latent positions, each with four action slots.

## Temporal and cache contract

Infra exposes `CineV3Repository(root)`, `fit_cine_action_statistics(repo, episode_indices)`, and `build_cine_latent_train_val_datasets(data_config)` for `dataset_type: cine_v3_latent`. The manifest schema is `cine_v3_latent_v1`. Payload tensors include video latents `[C,9,Hl,Wl]`, independently encoded condition latents `[C,1,Hl,Wl]`, text context, and exact raw frame IDs. Each payload includes its root-qualified `source_identity`; the loader must verify it against the selected split's repository identity so equal episode IDs from different roots cannot cross splits.

Each sample uses 33 consecutive raw RGB frames at 30 FPS. The VAE encodes the full window; target latent anchors are `start, start+4, …, start+32`, and the separate condition frame is `start-1`. The prefix state is from `start-1`; the nine proprio context states correspond to the nine latent anchors. Actions have four leading zero/masked slots followed by 32 unchanged raw commands from `start` through `start+31`. The ninth latent has no outgoing action target. Native temporal expansion aligns the prefix and per-latent context using the previous-boundary contract.

`check-data` and training must reject stale manifests, incomplete or root-mismatched source identities, invalid selections, incorrect encoder fingerprints, incorrect tensor/frame shapes, or data/config temporal geometry mismatches. Validation always reuses train-derived action statistics and remains a separate root.

For Gaussian normalization, model outputs remain normalized because the native ActionTargetConfig stays at none to avoid applying normalization twice. Before sending actions to a controller, call gradientwam.cine_inference.denormalize_cine_actions(actions, settings=settings, run_identity=json.load(open(RUN_OUTPUT_ROOT / "run_identity.json"))). This verifies the manifest digest, train/validation source identities, selection, semantic label, and normalization mode, then uses only the selected train mean/std. With action_normalization set to none, the helper validates identity and returns actions unchanged.

## Existing route and verification

The existing `gradientwam.runner` is the LIBERO latent smoke and remains unchanged. LIBERO episode caches and prompt indexes are not used by Cine. `gradientwam-cine` uses the Cine v3 roots and cache schema directly.

The focused CPU verification path covers all four method configs, explicit preparation routing, native source/cache identities and temporal alignment, then loads a real `CineV3LatentDataset` through the shared runtime factory with a tiny native model, performs a CPU optimizer update, and resumes from step 1 to step 2. Other focused checks cover heldout-prior evaluation, model-only inference, and two-rank Gloo CAGrad synchronization. Full-size model/data encoding and CUDA runs are separate operations.
