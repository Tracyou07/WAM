# GradientWAM

GradientWAM studies whether observation-conditioned variational routing between shared and private video K/V projections improves world–action learning. This repository contains an independent research derivative of OpenWAM, a frozen method contract, four experiment arms, synchronized training for reader-operated 1/2/8-GPU machines, explicit heldout validation, and a separate two-update recovery smoke.

本项目研究：根据真实观测，在共享／私有视频 K/V 投影之间选择，是否改善世界模型与动作模型的联合学习。交付包含固定环境与资产准备、四臂配置、1/2/8 卡同步训练、显式训练／验证划分及恢复入口。下面的 quickstart 串起检查和训练；真实 8 卡运行仍需在读者机器验证。

**Status / 状态（2026-10-09 delivery snapshot）:** the research environment accepted the real episode-378 CPU data path; preprocessing of nine other selected episodes remains partial. **Zero successful real OpenWAM optimizer updates and no control-task results are established by this release.** Tiny CPU checks establish interface/numerical correctness only. The external public checkpoint supplies pretrained video weights, not a trained action policy. GPU memory, full-width forward/backward, CUDA resume and control performance remain unverified.

## Attribution and license / 来源与许可

OpenWAM Team, Stanford Vision and Learning Lab (SVL), Stanford University. OpenWAM, version 0.2.0, 2026. https://github.com/OpenWAM/OpenWAM

GradientWAM is an independent modified distribution, not the official OpenWAM or Stanford project. Source is distributed under **AGPL-3.0-only**; preserve [LICENSE](LICENSE), [NOTICE](NOTICE), [LICENSES](LICENSES), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [CITATION.bib](CITATION.bib). External models/datasets have their own terms. See [provenance](docs/provenance.md).

## Layout / 目录

```text
src/open_wam/          native framework plus round02 routing integration
src/gradientwam/       portable configuration, preparation, data check, training/resume
configs/gradientwam/   four arms and shared native experiment
algorithm/            frozen method contract and CPU mathematical reference
scripts/              upstream preparation and framework utilities
tests/                upstream/round02 tests and portable delivery checks
docs/                 method, provenance and validation boundaries
```

## Install / 安装

Use Linux x86_64 with glibc 2.28 or newer and CPython 3.12.x (including WSL for CPU checks). The fixed installation profile uses PyTorch 2.11.0 and pins training/preparation dependencies. Start from a source checkout:

```bash
git clone --depth 1 https://github.com/Tracyou07/WAM.git
cd WAM
# CUDA 12.8 build for reader-operated GPU training:
bash scripts/setup_env.sh --device cu128 --python python3.12
source .venv/bin/activate
gradientwam --help
# No assets or GPU needed for these engineering checks:
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest tests/test_gradientwam_delivery.py tests/test_gradientwam_distributed.py -q
```

For a CPU-only environment, use `--device cpu --venv .venv-cpu` and activate `.venv-cpu/bin/activate`. `--dry-run` prints the installation plan. Installation downloads library wheels; model/tokenizer/dataset downloads remain explicit. A CUDA-enabled PyTorch build alone does not verify the driver, GPU capacity or an eight-GPU run. Windows users should run these commands in WSL: native Windows cannot import the upstream POSIX `fcntl` cache module. The commands below use Bash. Preparation relies on scripts in the editable checkout. A wheel supports package imports/config checks, but preparation requires the checkout.

## External assets / 外部资产

Assets are not bundled or downloaded automatically. Supply absolute direct paths, without symlinks:

Follow the [environment and asset guide](docs/reproduction_setup.md) for pinned public download commands, file hashes, CPU frontend conversion and a compatible LeRobot input. It supplies the complete route from a fresh checkout; model assets total about 32.26 GiB before conversion and dataset preparation.

| Variable | Required content |
|---|---|
| `GW_CHECKPOINT` | Public OpenWAM video pretraining `model_state.pt` |
| `GW_CHECKPOINT_SHA256` | Independently verified file digest |
| `GW_DATASET_ROOT` | LeRobot v2.1 LIBERO dataset at 20 fps; `meta`, Parquet `data`, two-camera `videos` |
| `GW_FRONTEND_ROOT` | Compatible Wan/LingBot frontend assets with `vae/` and `text_encoder/` |
| `GW_TOKENIZER_ROOT` | Matching local tokenizer files |
| `GW_PREPARATION_ROOT` | Fresh preparation destination, separate from all source assets |
| `GW_PROMPT_FINGERPRINT` | Verified encoder fingerprint returned by preparation; required for check-data/train |
| `GW_OUTPUT_ROOT` | Fresh training destination, separate from assets/preparation |

Public-compatible datasets are for checking the released pipeline; they are not automatically the same inputs as the earlier research environment. Dataset revision, resolution, episode inventory and split must be matched before claiming benchmark reproduction.

The frozen public video checkpoint is [OpenWAM-Pretraining, revision f27c127](https://huggingface.co/OpenWAM-Stanford/OpenWAM-Pretraining/tree/f27c127aa867a9e925bfe2fb7809861b94f1fe2d), `model_state.pt` (20,433,187,287 bytes), SHA256 `1d22c4159fd77beba6e5e41b484c35280cce8f15ba21c5d2adfd4138ca82c73f`. The asset guide pins Wan2.2 VAE/UMT5/tokenizer files and provides `python -m gradientwam.prepare_assets` to convert the downloaded raw weights into native frontend folders. Its default prints a plan; only `--execute` performs CPU conversion and strict local tensor reload. Checkpoint loading does not install frontend assets.

Example placeholders—replace with your actual direct paths:

```bash
export GW_CHECKPOINT=/absolute/assets/openwam/model_state.pt
export GW_CHECKPOINT_SHA256=1d22c4159fd77beba6e5e41b484c35280cce8f15ba21c5d2adfd4138ca82c73f
export GW_DATASET_ROOT=/absolute/datasets/libero
export GW_FRONTEND_ROOT=/absolute/assets/frontend
export GW_TOKENIZER_ROOT=/absolute/assets/tokenizer
export GW_PREPARATION_ROOT=/absolute/prepared/shared-split
export GW_OUTPUT_ROOT=/absolute/runs/variational-001
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
CFG=configs/gradientwam/variational_sharing.yaml
gradientwam check-config --config "$CFG"
```

`check-config` parses the native schema without loading a model or checking asset existence. Seeds are 20261008. The settings field `episode_id` defaults to 378 and is used only by the single-sample recovery smoke; split preparation and distributed training use the explicit split JSON. Camera keys are `observation.images.image` and `observation.images.wrist_image`.

## Prepare and inspect / 预处理与检查

For a train/heldout experiment, copy [episodes.example.json](configs/gradientwam/episodes.example.json), replace its illustrative IDs with episodes present in your dataset, and retain this schema:

```json
{"schema_version": 1, "train_episode_ids": [0,1,2,3,4,5,6,7], "heldout_episode_ids": [8,9]}
```

Both lists must be nonempty, unique, nonnegative integer IDs, and disjoint. These example IDs are not a published benchmark partition. Prepare one shared cache for all four arms:

```bash
SPLIT=/absolute/configs/episodes.json
gradientwam prepare --config "$CFG" --episodes-file "$SPLIT" --device cpu
# Only this explicit command performs encoding:
gradientwam prepare --config "$CFG" --episodes-file "$SPLIT" --device cpu --execute
export GW_PROMPT_FINGERPRINT='<verified encoder fingerprint>'
gradientwam check-data --config "$CFG" --episodes-file "$SPLIT"
```

The native encoder accepts contiguous ranges; the wrapper groups only selected consecutive IDs and never fills unselected gaps. Condition augmentation covers every selected camera pair. Task and empty prompts are genuinely encoded, including heldout instructions; heldout actions never belong in the training split. The prepared split is saved as `metadata/episodes.json`. A missing episode or invalid split is an error, not a fallback to episode 378. Split-mode `check-data` audits every selected episode's native data alignment; it is not a performance evaluation. Input data must be **LeRobot v2.1, 20 fps, action dimension 7, state dimension 8**, with the two exact camera keys above.

Preparation can be expensive even on CPU. It requires a fresh destination; an interrupted destination is not automatically overwritten or resumed. `check-data` verifies native raw action/state alignment, condition time axes, masks, finite tensors and nonzero task/empty prompt embeddings. Existing verified caches may be reused if they follow `latents/<dataset-directory-name>/latents` and `prompt_cache` under the preparation root and contain exactly the declared split episodes.

## Four arms / 四臂

| Config filename under `configs/gradientwam/` | Question |
|---|---|
| `native_joint.yaml` | Native single route under the same decoupled mask and restricted freeze policy |
| `same_capacity_deterministic_kv_blend.yaml` | Is deterministic K/V blending sufficient? |
| `variational_sharing.yaml` | Does latent shared/private routing help? |
| `forced_private_world_gradient_off.yaml` | Is any benefit explained by isolating action gradients from the final world K/V? |

Despite its historical name, `native_joint` uses the **decoupled_same_step** program. The native arm has fewer parameters; do not call all four arms capacity matched. World loss is retained in the forced-private arm. All arms retain the frozen weighted native action/world loss recipe. A capacity-matched output-mixture fifth control remains unresolved; G-det/G-var and VFP-derived extensions are future research, not released methods.

## Quick training and heldout validation / 快速训练验证

After following the [asset guide](docs/reproduction_setup.md), set the `GW_*` paths above, select an available GPU set on your machine, and use one explicit split for all arms. A fresh cache can be prepared by the quickstart script; a completed cache is reused without encoding:

```bash
export GW_PREPARATION_ROOT=/absolute/prepared/shared-split
export GW_OUTPUT_ROOT=/absolute/runs/variational-001
CFG=configs/gradientwam/variational_sharing.yaml
SPLIT=configs/gradientwam/episodes.example.json  # illustrative 8 train / 2 heldout
# First preparation is explicit and can be slow; this command stops before training:
PREPARE=1 CHECK_ONLY=1 bash scripts/quickstart.sh "$CFG" "$SPLIT"
# A completed cache needs only checks, then synchronized training and heldout proxies:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 TRAIN_STEPS=2 \
  timeout --signal=TERM --kill-after=3s 60m bash scripts/quickstart.sh "$CFG" "$SPLIT"
```

GPU indices above are examples for the reader's own free devices. `CHECK_ONLY=1` checks config/data without constructing the full training model. The script reads the prompt fingerprint from completed preparation metadata when it is not already exported. With an independently verified existing cache, export `GW_PROMPT_FINGERPRINT` explicitly. Missing assets or an incomplete split fail with an error.

`NPROC_PER_NODE=1`, `2` or `8` selects the process count; expose that many GPUs. `TRAIN_STEPS` is the total number of successful optimizer updates. The supplied native recipe retains **10 accumulated microbatches per update per rank** and per-rank batch size 1, so global batch is `processes × per-rank batch × 10`: **10/20/80 samples per update at 1/2/8 ranks**. Process counts therefore define different batch sizes. For paired method comparisons, retain the same process count, split, initialization seed, assets and batch recipe, with separate `GW_OUTPUT_ROOT` directories. This training over a split is separate from the repeated-single-sample recovery smoke.

For explicit launch configuration, copy [gradientwam_distributed.example.yaml](configs/gradientwam_distributed.example.yaml). If the copy is outside the repository's `configs/` directory, use an absolute `settings_config` path (relative paths resolve from the checkout or launch YAML directory):

```yaml
settings_config: /absolute/WAM/configs/gradientwam/variational_sharing.yaml
episode_split_json: /absolute/prepared/shared-split/metadata/episodes.json
steps: 2
eval_seed: 20261009
```

Then run:

```bash
RUN_CONFIG=/absolute/configs/distributed.yaml NPROC_PER_NODE=8 bash scripts/train_8gpu.sh
# Set steps to a larger total (e.g. 50) in the YAML before continuing a completed short run:
RUN_CONFIG=/absolute/configs/distributed.yaml NPROC_PER_NODE=8 \
  RESUME_CHECKPOINT=/absolute/runs/variational-001/checkpoints/checkpoint_step_2 \
  bash scripts/train_8gpu.sh
```

Resume retains the arm/assets/split, batch recipe and GPU process topology. The output directory may be retained or changed with `GW_OUTPUT_ROOT`. `steps` is a total budget, so set a larger value in the launch YAML to continue beyond a completed short run; it is not added to the restored step count. Resuming with the same completed budget performs no additional optimizer updates. Only trusted matching full-state checkpoints are supported. The native periodic save interval is 100 updates, plus a final checkpoint. DDP keeps a complete model on each GPU: eight cards do not combine their VRAM into one model allocation. Per-card feasibility and full-width CUDA behavior remain unverified.

The runner uses the explicit JSON episode membership and rank-shards a native weighted replacement stream; repeated draws are allowed by that sampling policy. Method attachment precedes optimizer construction and DDP wrapping; training forwards pass through the wrapped model. Rank zero writes logs/checkpoints, with each rank's RNG/cursor state retained. A resumable distributed checkpoint needs both native `full_training_state.pt` and `rank_runtime_state.pt`; the native manager also writes a model-only checkpoint, so budget disk space for both layouts and periodic retention. Independent heldout episodes produce `heldout_proxy_metrics.json` with action/world flow-denoising proxies. The variational action proxy uses the **observed prior**, without selecting a branch using heldout targets. These are offline losses; closed-loop task success requires a separate simulator/robot evaluation.

## Single GPU recovery smoke / 单卡恢复检查

This separate check repeats one sample from the selected episode (default 378). Use a completed cache containing that episode, or prepare it in a separate fresh directory:

```bash
export GW_PREPARATION_ROOT=/absolute/prepared/smoke-episode378
export GW_OUTPUT_ROOT=/absolute/runs/variational-smoke-001
gradientwam prepare --config "$CFG" --device cpu             # print a plan
gradientwam prepare --config "$CFG" --device cpu --execute   # explicit encoding
export GW_PROMPT_FINGERPRINT='<verified encoder fingerprint>'
gradientwam check-data --config "$CFG"
```

Change `run.episode_id` in a copied settings YAML if your dataset lacks episode 378. This path is a recovery check, not a train/heldout experiment.

Run only after selecting an available BF16-capable GPU and checking host identity, active jobs, environment and disk space. No training is started by installation or CPU checks. Full-state output requires at least 65 GiB free; one checkpoint is capped at 64 GiB. Actual full-width GPU memory feasibility is unverified.

```bash
export CUDA_VISIBLE_DEVICES="${FREE_GPU_INDEX:?Set FREE_GPU_INDEX after confirming the device is free}"
timeout --signal=TERM --kill-after=3s 60m gradientwam train --config "$CFG"
```

This command really constructs a CPU VisualTower, verifies/loads public video weights, calls the native pipeline factory, calls `configure_variational_sharing(arm=..., expected_layers=30)`, and only then constructs the optimizer. Use **gradientwam train**, not the upstream `openwam-train`, for these arms.

It performs ten accumulated microbatches → update 1 → full-state save → cold model reconstruction and strict restore → ten microbatches → update 2. It then stops. Parameters/Adam moments remain FP32; computation uses native BF16 autocast. AdamW, scheduler, sampler cursor and RNG are restored. This internal resume is not proof of equality with an uninterrupted CUDA run. The 60-minute internal timer is not a hard kill for blocking CUDA/native calls; the example adds an external timeout.

Outputs: `progress.jsonl`, `step1_full_state.pt`, and on success `result.json`. Never commit these. To resume a successfully saved step-1 checkpoint after interruption, keep the same assets, preparation paths, arm, seeds, episode and native configuration; use a new output directory:

```bash
export GW_OUTPUT_ROOT=/absolute/runs/variational-smoke-resume-001
timeout --signal=TERM --kill-after=3s 60m gradientwam train --config "$CFG" \
  --resume /absolute/runs/variational-smoke-001/step1_full_state.pt
```

The resume command executes update 2 only. A checkpoint must come from a trusted matching run. For paired research comparisons, reuse identical prepared inputs, checkpoint and seeds; use separate fresh output directories per arm, report capacity/runtime and evaluate held-out closed-loop control separately.

## Method and validation / 方法与验证

See [frozen contract](algorithm/method_contract_v02.md), [native integration](docs/variational_sharing.md), and [validation](docs/validation.md). The frozen contract contains historical local audit filenames; those private receipts are not bundled. Public CPU checks are runnable with:

```bash
python -m pytest tests/test_gradientwam_delivery.py tests/test_variational_native_pipeline.py tests/test_variational_packed_routes.py -q
python algorithm/operator_v02.py
```

Loss reduction, posterior preference and gradient cosine do not demonstrate control improvement. If variational sharing does not outperform deterministic blending or forced private routing, its proposed advantage remains unsupported. No performance claim is made in this release.
