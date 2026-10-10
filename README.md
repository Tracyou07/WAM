# GradientWAM: VRFM + CAGrad on OpenWAM

OpenWAM is the backbone. A continuous variational latent conditions both video and action prediction; CAGrad coordinates their gradients on common generator parameters. The new method has **no additional private K/V path, routing gate, or policy expert**.

本项目用变分潜变量区分可能的未来，用 CAGrad 协调视频与动作联合学习。提供 baseline、仅 VRFM、仅 CAGrad、VRFM+CAGrad 四组配置，以及数据准备、1/2/8 卡训练、独立留出验证和断点恢复入口。

**RM75/Cine LeRobot v3 数据请使用独立入口 `gradientwam-cine`：** 支持单 `color` 相机、30 FPS、7 维位置/四元数状态与原始 7 维关节命令。配置位于 `configs/cine_v3/`；完整命令见 [Cine v3 使用说明](docs/cine_v3.md)。下文的 `gradientwam` 命令继续用于原 LIBERO 数据，两个入口共用模型与训练算法。

**Status:** this is a research implementation, not a validated control policy. Engineering checks are listed in [validation](docs/validation.md). Full-width CUDA execution, actual eight-GPU training and robot/simulator gains must be established separately. Public OpenWAM weights initialize the video model, not a trained action policy. Older externally launched v0.2 experiments are a different method.

## Method

- **VRFM:** a training-only Gaussian posterior reads paired clean/noisy video/action inputs and context. Reparameterized z conditions both native streams. KL regularization uses a fixed standard Gaussian prior. Heldout prediction and deployment use the prior; one z remains fixed over the generation trajectory.
- **CAGrad:** accumulate the two weighted task gradients over microbatches and average across ranks before solving the two-task optimization. Coordinate only common generator parameters; keep ordinary gradients for other parameters and the posterior. Apply gradient clipping after composition, then AdamW.
- CAGrad is symmetric between tasks and does not guarantee action non-degradation after AdamW. This combination is an adaptation of existing methods, not a claim that either algorithm originated here.

See [method and gradient contract](docs/vrfm_cagrad.md), [CAGrad reference](docs/cagrad_reference.md), and [provenance](docs/provenance.md).

| Config under `configs/gradientwam/` | Continuous z | CAGrad |
|---|---:|---:|
| `baseline.yaml` | No | No |
| `vrfm.yaml` | Yes | No |
| `cagrad.yaml` | No | Yes |
| `vrfm_cagrad.yaml` | Yes | Yes |

Use one fixed split, process count, initial checkpoint, training budget and noise recipe across comparisons. VRFM adds a posterior and conditioning projections; the two VRFM arms match each other in architecture. The four arms are not all parameter-count matched.

## Install

Linux x86_64 / CPython 3.12.x is the supported reproduction environment. Native Windows is not supported by the upstream POSIX cache locks; use WSL.

Python must include `venv`/`ensurepip` (on Ubuntu with system Python 3.12, install `python3.12-venv` first). An existing Python 3.12 virtual environment can also be supplied with `--venv`.

```bash
git clone --depth 1 https://github.com/Tracyou07/WAM.git
cd WAM
bash scripts/setup_env.sh --device cu128 --python python3.12
source .venv/bin/activate
gradientwam --help
```

For CPU checks, select `--device cpu --venv .venv-cpu` instead. The setup script installs pinned library wheels, not datasets or pretrained weights. No training starts during installation.

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest tests/test_cagrad.py tests/test_vrfm*.py tests/test_gradientwam*.py -q
```

These checks use tiny models and CPU distributed workers; they do not download the full model or establish CUDA memory fit.

## Assets and preparation

Follow the [asset guide](docs/reproduction_setup.md) for pinned download, conversion and data preparation commands. Supply absolute direct paths. Do not put datasets, weights or training outputs in Git.

```bash
export GW_CHECKPOINT=/absolute/assets/openwam/model_state.pt
export GW_CHECKPOINT_SHA256=1d22c4159fd77beba6e5e41b484c35280cce8f15ba21c5d2adfd4138ca82c73f
export GW_DATASET_ROOT=/absolute/datasets/libero
export GW_FRONTEND_ROOT=/absolute/assets/frontend
export GW_TOKENIZER_ROOT=/absolute/assets/tokenizer
export GW_PREPARATION_ROOT=/absolute/prepared/shared-split
export GW_OUTPUT_ROOT=/absolute/runs/vrfm-cagrad-001
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CFG=configs/gradientwam/vrfm_cagrad.yaml
SPLIT=/absolute/configs/episodes.json
gradientwam check-config --config "$CFG"
```

The public checkpoint is [OpenWAM-Pretraining revision f27c127](https://huggingface.co/OpenWAM-Stanford/OpenWAM-Pretraining/tree/f27c127aa867a9e925bfe2fb7809861b94f1fe2d). Frontend/model assets require about 32.26 GiB before conversion and dataset preparation. Asset preparation does not create a trained action expert.

Input: LeRobot v2.1, 20 fps, action dimension 7, state dimension 8, cameras `observation.images.image` and `observation.images.wrist_image`. Copy [episodes.example.json](configs/gradientwam/episodes.example.json), then replace its illustrative IDs with IDs actually present in your data:

```json
{"schema_version": 1, "train_episode_ids": [0,1,2,3,4,5,6,7], "heldout_episode_ids": [8,9]}
```

The lists must be nonempty, unique and disjoint. Prepare once and reuse the same verified cache for every arm:

```bash
gradientwam prepare --config "$CFG" --episodes-file "$SPLIT" --device cpu
# Explicit encoding into a fresh output directory:
gradientwam prepare --config "$CFG" --episodes-file "$SPLIT" --device cpu --execute
export GW_PROMPT_FINGERPRINT='<verified encoder fingerprint>'
gradientwam check-data --config "$CFG" --episodes-file "$SPLIT"
```

Preparation can be slow. An incomplete cache is not a valid training dataset. Missing episodes fail; there is no fallback to a single episode.

## Train and validate on eight GPUs

Once assets and the cache are ready, this command performs checks, synchronized training and heldout flow-denoising evaluation:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 TRAIN_STEPS=2 \
  timeout --signal=TERM --kill-after=3s 60m \
  bash scripts/quickstart.sh "$CFG" "$SPLIT"
```

Select only free devices on your own machine. `NPROC_PER_NODE=1` or `2` uses fewer GPUs; `CHECK_ONLY=1` stops before model construction. `PREPARE=1 CHECK_ONLY=1` explicitly prepares a fresh cache. The two-step default checks the pipeline; increase `TRAIN_STEPS` for a deliberate research run.

The native recipe uses batch size 1 per rank and 10 accumulated microbatches, yielding global batch 10/20/80 at 1/2/8 ranks. Keep the same topology across method comparisons. DDP replicates the model on each device; eight GPUs do not pool their memory. CAGrad requires separate task-gradient storage and extra differentiation; its resource cost must be measured on the full model.

All four arms train the native action expert/packed action blocks and only the last video block's self-attention K/V. Other video parameters stay frozen; VRFM arms also train the posterior and latent projections. This bounded training scope adds no private route and is checked on resume. It is not full video-backbone fine-tuning.

For explicit configuration, copy [gradientwam_distributed.example.yaml](configs/gradientwam_distributed.example.yaml) and set your settings file, split and total update budget:

```bash
RUN_CONFIG=/absolute/configs/distributed.yaml NPROC_PER_NODE=8 \
  bash scripts/train_8gpu.sh

# Increase the total steps in distributed.yaml before continuing:
RUN_CONFIG=/absolute/configs/distributed.yaml NPROC_PER_NODE=8 \
  RESUME_CHECKPOINT=/absolute/runs/vrfm-cagrad-001/checkpoints/checkpoint_step_2 \
  bash scripts/train_8gpu.sh
```

Resume requires a trusted matching full-state checkpoint, including model/optimizer/scheduler and rank RNG/cursor state. Method, latent dimension, KL/CAGrad settings, data split and topology are checked. A changed method is a new run, not a silent checkpoint conversion. Checkpoints are large; provision space for both model-only and full-state saves.

Heldout outputs are `heldout_proxy_metrics.json`. VRFM evaluation uses the prior without encoding heldout targets into z. These are offline denoising losses, **not robot task success rates**.

To reload trained weights for generation, use the method-aware loader:

```python
from gradientwam.inference import load_policy_for_inference

policy = load_policy_for_inference(
    "configs/gradientwam/vrfm_cagrad.yaml",
    "/absolute/runs/vrfm-cagrad-001/checkpoints/checkpoint_step_2/model_state.pt",
    device="cuda",
)
```

It restores the declared architecture strictly, including VRFM modules, without constructing an optimizer or downloading assets. Supply encoded observations, robot state and prompt embeddings to the native inference API; see the [generation example](docs/vrfm_model_handoff.md#model-only-checkpoint-inference). Use `model_state.pt`, not the full optimizer checkpoint. The unmodified `openwam-eval` command does not attach these method modules automatically.

## Single-device recovery check

The separate recovery command repeats a selected sample (default episode 378), performs update 1, saves full state, reconstructs/restores the model, performs update 2 and stops:

```bash
# Use a verified preparation root containing the selected episode.
export GW_OUTPUT_ROOT=/absolute/runs/vrfm-cagrad-recovery
export CUDA_VISIBLE_DEVICES="$FREE_GPU_INDEX"
timeout --signal=TERM --kill-after=3s 60m gradientwam train --config "$CFG"
```

This is not a heldout experiment. Change `run.episode_id` if necessary. Full-state output requires at least 65 GiB free and the smoke checkpoint cap is 64 GiB. Use the distributed path for dataset training. Both paths require measured per-device memory feasibility.

## Legacy method

The former shared/private K/V route configurations and [v0.2 contract](algorithm/method_contract_v02.md) remain explicitly historical for reproducibility. They are not VRFM, and none of the four new arms enables them. Existing external v0.2 runs and frozen research assets are unaffected.

## Attribution

GradientWAM is an independent modified distribution of [OpenWAM](https://github.com/OpenWAM/OpenWAM), not an official Stanford release. Preserve [LICENSE](LICENSE), [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and [CITATION.bib](CITATION.bib). Code is AGPL-3.0-only except specifically attributed third-party components; model/data licenses apply separately.

- Guo and Schwing, [Variational Rectified Flow Matching, ICML 2025](https://proceedings.mlr.press/v267/guo25i.html).
- Liu et al., [Conflict-Averse Gradient Descent for Multi-task Learning, NeurIPS 2021](https://arxiv.org/abs/2110.14048), [official code](https://github.com/Cranial-XIX/CAGrad).
