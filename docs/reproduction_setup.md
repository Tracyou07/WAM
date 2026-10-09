# Reproduction environment and external assets

Use this guide from the source checkout. Environment installation, external
downloads, frontend conversion, dataset preparation, and training are separate
explicit commands. `setup_env.sh` installs dependencies and prints the remaining
asset checklist. It does not download weights or data or start a GPU job.

## Environment

The fixed profile supports **Linux x86_64, glibc >= 2.28, CPython 3.12.x**.
The reference CPU audit used Python 3.12.2. Other 3.12 patch releases use the
same wheel ABI; record the actual Python patch with your experiment.
Ubuntu 22.04/24.04 satisfy the glibc requirement. Windows and macOS are outside
this training profile.

| Component | Fixed version / choice |
| --- | --- |
| PyTorch | 2.11.0+cu128 for NVIDIA training; 2.11.0+cpu for CPU checks |
| Compiled CUDA runtime | 12.8 for the cu128 wheel |
| Diffusers / Transformers | 0.37.1 / 5.10.4 |
| Accelerate / Safetensors | 1.13.0 / 0.8.0 |
| NumPy / PyArrow / PyAV | 2.5.3 / 25.0.1 / 19.0.1 |
| SentencePiece / OpenCV headless | 0.2.2 / 4.14.0.94 |
| pip / editable build backend | 24.0 / Hatchling 1.27.0 |
| CPU engineering tests | pytest 9.1.1 |

[`requirements/reproduce-linux.txt`](../requirements/reproduce-linux.txt) pins the
common dependency closure from the installed CPU audit environment, plus the
editable build backend and pytest dependencies. The exact official Torch wheel
selects its device dependencies. In cu128 this includes CUDA Toolkit packages
12.8.1, cuDNN 9.19.0.56, NCCL 2.28.9 and Triton 3.6.0; Torch also constrains
CUDA bindings to >=12.9.4,<13. This is a version-pinned installation profile,
not a hash lock for every platform or a simulator environment. Keep the
resulting `pip freeze` with each run. These package versions were checked against
public distributions; do not replace the bounded Diffusers line with >=0.38
without revalidating the Wan numerical path.

Install OS packages and an existing Python 3.12 interpreter first. For example,
on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates git libstdc++6 libgomp1 ffmpeg
# Optional Python manager, installed following https://docs.astral.sh/uv/:
uv python install 3.12.2
export PYTHON_BIN="$(uv python find 3.12.2)"
```

Alternatively set `PYTHON_BIN` to your own CPython 3.12 interpreter with `venv`
and `ensurepip` available. The bootstrap does not install Python or OS packages.
No system CUDA toolkit, FlashAttention, Apex, or DeepSpeed build is required
for this native PyTorch profile. GPU machines still need an NVIDIA driver;
570.124.06 or newer is the CUDA 12.8.1 toolkit's corresponding Linux driver
profile in the [NVIDIA release notes](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html).
Driver/architecture compatibility and the actual memory requirement must be
checked on the target training machine. Changing GPU architecture, Python
patch, driver, or kernels is not evidence of identical floating-point results.

```bash
git clone --depth 1 https://github.com/Tracyou07/WAM.git
cd WAM
bash scripts/setup_env.sh --device cu128 --python "$PYTHON_BIN"
source .venv/bin/activate
CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_gradientwam_delivery.py tests/test_gradientwam_distributed.py
python -m pip freeze > environment.freeze.txt
```

For a CPU-only check environment instead:

```bash
bash scripts/setup_env.sh --device cpu --venv .venv-cpu --python "$PYTHON_BIN"
source .venv-cpu/bin/activate
```

`--dry-run` prints the installation commands without creating an environment.
Use separate environments for cpu and cu128: the script rejects an existing
environment with a different Torch profile. CPU imports and these engineering
tests do not validate eight-GPU training, available VRAM, or control success.
See the [official PyTorch installation profiles](https://pytorch.org/get-started/previous-versions/)
and [CPU](https://download.pytorch.org/whl/cpu/torch/) /
[cu128](https://download.pytorch.org/whl/cu128/torch/) wheel indexes.

## Choose storage paths

Use absolute paths on your machine. Source assets, converted frontends,
preparation output, and training output must be separate direct directories.
No symlinks are needed. The following example uses your home directory; adjust
it before downloading anything:

```bash
export GW_ASSET_ROOT="$HOME/gradientwam-assets"
export GW_CHECKPOINT="$GW_ASSET_ROOT/openwam/model_state.pt"
export GW_CHECKPOINT_SHA256=1d22c4159fd77beba6e5e41b484c35280cce8f15ba21c5d2adfd4138ca82c73f
export GW_RAW_WAN_ROOT="$GW_ASSET_ROOT/wan22-raw"
export GW_FRONTEND_ROOT="$GW_ASSET_ROOT/wan22-native"
export GW_TOKENIZER_ROOT="$GW_RAW_WAN_ROOT/google/umt5-xxl"
export GW_DATASET_ROOT="$GW_ASSET_ROOT/libero-public-v21"
export GW_PREPARATION_ROOT="$HOME/gradientwam-prepared/public-split"
export GW_OUTPUT_ROOT="$HOME/gradientwam-runs/public-split"
mkdir -p "$GW_ASSET_ROOT"
```

The selected model/raw frontend/tokenizer downloads below total
**34,635,400,956 bytes** (about 32.26 GiB). Keep another approximately 14.18 GB
for converted frontend tensors, plus the dataset, environments, preprocessing
caches and checkpoints. A full model/optimizer checkpoint can be tens of GiB;
budget for the checkpoint layout and retention policy of your chosen launcher.
Eight GPUs alone do not specify usable memory or storage capacity. All source
hashes below come from pinned public metadata, with the two small tokenizer
JSON files hashed directly; this guide does not bundle external assets.

## Video pretraining checkpoint

Public source: [OpenWAM-Stanford/OpenWAM-Pretraining](https://huggingface.co/OpenWAM-Stanford/OpenWAM-Pretraining/tree/f27c127aa867a9e925bfe2fb7809861b94f1fe2d),
revision **`f27c127aa867a9e925bfe2fb7809861b94f1fe2d`**.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| model_state.pt | 20,433,187,287 | `1d22c4159fd77beba6e5e41b484c35280cce8f15ba21c5d2adfd4138ca82c73f` |

This checkpoint contains **video-only pretraining weights**. It is not a
pretrained action policy and does not contain the VAE, text encoder, tokenizer,
or a resumable GradientWAM optimizer state. Native post-training initializes
action-side weights from the video tower and trains the action policy.

Download only this file when you explicitly choose to obtain it:

```bash
hf download OpenWAM-Stanford/OpenWAM-Pretraining model_state.pt \
  --revision f27c127aa867a9e925bfe2fb7809861b94f1fe2d \
  --local-dir "$GW_ASSET_ROOT/openwam"
printf '%s  %s\n' "$GW_CHECKPOINT_SHA256" "$GW_CHECKPOINT" | sha256sum -c -
```

## VAE, text encoder and matching tokenizer

Use the raw [Wan-AI/Wan2.2-TI2V-5B repository](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/tree/921dbaf3f1674a56f47e83fb80a34bac8a8f203e),
revision **`921dbaf3f1674a56f47e83fb80a34bac8a8f203e`**. Download the components
below, without the unrelated Wan transformer weights:

| Relative file | Bytes | SHA256 |
| --- | ---: | --- |
| Wan2.2_VAE.pth | 2,818,839,170 | `20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36` |
| models_t5_umt5-xxl-enc-bf16.pth | 11,361,920,418 | `7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d` |
| google/umt5-xxl/special_tokens_map.json | 6,623 | `7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d` |
| google/umt5-xxl/spiece.model | 4,548,313 | `e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458` |
| google/umt5-xxl/tokenizer.json | 16,837,417 | `6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b` |
| google/umt5-xxl/tokenizer_config.json | 61,728 | `ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05` |

```bash
hf download Wan-AI/Wan2.2-TI2V-5B \
  Wan2.2_VAE.pth models_t5_umt5-xxl-enc-bf16.pth \
  google/umt5-xxl/special_tokens_map.json google/umt5-xxl/spiece.model \
  google/umt5-xxl/tokenizer.json google/umt5-xxl/tokenizer_config.json \
  --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
  --local-dir "$GW_RAW_WAN_ROOT"
# Print a plan first; no hashing, loading, downloading, or output writes:
python -m gradientwam.prepare_assets --raw-root "$GW_RAW_WAN_ROOT" --output "$GW_FRONTEND_ROOT"
# Explicit CPU conversion and strict disk round-trip verification:
CUDA_VISIBLE_DEVICES='' python -m gradientwam.prepare_assets \
  --raw-root "$GW_RAW_WAN_ROOT" --output "$GW_FRONTEND_ROOT" --threads 4 --execute
```

The native frontend loader expects `vae/config.json` plus Safetensors and
`text_encoder/config.json` plus Safetensors shards/index. It cannot directly
load these raw `.pth` files. The included converter adapts the
[Diffusers 0.37.1 Wan2.2 recipe](https://github.com/huggingface/diffusers/blob/v0.37.1/scripts/convert_wan_to_diffusers.py),
whose source SHA256 is `5eb6873dce98d12658f9b740eacf4b8db788c637feefd74f3f446abe34551080`.
It checks all input size/hash values, VAE/T5 native keys and shapes, preserves
FP32 VAE and BF16 T5 tensors, and verifies every value and dtype after a local
strict reload. It preserves the shared T5 embedding alias. The tokenizer stays
at its direct source path and is validated locally before conversion.

For planning, allow **32-64 GiB of host RAM** for the 11.36 GB encoder and its
strict reload. This is an estimate from source and reload tensor sizes, not a
measured minimum: allocator, serialization and OS page-cache behavior can
change the peak. Assets after conversion occupy approximately 45.5 GiB.
Allow roughly **100-200 GiB of free disk** when retaining full training
checkpoints, environments and a small preparation cache; larger datasets,
checkpoint retention or replicated rank outputs require a separate budget.
Conversion processes one component at a time. It neither
quantizes weights nor establishes forward/output equivalence. A successful
`conversion_manifest.json` has status `converted_and_strictly_reloaded`; an
interrupted output has an in-progress manifest and must not be used as a
completed frontend. Existing output directories are always rejected.

The upstream [OpenWAM asset guide](https://openwam.github.io/OpenWAM/pretraining/training/)
also describes Diffusers/lingbot frontend assets. Those are alternative
initialization sources. This exact route uses the raw files and tokenizer
above: a similarly named Diffusers snapshot can have different tokenizer files
and is not an interchangeable identity for a paired experiment.

## Public compatible LeRobot input

Use [IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot/tree/e1a223d30b896c1613f270a2bfc63d382b3de7e1),
revision **`e1a223d30b896c1613f270a2bfc63d382b3de7e1`**. Its pinned metadata is:

| Field | Value |
| --- | --- |
| Layout / fps | LeRobot v2.1 / 20 |
| Episodes / frames / tasks | 379 / 101,469 / 10 |
| Cameras | `observation.images.image`, `observation.images.wrist_image` |
| Video frames | 256 x 256 RGB; AV1, yuv420p |
| Action / raw state | float32 `action[7]` / `observation.state[8]` |
| Episode data | `data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet` |
| Video data | `videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4` |

These public files already have the loader's required layout; no camera
renaming or v3 conversion is needed. Every episode metadata row supplies real
task descriptions. The repository API reported about 0.635 GB total stored
data at verification time; selected-download bytes can differ from this
repository-wide storage figure.

**This is a public compatible input route, not identical-input reproduction of
the prior research dataset.** That input was described as 388 episodes,
104,280 frames and 512px images. Its exact public source revision and conversion
provenance have not been established. No claim of equal episode numbering,
pixels, samples, or metrics is made here. The provided LeRobot v2.1 input
satisfies the public loader schema and allows a separately identified
experiment. The repository's native latent/prompt encoders below perform
preparation; the unrelated mixed-video LIBERO manifest converter is not a
raw action/state dataset converter. Generic `lerobot/libero_10` or v3 datasets
must not be substituted into this recipe without an explicit validated conversion.

Download and verify the three metadata files:

```bash
hf download IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot --repo-type dataset \
  --revision e1a223d30b896c1613f270a2bfc63d382b3de7e1 --local-dir "$GW_DATASET_ROOT"
cd "$GW_DATASET_ROOT"
sha256sum -c <<'EOF'
e5643c4e72e65f133ce0b1a5b9d9aaa5b2f065ed7f0f00ebbc231bde61af96d5  meta/info.json
5589f8f87cfddb34812782462160bf55b0d3082e404240682d1d0a89faba8265  meta/episodes.jsonl
45f9eb4d4b6b04999f64640c0aae380555372b7b273a904f5f459ad05d4a0a6a  meta/tasks.jsonl
EOF
cd -
```

Choose a **disjoint** `train_episode_ids` / `heldout_episode_ids` split JSON.
[`configs/gradientwam/episodes.example.json`](../configs/gradientwam/episodes.example.json)
uses train episodes 0-7 and heldout episodes 8-9. It is an engineering example,
not a benchmark split or a robot-control evaluation.

```bash
python -m gradientwam.runner check-config --config configs/gradientwam/vrfm_cagrad.yaml
python -m gradientwam.runner prepare --config configs/gradientwam/vrfm_cagrad.yaml \
  --episodes-file configs/gradientwam/episodes.example.json --device cpu
# Review the printed commands, then explicitly run the native encoders.
python -m gradientwam.runner prepare --config configs/gradientwam/vrfm_cagrad.yaml \
  --episodes-file configs/gradientwam/episodes.example.json --device cpu --execute
```

Preparation covers the union of both split lists without encoding gaps between
selected IDs. It uses the existing native two-camera latent encoder, one-frame
condition encoder and prompt-cache encoder. The fixed preparation resolution
is 128px with letterbox padding; input resolution is still part of dataset
identity. CPU encoding of real frontends can be slow. A reader can explicitly
select an available CUDA device for preparation after checking local resources.
GPU encoding has not been validated by this environment guide.

Read `prompt_encoder_fingerprint` from the printed preparation result (also in
`$GW_PREPARATION_ROOT/metadata/preparation.json`), then:

```bash
export GW_PROMPT_FINGERPRINT="$(python -c 'import json,os; print(json.load(open(os.path.join(os.environ["GW_PREPARATION_ROOT"],"metadata/preparation.json")))["prompt_encoder_fingerprint"])')"
python -m gradientwam.runner check-data --config configs/gradientwam/vrfm_cagrad.yaml \
  --episodes-file configs/gradientwam/episodes.example.json
```

`check-config` only parses configuration. `check-data` checks prepared samples
against aligned raw action/state rows; neither establishes a trained policy.
Follow the repository's distributed training instructions after these checks.
Keep the public dataset revision, split JSON, actual asset hashes, generated
cache fingerprint and environment freeze with training/validation results.

## Verification and provenance boundaries

This delivery checks public package/wheel availability, dependency metadata,
shell/CLI behavior, audited frontend key/config mapping and small CPU tensor
round trips. It does not execute a fresh CPU/GPU installation, rerun real frontend
conversion, download large assets, re-encode the dataset, or certify an
eight-GPU model update. Offline heldout action loss and simulator control
success require separate evidence. The minimal profile omits LIBERO simulator,
MuJoCo, tracking and visualization extras; follow their native setup before
attempting a closed-loop rollout.

Upstream native source is OpenWAM commit
`4b3814a82268f3523df5fcdadcbeb2c021ac7737`. Preserve repository `LICENSE`,
`NOTICE`, `LICENSES/` and `THIRD_PARTY_NOTICES.md`. External weights and data
retain the terms published by their respective upstream projects; this code's
AGPL license does not relicense them. See also the
[LIBERO upstream project](https://github.com/Lifelong-Robot-Learning/LIBERO) and
[LeRobot LIBERO documentation](https://huggingface.co/docs/lerobot/libero).
