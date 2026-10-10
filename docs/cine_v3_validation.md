# Cine v3 validation scope

The parallel entry adapts RM75/Cine LeRobot v3 to the existing OpenWAM + VRFM + CAGrad implementation. This document distinguishes executed engineering checks from full-size training and task performance.

## Executed checks

| Surface | Evidence and scope |
| --- | --- |
| Real source data | Bounded read-only checks of one training and one validation episode: v3 metadata/tasks, seven action/state values, 30-FPS timestamps and a few decoded RGB frames. No dataset transfer or modification. |
| V3 reader | Synthetic fixtures cover shared/sparse Parquet shards, nonzero video offsets, frame continuity and root-qualified episode identity. Equal numeric episode IDs alone do not cause a train/validation collision. |
| Preparation | A real tiny Wan VAE encodes 33 consecutive frames to nine target latents plus a separate single-frame condition. Native tiny local Wan/UMT5/tokenizer assets load and encode offline. This is not full-size public-asset encoding. |
| Supervision | Exact raw action rows, masked history slots, seven-dimensional state and per-latent state anchors are checked. Native training chunks use previous-boundary state rather than future state. |
| Method integration | A tiny native model performs forward/backward and CAGrad/AdamW updates. Ordinary nonshared gradients are retained. Prior-only generation has the declared Cine action/video geometry. |
| Continuation | The real Cine dataset, loader, shared runtime, optimizer and full-state checkpoint path execute on a tiny CPU model: save step one, restore and continue to step two. Separately, native VRFM+CAGrad continuation is compared with an uninterrupted second update. Entry config checks allow increasing the update budget while preserving the data/method identity. |
| Existing entry | Focused LIBERO settings, sampling, distributed runtime and two-rank Gloo regressions cover the shared runtime extraction. |

The independent review and exact temporal reasoning are in [cine_v3_review.md](cine_v3_review.md). The implementation tests are `tests/test_cine_*.py`; existing method/runtime regressions remain separate.

On 2026-10-10, the pinned WSL/Linux CPU environment completed:

- 195 combined adapter, preparation, native method and existing runtime checks in 84.43 seconds. One existing BF16 RMSNorm warning selected a non-fused implementation.
- 16 final Cine entry checks in 27.76 seconds, including the real shared-runtime update/save/restore test.
- 26 final distributed/sampling checks in 54.10 seconds after the last shared-entry import fix.
- Wheel and source-distribution builds, installed `gradientwam-cine --help`, shell syntax, and packaged module/config/launcher presence checks.

These groups overlap and must not be added into a unique-test total. The native runtime test uses a small model without loading the large public initialization; it does not certify that the full model fits a GPU.

To rerun the changed entry and shared runtime checks in the installed CPU environment:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest tests/test_cine_*.py tests/test_gradientwam_distributed.py \
    tests/test_distributed_sampling.py -q
```

## Limits

- No full-size OpenWAM GPU training, eight-GPU NCCL run, CUDA memory-fit claim, throughput measurement or complete-dataset encoding was performed for this delivery.
- No RM75 robot/simulator task success rate is established. Offline denoising losses and passing software tests are not policy success rates.
- Actions are preserved as `raw_joint_command`. Field names and archived configuration do not independently establish whether the producer commands absolute joint angles or joint increments.
- Source identity includes small source metadata fingerprints. Train/validation physical-file interval checks detect the same video reused through paths/hardlinks; they do not constitute exhaustive content deduplication of independently copied videos.
- Preparation defaults to an explicit subset unless `--all-windows` is requested. All-windows coverage is defined on the configured window-start stride, not every possible overlapping frame start.

Use the [Cine guide](cine_v3.md) to prepare a bounded cache, check it, then measure full-model fit before increasing the experiment budget.
