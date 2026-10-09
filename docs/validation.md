# Delivery validation

## Current VRFM + CAGrad revision (2026-10-09)

Validation below concerns engineering behavior, not control quality. The new method changes native model/training files; the historical byte-identical snapshot statement further below does not apply to this revision.

- A separate Linux/WSL CPython 3.12.3 environment installed the complete pinned CPU profile, including PyTorch 2.11.0+cpu. Editable installation, `pip check`, exact dependency-version checks, and frontend/package imports passed with no CUDA initialization. This exposed and fixed a missing `editables` build dependency in the old install profile.
- `python -m pytest tests/test_cagrad.py tests/test_vrfm_conditioning.py tests/test_vrfm_native.py tests/test_gradientwam_vrfm_cagrad_settings.py -q`: **79 passed in 25.89 s** in that CPU environment, with offline flags and one BLAS/OpenMP thread. The only warning concerned a BF16/FP32 RMSNorm fused-kernel fallback; finite backward checks passed. Coverage includes an independent numerical CAGrad reference, analytic KL, train/eval boundaries, both streams' posterior gradients, fixed-z generation with CFG/cache, checkpoint recomputation/RNG restoration, variable-batch loss aggregation, and method configuration validation. Native forwards use tiny models, not the full pretrained backbone.
- CUDA installation, full-width weights, real-data updates, eight-GPU execution, and simulator performance remain unverified for this revision. No external training job or frozen research artifact was changed.
- All four public `gradientwam check-config` commands succeeded with nonexistent placeholder asset paths and reported `model_constructed=False`. Linux shell syntax checks passed for setup, quickstart and distributed launch scripts.
- PM integration regression: **371 passed in 238.70 s**, one non-fused RMSNorm warning, using the pinned CPU environment and command below. This covers the new method tests plus existing checkpoint, native pipeline, packed-block, variable-batch, decoder, training-runtime and trainability checks. It includes actual two-rank Gloo synchronization and tiny native four-arm optimizer updates. The complete upstream repository suite is not claimed green: its historical PyPI-release tests require a workflow file omitted from this distribution.

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
python -m pytest tests/test_cagrad.py tests/test_vrfm*.py tests/test_gradientwam*.py \
  tests/test_checkpoint_runtime.py tests/test_variational_native_pipeline.py \
  tests/test_dual_expert_packed_block.py tests/test_variable_batch_pipeline.py \
  tests/test_decoder_artifact_contracts.py tests/test_training_runtime.py \
  tests/test_training_controls.py -q
```

The independent review identified a variable-batch activity aggregation edge case after that integration run. It is fixed: activity is derived recursively from native per-sample masks. The final candidate set also excludes structurally action-only parameters without discarding shared parameters whose gradients happen to be zero.

Final post-fix PM regression: **276 passed in 192.10 s**, exit code 0, using the same pinned CPU environment:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
python -m pytest tests/test_gradientwam*.py tests/test_training_runtime.py \
  tests/test_variable_batch_pipeline.py tests/test_cagrad.py -q
```

This final command exercises actual tiny native updates for all four arms, VRFM+CAGrad step-1 checkpoint/reconstructed step-2 equality, two-rank Gloo global-gradient parity, zero-gradient retention and masked-task fallback. The independent reviewer additionally verified six native accumulated updates across bucket/padded/packed modes and unchanged ordinary posterior/action-projection gradients, then closed the P2 finding. See the [independent review](vrfm_cagrad_review.md) for scope and source hashes. Test counts from overlapping commands must not be added together.

`python -m hatchling build` produced a wheel and source distribution. Archive inspection verified both console entrypoints and the new solver, trainer, inference and VRFM modules in the 515-file wheel; the 890-file source distribution includes the four-arm configuration, pinned requirements, preparation/launch scripts and tests.

The 49 changed/new text files passed a bounded public-export scan for private paths, host/chat identifiers, common token/key patterns, binary assets and files over 1 MiB. The changed-file whitespace check passed. These checks do not certify the complete upstream repository test suite or prove absence of every possible secret.

## Historical delivery at f85b35c

The following are receipts from the earlier shared/private method delivery. They describe that revision only, and are retained for provenance rather than reused as evidence for VRFM + CAGrad.

### Checks reported for the historical delivery

- `python -m pytest tests/test_gradientwam_delivery.py -q`: **15 passed** in 103.28 s, Linux/WSL, Python 3.12.3, PyTorch 2.14.1+cpu. CUDA visibility disabled; offline model flags enabled.
- Four public `gradientwam check-config` commands: success with nonexistent absolute placeholder assets; confirms native schema parsing without model construction.
- `python algorithm/operator_v02.py`: **6 passed** (native weighted loss/gradient parity, empty masks, evidence/ELBO gradient identity, reduction counterexample, visibility checks and negative controls).
- Editable installation with `[train,pretrain]`: success on Linux. Characterized packages include diffusers 0.37.1, transformers 5.10.4, numpy 2.5.3, pyarrow 25.0.1 and pytest 9.1.1. These observations are not a dependency lock.
- Infra asset audit: **7 CPU checks passed** in the reference Python 3.12.2 environment. Checks covered all 196 VAE and 243 T5 mapped key/config/meta-shape contracts, unknown-key rejection, input hashes/output separation, and small FP32/BF16 native serialization/reload with embedding aliases. No real frontend weights were loaded; temporary test tensors totaled about 7.54 MB. This private source audit is reported as provenance, not bundled as a public test dependency.
- `scripts/setup_env.sh`: Linux shell syntax, cpu/cu128 dry-run plans and dependency metadata checked. The released installation profile pins PyTorch **2.11.0**, distinct from the **2.14.1+cpu** owner test environment above. A fresh installation of the full profile and any CUDA/full-model update were not executed for delivery.
- `scripts/quickstart.sh`: Linux shell syntax and `--help` checked; an asset-free `CHECK_ONLY=1` invocation parsed the config then stopped with the explicit missing-preparation error before any model construction. Actual raw-data preparation and the full model launch remain reader-operated, unverified paths.
- `python -m pytest tests/test_gradientwam_distributed.py -q`: **3 passed** on Windows with PyTorch **2.14.1+cpu**, and **3 passed in 14.06 s** in the existing Linux reference environment with PyTorch **2.11.0+cu128**, as recorded in the training handoff. The Linux run used Gloo, `CUDA_VISIBLE_DEVICES=''`, disabled user-site packages and single-thread BLAS settings; no packages were installed and no GPU was used. This existing-environment test does not establish fresh-installer success.
- `uv build --build-constraints requirements/reproduce-linux.txt`: wheel and source distribution built successfully with the fixed build dependencies. The 511-file wheel contains both `gradientwam` and `open_wam`, including both GradientWAM console entrypoints. The source distribution includes fixed requirements, launch scripts and example configuration.
- Public export scan: no detected private server/local-user paths, chat IDs, common access-token/private-key patterns, prohibited large artifacts or files over 1 MiB. Pattern scans are a bounded check, not proof of absence of every possible secret.
- 447 native Python files match the round02 snapshot byte for byte; frozen method contract hash preserved. Eleven small upstream fixtures are individually allowlisted by SHA256.

### What the historical tests actually executed

The two-rank Gloo test executes the actual `RankAwareTrainingRuntime` and native step loop with a stochastic scalar model/dataset, accumulation 2, a global-batch reference update, and exact resumed parameter equality. The other two checks cover split validation and resume identity changes. The Windows worker supplies a test-only `fcntl` stub and never uses the cache lock; this is not native Windows runtime support. Linux uses the real POSIX module.

The four assembly tests execute the public `_build_stack` against the real native factory, public loader, sharing implementation and optimizer. They replace full 30-layer geometry with a two-layer CPU model, intercept the requested 30-layer/count contract, and replace GPU/BF16 strategy with CPU/FP32. This verifies attachment/order/ownership, not full-width execution.

The update/resume test executes the delivered update loop and checkpoint helper with a tiny linear model: ten microbatches, update 1, checkpoint, update 2; it compares the restored second update and parameters exactly against uninterrupted CPU execution, including restored RNG and scheduler. Metadata mismatches are rejected. This does not demonstrate native transformer or CUDA resume equivalence.

The five split checks cover strict schema/disjointness, exact selected-episode range grouping, condition augmentation over every selected pair, missing raw episode rejection and CLI split routing. They execute preparation command construction without performing frontend encoding.

The broader upstream/round02 suite is retained but was not rerun in full for this delivery. Native Windows runtime collection fails at the upstream POSIX `fcntl` import; use Linux/WSL. No runtime compatibility shim or native-source rewrite was introduced.

Full-tree whitespace checks retain two pre-existing native-source warnings (`models/__init__.py` and vendored `third_party/lingbot/model.py`); delivery files are checked separately so source hashes remain unchanged.

### Historical limitations

No new real data encoding, full-width model load, GPU work, real OpenWAM optimizer update, simulator rollout or performance evaluation was performed for repository delivery. Earlier research receipts establish one real CPU data path only; they are not public release validation artifacts. No claimed benefit over the deterministic/private controls, no fully matched output-mixture control, and no trained action checkpoint are supplied.
