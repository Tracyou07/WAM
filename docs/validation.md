# Delivery validation (2026-10-09)

## Fresh checks for this delivery

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

## What the new tests actually execute

The two-rank Gloo test executes the actual `RankAwareTrainingRuntime` and native step loop with a stochastic scalar model/dataset, accumulation 2, a global-batch reference update, and exact resumed parameter equality. The other two checks cover split validation and resume identity changes. The Windows worker supplies a test-only `fcntl` stub and never uses the cache lock; this is not native Windows runtime support. Linux uses the real POSIX module.

The four assembly tests execute the public `_build_stack` against the real native factory, public loader, sharing implementation and optimizer. They replace full 30-layer geometry with a two-layer CPU model, intercept the requested 30-layer/count contract, and replace GPU/BF16 strategy with CPU/FP32. This verifies attachment/order/ownership, not full-width execution.

The update/resume test executes the delivered update loop and checkpoint helper with a tiny linear model: ten microbatches, update 1, checkpoint, update 2; it compares the restored second update and parameters exactly against uninterrupted CPU execution, including restored RNG and scheduler. Metadata mismatches are rejected. This does not demonstrate native transformer or CUDA resume equivalence.

The five split checks cover strict schema/disjointness, exact selected-episode range grouping, condition augmentation over every selected pair, missing raw episode rejection and CLI split routing. They execute preparation command construction without performing frontend encoding.

The broader upstream/round02 suite is retained but was not rerun in full for this delivery. Native Windows runtime collection fails at the upstream POSIX `fcntl` import; use Linux/WSL. No runtime compatibility shim or native-source rewrite was introduced.

Full-tree whitespace checks retain two pre-existing native-source warnings (`models/__init__.py` and vendored `third_party/lingbot/model.py`); delivery files are checked separately so source hashes remain unchanged.

## Explicitly unverified

No new real data encoding, full-width model load, GPU work, real OpenWAM optimizer update, simulator rollout or performance evaluation was performed for repository delivery. Earlier research receipts establish one real CPU data path only; they are not public release validation artifacts. No claimed benefit over the deterministic/private controls, no fully matched output-mixture control, and no trained action checkpoint are supplied.
