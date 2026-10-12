# Matched DDP Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish matched bounded DDP baseline/VRFM+CAGrad recipes with trained state projection and preserved full-state resume.

**Architecture:** Extend the existing shared trainability controls and identity;
reuse the existing runtime and checkpoint implementation for both data entries.
Keep old settings identities unchanged when the new scope is disabled.

**Tech Stack:** Python 3.12, PyTorch 2.11, pytest, CPU/Gloo validation.

**Spec:** ../specs/2026-10-12-paired-ddp-resume-design.md

## Global Constraints

- DDP; two-update entry defaults; AdamW 1e-5; identical base scope and recipes.
- Keep LIBERO and RM75/Cine entrypoints and exact rank-local resume.
- No changes to original dirty checkout, historical v0.2 contract, or active jobs.
- GPU training uses three free H20 GPUs after a live preflight; this task does
  not launch a formal training run.

## Review Focus

- Old settings must retain old checkpoint identities and resume behavior.
- A state projection used by only one task must keep its ordinary gradient.
- VRFM initialization must not alter the native RNG or existing parameter values.
- Resume must restore Adam moments, scheduler and sampler/RNG, not only weights.
- Both data-entry shipped arm configs must match after removing method switches.

### Task 1: State-projection scope and isolated initialization

**Files:** settings.py, cagrad_training.py, runtime_factory.py, train_smoke.py,
dual_expert/vrfm.py, tests/test_gradientwam_training_methods.py,
tests/test_gradientwam_vrfm_cagrad_settings.py, tests/test_vrfm.py.

- [ ] Add tests for opt-in scope identity, real projection gradients/updates,
  ordinary single-task handling, and independent VRFM initialization.
- [ ] Run focused tests and observe the missing-feature failures.
- [ ] Add validated `train_proprio_projection` with an old-scope-compatible default;
  propagate it through both shared and single-device construction paths.
- [ ] Preserve the native RNG while attaching VRFM; test both arms' base values.
- [ ] Run the focused suite and commit the implementation with its tests.

### Task 2: Paired entry recipes and accurate documentation

**Files:** configs/gradientwam/{baseline,vrfm,cagrad,vrfm_cagrad}.yaml,
configs/cine_v3/{baseline,vrfm,cagrad,vrfm_cagrad}.yaml, README.md,
docs/vrfm_cagrad.md, docs/cine_v3.md, tests/test_gradientwam_paired_recipes.py.

- [ ] Test resolved arm equality apart from method, scope activation, and actual
  two-update entry budgets for both entries.
- [ ] Observe failing scope activation before changing shipped recipes.
- [ ] Enable state-projection training equally and document the approved table,
  actual global-batch calculation, explicit budget changes and legacy resume.
- [ ] Run config and entry tests and commit.

### Task 3: VRFM+CAGrad DDP continuation regression

**Files:** tests/test_gradientwam_distributed.py; checkpoint/runtime files only
if the real continuation regression exposes a defect.

- [ ] Add a small real two-task posterior/projection model to the existing Gloo
  runtime test; stop at optimizer step one and resume through further updates.
- [ ] Compare next batches/noise, all model values, optimizer state, scheduler,
  step/cursor state, and cross-rank agreement with uninterrupted training.
- [ ] Exercise matching-scope continuation and changed-scope rejection.
- [ ] Fix any evidenced defect at the shared owner, with a failing regression.
- [ ] Run scoped regressions, then the relevant broader suite and commit.

### Task 4: Review and remote delivery

- [ ] Review the entire branch against the latest table and resume requirement.
- [ ] Publish the branch/PR, attach it to this chat, and verify remote commit.
- [ ] Report checks performed and full-width CUDA/benchmark limits precisely.
