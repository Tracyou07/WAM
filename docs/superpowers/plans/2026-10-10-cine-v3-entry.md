# Parallel Cine v3 training entry

**Goal:** add an independent, executable RM75/Cine LeRobot v3 preparation and training entry alongside the existing LIBERO entry, reusing the accepted OpenWAM + VRFM + CAGrad model and optimizer implementation.

**Architecture:** a native dataset adapter reads v3 episode/task/video metadata and returns the existing latent sample contract. An independent `gradientwam-cine` CLI owns explicit Cine settings, preparation, data checks and training orchestration. Shared runtime hooks may be extracted with unchanged LIBERO defaults; do not duplicate the full trainer or pretend v3 data is a v2 raw dataset.

**User-approved scope:** develop this parallel entry now. Keep the existing entry, public four-method controls and native model boundaries. No training starts as a side effect of implementation or installation.

## Data contract

- LeRobot v3.0 Parquet/video shards; one `observation.images.color` camera, source RGB 480x640 at 30 FPS.
- State is seven raw values `[x,y,z,qx,qy,qz,qw]`; preserve quaternion representation and do not invent gripper values.
- Action is the seven raw values named `joint_1` through `joint_7`. Preserve values; do not difference them, convert to EEF, or independently claim increment semantics without producer evidence. Record an explicit declared semantic label in run/cache identity.
- A 33-consecutive-frame window and 224x448 processed images are the initial profile. Distinguish raw-frame count from Wan causal VAE latent count. Freeze and test action/video/state timestamps, prefix masks and episode boundaries using the native temporal contract.
- Train and validation are separate explicit roots. Numeric episode IDs can overlap across roots; source/trajectory identity must distinguish them. No validation fallback to training, no concatenating overlapping scale subsets, and no normalization statistics fitted on validation.
- Preparation writes fresh derived outputs outside raw roots, records encoder and source identities, and never edits or symlinks source assets. Reject incomplete or incompatible prepared caches.

## Ownership and deliverables

- [x] Infra: `src/open_wam/data/cine_v3*.py`, minimal native registration/config additions and `tests/test_cine_v3*.py`. Deliver cross-shard reader, timestamp-correct bounded video decoding, explicit split planning, native latent dataset and cache contract. Publish consumed/produced interfaces in `docs/cine_v3_data_handoff.md` before dependent integration.
- [x] Training: `src/gradientwam/cine_*.py`, minimal shared runtime extraction and `tests/test_cine_training*.py`. Deliver check-config, prepare, check-data and train commands, real frontend preparation calls, configurable torchrun launching, prior-only heldout evaluation, strict resume and inference interoperability. Define CLI/settings schema early for PM-owned examples.
- [x] PM: public `configs/cine_v3/`, console entrypoint, separate launcher, README and reproducible setup/validation documentation. Integrate APIs, run targeted regression and prepare the fresh branch for PR publication.
- [x] Algorithm: independent data/temporal/leakage/gradient review in `docs/cine_v3_review.md`; minimal additional tests only for gaps or reproduced failures. Check actual data generation evidence when available, without turning review into unrelated research.

## Verification

Use focused failing tests before behavioral implementation. Include v3 episodes sharing a Parquet/video shard and crossing row groups, nonzero video offsets, strict state/action shape validation, train-only normalization, root-qualified episode identities, exact 30-FPS window alignment and no future-state leakage. Test a real tiny native forward/backward/optimizer update and checkpoint continuation through the new entry, plus two-rank CPU synchronization when shared runtime integration changes. Preserve existing LIBERO checks.

Run one bounded read-only H20 check against actual metadata, rows and a few decoded frames. Do not download full data/model assets, initialize CUDA, launch a scale queue or alter existing remote training. Full-size encoding, CUDA memory fit and robot performance are separate evidence and must remain explicitly unverified unless actually executed under a later valid resource release.

## Integration decisions

The implementation branch is `feat/cine-v3-entry` from accepted main `34df707ca929b0f5f8261addd2904ea30182051d`. Work in the existing isolated delivery checkout shared by the three authorized employee chats. Only PM stages, commits and publishes; each owner stays within its file boundary. Concrete interface refinements belong in the corresponding handoff, with PM resolving conflicts before caller code is duplicated.

Implementation and CPU acceptance completed on 2026-10-10. Full-width CUDA and eight-GPU execution remain unverified; see docs/cine_v3_validation.md. Publication uses the accepted branch/PR workflow; no GPU task starts as part of this delivery.
