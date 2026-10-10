# Cine v3 independent data and interface review

Date: 2026-10-10. Branch: `feat/cine-v3-entry`, base
`34df707ca929b0f5f8261addd2904ea30182051d`. Review completed against the shared
working tree. Only this document and `tests/test_cine_review*.py`
are reviewer-owned. No implementation source, raw data or archived FastWAM
file is modified. No GPU, full-data encoding or asset download is performed.

## Initial assessment for PM

The real data is compatible with a native single-camera, state7/action7 adapter.
Keep `raw_joint_command` as the default semantic label and preserve the seven
values. The checked metadata and archived configuration do not independently
prove absolute versus incremental commands. This does not block raw supervision.

The main interface risk is confusing three time axes: 33 raw frames, nine VAE
target latents, and ten model-visible latents after the native external prefix.
The minimal explicit temporal freeze below matches the existing caller's
structure and passes the executable native model test. Initial adapter review
and reproduced regressions are recorded below as historical findings. All six
data regressions are closed. The new-entry check also exposed a stale helper
import; its removal and the final real shared-runtime continuation test are
recorded in the integration closure below. No implementation blocker remains.

## Confirmed initial implementation findings

Initial reviewed source hashes: `cine_v3.py`
`b001bda9b9f9157d3c61ab3eebd0adfd85f3b3f7b9f33e1f589342db41786396`;
`cine_v3_latent.py`
`046c601e07ceb8115f0efa8fb46abe06e7e91c1bc3eeb5b8d4cfd174fcbec4a3`.
All four original data-review tests failed for the following triggers on this snapshot.
Owners must fix implementation source; reviewer edits remain restricted to
tests/docs.

### [P2, resolved] Prefix-only state replaced the native boundary-state sequence

`src/open_wam/data/cine_v3_latent.py:73-75` expands state at `start-1` to every
target latent. The computed anchor positions are unused for state. This avoids
current-chunk future-state injection, but discards later observations and changes
the native/handoff per-chunk data semantics without an approved change. For
fixture start 1, the frame-state sequence should be `[1,5,9,13,17,21,25,29,33]`;
the current adapter supplies nine copies of state 0. The native projector must
select the preceding chunk boundary. Emit the local grid/chunk/action-range
metadata below. Regression: `test_cine_sample_preserves_native_state_anchor_sequence`.

### [P2, resolved] Same-ID payloads could be moved across splits without rejection

`cine_v3_latent.py:41-47` checks encoding, numeric episode ID and frame IDs, but
does not bind a payload to its source/episode provenance or task/prompt.
Replacing validation episode-0 payload bytes with train episode-0 payload bytes
is accepted, pairing train video/text with validation raw actions. Manifest root
checks do not bind individual tensors. Validate per-payload source/episode and
task/prompt identity against the matching repository; include episode metadata
in source/cache provenance, since info/tasks hashes alone do not cover changed
episode/video/time mappings. Regression:
`test_train_payload_cannot_be_loaded_as_same_id_validation_episode`.

### [P2, resolved] Root names did not reject a physically shared video interval

`cine_v3_latent.py:97-100` only compares resolved directory ancestry. Different
roots with hardlinked video files and overlapping episode timestamp intervals
are accepted. Root-qualified IDs prevent false overlap, not physical leakage.
Validate selected episodes' video/file identities and time intervals; at minimum
reject identical device/inode intervals across splits. This does not require a
full video-content audit. Regression:
`test_overlapping_hardlinked_video_is_rejected_across_split_roots`.

### [P2, resolved] Duplicate cache windows were accepted as extra examples

`cine_v3_latent.py:140-150` deduplicates episode IDs, not sample entries.
Appending the same train entry twice changes sampling weights without changing
the claimed selection/statistics. Reject duplicate payload paths and duplicate
`(source, episode, raw_start)` windows before constructing datasets. Regression:
`test_duplicate_sample_window_is_rejected`.

Closure on the revised source: **4 passed in 6.67s** for
`tests/test_cine_review_data.py` before the two additional tests below were
added. The adapter now supplies real state anchors and explicit prefix/chunk
metadata, binds payload source identity, rejects duplicate windows/files and
checks overlapping physical video intervals by device/inode/time.
Revised hashes were `cine_v3.py`
`d61d47fdfe8f218ee7b6c8f28367be129ab519264948425912105d8b37aee0d0`,
`cine_v3_latent.py`
`bc089b6ab3de1f17ada008e640bca568db306a4b9d0e945f01922e3d49d8b7be`.

## Subsequent cache-contract findings, resolved

### [P2, resolved] Episode/video mapping changes did not invalidate prepared tensors

`cine_v3.py:62-63` binds root, info and task hashes, but not episode metadata.
Changing both video from/to timestamps by one source frame preserves that
identity. The prepared sample is then still accepted although the claimed RGB
window shifted. Include episode metadata hashes or a canonical per-episode
video/row mapping digest in source/payload/manifest identities. Regression:
`test_episode_video_mapping_change_invalidates_prepared_cache` fails with
`DID NOT RAISE ValueError` on the revised hashes above.

### [P2, resolved] Partial cache could claim complete all-window coverage

`cine_v3_latent.py:158,174-200` requires `complete: true` but does not validate
selection scope/grid against source episodes and sample entries.
`cine_runner.py:127,263-276` subsequently reports/records that declaration.
For a 40-frame episode and stride one, valid full-window starts are 1..7.
One cached window at start 1 labeled `all_windows` is accepted as valid.
Validate declared episode/start/count selections against entries, and full
coverage against the source's eligible window grid when scope is all-windows.
Regression: `test_partial_manifest_cannot_claim_all_windows` fails with
`DID NOT RAISE ValueError` on the revised hashes above. Both new regressions
were reproduced in **6.63s**, with four earlier tests deselected.

Closure: **15 passed in 32.30s** for `tests/test_cine_review_data.py` and
`tests/test_cine_preparation.py`. All six independent data regressions pass.
The repository now hashes the relative names and SHA256 values of all episode
metadata files into `episodes_sha256`; manifest/payload source checks consume
that identity. The native selection validator rebuilds episode/start/count
declarations and the full eligible grid for `all_windows`. The same run verifies
explicit plan-only selection, no overwrite, actual small Wan encode roundtrip
through cache/sample loading, and real local tiny Wan/UMT5/tokenizer loading.

## New-entry integration closure

The final isolated CPU checks pass:

- tests/test_cine_training_entry.py: 16 passed in 27.76s. This includes a real CineV3LatentDataset and loader, the shared runtime factory with a tiny native model, a CPU optimizer update, and full-state step-1 to step-2 resume.
- Runtime regressions: 26 passed in 54.10s. This includes the LIBERO non-torchrun guard and two-rank Gloo CAGrad synchronization.
- The broader 195-case combined regression run passed earlier. Infra's latest 102-case data suite also passed; its cases overlap the other sets, so these totals should not be added together.

The stale cine_data_contract import was removed from check-data and Gaussian action conversion. The final entry suite passed after that cleanup, so the native builder remains the source of truth for manifest selection and provenance checks.

There are no unresolved implementation blockers in the Cine training entry or its data/runtime interface. The raw producer's absolute-versus-incremental command semantics remain unverified from the available metadata and producer code; training preserves the declared seven action values without transforming them.

## Read-only real-data evidence

Read on H20 (`${COMPUTE_HOST}`, the configured read-only account) with the existing
`${READONLY_PYTHON}`, PyArrow 23.0.0. Raw action/state
inspection was limited to frames 0, 1 and 2 of episode 0 in each of two roots:
six raw rows total. Additional reads were metadata and Parquet footers, plus
SHA256 of the two corresponding small videos; no video was downloaded.
The same two videos were also decoded remotely for only their first three RGB
frames each. Both use libdav1d, average rate 30, time base 1/15360 and 399 frames.
PTS values 0/512/1024 give 0, 1/30 and 2/30 seconds, matching the sampled rows;
all six decoded arrays are 480x640x3. Only tiny textual fingerprints returned.

| Property | Observed value |
|---|---|
| Roots | `${DATA_ROOT}/train_0.6k`, `${DATA_ROOT}/validation_200` |
| Format / robot | LeRobot `v3.0` / `rm75_mujoco_cinematography` |
| Camera | Only `observation.images.color`, RGB 480x640, AV1, 30 FPS |
| State | Float32 seven values `[x,y,z,qx,qy,qz,qw]` |
| Action | Float32 seven values `joint_1` ... `joint_7` |
| Train metadata | 600 episodes, 178,718 rows; contiguous global metadata ranges |
| Validation metadata | 200 episodes, 58,737 rows; contiguous global metadata ranges |
| Sampled row times | 0, approximately 1/30 and 2/30 seconds |
| Episode 0 in both roots | Numeric ID 0, length 399, same task text, independent physical video/data |

Both roots use four data shards numbered **000, 004, 005, 006**, not a contiguous
file-number range. Train shard 004 has 55,105 rows, with global indices
87,409..142,513. Its first episode is 264, global interval `[87409,87832)`.
Its local shard slice begins at zero. A reader that directly uses global
`dataset_from_index` as the local Parquet slice will read the wrong rows or none.
The corresponding validation shard 004 begins at global index 28,592.

Current metadata has one video file per episode and all video start offsets are
zero. That establishes the current sample's layout, not correctness for generic
v3 shared videos, nonzero offsets or cross-shard episodes. Those cases need
synthetic interface tests in the adapter. Row timestamps must be interpreted
relative to the episode and added to the video's metadata start timestamp.

### Identity and leakage

The sampled train and validation episodes both use ID 0, the same length and
task prompt. Their first three raw rows differ. Their video SHA256 values are:

| Split | Video SHA256 | Physical inode on device 64529 |
|---|---|---:|
| train_0.6k | `6b724fbca5c21abe3f654a94fb381dabb31086aa9049bb6752736f2c58fc81bf` | 106848154 |
| validation_200 | `c5b355262fbdf6a105245bea11a9eb728945f35f8d8284f2854c8991aa393ba6` | 106977942 |

The associated Parquet inodes are 106848140 and 106977928. Thus numeric episode
ID intersection is not proof of leakage. Conversely, different root strings
alone do not rule out symlink aliases, hardlinks, shared physical video/time
intervals or copied trajectories. Cache identity should bind resolved roots,
metadata/encoder identity and episode/video/time ranges; reject physically
identical train/validation ranges. A full content-based split audit is not
established by this two-episode sample.

Statistics must record selected train root/episodes and be reused unchanged for
validation. Do not fit against validation or concatenate the nested train
scales as though they were disjoint.

### Command semantics

The inspected metadata supplies names and shapes, not units, control mode,
reference joint positions or a producer formula. Both sampled episodes start
with near-zero commands, which is compatible with several command protocols
and does not prove increments. State is Cartesian pose/quaternion, so its
difference cannot establish a joint-angle increment formula.

Archived read-only references under
`${ARCHIVED_REFERENCE_ROOT}` declare `joint_delta` in
`configs/data/cine_v3_video_action.yaml:5` and
`configs/task/cine_v3_joint_delta.yaml:13`. However,
`scripts/experiments/fastwam_video_action/audit_cine_v3_actions.py:89` accepts
only that expected label and distinguishes an assumed protocol (`:133`). This
is declaration/diagnostic evidence, not inspection of the raw command producer.
No corresponding RM75 data-generation implementation was found among the
checked Cine/RM75 files in that archived checkout. The absolute/incremental
producer semantics remain unverified; do not silently difference, integrate,
convert to EEF, add a gripper dimension, or relabel a declared protocol as proved.

## Minimal temporal freeze for the three implementation owners

Use one complete, episode-contained window starting at raw frame `s >= 1`.
All IDs below are episode-local and all timestamps retain 30 FPS.

| Field | Proposed frozen meaning |
|---|---|
| Raw target frames | `s..s+32`, all 33 consecutive frames, resize to 224x448 without temporal subsampling |
| Target VAE encode | Encode all 33 together; actual output must have 9 latents, `1+(33-1)/4` |
| External condition | Independently encode only frame `s-1`; exactly one condition latent |
| `batch.state` | Raw xyz/quaternion state at `s-1`, paired with the external condition |
| Native model sequence | External prefix + nine targets = 10 model-visible latent frames |
| Target state anchors | Native anchor helper gives `s,s+4,s+8,...,s+32`; retain frame granularity |
| Action shape | `[36,7]`: first four slots zero with mask zero, then raw rows `action[s:s+32]` in original order |
| Main supervised target range | Video target `[0,9)` retains the native external-prefix default; actions use `[1,9)` because target 0 is the causal singleton with masked action slots |
| Attention geometry | `history_frames=1`, `chunk_origin_frame=1`, `singleton_chunk_frame=0`; sampled chunk/window are explicit |
| Local action grid | `frame_shift=1` to account for the separate external video prefix; never use raw frame `s` as a latent-grid offset |
| Factory / inference geometry | PM accepted data/model action horizon 36 and inference chunk 9, preserving four actions per latent; training chunk 4 is separately declared |

`conditioning.py:508-512` prepends the external condition once per sample, not
one condition latent per chunk. `proprio_conditioning.py:100` prepends
`batch.state`; `:168` projects frame-level state to the immediately preceding
chunk boundary. Therefore do not broadcast a future endpoint state over the
chunk or synthesize several condition images from future targets.

For chunk size two and target chunk origin one, first future targets 1/2 use
state at raw `s`; targets 3/4 use state at `s+8`, 5/6 at `s+16`, and 7/8 at
`s+24`. The external prefix and target-0 warm-up use the prefix state at `s-1`.
These are boundary observations preceding the predicted chunk. They are not
the current chunk's endpoint state. First dummy action group is a temporal
masking device, not a fabricated eighth command dimension.

Stamp `latent_loss_frame_start/end=0/9`, `action_loss_frame_start/end=1/9`,
history one and the generic action/history range 1/9 explicitly. This retains
native video supervision after the external prefix (`sequence_layout.py:159`),
instead of accidentally removing singleton video supervision together with
the first masked action group.
Short-tail windows must be rejected or explicitly padded and masked; no implicit
cross-episode reads are acceptable. The last anchor `s+32` has no outgoing
action target inside this 32-command window.

The existing factory rejects data horizon 36 with decoder horizon 8 through
`PolicyPipelineRequirements.validate_source_action_shape`. An independent
fixture reproduced the error before model construction. PM accepted the
minimal 36/9 profile, avoiding a generic factory change. Do not claim that
training chunk four and inference chunk nine are the same chunk policy.

The independent actual small Wan VAE check passes: all 33 frames encode to nine
latents, subsampling to nine before encoding yields three, and independent
one-frame encoding yields one. Its first latent matches one-frame encoding
within FP32 rounding (initial max difference 2.98e-8); demanding bit identity
was an overly strict reviewer assertion, not a source defect.

## Independent CPU model/interface verification

Command in the existing pinned CPU environment:

```text
python -m pytest tests/test_cine_review_temporal.py -q -p no:cacheprovider
```

Result: **3 passed in 20.48s**, CUDA uninitialized. One check uses a real small
randomly initialized `AutoencoderKLWan` through the native encode helper; two
use an actual tiny native OpenWAM+VRFM+CAGrad pipeline with state7/action7,
data/model action horizon 36 and inference chunk nine. Training chunk sizes two
and four both pass. This is synthetic small-model engineering evidence, not
pretrained/full-width encoding or the new CLI's end-to-end result.

The native tests confirm model-visible video length ten, correct first four
action masks, original 32 command values, and the exact previous-boundary state
indices after the external prefix is appended. Posterior and action-projection
gradients remain the ordinary total-loss gradients after CAGrad composition;
clipped AdamW advances. Heldout evaluation refuses posterior access and has zero
KL. Real generation refuses posterior access, draws one prior latent and outputs
nine video latents plus 36 seven-channel action slots. The initial four masked
training slots are not evidence that all 36 generated commands are valid robot
control outputs; control evaluation is outside this audit.

The first native harness attempt used a nonexistent decoder-artifact attribute
for unchanged action values. This was corrected to check the input tensor;
the model's temporal/state/mask assertions had already passed. The four adapter
regression failures are separate implementation findings, not this harness error.

Additional isolated native continuation check after the shape/temporal test was
extended: **1 passed in 16.87s**. With the current state7/action7, 36/9 profile
and VRFM+CAGrad, public checkpoint restoration yields bit-identical second-step
loss, all model tensors, AdamW states, scheduler and next Torch RNG versus direct
continuation. This remains an actual-model/single-CPU checkpoint check. It does
not substitute for the separate dataset/loader/shared-runtime check above.

This proposal uses existing latent/time/proprio helpers and leaves the native
posterior/prior/gradient interfaces intact. Validation/generation must still
use the fixed prior, one latent per trajectory and ordinary posterior gradients;
CAGrad remains restricted to the accepted common generator subset. Old CPU
test counts do not certify this new adapter or CLI.

## Final review status

The implementation review is complete with no unresolved training-entry or Cine data-interface blockers. The evidence is from isolated CPU tests, tiny native model/runtime integration, and existing Gloo/distributed regressions; it does not establish full-width training throughput or closed-loop control success. Raw action values remain unchanged while their producer semantics remain unverified.
